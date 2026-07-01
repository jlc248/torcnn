import os, sys
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import time
import s3fs
import pyart
import numpy as np
import pandas as pd
import keras
import pickle
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import rad_utils 
import shap
import joblib
import tensorflow as tf

logger = logging.getLogger(__name__)

# --- AWS Configuration ---
fs = s3fs.S3FileSystem(anon=True)
#-------------------------------------------------------------------------------------------------------
def find_s3_nexrad(radar, dt):
    """Search NOAA S3 bucket for the closest L2 file."""
    bucket = "unidata-nexrad-level2"
    path = f"{bucket}/{dt.year}/{dt.month:02d}/{dt.day:02d}/{radar.upper()}/"
    try:
        files = fs.ls(path)
        if not files: return None
        file_dts = []
        for f in files:
            parts = os.path.basename(f).split('_')
            if len(parts) < 3: continue
            f_dt = datetime.strptime(f"{parts[0][4:]}_{parts[1]}", "%Y%m%d_%H%M%S")
            file_dts.append((f, f_dt))
   
        # Filter for files that occurred at or BEFORE the target datetime
        before_dt = [x for x in file_dts if x[1] <= dt]
 
        if not before_dt: 
            # If no files exist before dt on this day, you might want to 
            # search the previous day's folder, but for now we return None.
            return None

        # Pick the one with the maximum time (the "latest" of the "before" files)
        best_file = max(before_dt, key=lambda x: x[1])

        time_diff = (dt - best_file[1]).total_seconds()
        return f"s3://{best_file[0]}" if time_diff < 600 else None

    except:
        print('WARNING: exception when getting s3 URL')
        return None

#-------------------------------------------------------------------------------------------------------
def get_sweep_indexes(radar: pyart.core.radar.Radar,
                       vel_dt: datetime, 
) -> tuple[int, int]:
    """
    Use the velocity sweep datetime to determine the sweeps of the other
    fields in the pyart object. Works for split-cut or normal NEXRAD VCPs.
    Args:
        radar: pyart radar object with every field
        vel_dt: datetime of the desired velocity sweep. 
    Returns:
        z_idx: index of the Z fields
        v_idx: index of the V fields 
    """

    # 1. Find all tilts near 0.5 degrees
    idx05 = np.where(np.abs(radar.fixed_angle['data'] - 0.5) < 0.2)[0]
    all_times = pyart.util.datetimes_from_radar(radar)
    
    # 2. Map every 0.5 tilt to its Nyquist value
    sweep_nyquists = {}
    for i in idx05:
        ray_idx = radar.sweep_start_ray_index['data'][i]
        sweep_nyquists[i] = radar.instrument_parameters['nyquist_velocity']['data'][ray_idx]
    
    # 3. Find the Doppler candidate (Anchor) closest to vel_dt
    # We define 'Doppler-capable' as any sweep that isn't the 'low-nyquist' partner
    # Logic: if there are two sweeps close in time, the one with the higher Nyquist is V
    time_diffs = []
    for i in idx05:
        start_idx = radar.sweep_start_ray_index['data'][i]
        diff = abs((all_times[start_idx] - vel_dt).total_seconds())
        time_diffs.append(diff)

    # This is the sweep that matches your timestamp
    v_idx = idx05[np.argmin(time_diffs)]
    
    # Extract and check
    radar_sweep = radar.extract_sweeps([v_idx])
    velocity_data = radar_sweep.fields['velocity']['data']
    if np.ma.count(velocity_data) == 0:
        # If the data is empty, we can't dealias
        # Check idx+1
        v_idx += 1
        radar_sweep = radar.extract_sweeps([v_idx])
        velocity_data = radar_sweep.fields['velocity']['data']
        if np.ma.count(velocity_data) == 0:
            # If the model NEEDS dealiased data, we raise.
            raise ValueError(f'Sweep {tilt_ind} has 0 valid velocity gates.')

    # 4. Now find the BEST Reflectivity partner for this specific v_idx
    # Check if there is a sweep close by with a LOWER Nyquist
    v_time = all_times[radar.sweep_start_ray_index['data'][v_idx]]
    v_nyq = sweep_nyquists[v_idx]
    
    z_idx = v_idx # Default to same sweep (Non-split cut)
    
    for i in idx05:
        if i == v_idx: continue
        i_time = all_times[radar.sweep_start_ray_index['data'][i]]
        i_nyq = sweep_nyquists[i]
        
        # If this other sweep is within 30 seconds AND has a lower Nyquist,
        # it is the high-sensitivity Surveillance cut we want for Z.
        if abs((v_time - i_time).total_seconds()) < 30 and i_nyq < (v_nyq - 2.0):
            z_idx = i
            break

    return z_idx, v_idx
#-------------------------------------------------------------------------------------------------------
def get_sector_data(radar, lat, lon, raddt, hs, inputs, bsinfo):
    
    # Variable Mapping
    field_mapping = {
        'Reflectivity': 'reflectivity',
        'Velocity': 'dealiased_velocity',
        'SpectrumWidth': 'spectrum_width',
        'RhoHV': 'cross_correlation_ratio',
        'Zdr': 'differential_reflectivity'
    }

    container = {}
    model_inputs = {}
    plot_dict = {}

    # Now perform dealiasing
    t0=time.time()
    d_vel, radar_with_dvel, _ = rad_utils.dealias_velocity_pyart(
                         raddt=raddt,
                         radar=radar,
                         tilt=0.5
    )
    
    secs = time.time()-t0
    print(f"Dealasing took {secs:.2f} seconds.")
    
    # Find the Surveillance sweep that belongs with it
    # In NEXRAD Split-Cut, Surveillance (Z) usually immediately precedes Doppler (V)
    z_idx, v_idx = get_sweep_indexes(radar, raddt)
   
    field_to_sweep = {
        'reflectivity': z_idx,
        'cross_correlation_ratio': z_idx,
        'differential_reflectivity': z_idx,
        'velocity': v_idx,
        'dealiased_velocity': v_idx,
        'spectrum_width': v_idx,
    }    

    # Used for range, range_folded_mask, etc. 
    v_sweep = radar.extract_sweeps([v_idx])
    v_az_idx, v_gate_idx, _, _ = rad_utils.get_azimuth_range_from_pyart(lat, lon, v_sweep)

    # Process each group in 'inputs'
    for ii, inp_group in enumerate(inputs):
        X = None
        for varname in inp_group:
            
            # A. Special Channels (Masks/Range)
            if varname in ['range', 'range_inv', 'range_folded_mask', 'out_of_range_mask']:
                # Calculate these relative to the Velocity sweep coordinates
                v_data = v_sweep.fields['velocity']['data'] #.data
                naz, ngates = v_data.shape
                v_az_inds = np.arange(v_az_idx - hs[0], v_az_idx + hs[0]) % naz
                v_rng_slice = slice(v_gate_idx - hs[1], v_gate_idx + hs[1])
               
                if varname == 'out_of_range_mask':
                    if v_rng_slice.start < 0: # Fix if too close to radar, fill with 0s
                        nfill_rng = v_rng_slice.start * -1
                        out_of_range = np.ones((hs[0]*2, nfill_rng), dtype=np.float32)
                        in_range = np.zeros((hs[0]*2, hs[1]*2 - nfill_rng), dtype=np.float32)
                        data_patch = np.concatenate((out_of_range, in_range), axis=1)
                    elif v_rng_slice.stop > ngates: # Fix if too from radar, fill with 0s
                        nfill_rng = v_rng_slice.stop - ngates
                        in_range = np.zeros((hs[0]*2, ngates - v_rng_slice.start), dtype=np.float32)
                        out_of_range = np.zeros((hs[0]*2, nfill_rng), dtype=np.float32)
                        data_patch = np.concatenate((in_range, out_of_range), axis=1)
                    else:
                        data_patch = np.full((hs[0]*2, hs[1]*2), 0., dtype=np.float32)

                elif varname == 'range_folded_mask':
                    is_range_folded  = np.isclose(v_data.data, -64.0, atol=0.1).astype(np.float32)
                    if v_rng_slice.start < 0: # Fix if too close to radar, fill with 0s
                        nfill_rng = v_rng_slice.start * -1
                        v_rng_slice = slice(0, v_gate_idx + hs[1])
                        partial_patch = is_range_folded[v_az_inds, v_rng_slice]
                        data_patch = np.concatenate([np.zeros((hs[0]*2, nfill_rng), dtype=np.float32), partial_patch], axis=1)
                    elif v_rng_slice.stop > ngates: # Fix if too from radar, fill with 0s
                        partial_patch = is_range_folded[v_az_inds, v_rng_slice.start:ngates]
                        nfill = np.zeros((v_az_inds, v_rng_slice.stop - ngates), dtype=np.float32)
                        data_patch = np.concatenate([partial_patch, nfill], axis=1)
                    else:
                        data_patch = is_range_folded[v_az_inds, v_rng_slice]

                elif varname in ['range', 'range_inv']:
                    if v_rng_slice.start < 0: # Fix if too close to radar, fill with 0s
                        nfill = np.zeros((v_rng_slice.start*-1))
                        partial_patch = v_sweep.range['data'][0:v_rng_slice.stop]
                        r_meters = np.concatenate((nfill, partial_patch))
                        
                    elif v_rng_slice.stop > ngates: # Fix if too from radar, fill with 0s
                        partial_patch = v_sweep.range['data'][v_rng_slice.start:ngates]
                        nfill = np.zeros((v_rng_slice.stop - ngates))
                        r_meters = np.concatenate((partial_patch, nfill))
                    else:
                        r_meters = v_sweep.range['data'][v_rng_slice]

                    # Convert to km
                    r_patch = np.repeat(np.expand_dims(r_meters / 1000., axis=0), hs[0]*2, axis=0)
                    rmin, rmax = bsinfo['range']['vmin'], bsinfo['range']['vmax']
                    norm_r = np.clip((r_patch - rmin) / (rmax - rmin), 0, 1)
                    data_patch = norm_r if varname == 'range' else 1.0 / (norm_r + 1e-6)
            
            # B. Standard Radar Fields
            else:
                py_f = field_mapping[varname]
                target_sweep_idx = field_to_sweep[py_f]
                target_sweep = radar.extract_sweeps([target_sweep_idx])
 
                # CRITICAL: Re-calculate indices for THIS specific sweep
                local_az_idx, local_gate_idx, _, _ = rad_utils.get_azimuth_range_from_pyart(lat, lon, target_sweep)

                # Get the patch
                n_rays = target_sweep.azimuth['data'].shape[0]
                local_az_inds = np.arange(local_az_idx - hs[0], local_az_idx + hs[0]) % n_rays
                local_rng_slice = slice(local_gate_idx - hs[1], local_gate_idx + hs[1])
                # For zoom-in (for imaging)
                # Hard-coded h[0]//2 and h[1]//s
                local_az_inds_zoom = np.arange(local_az_idx - hs[0]//2, local_az_idx + hs[0]//2) % n_rays
                local_rng_slice_zoom = slice(local_gate_idx - hs[1]//2, local_gate_idx + hs[1]//2)

                # Fix if too close to radar, fill with 0s
                if local_rng_slice.start < 0:
                    nfill_rng = local_rng_slice.start * -1
                    local_rng_slice = slice(0, local_gate_idx + hs[1])
                    partial_patch = target_sweep.fields[py_f]['data'][local_az_inds, local_rng_slice]
                    raw_patch = np.concatenate([np.zeros((hs[0]*2, nfill_rng), dtype=np.float32)-99900.0, partial_patch], axis=1)
                else:
                    raw_patch = target_sweep.fields[py_f]['data'][local_az_inds, local_rng_slice]
              
                 # Get the mask, because the dealiasing removed it for velocity
               # if py_f == 'dealiased_velocity':
               #     #ref_mask = target_sweep.fields['reflectivity']['data'].mask[local_az_inds, local_rng_slice]
               #     # Force the velocity to be masked where reflectivity is masked
               #     raw_patch = d_vel[local_az_inds, local_rng_slice]
               #     plt.imshow(d_vel)
               #     plt.show()
               #     plt.close()
               #     plt.imshow(raw_patch)
               #     plt.show(); sys.exit()
               #     raw_patch = np.ma.masked_array(raw_patch, mask=ref_mask)

                clean_patch = np.ma.filled(raw_patch, fill_value=-99900.0)
 
                # Store for plotting (using master plot_dict)
                if varname not in plot_dict:
                    plot_dict[varname] = {'data': clean_patch}
                    if ii == 0 and varname == 'Reflectivity': # Use Z sweep for plot coords
                        plot_dict['az_lower'] = target_sweep.azimuth['data'][local_az_inds[0]]
                        plot_dict['az_upper'] = target_sweep.azimuth['data'][local_az_inds[-1]]
                        plot_dict['rng_lower'] = target_sweep.range['data'][local_rng_slice].min()/1000.
                        plot_dict['rng_upper'] = target_sweep.range['data'][local_rng_slice].max()/1000.
                        # For zoom-in
                        plot_dict['az_lower_zoom'] = target_sweep.azimuth['data'][local_az_inds_zoom[0]]
                        plot_dict['az_upper_zoom'] = target_sweep.azimuth['data'][local_az_inds_zoom[-1]]
                        plot_dict['rng_lower_zoom'] = target_sweep.range['data'][local_rng_slice_zoom].min()/1000.
                        plot_dict['rng_upper_zoom'] = target_sweep.range['data'][local_rng_slice_zoom].max()/1000.

                # Normalize
                vmin, vmax = bsinfo[varname]['vmin'], bsinfo[varname]['vmax']
                if varname == 'Velocity':
                    data_patch = np.clip(clean_patch / max(abs(vmin), abs(vmax)), -1, 1)
                    missing_mask = (clean_patch == -99900).astype(bool) 
                    #pickle.dump(data_patch, open('velocity_pyart.pkl','wb'))
                else:
                #    print(varname, vmin, vmax)
                #    plt.imshow(clean_patch)
                #    plt.show()
                    data_patch = np.clip((clean_patch - vmin) / (vmax - vmin), 0, 1)
                    #if varname=='Reflectivity':
                    #    pickle.dump(data_patch, open('ref_pyart.pkl', 'wb'))
               
            #data_patch = np.flipud(np.fliplr(data_patch))
            #print(varname)
            #plt.imshow(data_patch)
            #plt.show()
            

            p4d = np.expand_dims(data_patch.astype(np.float32), axis=(0, -1))
            X = p4d if X is None else np.concatenate((X, p4d), axis=-1)
            
        model_inputs['radar' if ii == 0 else 'coords'] = X

    return model_inputs, plot_dict, missing_mask

#---------------------------------------------------------------------------------------------------------
def plot_importance_maps(importance_values, input_tensor, channel_names, out_path, only_top_two=False):
    """
    Plots the spatial distribution of SHAP values for each radar channel.
    """
    
    img = input_tensor['radar'][0]
    
    n_channels = importance_values.shape[-1]
    
    # Calculate global importance per channel to find "Top Two"
    # We use mean absolute SHAP value as the importance metric
    importance = [np.abs(importance_values[:, :, i]).mean() for i in range(n_channels)]
    indices = np.argsort(importance)[::-1] # Sort descending
    
    if only_top_two:
        indices = indices[:2]
        n_plots = 2
    else:
        n_plots = n_channels

    fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 5, 5), squeeze=False)
    
    for i, idx in enumerate(indices):
        ax = axes[0, i]
        # Calculate a symmetric vmin/vmax for the SHAP heatmap
        vmax = np.max(np.abs(importance_values[:, :, idx])) * 0.75
        
        # Plot the SHAP values as a heatmap
        im = ax.imshow(importance_values[:, :, idx], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        
        # Optional: Overlay a contour of the original data (e.g., Reflectivity) 
        # to provide context if the channel is recognizable
        ax.set_title(f"{channel_names[idx]}\nMean |IMPORTANCE|: {importance[idx]:.4f}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

#---------------------------------------------------------------------------------------------------------
def get_integrated_gradients(model, input_tensor, baseline=None, steps=50, target_logits=True):
    """
    Calculates Integrated Gradients for a pixel-by-pixel and channel-by-channel importance map.
    Handles Keras 3 / TF tensors safely without silent broadcasting bugs.
    """
    # 1. Ensure inputs have a batch dimension (1, H, W, C) for consistency
    if len(input_tensor.shape) == 3:
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
    if baseline is None:
        baseline = np.zeros_like(input_tensor)
    elif len(baseline.shape) == 3:
        baseline = np.expand_dims(baseline, axis=0)

    # Prepare logit model if requested to avoid sigmoid saturation
    if target_logits and hasattr(model.layers[-1], 'activation') and model.layers[-1].activation.__name__ == 'sigmoid':
        # Reconstruct a temporary model that outputs the raw logits before sigmoid
        logit_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-1].input)
    else:
        logit_model = model

    # 2. Create interpolated images along the path
    alphas = np.linspace(0, 1, steps, dtype=np.float32)
    
    # Calculate the true delta between the full image and full baseline
    delta = input_tensor[0] - baseline[0]  # Shape: (H, W, C)
    
    # Generate path: Shape (steps, H, W, C)
    interpolated = baseline[0] + alphas[:, np.newaxis, np.newaxis, np.newaxis] * delta
    interpolated = tf.cast(interpolated, tf.float32)

    # 3. Calculate gradients along the path
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        predictions = logit_model(interpolated) # Shape: (steps, 1)

    grads = tape.gradient(predictions, interpolated) # Shape: (steps, H, W, C)

    # 4. Average gradients and multiply by the true delta
    avg_grads = tf.reduce_mean(grads, axis=0) # Shape: (H, W, C)
    integrated_grad = delta * avg_grads        # Shape: (H, W, C) - Safe element-wise math

    return integrated_grad.numpy()
#---------------------------------------------------------------------------------------------------------
# --- MAIN PIPELINE ---

def process_events(event_list, model_path, config_path, out_dir, calibrator=None, run_importance=False, top_two_only=False):
    # If calibrator, load
    if calibrator:
        calibrator_obj = joblib.load(calibrator)    

    # Load Model & Config
    conv_model = keras.models.load_model(model_path, compile=False)
    config = pickle.load(open(config_path, 'rb'))
    bsinfo = config['byte_scaling_vals']
    inputs = config['inputs']
    channel_names = config['channels']
    ps = config['ps']
    hs = (ps[0]//2, ps[1]//2)
    
    os.makedirs(out_dir, exist_ok=True)

    # --- Initialize SHAP Explainer ---
    # We use a small background of zeros or a representative sample. 
    # Since radar data is normalized, zeros often represent "no signal".
    background = np.zeros((1, ps[0], ps[1], len(channel_names)))
    explainer = shap.GradientExplainer(conv_model, background)

    for event in event_list:
     
        s3_url = find_s3_nexrad(event['radar'], event['time'])
        if not s3_url: continue
       
        try:
            with fs.open(s3_url, 'rb') as f:
                # Load the WHOLE radar object
                radar = pyart.io.read_nexrad_archive(f)

            # Extract Scaled Tensor and Unscaled Plot Data
            tensors, plot_data, missing_mask = get_sector_data(radar, event['lat'], event['lon'], event['time'], hs, inputs, bsinfo)
            if tensors is None: continue
           
            # Run Prediction
            preds = conv_model.predict(tensors, verbose=0)
            prob1 = prob = np.squeeze(preds)
           
            if calibrator:
                print(prob1)
                prob2 = prob = calibrator_obj.predict(np.array([prob1]))[0]
                print(prob2)

            # This is for a one-off run to get the background  
            #for ii in range(len(inputs[0])):
            #    plt.imshow(tensors['radar'][0,...,ii])
            #    plt.show()
            #pickle.dump(tensors['radar'], open('ig_background_Z-V-Rho-SW-range-mask.pkl','wb'))
            #print(tensors['radar'].shape)
            #sys.exit()

            # Generate importance values
            if run_importance:

                baseline = pickle.load(open('static/ig_background_Z-V-Rho-SW-range-mask.pkl', 'rb'))
                ig_map = get_integrated_gradients(conv_model,
                                                  tensors['radar'],
                                                  baseline=None, #baseline,
                                                  steps=50,
                                                  target_logits=True
                )
                # Add IGs to plot_data
                for kk, inp in enumerate(inputs):
                    for ii, chan in enumerate(inp):
                        if kk == 0:  
                            try:
                                plot_data[f"{chan}_importance"] = {"data":ig_map[..., ii]}
                            except KeyError:
                                pass
                
                # 2. Sum up everything (all pixels, all NEXRAD channels)
                #total_ig_sum = np.sum(ig_map)
                
                # 3. Get the actual model predictions
                #prob_baseline = np.squeeze(conv_model.predict(np.zeros_like(tensors['radar'])))
                #prob_diff = prob1 - prob_baseline
                
                #print(f'sum of all IGs:  {total_ig_sum:.4f}')
                #print(f'Actual Prob Diff: {prob_diff:.4f}')
                # These two numbers should be nearly identical (ignoring minor approximation errors from step size)!
                #sys.exit()

                #vmax = ig_map[...,1].max() * 0.75
                #vmin = vmax*-1
                #plt.imshow(ig_map[...,1], vmin=vmin, vmax=vmax, cmap='RdBu_r')
                #plt.show()
                
                # Plot the IGs
                imp_fname = f"{out_dir}/IG_{event['radar']}_{event['time'].strftime('%Y%m%d_%H%M%S')}.png"
                #plot_importance_maps(ig_map, tensors, channel_names, shap_fname, only_top_two=top_two_only)
                fig = plt.figure(figsize=(14, 6)) # (14,6) for 4 vars, (7,6) for 2 vars
                plot_channels = ['Reflectivity', 'Velocity', 'RhoHV', 'SpectrumWidth', 'Reflectivity_importance',
                                 'Velocity_importance', 'RhoHV_importance', 'SpectrumWidth_importance']
                rad_utils.plot_radar(
                    plot_data,
                    channels=plot_channels,
                    fig=fig,
                    n_rows=2,
                    n_cols=4,
                    missing_mask=missing_mask,
                    zoom_in=True,
                    include_title=False,
                    include_cbar=False,
                    full_ppi=False,
                )

                # Add title
                time_str = event['time'].strftime('%Y-%m-%d %H:%M')
                plt.suptitle(
                    f"Radar: {event['radar']} | {time_str} UTC\n"
                    f"Lat: {event['lat']}, Lon: {event['lon']} | Tornado Prob: {prob:.1%}",
                    fontsize=12, y=0.97
                )

                # Save Image
                plt.savefig(imp_fname, dpi=300, bbox_inches='tight')
                logger.info(f"Saved importance plot to {imp_fname}")


            # Create the Plot
            plot_channels = ['Reflectivity', 'Velocity', 'RhoHV', 'SpectrumWidth']
            fig = plt.figure(figsize=(7, 6))
            rad_utils.plot_radar(
                plot_data, 
                channels=plot_channels, 
                fig=fig, 
                n_rows=2,
                n_cols=2,
                missing_mask = missing_mask,
                include_title=False,
                include_cbar=True, 
                full_ppi=False
            )
            
            # Add dynamic title with probability
            time_str = event['time'].strftime('%Y-%m-%d %H:%M')
            plt.suptitle(
                f"Radar: {event['radar']} | {time_str} UTC\n"
                f"Lat: {event['lat']}, Lon: {event['lon']} | Tornado Prob: {prob:.1%}",
                fontsize=12, y=0.95
            )
            
            # Save Image
            fname = f"{out_dir}/{event['radar']}_{event['time'].strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Saved prediction plot to {fname}")

        except Exception as e:
            logger.error(f"Failed to process event at {event['time']}: {e}", exc_info=True)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


    # Your list of specific dates/locations
    my_events = [
    #    {'time': datetime(2026, 4, 2, 20, 4), 'lat': 40.95, 'lon': -92.58, 'radar': 'KDVN'},
    #    {'time': datetime(2026, 4, 2, 20, 48), 'lat': 41.29, 'lon': -91.95, 'radar': 'KDVN'},
    #    {'time': datetime(2026, 4, 2, 21, 7, 16), 'lat': 41.414, 'lon': -91.7322, 'radar': 'KDVN'},
    #    {'time': datetime(2026, 4, 2, 21, 23), 'lat': 41.51, 'lon': -91.458, 'radar': 'KDVN'},
    #    {'time': datetime(2026,5,26,0,27,47), 'lat': 31.232, 'lon':-85.314, 'radar':'KEOX'},
    #     {'time': datetime(2026,6,4,22,13,0), 'lat': 39.297, 'lon':-96.823, 'radar':'KTWX'},
         # Enid OK tornado
         {'time': datetime(2026,4,24,1,1,30), 'lat':36.316, 'lon':-97.976, 'radar':'KVNX'},
         {'time': datetime(2026,4,24,1,3,4), 'lat':36.326, 'lon':-97.942, 'radar':'KVNX'},
         {'time': datetime(2026,4,24,1,4,45), 'lat':36.3218, 'lon':-97.957, 'radar':'KVNX'},
         {'time': datetime(2026,4,24,1,6,39), 'lat':36.32, 'lon':-97.942, 'radar':'KVNX'},
         {'time': datetime(2026,4,24,1,7,58), 'lat':36.325, 'lon':-97.934, 'radar':'KVNX'},
         {'time': datetime(2026,4,24,1,9,32), 'lat':36.326, 'lon':-97.929, 'radar':'KVNX'},
         {'time': datetime(2026,4,24,1,11,32), 'lat':36.331, 'lon':-97.921, 'radar':'KVNX'},
         {'time': datetime(2026,4,24,1,12,58), 'lat':36.332, 'lon':-97.912, 'radar':'KVNX'}, 
         {'time': datetime(2026,4,24,1,14,14), 'lat':36.333, 'lon':-97.901, 'radar':'KVNX'},
         {'time': datetime(2026,4,24,1,15,46), 'lat':36.333, 'lon':-97.896, 'radar':'KVNX'},
         {'time': datetime(2026,4,24,1,17,24), 'lat':36.336, 'lon':-97.893, 'radar':'KVNX'},
         {'time': datetime(2026,4,24,1,19,18), 'lat':36.343, 'lon':-97.881, 'radar':'KVNX'},
         {'time': datetime(2026,4,24,1,20,37), 'lat':36.347, 'lon':-97.873, 'radar':'KVNX'},
    #     {'time': datetime(2026,6,30,15,40), 'lat':40.277, 'lon':-86.127, 'radar':'KIND'}, # IG BASELINE
    ]
   
    #model_dir = 'static/model' 
    #calibrator = None
    model_dir = '/work2/jcintineo/torcnn/tests/2011-19/test07'
    calibrator = '/work2/jcintineo/torcnn/tests/2011-19/test07/tornado_calibrator_isotonic.joblib'
    process_events(
        my_events, 
        model_path=f'{model_dir}/fit_conv_model.keras',
        config_path=f'{model_dir}/model_config.pkl',
        out_dir='./tornado_plots/OK_20260424',
        calibrator=calibrator,
        run_importance=True,
        top_two_only=True, #non functional
    )
