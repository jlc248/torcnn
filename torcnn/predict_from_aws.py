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

                # Normalize
                vmin, vmax = bsinfo[varname]['vmin'], bsinfo[varname]['vmax']
                if varname == 'Velocity':
                    data_patch = np.clip(clean_patch / max(abs(vmin), abs(vmax)), -1, 1)
                    #pickle.dump(data_patch, open('velocity_pyart.pkl','wb'))
                else:
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

    return model_inputs, plot_dict

#---------------------------------------------------------------------------------------------------------
def plot_shap_maps(shap_values, input_tensor, channel_names, out_path, only_top_two=False):
    """
    Plots the spatial distribution of SHAP values for each radar channel.
    """
    # shap_values[0] corresponds to the 'radar' input [batch, H, W, Channels]
    # We take [0] for the first (and only) sample in the batch
    sv = shap_values[0][0] 
    img = input_tensor['radar'][0]
    
    n_channels = sv.shape[-1]
    
    # Calculate global importance per channel to find "Top Two"
    # We use mean absolute SHAP value as the importance metric
    importance = [np.abs(sv[:, :, i]).mean() for i in range(n_channels)]
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
        vmax = np.max(np.abs(sv[:, :, idx]))
        
        # Plot the SHAP values as a heatmap
        im = ax.imshow(sv[:, :, idx], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        
        # Optional: Overlay a contour of the original data (e.g., Reflectivity) 
        # to provide context if the channel is recognizable
        ax.set_title(f"{channel_names[idx]}\nMean |SHAP|: {importance[idx]:.4f}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()

#---------------------------------------------------------------------------------------------------------
def get_integrated_gradients(model, input_tensor, baseline=None, steps=50):
    """
    Calculates Integrated Gradients for a pixel-by-pixel importance map.
    """
    if baseline is None:
        baseline = np.zeros_like(input_tensor)

    # 1. Create interpolated images between baseline and input
    alphas = np.linspace(0, 1, steps)
    # Shape: (steps, H, W, C)
    interpolated = baseline + alphas[:, np.newaxis, np.newaxis, np.newaxis] * (input_tensor - baseline)
    interpolated = tf.cast(interpolated, tf.float32)

    # 2. Calculate gradients for all interpolated images
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        predictions = model(interpolated)
        # We want the gradient of the tornado probability
        # If your model has one output, predictions is [steps, 1]
    
    grads = tape.gradient(predictions, interpolated)

    # 3. Average the gradients and multiply by (input - baseline)
    avg_grads = tf.reduce_mean(grads, axis=0)
    integrated_grad = (input_tensor[0] - baseline[0]) * avg_grads
    
    return integrated_grad.numpy()
#---------------------------------------------------------------------------------------------------------
# --- MAIN PIPELINE ---

def process_events(event_list, model_path, config_path, out_dir, run_shap=False, top_two_only=False):
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
            tensors, plot_data = get_sector_data(radar, event['lat'], event['lon'], event['time'], hs, inputs, bsinfo)
            if tensors is None: continue
           
            # Run Prediction
            preds = conv_model.predict(tensors, verbose=0)
            prob = np.squeeze(preds)
           # print(prob);sys.exit() 
            # Generate SHAP values
            if run_shap:
               # SHAP Explainer needs a function that takes a numpy array and returns a prediction
               # def f(x):
               #     # Ensure the input is float32 for the model
               #     return conv_model.predict(x.astype(np.float32), verbose=0)

               # # Define the Masker
               # ## This tells SHAP how to "hide" pixels. We'll use a "blur" 
               # ## approach which is often better for spatial features than just solid black.
               # ## You can also use: masker = shap.maskers.Image("zeros", shape=tensors['radar'][0].shape)
               # masker = shap.maskers.Image("blur(10,10)", shape=tensors['radar'][0].shape)

               # # Create the Explainer
               # explainer = shap.Explainer(f, masker)               

               # # Calculate SHAP values
               # # max_evals: higher = better quality but slower. 500 is a good starting point.
               # # batch_size: increase this if you have a good GPU to speed up the process.
               # shap_results = explainer(tensors['radar'], max_evals=5000, batch_size=512)

               # shap_map_data = shap_results.values[0][...,0]

               # plt.imshow(shap_map_data[...,0])

                ig_map = get_integrated_gradients(conv_model, tensors['radar'], steps=50)
                vmax = ig_map[...,1].max() * 0.75
                vmin = vmax*-1
                plt.imshow(ig_map[...,1], vmin=vmin, vmax=vmax, cmap='RdBu_r')
                plt.show()
                
                # Using GradientExplainer
                ## Background (10-20 samples of zeros is standard)
                #background = np.zeros((10, 128, 256, 7))

                #explainer = shap.GradientExplainer(bda x: conv_model(x)), background)

                ## Explain (This will give you a 128x256 result immediately)
                #shap_values = explainer.shap_values(tensors['radar'])               
 
                #plt.imshow(shap_values[0,...,0])
                #plt.show()
                sys.exit()

                shap_fname = f"{out_dir}/SHAP_{event['radar']}_{event['time'].strftime('%Y%m%d_%H%M%S')}.png"
                plot_shap_maps(shap_values, tensors, channel_names, shap_fname, only_top_two=top_two_only)
                logger.info(f"Saved SHAP plot to {shap_fname}")

            # Create the Plot
            plot_channels = ['Reflectivity', 'Velocity', 'RhoHV', 'SpectrumWidth']
            fig = plt.figure(figsize=(7, 6))
            
            rad_utils.plot_radar(
                plot_data, 
                channels=plot_channels, 
                fig=fig, 
                n_rows=2,
                n_cols=2,
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
            plt.savefig(fname, dpi=200, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Saved prediction plot to {fname}")

        except Exception as e:
            logger.error(f"Failed to process event at {event['time']}: {e}", exc_info=True)

if __name__ == "__main__":
    # Your list of specific dates/locations
    my_events = [
    #    {'time': datetime(2026, 4, 2, 20, 4), 'lat': 40.95, 'lon': -92.58, 'radar': 'KDVN'},
    #    {'time': datetime(2026, 4, 2, 20, 48), 'lat': 41.29, 'lon': -91.95, 'radar': 'KDVN'},
    #    {'time': datetime(2026, 4, 2, 21, 7, 16), 'lat': 41.414, 'lon': -91.7322, 'radar': 'KDVN'},
    #    {'time': datetime(2026, 4, 2, 21, 23), 'lat': 41.51, 'lon': -91.458, 'radar': 'KDVN'},
    #    {'time': datetime(2026,5,26,0,27,47), 'lat': 31.232, 'lon':-85.314, 'radar':'KEOX'},
         {'time': datetime(2026,6,4,22,13,0), 'lat': 39.297, 'lon':-96.823, 'radar':'KTWX'},
    ]
   
    model_dir = 'static/model' 
    #model_dir = '/work2/jcintineo/torcnn/tests/2011-19/test01'
    process_events(
        my_events, 
        model_path=f'{model_dir}/fit_conv_model.keras',
        config_path=f'{model_dir}/model_config.pkl',
        out_dir='./tornado_plots/KS_20260604',
        run_shap=False,
        top_two_only=True,
    )
