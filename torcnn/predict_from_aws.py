import os, sys
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
from geopy.distance import geodesic
from geopy.point import Point

# Assuming these are imported from your rad_utils.py
import rad_utils 

logger = logging.getLogger(__name__)

# --- AWS Configuration ---
fs = s3fs.S3FileSystem(anon=True)

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
    
        best_file = min(file_dts, key=lambda x: abs(x[1] - dt))
        return f"s3://{best_file[0]}" if abs(best_file[1] - dt).seconds < 600 else None
    except:
        print('WARNING: exception when getting s3 URL')
        return None

def get_sector_data(radar, lat, lon, timestamp, hs, inputs, bsinfo):
    
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

    # We'll use the 'Velocity' sweep as the master for coordinate/mask shapes
#    v_sweep_idx = field_to_sweep.get('velocity', idx05[0])
#    v_sweep = radar.extract_sweeps([v_sweep_idx])

    # Now perform dealiasing
    t0=time.time()
    d_vel, radar_sweep = rad_utils.dealias_velocity_pyart(
                                   raddt=timestamp,
                                   radar=radar,
                                   tilt=0.5
    )
    secs = time.time()-t0
    print(f"Dealasing took {secs:.2f} seconds.")

    # Create a full-volume mask filled with the radar's default fill value
    # This matches the (11160, 1832) shape required by the radar object
    full_volume_shape = radar.fields['velocity']['data'].shape
    d_vel_full = np.ma.masked_all(full_volume_shape)

    # Identify which rays in the volume belong to the 0.5 deg sweep
    # NEXRAD often has multiple 0.5 tilts; we'll find the one that matches 'velocity'
    # We look for the sweep index where the fixed_angle is ~0.5 AND it has velocity data
    vel_sweeps = [i for i, swp in enumerate(radar.sweep_number['data']) 
                  if np.abs(radar.fixed_angle['data'][i] - 0.5) < 0.1 
                  and 'velocity' in radar.extract_sweeps([i]).fields]

    # Identify tilts
    tilts = radar.fixed_angle['data']
    idx05 = np.where(np.abs(tilts - 0.5) < 0.1)[0]

    # Filter for the one that actually contains Doppler data
    # We check 'velocity' and ensure it's not mostly empty
    vel_sweeps = []
    for i in idx05:
        swp = radar.extract_sweeps([i])
        if 'velocity' in swp.fields:
            # Check if it has actual data (not just masked)
            if np.ma.count(swp.fields['velocity']['data']) > 0:
                vel_sweeps.append(i)

    # Add the de-aliased field back to the radar object
    if vel_sweeps:
        target_swp_idx = vel_sweeps[0]
        start_ray = radar.sweep_start_ray_index['data'][target_swp_idx]
        end_ray = radar.sweep_end_ray_index['data'][target_swp_idx]
        
        # Insert the dealiased sweep data into the correct rows of the volume
        d_vel_full[start_ray : end_ray + 1, :] = d_vel
        
        # Add to the RADAR object. extract_sweeps will now "see" this field.
        radar.add_field_like('velocity', 'dealiased_velocity', d_vel_full, replace_existing=True)
    else:
        raise ValueError("Could not find a 0.5 degree sweep containing velocity.")

    # We will use this to find which sweep contains which field
    field_to_sweep = {}
    for i in idx05:
        swp_fields = radar.extract_sweeps([i]).fields.keys()
        for f in swp_fields:
            # Prefer sweeps with more data for that field
            if f not in field_to_sweep:
                field_to_sweep[f] = i
            else:
                current_cnt = np.ma.count(radar.extract_sweeps([field_to_sweep[f]]).fields[f]['data'])
                new_cnt = np.ma.count(radar.extract_sweeps([i]).fields[f]['data'])
                if new_cnt > current_cnt:
                    field_to_sweep[f] = i

     
    v_sweep = radar.extract_sweeps([field_to_sweep['velocity']])
    v_az_idx, v_gate_idx, _, _ = rad_utils.get_azimuth_range_from_pyart(lat, lon, v_sweep)

    # Process each group in 'inputs'
    for ii, inp_group in enumerate(inputs):
        X = None
        for varname in inp_group:
            
            # A. Special Channels (Masks/Range)
            if varname in ['range', 'range_inv', 'range_folded_mask', 'out_of_range_mask']:
                # Calculate these relative to the Velocity sweep coordinates
                v_patch = v_sweep.fields['velocity']['data'].data
                v_az_inds = np.arange(v_az_idx - hs[0], v_az_idx + hs[0]) % v_patch.shape[0]
                v_rng_slice = slice(v_gate_idx - hs[1], v_gate_idx + hs[1])
                local_v_patch = v_patch[v_az_inds, v_rng_slice]
                
                if varname == 'out_of_range_mask':
                    data_patch = np.full(local_v_patch.shape, 0., dtype=np.float32) #FIXME all set to 0 for now
                elif varname == 'range_folded_mask':
                    #is_range_folded  = np.isclose(local_v_patch, -64.0, atol=0.1)
                    #data_patch = is_range_folded.astype(np.float32) 
                    data_patch = np.full(local_v_patch.shape, 0., dtype=np.float32) #FIXME all set to 0 for now
                elif varname in ['range', 'range_inv']:
                    r_meters = v_sweep.range['data'][v_rng_slice] 
                    # Convert to km
                    r_patch = np.repeat(np.expand_dims(r_meters / 1000., axis=0), hs[0]*2, axis=0)
                    rmin, rmax = bsinfo['range']['vmin'], bsinfo['range']['vmax']
                    norm_r = np.clip((r_patch - rmin) / (rmax - rmin), 0, 1)
                    data_patch = norm_r if varname == 'range' else 1.0 / (norm_r + 1e-6)
            
            # B. Standard Radar Fields
            else:
                py_f = field_mapping.get(varname, varname.lower())
                target_sweep_idx = field_to_sweep.get(py_f, idx05[0])
                target_sweep = radar.extract_sweeps([target_sweep_idx])
                
                # CRITICAL: Re-calculate indices for THIS specific sweep
                local_az_idx, local_gate_idx, _, _ = rad_utils.get_azimuth_range_from_pyart(lat, lon, target_sweep)
                
                local_az_inds = np.arange(local_az_idx - hs[0], local_az_idx + hs[0]) % target_sweep.azimuth['data'].shape[0]
                local_rng_slice = slice(local_gate_idx - hs[1], local_gate_idx + hs[1])

                raw_patch = target_sweep.fields[py_f]['data'][local_az_inds, local_rng_slice]

                 # Get the mask, because the dealiasing removed it for velocity
                if py_f == 'dealiased_velocity':
                    ref_mask = target_sweep.fields['reflectivity']['data'].mask[local_az_inds, local_rng_slice]
                    # Force the velocity to be masked where reflectivity is masked
                    raw_patch = np.ma.masked_array(raw_patch, mask=ref_mask)

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
#                    pickle.dump(data_patch, open('velocity_pyart.pkl','wb'))
                else:
                    data_patch = np.clip((clean_patch - vmin) / (vmax - vmin), 0, 1)
                    if varname=='Reflectivity':
                        pickle.dump(data_patch, open('ref_pyart.pkl', 'wb'))
            #data_patch = np.flipud(np.fliplr(data_patch))
            #print(varname)
            #plt.imshow(data_patch)
            #plt.show()
            

            p4d = np.expand_dims(data_patch.astype(np.float32), axis=(0, -1))
            X = p4d if X is None else np.concatenate((X, p4d), axis=-1)
            
        model_inputs['radar' if ii == 0 else 'coords'] = X

    return model_inputs, plot_dict

# --- MAIN PIPELINE ---

def process_events(event_list, model_path, config_path, out_dir):
    # Load Model & Config
    conv_model = keras.models.load_model(model_path, compile=False)
    config = pickle.load(open(config_path, 'rb'))
    bsinfo = config['byte_scaling_vals']
    inputs = config['inputs']
    ps = config['ps']
    hs = (ps[0]//2, ps[1]//2)
    
    os.makedirs(out_dir, exist_ok=True)

    for event in event_list:
     
        s3_url = find_s3_nexrad(event['radar'], event['time'])
        if not s3_url: continue
       
        try:
            with fs.open(s3_url, 'rb') as f:
                # Load the WHOLE radar object
                radar = pyart.io.read_nexrad_archive(f)

            # 1. Extract Scaled Tensor and Unscaled Plot Data
            tensors, plot_data = get_sector_data(radar, event['lat'], event['lon'], event['time'], hs, inputs, bsinfo)
            if tensors is None: continue
           
            # 2. Run Prediction
            preds = conv_model.predict(tensors, verbose=0)
            prob = np.squeeze(preds)
           
            # 3. Create the Plot
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
        {'time': datetime(2026, 4, 2, 21, 7, 16), 'lat': 41.414, 'lon': -91.7322, 'radar': 'KDVN'},
    #    {'time': datetime(2026, 4, 2, 21, 23), 'lat': 41.51, 'lon': -91.458, 'radar': 'KDVN'},
    #    {'time': datetime(2026, 4, 2, 21, 36), 'lat': 41.61, 'lon': -91.37, 'radar': 'KDVN'},
    ]
   
    model_dir = 'static/model' 
    #model_dir = '/work2/jcintineo/torcnn/tests/2011-19/test01'
    process_events(
        my_events, 
        model_path=f'{model_dir}/fit_conv_model.keras',
        config_path=f'{model_dir}/model_config.pkl',
        out_dir='./tornado_plots'
    )
