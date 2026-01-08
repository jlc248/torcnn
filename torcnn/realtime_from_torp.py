import sys,os
import numpy as np
import pandas as pd
import tensorflow as tf
import glob
import argparse
import pickle
from generate_tfrecs import get_sector_patch
import utils
import time
from datetime import datetime, timedelta
import xarray as xr
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)
OUT_OF_RANGE = -999

#----------------------------------------------------------------------------------
def find_new_files(listened_file, processed_list):
    """
    Purge old files in the processed list and retrieve new filenames.

    Example line in listened file:
    <log time info> New File: /sas8tb/localdata_MRMS/realtime/radar/KSJT/TORPcsvShort/20260107-163041_KSJT_0050_tordetections_short.csv

    Args:
    - listened_file (str): The log file where new TORP files are logged.
    - processed_list (list): A list of files that have already been processed.

    Returns:
    - new_files (list): A list of new files to process.
    processed_list also gets returned modified, potentially
    """

    now = datetime.utcnow()
    expired = now - timedelta(minutes=12)

    # Get new files
    new_files = []
    search_str = "New File: "
    with open(listened_file, 'r') as f:
        new_files = [line.strip().split(search_str)[1] for line in f if search_str in line]
   
    # new_files[:] creates a new list for iteration, leaving the original free for modification 
    for new_file in new_files[:]: 
        dt = datetime.strptime(os.path.basename(new_file)[0:15], '%Y%m%d-%H%M%S')
        if dt < expired or new_file in processed_list:
            # If too old or already-processed that don't include it
            new_files.remove(new_file)
           
        elif dt < expired and new_file in processed_list:
            # Purge old files from processed list so that it doesn't grow too large
            processed_list.remove(new_file)
 
    return new_files


#----------------------------------------------------------------------------------

def make_prediction_tensors(config, new_files, processed_list):

    """
    Using the list of new_files, get the actual NEXRAD data and create a tensor
    ready for predictions.

    Args:
    - config (dict): model config dictionary, so we know which channels to get
    - new_files (list): A list of full paths to new TORP detects (per-radar per-time)
    - processed_list (list): The list to add each new_file to if it was successfully processed.

    Returns:
    - tensor_list (numpy.ndarray): tensor ndarray; shape = (ndetects, ny, nx, nchannels)
    """
    
    bsinfo = config['byte_scaling_vals']
    tilt = '00.50'
    n_az, n_gates = config['ps']
    hs = (n_az // 2, n_gates // 2)
    channels = config['channels']
    nchan = len(channels)

    # Dict for holding the tensor, lats, and lons for each torpfile
    samples=1
    # One tensor per torpfile, which will be aggregated at the end
    tensor_list = []
 
    for torpfile in new_files:

        # Get the detect locations
        df = pd.read_csv(torpfile)
        lats = df.lat
        lons = df.lon
        if np.isnan(lats).any() or np.isnan(lons).any():
            logger.warning(f'Missing coords in {torpfile}. Skipping.')
            continue

        # Create a numpy array for the tensor for the detects in THIS torpfile
        nsamples = len(lats)
        tensor = np.zeros((nsamples, n_az, n_gates, nchan), dtype=np.float32)

        # We need to find the WDSS2 netcdfs from the TORP detects time.
        # TORP detects time is the scan time of the velocity data, which 
        # can often be after the file time of the reflectivity data.
        # Assume Velocity is always used.

        vel_dt = datetime.strptime(os.path.basename(torpfile)[0:15], '%Y%m%d-%H%M%S')

        # Root dir for data
        rootdir = os.path.dirname(torpfile).split('TORPcsv')[0]

        for chIdx,chan in enumerate(channels):

            if chan in ['range', 'range_inv', 'range_folded_mask', 'out_of_range_mask']:
                # We handle these while doing the first channel, typically Ref or Vel
                pass
            else:
                # Normal radar moments

                # Find the closest file to vel_dt
                all_files = glob.glob(f"{rootdir}/{chan}/{tilt}/*netcdf")
                dts = [datetime.strptime(os.path.basename(ff), '%Y%m%d-%H%M%S.netcdf') for ff in all_files]
                closest_dt = min(dts, key=lambda dt: abs(vel_dt - dt))
                if abs(vel_dt - closest_dt).seconds > 180:
                    print(f"{vel_dt} is too far from {closest_dt}")
                    sys.exit(1)
                idx = dts.index(closest_dt)
                file_path = all_files[idx]

                # Open the netCDF file using xarray
                radds = xr.open_dataset(file_path)

                # Get the patches for each detect, and add them to the proper slices in tensor
                for ii in range(len(lats)):
             
                    rad_sector, theta_sector, r_sector, az_idx, gate_idx = get_sector_patch(lat=lats[ii],
                                                                                            lon=lons[ii],
                                                                                            ds=radds,
                                                                                            hs=hs,
                                                                                            varname=chan
                    )

                    # Assert that the shape is correct
                    try:
                        assert(rad_sector.shape == (hs[0]*2, hs[1]*2))
                    except AssertionError as err:
                        logger.warning(f'rad_sector.shape == {rad_sector.shape}, but should = {(hs[0]*2, hs[1]*2)}')
                        # Remove this slice from the tensor and move on                
                        tensor = np.delete(tensor, ii, axis=0)
                        continue

                    if chIdx == 0: 
                        # Get range and range_inv on the first channel iteration. Also, make out_of_range_mask.
                        ## 'range' is 1D, but we need it to be 2D. Expand to number of azimuths
                        if 'range' in channels or 'out_of_range_mask' in channels:
                            range_data = np.repeat(np.expand_dims(r_sector, axis=0), hs[0]*2, axis=0)
                            range_data = utils.bytescale(range_data,
                                                         bsinfo['range']['vmin'],
                                                         bsinfo['range']['vmax'],
                                                         min_byte_val=1, # we also invert range, so keep it > 0
                                                         max_byte_val=255
                            ).astype(np.float32)

                            # Add data to tensor
                            if 'range' in channels:
                                chan_slice = channels.index('range')
                                tensor[ii, ..., chan_slice] = range_data    
                            if 'range_inv' in channels: 
                                chan_slice = channels.index('range_inv')
                                tensor[ii, ..., chan_slice] = 1. / range_data
                            if 'out_of_range_mask' in channels:
                                chan_slice = channels.index('out_of_range_mask')
                                tensor[ii, ..., chan_slice] = (rad_sector == OUT_OF_RANGE).astype(np.float32)

                    # Encode range-folded region
                    if 'range_folded_mask' in channels and chan == 'Velocity':
                        range_folded_value = radds.attrs.get('RangeFolded', -99901.0)
                        chan_slice = channels.index('range_folded_mask')
                        # Add data to tensor
                        tensor[ii, ..., chan_slice] = (rad_sector == range_folded_value).astype(np.float32)


                    # Bytescale the moment data data
                    rad_sector_scaled = utils.bytescale(rad_sector,
                                                        bsinfo[chan]['vmin'],
                                                        bsinfo[chan]['vmax'],
                                                        min_byte_val=0,
                                                        max_byte_val=255
                    ) 

                    # Add moment data to tensor
                    tensor[ii, ..., chIdx] = rad_sector_scaled

        # The tensor for torpfile is ready
        tensor_list.append(tensor)

    logger.info('Created tensors')

    return tensor_list
#----------------------------------------------------------------------------------

def run_model(listened_file,
              model_file,
              outpatt,
):

    # One-time overhead
    ## Read TF model
    model = tf.keras.models.load_model(model_file, compile=False)
    ## Get the config file. Assume it's in the same directory.
    config = pickle.load(open(f'{os.path.dirname(model_file)}/model_config.pkl', 'rb'))
    ## Just a list to keep log of files we've processed
    processed_list = []

    while True:

        # Get list of files that are "new" and have not yet been processed
        new_files = find_new_files(listened_file, processed_list) 
       
        if new_files: 
            logger.info('There are {len(new_files)} new files to process...')

            # Create the prediction tensor with NEXRAD WDSS2 netcdf data
            tensor_list = make_prediction_tensors(config, new_files, processed_list)

            for tl in tensor_list:
                print(tl.shape)

            # Make predictions
            #probs = model.predict(pred_tensor)

            # Write the jsons
            #write_outputs(probs, new_files)
            
        logger.info('\nSleeping...')
        time.sleep(10)






#----------------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Uses TORP detects from output csvs and WDSS2-processed " + \
                                                 "radar data (netcdfs) to predict prob(tor) from CNN. " + \
                                                 "Velocity data is dealiased with WDSS2+RAP data."
    )
    parser.add_argument('listened_file',
                        help="This file generates a new line when a TORP csv is created (per-radar per-time). " + \
                             "E.g., /raid/sas8tb/jcintineo/torcnn_output/logs/MAIN-TORPLIST-A1.log", 
                        type=str
    )
    parser.add_argument('-m',
                        '--model',
                        help="import this CNN model. Default = static/model/fit_conv_model.keras",
                        default="static/model/fit_conv_model.keras",
                        type=str
    )
    parser.add_argument('-o',
                        '--outpatt',
                        help="Output pattern. Default = /raid/sas8tb/jcintineo/torcnn_output/{radar}/%%Y/%%Y%%m%%d/torcnn_%%Y%%m%%d-%%H%%M%%S.json",
                        default="/raid/sas8tb/jcintineo/torcnn_output/{radar}/%Y/%Y%m%d/torcnn_%Y%m%d-%H%M%S.json",
                        type=str
    )
    
    args = parser.parse_args()

    run_model(args.listened_file,
              args.model,
              args.outpatt
    )
