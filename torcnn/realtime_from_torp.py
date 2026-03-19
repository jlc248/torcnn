import sys,os
import numpy as np
import pandas as pd
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import glob
import argparse
import pickle
from generate_tfrecs import get_sector_patch
import utils
import time
from datetime import datetime, timedelta
import xarray as xr
import collections
sys.path.append('/mnt/home/phi/localdata/PHI_Processing/generalServices/getCNN/Tor/')
import addCnnTorToDB as add2db
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
        try:
            dt = datetime.strptime(os.path.basename(new_file)[0:15], '%Y%m%d-%H%M%S')
        except ValueError:
            logging.warning(f'Bad format: {os.path.basename(new_file)}. Skipping.')
            continue
        if dt < expired or new_file in processed_list:
            # If too old or already-processed then don't include it
            new_files.remove(new_file)
           
        elif dt < expired and new_file in processed_list:
            # Purge old files from processed list so that it doesn't grow too large
            processed_list.remove(new_file)
 
    return new_files


#----------------------------------------------------------------------------------

def make_prediction_tensors(config, new_files, processed_list, dataroot):

    """
    Using the list of new_files, get the actual NEXRAD data and create a dict 
    of tensors ready for predictions.

    Args:
    - config (dict): model config dictionary, so we know which channels to get
    - new_files (list): A list of full paths to new TORP detects (per-radar per-time)
    - processed_list (list): The list to add each new_file to if it was successfully processed.
    - dataroot (str): Root path where the NEXRAD L2 netcdfs live.

    Returns:
    - samples (dict): Contains the tensors with shape = (ndetects, ny, nx, nchannels)
                      and associated lats and lons
    """
    
    bsinfo = config['byte_scaling_vals']
    tilt = '00.50'
    n_az, n_gates = config['ps']
    hs = (n_az // 2, n_gates // 2)
    channels = config['channels']
    nchan = len(channels)

    # Dict for holding the tensor, lats, and lons for each torpfile
    samples = collections.OrderedDict()

    for torpfile in new_files:

        # Get the detect locations
        df = pd.read_csv(torpfile)
        lats = df.lat
        lons = df.lon
        radar = df.iloc[0].radar
        if np.isnan(lats).any() or np.isnan(lons).any():
            logger.warning(f'Missing coords in {torpfile}. Skipping.')
            continue
        else:
            # A sample gets instantiated so long as all lats and lons are valid.
            samples[torpfile] = {'df':df, 'is_valid':np.full(len(lats), True)}

        # Create a numpy array for the tensor for the detects in THIS torpfile
        nsamples = len(lats)
        tensor = np.zeros((nsamples, n_az, n_gates, nchan), dtype=np.float32)

        # We need to find the WDSS2 netcdfs from the TORP detects time.
        # TORP detects time is the scan time of the velocity data, which 
        # can often be after the file time of the reflectivity data.
        # Assume Velocity is always used.

        vel_dt = datetime.strptime(os.path.basename(torpfile)[0:15], '%Y%m%d-%H%M%S')
        samples[torpfile]['vel_dt'] = vel_dt

        for chIdx,chan in enumerate(channels):

            if chan in ['range', 'range_inv', 'range_folded_mask', 'out_of_range_mask']:
                # We handle these while doing the first channel, typically Ref or Vel
                pass
            else:
                # Normal radar moments

                cmin = bsinfo[chan]['vmin']
                cmax = bsinfo[chan]['vmax']

                # Find the closest file to vel_dt
                all_files = glob.glob(f"{dataroot}/{radar}/{chan}/{tilt}/*netcdf")
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
                        # This slice of the tensor will be 0s. Flag it. 
                        samples[torpfile]['is_valid'][ii] = False
                        continue


                    if chIdx == 0: 
                        # Get range and range_inv on the first channel iteration. Also, make out_of_range_mask.
                        ## 'range' is 1D, but we need it to be 2D. Expand to number of azimuths
                        if 'range' in channels or 'out_of_range_mask' in channels:
                            range_data = np.repeat(np.expand_dims(r_sector, axis=0), hs[0]*2, axis=0)

                            rmin = bsinfo['range']['vmin']
                            rmax = bsinfo['range']['vmax']

                            # Depending on the model, you will either normalize to bytes or  normalize to 0 --> 1
                           # range_data = utils.bytescale(range_data,
                           #                              rmin, 
                           #                              rmax, 
                           #                              min_byte_val=1, # we also invert range, so keep it > 0
                           #                              max_byte_val=255
                           # ).astype(np.float32)

                            # Normalize in physical units
                            range_data = (range_data - rmin) / (rmax - rmin)
                            range_data[range_data < 0] = 0
                            range_data[range_data > 1] = 1

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

                    # Depending on the model, you will either normalize to bytes or
                    # normalize to 0 --> 1 or -1 --> +1

                    # Normalize the moment data data
                    ## Bytescale
                    #rad_sector_scaled = utils.bytescale(rad_sector,
                    #                                    cmin, 
                    #                                    cmax, 
                    #                                    min_byte_val=0,
                    #                                    max_byte_val=255
                    #) 
                
                    ## Normalize in physical units
                    if chan == 'Velocity':
                        # Scale physical -100 to 100 -> -1.0 to 1.0
                        # Formula: (val - center) / half_range
                        rad_sector_scaled = rad_sector / max(abs(cmin), abs(cmax))
                        rad_sector_scaled[rad_sector_scaled < -1] = -1
                        rad_sector_scaled[rad_sector_scaled > 1] = 1
                    else:
                        # Scale physical min to max -> 0.0 to 1.0
                        # Formula: (val - min) / (max - min)
                        rad_sector_scaled = (rad_sector - cmin) / (cmax - cmin)
                        rad_sector_scaled[rad_sector_scaled < 0] = 0
                        rad_sector_scaled[rad_sector_scaled > 1] = 1

                    # Add moment data to tensor
                    tensor[ii, ..., chIdx] = rad_sector_scaled

        # The tensor for torpfile is ready
        if len(tensor.shape) == 4:
            samples[torpfile]['tensor'] = tensor
            processed_list.append(torpfile)
        else:
            # Remove if no good objects
            del samples[torpfile]

    logger.info(f'Created tensors for {len(samples)} files.')

    return samples


#----------------------------------------------------------------------------------
def predict(model, samples):
    """
    Make predictions from the tensors in samples (each sample is a torp file).

    Args:
    - model (TF model): The model to make predictions
    - samples (dict): Contains the input predictions and is_valid flags for the TORP detects.
                      Predictions will be added to this.
    Returns:
    - probs (list): The list of probabilities (0 to 1).
    - end_index (list): Marks the index of the last object for each torpfile
    """

    # Here, we want to combine all of the tensors for predict efficiency.
    # end_index will mark the index of the last object for each torpfile
    end_index = [0]

    tensor_list = []
    is_valid = np.array([], dtype=bool)
    for torpfile, sample in samples.items():
        end_index.append(end_index[-1] + len(sample['df']))
        is_valid = np.concatenate((is_valid, sample['is_valid']))
        tensor_list.append(sample['tensor'])

    # Combine tensors and compute probs
    if len(tensor_list):
        super_tensor = np.vstack(tensor_list) 

        probs = model.predict(super_tensor, verbose=1)
        probs[~is_valid] = -1 # invalid patches get assigned -1

        del super_tensor, tensor_list

        return probs, end_index
    else:
        logger.warning("No complete samples to predict!")
        return [], []


#----------------------------------------------------------------------------------
def write_outputs(probs, samples, end_index, outpatt):
    """
    Write out amended TORP csvs.

    Args:
        probs (list): A list of float probabilities, 0 to 1. May be -1 if invalid.
        samples (dict): Keys are the torpfiles (csvs). Contains the 'df' for each torpfile.
        end_index (list): List of the final index in probs for each torpfile.
        outpatt (str): Output directory pattern.
    Returns:
        None
    """

    cter = 1
    for torpfile, sample in samples.items():

        df = sample['df']
        radar = df.radar[0]
        # Assign the correct probs
        df['torcnn_probability'] = probs[end_index[cter-1]:end_index[cter]]
        # Iterate for next torpfile
        cter += 1

        outdir = sample['vel_dt'].strftime(outpatt).replace('{radar}',radar)
        os.makedirs(outdir, exist_ok=True)
        outbasefile = os.path.basename(torpfile).replace('short','torcnn')
        outfile = f"{outdir}/{outbasefile}"

        df.to_csv(outfile)
        logger.info(f"Wrote {outfile}")


#---------------------------------------------------------------------------------
def upload_to_SWRM(samples):

    # super_list is a list of dicts.
    # Each dict is a row/record from a TORP object.
    super_list = []
    for torpfile, sample in samples.items():
        super_list += sample['df'].to_dict(orient='records')

    add2db.main(super_list)
#----------------------------------------------------------------------------------

def run_model(listened_file,
              model_file,
              dataroot,
              outpatt,
              db_upload=False,
):

    # One-time overhead
    ## Read TF model
    model = tf.keras.models.load_model(model_file, compile=False)
    ## Get the config file. Assume it's in the same directory.
    config = pickle.load(open(f'{os.path.dirname(model_file)}/model_config.pkl', 'rb'))
    ## Just a list to keep a log of files we've processed
    processed_list = []

    #if db_upload:
    #    # Hard-coded path
    #    sys.path.append('/mnt/home/phi/localdata/PHI_Processing/generalServices/getCNN/Tor/')
    #    import addCnnTorToDB as add2db

    while True:

        # Get list of files that are "new" and have not yet been processed
        new_files = find_new_files(listened_file, processed_list) 
       
        if new_files: 
            logger.info(f'There are {len(new_files)} new files to process...')
        
            # Create the prediction tensors with NEXRAD WDSS2 netcdf data.
            # samples is a dict with each new_file as a key, pointing to the tensor at lats/lons
            samples = make_prediction_tensors(config, new_files, processed_list, dataroot)
            
            if len(samples):
                # Make predictions
                probs, end_index = predict(model, samples)

                # Write the csvs
                if len(probs):
                    write_outputs(probs, samples, end_index, outpatt)

                    if db_upload:
                        upload_to_SWRM(samples)
            
        logger.info('\nSleeping...')
        time.sleep(10)




#----------------------------------------------------------------------------------

if __name__ == "__main__":

    if 'TORCNN_DATA' not in os.environ:
        print('TORCNN_DATA env variable must be set.')
        sys.exit(1)

    myparser = argparse.ArgumentParser(description="Uses TORP detects from output csvs and WDSS2-processed " + \
                                                 "radar data (netcdfs) to predict prob(tor) from CNN. " + \
                                                 "Velocity data is dealiased with WDSS2+RAP data."
    )
    myparser.add_argument('listened_file',
                        help="This file generates a new line when a TORP csv is created (per-radar per-time). " + \
                             "E.g., /sas8tb/jcintineo/torcnn_output/logs/MAIN-TORPLIST-A1.log", 
                        type=str
    )
    myparser.add_argument('-m',
                        '--model',
                        help="import this CNN model. Default = static/model/fit_conv_model.keras",
                        default="static/model/fit_conv_model.keras",
                        type=str
    )
    myparser.add_argument('-d',
                        '--dataroot',
                        help="root path to NEXRAD L2 data. Default= /ssd1/localdata_MRMS/realtime/radar/.",
                        default='/ssd1/localdata_MRMS/realtime/radar/',
                        type=str
    )
    myparser.add_argument('-o',
                        '--outpatt',
                        help="Output directory pattern. Default = ${TORCNN_DATA}/products/{radar}/%%Y/%%Y%%m%%d/",
                        default=os.environ['TORCNN_DATA'] + "/products/{radar}/%Y/%Y%m%d/",
                        type=str
    )
    myparser.add_argument('-u',
                        '--db_upload',
                        help="Use helper functions to upload to SWRM database",
                        action="store_true",
    )
    
    args = myparser.parse_args()
        
    run_model(args.listened_file,
              args.model,
              args.dataroot,
              args.outpatt,
              db_upload=args.db_upload,
    )
