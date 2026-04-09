import os,sys
import matplotlib.pyplot as plt
# turn off GPUs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pickle
import pandas as pd
import numpy as np
from torp.torp_dataset import TORPDataset
import logging
from tqdm import tqdm
from datetime import datetime, timedelta
import glob
import xarray as xr
from generate_tfrecs import get_sector_patch
import utils
import keras
import tf_models

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)
OUT_OF_RANGE = -999

def create_tensor(row,
                  hs,
                  inputs,
                  bsinfo,
                  datapatt,
):

    """
    Collect the L2 radar data, create a tensor, and make predictions.
    Args:
        row (pandas Series or Frame): a row or data sample in the TORP dataset.
        hs (int): halfsize of the patch
        inputs (list): List of lists with stinrgs for the variable names
        bsinfo (dict): contains min and max scaling values for each varname
        datapatt (str): Full path data pattern. E.g., '/ssd1/localdata_MRMS/realtime/radar/{radar}/{varname}/00.50/%Y%m%d-%H%M%S.netcdf'
    """

    if row['radar'] == '-99900':
        raise ValueError("radar == -99900")

    # Anchoring timestamp
    try:
        raddt = datetime.strptime(row['current_time_of_detection'], '%Y%m%d-%H%M%S')
    except TypeError:
        print(row)
        print(row['current_time_of_detection'])
        sys.exit() 

    container = {}

    all_inputs = {}
    for ii, inp in enumerate(inputs):
        X = None
        for jj, varname in enumerate(inp):

            if varname in ['range', 'range_inv', 'range_folded_mask', 'out_of_range_mask']:
                # We should aleady have these data
                # Just append them to the tensor

                if X is None:
                    X = np.expand_dims(container[varname], axis=(0,-1))
                else:
                    X = np.concatenate(
                        (X, np.expand_dims(container[varname], axis=(0,-1))), axis=-1
                    )
            else:
                # Normal radar variables

                dpat = datapatt.replace('{radar}', row['radar']).replace('{varname}', varname)
                all_files = glob.glob(raddt.strftime(f"{os.path.dirname(dpat)}/*netcdf"))
                dts = [datetime.strptime(os.path.basename(ff), '%Y%m%d-%H%M%S.netcdf') for ff in all_files]
                try:
                    closest_dt = min(dts, key=lambda dt: abs(raddt - dt))
                except ValueError:
                    logging.error(f"ValueError: no closest_dt, {row['radar']}, {row['current_time_of_detection']}")
                if abs(raddt - closest_dt).seconds > 600:
                    raise ValueError(f"{raddt} is too far from {closest_dt}. {row['radar']} - {varname}")
                
                idx = dts.index(closest_dt)
                file_path = all_files[idx]
    
                # Open the netCDF file using xarray
                radds = xr.open_dataset(file_path)
    
                rad_sector, theta_sector, r_sector, az_idx, gate_idx = get_sector_patch(lat=row['lat'],
                                                                                        lon=row['lon'],
                                                                                        ds=radds,
                                                                                        hs=hs,
                                                                                        varname=varname
                )
                rad_sector = rad_sector.astype(np.float32)
    
                # Assert that the shape is correct
                try:
                    assert(rad_sector.shape == (hs[0]*2, hs[1]*2))
                except AssertionError as err:
                    print(f"{row['radar']}, {row['current_time_of_detection']}, {varname}.shape == {rad_sector.shape}")
                    return
    
                if ii == 0 and jj == 0:
                    # Get range and range_inv on the iter. Also, make out_of_range_mask.
                    # 'range' is 1D, but we need it to be 2D. Expand to number of azimuths
                    range_data = np.repeat(np.expand_dims(r_sector, axis=0), hs[0]*2, axis=0)
                    #container['range'] = utils.bytescale(range_data,
                    ##                                     bsinfo['range']['vmin'],
                    #                                     bsinfo['range']['vmax'],
                    #                                     min_byte_val=1, # we also invert range, so keep it > 0
                    #                                     max_byte_val=255
                    #).astype(np.float32)

                    # Normalize in physical units
                    rmin = bsinfo['range']['vmin']
                    rmax = bsinfo['range']['vmax']
                    range_data = (range_data - rmin) / (rmax - rmin)
                    range_data[range_data < 0] = 0
                    range_data[range_data > 1] = 1
                    container['range'] = range_data.astype(np.float32)
 
                    container['range_inv'] = 1. / range_data
    
                    container['out_of_range_mask'] = (rad_sector == OUT_OF_RANGE).astype(np.float32)
    
    
                # Bytescale the data
                #rad_sector_scaled = utils.bytescale(rad_sector,
                #                                    bsinfo[varname]['vmin'],
                #                                    bsinfo[varname]['vmax'],
                #                                    min_byte_val=0,
                #                                    max_byte_val=255
                #)
   
                # Normalize in physical units
                rmin = bsinfo[varname]['vmin']
                rmax = bsinfo[varname]['vmax']
                if varname == 'Velocity':
                    # Scale physical -100 to 100 -> -1.0 to 1.0
                    # Formula: (val - center) / half_range
                    normalized = rad_sector / max(abs(rmin), abs(rmax))
                else: 
                    rad_sector_scaled = (rad_sector - rmin) / (rmax - rmin)
                    rad_sector_scaled[rad_sector_scaled < 0] = 0
                    rad_sector_scaled[rad_sector_scaled > 1] = 1

                # Encode range-folded region
                if varname == 'Velocity':
                    range_folded_value = radds.attrs.get('RangeFolded', -99901.0)
                    container['range_folded_mask'] = (rad_sector == range_folded_value).astype(np.float32)
    

                if X is None:
                    X = np.expand_dims(rad_sector_scaled, axis=(0,-1))
                else:
                    X = np.concatenate(
                        (X, np.expand_dims(rad_sector_scaled, axis=(0,-1))), axis=-1
                    )

        if ii == 0:
            all_inputs['radar'] = X
        elif ii == 1:
            all_inputs['coords'] = X    
   
    return all_inputs 

#------------------------------------------------------------------------------------------------------------

# Concatenate a list of TORP csvs and make predictions.
# Add predictions to the csvs and save.

# Model for predctions
model = 'static/model/fit_conv_model.keras'
conv_model = keras.models.load_model(model, compile=False)
config = pickle.load(open(f'{os.path.dirname(model)}/model_config.pkl', 'rb'))
# Output directory
outdir=f'/sas8tb/jcintineo/offline/20260306-07_KINX/'
os.makedirs(outdir, exist_ok=True)
# Root dir for the input data
rootdir = '/ssd1/localdata_MRMS/realtime/radar/'

# Get and combine the csvs
csvs = np.sort(glob.glob(f'{rootdir}/KINX/TORPcsvShort/*csv'))
df_list = [pd.read_csv(f, index_col=False) for f in csvs] # index_col=False for trailing commas
combined_df = pd.concat(df_list, ignore_index=True)
# Drop missing rows
ds = combined_df.dropna(subset=['lat'])
logging.info(f'Combined {len(ds)} records')
logging.warning(f'{len(combined_df) - len(ds)} rows were missing/null.')
datapatt = rootdir + '/{radar}/{varname}/00.50/%Y%m%d-%H%M%S.netcdf'

inputs = config['inputs']
ps = config['ps']
hs = (ps[0]//2, ps[1]//2)
bsinfo = config['byte_scaling_vals'] # also used for scaling from 0 to 1

batch_size=1000

struct = {'radar': np.zeros((batch_size, ps[0], ps[1], len(inputs[0])), dtype=np.float32)}
if len(inputs) > 1: 
    struct['coords'] = np.zeros((batch_size, ps[0], ps[1], len(inputs[1])), dtype=np.float32)

all_preds = np.array([])

cter = 0
sample_cter = 0
for row in ds.itertuples():
    row_as_dict = row._asdict()

    try:
        all_inputs = create_tensor(row_as_dict,
                                   hs=hs,
                                   inputs=inputs,
                                   bsinfo=bsinfo,
                                   datapatt=datapatt,
        )
    except ValueError as err:
        print(err)
        struct['radar'][sample_cter] = np.zeros((1, ps[0], ps[1], len(inputs[0])), dtype=np.float32)
        if len(inputs) > 1:
            struct['coords'][sample_cter] = np.zeros((1, ps[0], ps[1], len(inputs[1])), dtype=np.float32)


    struct['radar'][sample_cter] = all_inputs['radar']
    if len(inputs) > 1:
        struct['coords'][sample_cter] = all_inputs['coords']

    cter += 1
    sample_cter += 1

    if cter % batch_size == 0:
        print(cter)
        # predict
        preds = conv_model.predict(struct, verbose=1)
        all_preds = np.concatenate((all_preds, np.squeeze(preds)))
        # reset struct and sample_cter
        struct['radar'] *= 0
        if len(inputs) > 1:
            struct['coords'] *= 0  
        sample_cter = 0
    elif cter >= len(ds):
        print(cter)
        # predict on remainder
        # remove excess elements
        struct['radar'] = struct['radar'][0:cter % batch_size]
        if len(inputs) > 1:
            struct['coords'] = struct['coords'][0:cter % batch_size]
        preds = conv_model.predict(struct, verbose=1)
        all_preds = np.concatenate((all_preds, np.squeeze(preds)))

print(all_preds.shape)
ds['torcnn_probability'] = all_preds

ds = ds.sort_values(by=['id', 'current_time_of_detection'])


ds.to_csv(f'{outdir}/df_combined.csv', index=False)
logging.info(f'Saved {outdir}/df_combined.csv')
