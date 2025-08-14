import os,sys
import matplotlib.pyplot as plt
# turn off GPUs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pickle
import pandas as pd
import numpy as np
import glob
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

OUT_OF_RANGE = -999

def create_tensor(row,
                  hs,
                  inputs,
                  bsinfo,
                  datapatt1,
                  datapatt2=None,
                  year_thresh=2019,
):

    """
    Collect the L2 radar data, create a tensor, and make predictions.
    Args:
        row (pandas Series or Frame): a row or data sample in the TORP dataset.
        hs (int): halfsize of the patch
        inputs (list): List of lists with stinrgs for the variable names
        bsinfo (dict): contains min and max scaling values for each varname
        datapatt1 (str): Full path data pattern. E.g., '/data/thea.sandmael/data/radar/%Y%m%d/{radar}/netcdf/{varname}/00.50/%Y%m%d-%H%M%S.netcdf'
        datapatt2 (str): Secondary data pattern. E.g., '/work/thea.sandmael/radar/%Y%m%d/{radar}/netcdf/{varname}/00.50/%Y%m%d-%H%M%S.netcdf'
        year_thresh (int): The threshold such that >= year_thresh will use datapatt2. Default is 2019.
    """

    if row['radar'] == '-99900':
        raise ValueError("radar == -99900")

    # Anchoring timestamp
    raddt = datetime.strptime(row['radarTimestamp'], '%Y%m%d-%H%M%S')

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

                # Check datapatt2 if defined
                if datapatt2:
                    if row['year'] >= year_thresh:
                        datapatt = datapatt2
                    else:
                        datapatt = datapatt1
                else:
                    datapatt = datapatt1
    
                dpat = datapatt.replace('{radar}', row['radar']).replace('{varname}', varname)
                all_files = glob.glob(raddt.strftime(f"{os.path.dirname(dpat)}/*netcdf"))
                dts = [datetime.strptime(os.path.basename(ff), '%Y%m%d-%H%M%S.netcdf') for ff in all_files]
                try:
                    closest_dt = min(dts, key=lambda dt: abs(raddt - dt))
                except ValueError:
                    logging.error(f"ValueError: no closest_dt, {row['radar']}, {row['radarTimestamp']}")
                if abs(raddt - closest_dt).seconds > 600:
                    raise ValueError(f"{raddt} is too far from {closest_dt}. {row['radar']} - {varname}")
                
                idx = dts.index(closest_dt)
                file_path = all_files[idx]
    
                # Open the netCDF file using xarray
                radds = xr.open_dataset(file_path)
    
                rad_sector, theta_sector, r_sector, az_idx, gate_idx = get_sector_patch(lat=row['latitude'],
                                                                                        lon=row['longitude'],
                                                                                        ds=radds,
                                                                                        hs=hs,
                                                                                        varname=varname
                )
                rad_sector = rad_sector.astype(np.float32)
    
                # Assert that the shape is correct
                try:
                    assert(rad_sector.shape == (hs[0]*2, hs[1]*2))
                except AssertionError as err:
                    print(f"{row['radar']}, {row['radarTimestamp']}, {varname}.shape == {rad_sector.shape}")
                    return
    
                if ii == 0 and jj == 0:
                    # Get range and range_inv on the iter. Also, make out_of_range_mask.
                    # 'range' is 1D, but we need it to be 2D. Expand to number of azimuths
                    range_data = np.repeat(np.expand_dims(r_sector, axis=0), hs[0]*2, axis=0)
                    container['range'] = utils.bytescale(range_data,
                                                         bsinfo['range']['vmin'],
                                                         bsinfo['range']['vmax'],
                                                         min_byte_val=1, # we also invert range, so keep it > 0
                                                         max_byte_val=255
                    ).astype(np.float32)
    
                    container['range_inv'] = 1. / range_data
    
                    container['out_of_range_mask'] = (rad_sector == OUT_OF_RANGE).astype(np.float32)
    
    
                # Bytescale the data
                rad_sector_scaled = utils.bytescale(rad_sector,
                                                    bsinfo[varname]['vmin'],
                                                    bsinfo[varname]['vmax'],
                                                    min_byte_val=0,
                                                    max_byte_val=255
                )

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

# Load dataset
dataset_type = 'Storm_Reports' # 'Storm_Reports' or 'pretornadic'

# Load torp dataset
dataset = TORPDataset(dirpath='/raid/jcintineo/torcnn/torp_datasets/',
                      years=[2023],
                      dataset_type=dataset_type
)
ds = dataset.load_dataframe()

# No spouts
ds = ds[ds.spout == 0]

# Define the list of columns to keep
columns_to_keep = ['latitude', 'longitude', 'tornado', 'radar', 'year', 'radarTimestamp', 'latitudeExtractCenter', 'longitudeExtractCenter', 
                   'latitudeAzShearMax', 'longitudeAzShearMax'] # these two are for the pre-tor dataset
# Find the columns to drop (columns not in the list to keep)
columns_to_drop = [col for col in ds.columns if col not in columns_to_keep]
# Drop the columns
ds = ds.drop(columns=columns_to_drop)

# 2011-2018
datapatt1 = '/data/thea.sandmael/data/radar/%Y%m%d/{radar}/netcdf/{varname}/00.50/%Y%m%d-%H%M%S.netcdf'
# 2019-2024
datapatt2 = '/work/thea.sandmael/radar/%Y%m%d/{radar}/netcdf/{varname}/00.50/%Y%m%d-%H%M%S.netcdf'

logger = logging.getLogger(__name__)

model = '/raid/jcintineo/torcnn/tests/2011-23/test07/fit_conv_model.keras'

outdir=f'{os.path.dirname(model)}/eval2023/nospout/'
os.makedirs(outdir, exist_ok=True)

conv_model = keras.models.load_model(model, compile=False)
config = pickle.load(open(f'{os.path.dirname(model)}/model_config.pkl', 'rb'))
inputs = config['inputs']
ps = config['ps']
hs = (ps[0]//2, ps[1]//2)
bsinfo = config['byte_scaling_vals']

batch_size=1000

struct = {'radar': np.zeros((batch_size, ps[0], ps[1], len(inputs[0]))), 
          'coords': np.zeros((batch_size, ps[0], ps[1], len(inputs[1])))
}

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
                                   datapatt1=datapatt1,
                                   datapatt2=datapatt2,
                                   year_thresh=2019,
        )
    except ValueError as err:
        print(err)
        struct['radar'][sample_cter] = np.zeros((1, ps[0], ps[1], len(inputs[0])), dtype=np.float32)
        struct['coords'][sample_cter] = np.zeros((1, ps[0], ps[1], len(inputs[1])), dtype=np.float32)

    struct['radar'][sample_cter] = all_inputs['radar']
    struct['coords'][sample_cter] = all_inputs['coords']

    cter += 1
    sample_cter += 1

    if cter % batch_size == 0:
        print(cter)
        # predict
        preds = conv_model.predict(struct, verbose=1)
        if np.count_nonzero(preds):
            ind = np.where(np.squeeze(preds) == np.nan)
            rad = struct['radar'][sample_cter]
            cord = struct['coords'][sample_cter]
            print(
            fig, ax = plt.subplots(nr
        all_preds = np.concatenate((all_preds, np.squeeze(preds)))
        # reset struct
        struct['radar'] *= 0; struct['coords'] *= 0  
        sample_cter = 0
    elif cter >= len(ds):
        print(cter)
        # predict on remainder
        # remove excess elements
        struct['radar'] = struct['radar'][0:cter % batch_size]
        struct['coords'] = struct['coords'][0:cter % batch_size]
        preds = conv_model.predict(struct, verbose=1)
        all_preds = np.concatenate((all_preds, np.squeeze(preds)))

pickle.dump(all_preds.astype(np.float32), open(f'{outdir}/preds.pkl', 'wb'))
ds.to_pickle(f'{outdir}/df_lite.pkl')

sys.exit()
# Common arguments to pass in
create_tensor_and_predict_args = dict(
    hs=halfsize,
    varnames=varnames,
    bsinfo=bsinfo,
    datapatt1=datapatt1,
    datapatt2=datapatt2,
    year_thresh=2019,
)
create_tensor_and_predict_tfrec = functools.partial(create_tensor_and_predict, **create_tensor_and_predict_args)

number_of_rows = len(ds)

max_workers = 10 #min(gfs_columns_extract_workers, os.cpu_count())

with tqdm(total=number_of_rows) as pbar: # progress bar
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for row in ds.itertuples():
            # Convert the Pandas namedtuple-like object to a regular dictionary.
            # This makes sure only simple, pickleable values are passed.
            row_as_dict = row._asdict() # This converts it to an OrderedDict, which is pickleable

            futures.append(executor.submit(create_tensor_and_predict, row_as_dict))

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error processing a row: {e}")
                logger.error(traceback.format_exc())
            finally:
                pbar.update(1)

