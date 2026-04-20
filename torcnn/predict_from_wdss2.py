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

def create_tensor(rad_file: str,
                  hs: tuple,
                  inputs: list,
                  bsinfo: dict,
                  lat: float,
                  lon: float,
) -> dict[str, np.ndarray]:

    """
    Collect the L2 radar data, create a tensor, and make predictions.
    Args:
        rad_file (str): Full path to one of the WDSS2 netcdfs
        hs (tuple): halfsizes of the patch
        inputs (list): List of lists with stinrgs for the variable names
        bsinfo (dict): contains min and max scaling values for each varname
        lat (float): Central latitudes for the patch
        lon (float): Central longitudes for the patch
    Returns:
        all_inputs (dict): Dictionary with named inputs, pointing to tensors
    """

    # Anchoring timestamp
    try:
        raddt = datetime.strptime(os.path.basename(rad_file), '%Y%m%d-%H%M%S.netcdf')
    except TypeError as err:
        print(err) 
        sys.exit(1) 

    container = {}

    all_inputs = {}

    # /home/john.cintineo/temp_KDVN/20260402/KDVN/netcdf/Velocity/00.50/20260402-210716.netcdf'
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(rad_file)))

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
                all_files = np.sort(glob.glob(f"{root_path}/{varname}/00.50/*netcdf"))
                dts = [datetime.strptime(os.path.basename(ff), '%Y%m%d-%H%M%S.netcdf') for ff in all_files]
                try:
                    closest_dt = min(dts, key=lambda dt: abs(raddt - dt))
                except ValueError:
                    logging.error(f"ValueError: no closest_dt, {varname} {rad_file}")
                if abs(raddt - closest_dt).seconds > 600:
                    raise ValueError(f"{raddt} is too far from {closest_dt}. {varname} {rad_file}")
                
                idx = dts.index(closest_dt)
                file_path = all_files[idx]
    
                # Open the netCDF file using xarray
                radds = xr.open_dataset(file_path)
    
                rad_sector, theta_sector, r_sector, az_idx, gate_idx = get_sector_patch(lat=lat,
                                                                                        lon=lon,
                                                                                        ds=radds,
                                                                                        hs=hs,
                                                                                        varname=varname
                )
                rad_sector = rad_sector.astype(np.float32)
    
                # Assert that the shape is correct
                try:
                    assert(rad_sector.shape == (hs[0]*2, hs[1]*2))
                except AssertionError as err:
                    print(f"Bad shape  {varname}.shape == {rad_sector.shape}")
                    return
    
                if ii == 0 and jj == 0:
                    # Get range and range_inv on the iter. Also, make out_of_range_mask.
                    # 'range' is 1D, but we need it to be 2D. Expand to number of azimuths
                    range_data = np.repeat(np.expand_dims(r_sector, axis=0), hs[0]*2, axis=0)

                    # Normalize in physical units
                    rmin = bsinfo['range']['vmin']
                    rmax = bsinfo['range']['vmax']
                    range_data = (range_data - rmin) / (rmax - rmin)
                    range_data[range_data < 0] = 0
                    range_data[range_data > 1] = 1
                    container['range'] = range_data.astype(np.float32)
 
                    container['range_inv'] = 1. / range_data
    
                    container['out_of_range_mask'] = (rad_sector == OUT_OF_RANGE).astype(np.float32)
    
    
                # Normalize in physical units
                rmin = bsinfo[varname]['vmin']
                rmax = bsinfo[varname]['vmax']
                if varname == 'Velocity':
                    # Scale physical -100 to 100 -> -1.0 to 1.0
                    # Formula: (val - center) / half_range
                    rad_sector_scaled = np.clip(rad_sector / max(abs(rmin), abs(rmax)), -1, 1) 
                else: 
                    rad_sector_scaled = np.clip((rad_sector - rmin) / (rmax - rmin), 0, 1)

                # Encode range-folded region
                if varname == 'Velocity':
                    range_folded_value = radds.attrs.get('RangeFolded', -99901.0)
                    container['range_folded_mask'] = (rad_sector == range_folded_value).astype(np.float32)
               #     pickle.dump(rad_sector_scaled, open('velocity_wdss2.pkl','wb'))
               # elif varname == 'Reflectivity':
               #     pickle.dump(rad_sector_scaled, open('ref_wdss2.pkl','wb'))

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

if __name__ == "__main__":

    # Model for predctions
    model = 'static/model/fit_conv_model.keras'
    conv_model = keras.models.load_model(model, compile=False)
    config = pickle.load(open(f'{os.path.dirname(model)}/model_config.pkl', 'rb'))

    # Output directory
    #outdir=f'/sas8tb/jcintineo/offline/20260306-07_KINX/'
    #os.makedirs(outdir, exist_ok=True)

    # Radar files
    rad_files = ['/home/john.cintineo/temp_KDVN/20260402/KDVN/netcdf/Velocity/00.50/20260402-210716.netcdf']
    lats = [41.414]
    lons=[-91.7322]
 
    inputs = config['inputs']
    ps = config['ps']
    hs = (ps[0]//2, ps[1]//2)
    bsinfo = config['byte_scaling_vals'] # also used for scaling from 0 to 1
    
    batch_size = np.min([len(rad_files), 1000])
    
    struct = {'radar': np.zeros((batch_size, ps[0], ps[1], len(inputs[0])), dtype=np.float32)}
    if len(inputs) > 1: 
        struct['coords'] = np.zeros((batch_size, ps[0], ps[1], len(inputs[1])), dtype=np.float32)
    
    all_preds = np.array([])
    
    cter = 0
    sample_cter = 0
    for rad_file, lat, lon in zip(rad_files, lats, lons):
    
        try:
            all_inputs = create_tensor(rad_file,
                                       hs,
                                       inputs,
                                       bsinfo,
                                       lat,
                                       lon,
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
            print(preds)
            sys.exit() 
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
    print(all_preds)
    
