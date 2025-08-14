import traceback
import logging
import functools
import concurrent
from collections import namedtuple
from tqdm import tqdm
import numpy as np
import xarray as xr
import os,sys
import tensorflow as tf
from torp.torp_dataset import TORPDataset
import utils
import rad_utils
import xarray
import pandas as pd
import time
from datetime import datetime, timedelta
import glob
import math
from geopy.distance import geodesic
from geopy.point import Point

OUT_OF_RANGE = -999

#--------------------------------------------------------------------------------------------------
def get_sector_patch(lat, lon, ds, hs, varname):

    """
    Returns a patch of radar data from native polar coordinates.
    
    Args:
        lat (float): latitude of the center of patch
        lon (float): longitude of the center of patch
        ds (xarray dataset): the radar data and attributes
        hs (tuple): tuple of ints for the halfsizes of the patch
        varname (str): variable name of the radar product
    Returns:
        rad_sector (np.ndarray): the patch/sector of radar data
        theta_sector (np.ndarray): the patch/sector of azimuths (in degrees)
        r_sector (np.ndarray): the patch/sector of ranges (in km)
    """

    az_hs, range_hs = hs 
 
    raddata = ds[varname].values
    theta = ds['Azimuth'].values
    gate_width = ds['GateWidth'].values.mean() # Assuming GateWidth is relatively constant
    range_to_first_gate = ds.attrs['RangeToFirstGate']
    #radar_name = ds.attrs['radarName-value']

    # Create an array of gate distances (r)
    gates = np.arange(ds.sizes['Gate'])
    r_meters = range_to_first_gate + (gates * gate_width)
    r_km = r_meters / 1000. # convert to km

    azimuth_idx, gate_idx, calc_az, calc_range = rad_utils.get_azimuth_range_from_latlon(lat, lon, ds=ds)

    # Azimuth slicing
    num_azimuths = len(theta)
    start_az_slice = azimuth_idx - az_hs
    end_az_slice = azimuth_idx + az_hs

    # Gate slicing (clamped)
    num_gates_limited = len(r_km)
    start_gate_slice = max(0, gate_idx - range_hs)
    end_gate_slice = min(num_gates_limited, gate_idx + range_hs)
    

    # See if we're too close or too far from the radar. If so, pad with missing value.
    if gate_idx - range_hs < 0:
        gate_pad_low = True
        gate_pad_high = False
        n_extra_gates = range_hs - gate_idx
    elif gate_idx + range_hs > num_gates_limited:
        gate_pad_low = False
        gate_pad_high = True
        n_extra_gates = gate_idx + range_hs - num_gates_limited
    else:
        gate_pad_low = gate_pad_high = False

    if end_az_slice > num_azimuths:
        # Indicates that we're wrapping around 0 degrees
        rad_sector = np.concatenate([
                        raddata[start_az_slice:num_azimuths, start_gate_slice:end_gate_slice],
                        raddata[0:(end_az_slice-num_azimuths), start_gate_slice:end_gate_slice]
        ])
        if gate_pad_low:
            padding = np.full((az_hs*2, n_extra_gates), OUT_OF_RANGE)
            rad_sector = np.concatenate([padding, rad_sector], axis=1)
        elif gate_pad_high:
            padding = np.full((az_hs*2, n_extra_gates), OUT_OF_RANGE)
            rad_sector = np.concatenate([rad_sector, padding], axis=1)

        theta_sector = np.concatenate([
                          theta[start_az_slice:num_azimuths],
                          theta[0:(end_az_slice-num_azimuths)]
        ])

    elif start_az_slice < 0:
        # Indicates that we're wrapping around 0 degrees
        rad_sector = np.concatenate([
                        raddata[start_az_slice:, start_gate_slice:end_gate_slice],
                        raddata[0:end_az_slice, start_gate_slice:end_gate_slice]
        ])
        if gate_pad_low:
            padding = np.full((az_hs*2, n_extra_gates), OUT_OF_RANGE)
            rad_sector = np.concatenate([padding, rad_sector], axis=1)
        elif gate_pad_high:
            padding = np.full((az_hs*2, n_extra_gates), OUT_OF_RANGE)
            rad_sector = np.concatenate([rad_sector, padding], axis=1)

        theta_sector = np.concatenate([
                          theta[start_az_slice:],
                          theta[0:end_az_slice]
        ])
    else:
        rad_sector = raddata[start_az_slice:end_az_slice, start_gate_slice:end_gate_slice]
        if gate_pad_low:
            padding = np.full((az_hs*2, n_extra_gates), OUT_OF_RANGE)
            rad_sector = np.concatenate([padding, rad_sector], axis=1)
        elif gate_pad_high:
            padding = np.full((az_hs*2, n_extra_gates), OUT_OF_RANGE)
            rad_sector = np.concatenate([rad_sector, padding], axis=1)

        theta_sector = theta[start_az_slice:end_az_slice]

    # Range should be unaffected by azimuth wrapping
    r_sector = r_km[start_gate_slice:end_gate_slice]
    if gate_pad_low:
        r_sector = np.concatenate([np.zeros(n_extra_gates), r_sector])
    elif gate_pad_high:
        gw = gate_width / 1000. # convert to km
        r_sector = np.concatenate([r_sector, np.full(n_extra_gates, r_sector[-1] + gw)])

    return rad_sector, theta_sector, r_sector, azimuth_idx, gate_idx

#--------------------------------------------------------------------------------------------------
def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.numpy()]))

#--------------------------------------------------------------------------------------------------
def bytes_feature(value):
    """Returns a bytes_list from a string / byte / serialized tensor."""
    if isinstance(value, str):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))
    elif isinstance(value, bytes):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    else:
        # Fallback for unexpected types, but generally you should ensure 'value' is bytes or str
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value).encode('utf-8')]))

#--------------------------------------------------------------------------------------------------
def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    # Ensure value is iterable for Int64List, even if it's a single scalar
    if isinstance(value, (list, np.ndarray)): # Check for list or numpy array of ints
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value.flatten()))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
  
#--------------------------------------------------------------------------------------------------
def float_feature(value):
    """Returns a float_list from a float / double."""
    if isinstance(value, (list, np.ndarray)): # Check for list or numpy array of floats
        return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))
    else:
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

#--------------------------------------------------------------------------------------------------
def create_example(channels):
    features = {}
    for key,item in channels.items():
        if isinstance(item, bytes):
            # If the item is already bytes (e.g., a serialized 2D array or image bytes)
            features[key] = bytes_feature(item)
        elif isinstance(item, (float, np.float64, np.float32)):
            features[key] = float_feature(item)
        elif isinstance(item, (int, np.int64, np.int32)):
            features[key] = int64_feature(item)
        elif isinstance(item, str):
            features[key] = bytes_feature(item)
        elif isinstance(item, np.ndarray):
            # This case handles raw numpy arrays that *haven't* been serialized yet.
            # If you *sometimes* have raw numpy arrays and *sometimes* pre-serialized bytes,
            # this would be the place to serialize the raw numpy array.
            # Example: Serialize a float32 numpy array to bytes if not already bytes
            # print(f"Warning: Item for key '{key}' is a numpy array. Serializing it to bytes.")
            tensor_item = tf.convert_to_tensor(item) # TensorFlow will infer dtype or you can specify
            serialized_tensor = tf.io.serialize_tensor(tensor_item)
            features[key] = bytes_feature(serialized_tensor.numpy())
        else:
            # Handle any other types or raise an error for unsupported types
            raise ValueError(f"Unsupported data type for key '{key}': {type(item)}")

    return tf.train.Example(features=tf.train.Features(feature=features))

#--------------------------------------------------------------------------------------------------
def collect_and_write_tfrec(row,
                            hs,
                            varnames,
                            bsinfo,
                            datapatt1,
                            outpatt,
                            datapatt2=None,
                            year_thresh=2019,
):

    """
    Collect the L2 radar data and write a TFRecord.
    Args:
        row (pandas Series or Frame): a row or data sample in the TORP dataset.
        hs (int): halfsize of the patch
        varnames (list): List of strings for the variable names
        bsinfo (dict): contains min and max scaling values for each varname
        datapatt1 (str): Full path data pattern. E.g., '/data/thea.sandmael/data/radar/%Y%m%d/{radar}/netcdf/{varname}/00.50/%Y%m%d-%H%M%S.netcdf'
        outpatt (str): Pattern for root outdir. E.g., '/raid/jcintineo/torcnn/tfrecs/%Y/%Y%m%d/'
        datapatt2 (str): Secondary data pattern. E.g., '/work/thea.sandmael/radar/%Y%m%d/{radar}/netcdf/{varname}/00.50/%Y%m%d-%H%M%S.netcdf'
        year_thresh (int): The threshold such that >= year_thresh will use datapatt2. Default is 2019.
    """

    if row['radar'] == '-99900':
        return

    cols = row.keys()

    info = {} # contains predictor and target data, and metadata
    info['stormID'] = row['stormID']
    info['az_hs'] = hs[0]
    info['range_hs'] = hs[1]
    info['radarTimestamp'] = row['radarTimestamp']
    info['radar'] = row['radar']
    info['state'] = row['state']
    info['county'] = row['county']
    info['CWA'] = row['CWA']
    info['stormType'] = row['stormType']
    info['AzShear_max'] = row['AzShear_max']
    info['AzShear_mean'] = row['AzShear_mean']
    info['populationDensity_2pt5_min'] = row['populationDensity_2pt5_min']
 
    if 'tornadoWidth' in cols:
        # StormReports csvs
        info['obj_lat'] = row['latitudeExtractCenter']
        info['obj_lon'] = row['longitudeExtractCenter']
        info['rangeExtractCenter'] = row['rangeExtractCenter']                      # Range from the radar for the storm object
        info['distToExtractCenter'] = row['distToExtractCenter']                    # distance to a report, in meter
        info['tornado'] = row['tornado']
        info['hail'] = row['hail']
        info['wind'] = row['wind']
        info['spout'] = row['spout']                                                    # landspout or waterspout
        info['severeType'] = row['severeType']                                          # '-99900', 'Hail', 'Tornado', 'Wind'
        info['tornadoWidth'] = row['tornadoWidth']
        info['tornadoLength'] = row['tornadoLength']
        info['durationMin'] = row['durationMin']                                        # Duration of the tornado in minutes
        info['stormEventsReportID'] = row['stormEventsReportID']                        # From Storm Data: ID assigned by NWS for each individual storm event contained within a storm episode.
        info['oneTorID'] = row['oneTorID']                                              # Common ID for all segments of a tornado. Same as the first segment's stormEventsReportID.
        info['minutesFromReport'] = row['minutesFromReport']                            # The minimum minutes from either the start or end time of a report swath.
        info['preTornadoTracked'] = row['preTornadoTracked']                            # was this stormID tracked pre-tornado?
        info['overallWarningLeadTime'] = row['overallWarningLeadTime']
        info['pointWarningLeadTime'] = row['pointWarningLeadTime']
        info['magnitude'] = -1 if row['magnitude'] == 'U' else float(row['magnitude'])
        if row['spout']:
            label = 'spout'
        elif row['tornado']:
            label = 'tor'
        else:
            label = 'nontor'
    else:
        # pretornado csvs
        info['obj_lat'] = row['latitudeAzShearMax']
        info['obj_lon'] = row['longitudeAzShearMax']
        info['rangeExtractCenter'] = row['rangeAzShearMax']
        info['distToExtractCenter'] = row['distToAzShearMax']
        info['minPreTornado'] = row['minPreTornado']
        info['tornado'] = 1
        ceil_minPreTor = int(np.ceil(row['minPreTornado'] / 15) * 15)
        if ceil_minPreTor == 0:
            ceil_minPreTor = 15
        elif ceil_minPreTor > 60:
            ceil_minPreTor = 120
        label = f'pretor_{ceil_minPreTor}'

    # Anchoring timestamp
    raddt = datetime.strptime(row['radarTimestamp'], '%Y%m%d-%H%M%S')

    for ii, varname in enumerate(varnames):
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
            return
        if abs(raddt - closest_dt).seconds > 600:
            logging.error(f"{raddt} is too far from {closest_dt}. {row['radar']} - {varname}")
            return
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

        # Assert that the shape is correct
        try:
            assert(rad_sector.shape == (hs[0]*2, hs[1]*2))
        except AssertionError as err:
            print(f"{row['radar']}, {row['radarTimestamp']}, {varname}.shape == {rad_sector.shape}")
            return

        if ii == 0:
            # Store indexes
            info['az_idx'] = az_idx
            info['gate_idx'] = gate_idx
            # Store range on first iter
            # 'range' is 1D!
            info['range'] = np.expand_dims(utils.bytescale(r_sector,
                                                           bsinfo['range']['vmin'],
                                                           bsinfo['range']['vmax'],
                                                           min_byte_val=1, # we also invert range, so keep it > 0
                                                           max_byte_val=255
                                           ), axis=-1
            )
            info['out_of_range_mask'] = np.expand_dims( (rad_sector == OUT_OF_RANGE).astype(np.uint8), axis=-1)
     

        # Bytescale the data
        rad_sector_scaled = np.expand_dims(utils.bytescale(rad_sector,
                                                           bsinfo[varname]['vmin'],
                                                           bsinfo[varname]['vmax'],
                                                           min_byte_val=0,
                                                           max_byte_val=255
                                           ), axis=-1
        )

        # Encode range-folded region
        if varname == 'Velocity':
            range_folded_value = radds.attrs.get('RangeFolded', -99901.0)
            range_folded = (rad_sector == range_folded_value).astype(np.uint8)
            info['range_folded_mask'] = np.expand_dims(range_folded, axis=-1)
        

        # Add to info
        info[varname] = rad_sector_scaled


    # Write out the tfrec
    outdir = f"{raddt.strftime(outpatt)}/{label}"
    os.makedirs(outdir, exist_ok=True)
    outfile = f"{outdir}/{row['radar']}_{np.round(row['latitude'],2)}_{np.round(row['longitude'],2)}_{row['radarTimestamp']}.tfrec"

    with tf.io.TFRecordWriter(outfile) as writer:
        example = create_example(info)
        writer.write(example.SerializeToString())

    logging.info(outfile)

#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    # Drive the parallel processing
    
    dataset_type = 'pretornadic' # 'Storm_Reports' or 'pretornadic'
    
    # Load torp dataset
    dataset = TORPDataset(dirpath='/raid/jcintineo/torcnn/torp_datasets/',
                          years=[2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018],
                          #years=[2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
                          dataset_type=dataset_type
    )
    ds = dataset.load_dataframe()
    
    try: 
        # Make EFU = -1
        ds.loc[ds.magnitude == 'U', 'magnitude'] = -1
        # Convert to numeric
        ds.magnitude = pd.to_numeric(ds.magnitude)
    except AttributeError as err:
        logging.error(str(err))
        logging.warning('continuing...')
    
    
    
    # 2011-2018
    datapatt1 = '/data/thea.sandmael/data/radar/%Y%m%d/{radar}/netcdf/{varname}/00.50/%Y%m%d-%H%M%S.netcdf'
    # 2019-2024
    datapatt2 = '/work/thea.sandmael/radar/%Y%m%d/{radar}/netcdf/{varname}/00.50/%Y%m%d-%H%M%S.netcdf'
    
    outpatt = f'/raid/jcintineo/torcnn/tfrecs/%Y/%Y%m%d/'
    
    hs = halfsize = (64, 128)
    
    varnames = ['Velocity',
                'AzShear',
                'DivShear',
                'SpectrumWidth',
                'Reflectivity',
                'RhoHV',
                'PhiDP',
                'Zdr',
    ]
    
    # Get byte-scaling info
    bsinfo = utils.get_bsinfo()
    
    logger = logging.getLogger(__name__)
    logger.info(f"begin extract data and write TFRecords")
    
    # Common arguments to pass in
    # Do these varnames need to match those in the function?
    # I assume so. 
    collect_and_write_tfrec_args = dict(
        hs=halfsize,
        varnames=varnames,
        bsinfo=bsinfo,
        datapatt1=datapatt1,
        outpatt=outpatt,
        datapatt2=datapatt2,
        year_thresh=2019,
    )
    partial_collect_and_write_tfrec = functools.partial(collect_and_write_tfrec, **collect_and_write_tfrec_args)
    
    number_of_rows = len(ds)
    
    max_workers = 20 #min(gfs_columns_extract_workers, os.cpu_count())
    
    with tqdm(total=number_of_rows) as pbar: # progress bar
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
    
            for row in ds.itertuples():
                # Convert the Pandas namedtuple-like object to a regular dictionary.
                # This makes sure only simple, pickleable values are passed.
                # You can also cherry-pick specific columns if 'row' has too many.
                row_as_dict = row._asdict() # This converts it to an OrderedDict, which is pickleable
                
                futures.append(executor.submit(partial_collect_and_write_tfrec, row_as_dict))
    
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing a row: {e}")
                    logger.error(traceback.format_exc())
                finally: 
                    pbar.update(1)

