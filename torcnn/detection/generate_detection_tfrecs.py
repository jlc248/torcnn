import traceback
import logging
import functools
import concurrent
from tqdm import tqdm
import numpy as np
import os,sys
import tensorflow as tf
sys.path.insert(0, '../')
sys.path.append('../vda')
import utils
import rad_utils
import time
from datetime import datetime, timedelta
import glob
import pyart

OUT_OF_RANGE = -999

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
#def create_example(channels):
#    features = {}
#    for key,item in channels.items():
#        if isinstance(item, bytes):
#            # If the item is already bytes (e.g., a serialized 2D array or image bytes)
#            features[key] = bytes_feature(item)
#        elif isinstance(item, (float, np.float64, np.float32)):
#            features[key] = float_feature(item)
#        elif isinstance(item, (int, np.int64, np.int32)):
#            features[key] = int64_feature(item)
#        elif isinstance(item, str):
#            features[key] = bytes_feature(item)
#        elif isinstance(item, np.ndarray):
#            # This case handles raw numpy arrays that *haven't* been serialized yet.
#            # If you *sometimes* have raw numpy arrays and *sometimes* pre-serialized bytes,
#            # this would be the place to serialize the raw numpy array.
#            # Example: Serialize a float32 numpy array to bytes if not already bytes
#            # print(f"Warning: Item for key '{key}' is a numpy array. Serializing it to bytes.")
#            tensor_item = tf.convert_to_tensor(item) # TensorFlow will infer dtype or you can specify
#            serialized_tensor = tf.io.serialize_tensor(tensor_item)
#            features[key] = bytes_feature(serialized_tensor.numpy())
#        else:
#            # Handle any other types or raise an error for unsupported types
#            raise ValueError(f"Unsupported data type for key '{key}': {type(item)}")
#
#    return tf.train.Example(features=tf.train.Features(feature=features))

def create_example(radar_data, labels):
    """
    radar_data: dictionary of 2D numpy arrays 
    labels: List of lists or 2D array [[cls, x, y, w, h], ...]
    """

    features ={} 

    # Convert radar image to raw bytes
    for key,item in radar_data.items():
        # Ensure it is float32 for consistency
        radar_bytes = item.astype(np.float32).tobytes()
        features[key] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[radar_bytes]))

    # Flatten labels into a 1D list [cls1, x1, y1, w1, h1, cls2, x2, ...]
    # If no detections, this remains an empty list
    flattened_labels = np.array(labels, dtype=np.float32).flatten()
    
    # The 'labels' are stored as a variable-length list of floats
    features['labels'] = tf.train.Feature(float_list=tf.train.FloatList(value=flattened_labels))

    return tf.train.Example(features=tf.train.Features(feature=features))
#--------------------------------------------------------------------------------------------------
def collect_and_write_tfrec(target_file,
                            varnames,
                            datapatt1,
                            outpatt,
                            datapatt2=None,
                            year_thresh=2019,
):

    """
    Collect the L2 radar data and write a TFRecord.
    Args:
        target_file (str): txt file where each row is info about a circulation object.
        varnames (dict): List of pairs of our outnames and pyart variable names.
        datapatt1 (str): Full path data pattern. E.g., '/data/thea.sandmael/data/radar/%Y%m%d/{radar}/raw/{radar}*_V06'
        outpatt (str): Pattern for root outdir. E.g., '/raid/jcintineo/torcnn/detection/tfrecs/%Y/%Y%m%d/{radar}_%Y%m%d-%H%M%S.tfrec'
        datapatt2 (str): Secondary data pattern. E.g., '/work/thea.sandmael/radar/%Y%m%d/{radar}/raw/{radar}*_V06'
        year_thresh (int): The threshold such that >= year_thresh will use datapatt2. Default is 2019.
    """

    # Anchoring timestamp
    basefile = os.path.basename(target_file)
    dt_str = basefile[5:]
    radar_str = basefile[0:4]
    raddt = datetime.strptime(dt_str, '%Y%m%d-%H%M%S.txt')
    # NB: the raw files will often have timestamps slightly before this raddt due to SAILS modes

    opat = outpatt.replace('{radar}', radar_str)

    # Check datapatt2 if defined
    if datapatt2:
        if raddt.year >= year_thresh:
            datapatt = datapatt2 
        else:
            datapatt = datapatt1
    else:
        datapatt = datapatt1

    dpat = datapatt.replace('{radar}', radar_str)

    # Find closest file <= raddt
    all_files = glob.glob(raddt.strftime(dpat))
    if raddt.hour == 0: # check previous day if we're near 00Z
        all_files += glob.glob((raddt - timedelta(days=1)).strftime(dpat))
    all_files = np.sort(all_files)
                             #KVNX20240515_233356_V06
    dts = [datetime.strptime(os.path.basename(ff)[4:], '%Y%m%d_%H%M%S_V06') for ff in all_files]
   
    past_dts = [dt for dt in dts if dt <= raddt]
   
    if past_dts:
        closest_dt = max(past_dts)
        if (raddt - closest_dt).seconds > 240:
            logging.error(f"{raddt} is too far from {closest_dt}. {radar_str}")
            return
        else:
            ind = dts.index(closest_dt)
            file_path = all_files[ind]
    else:
        logging.error(f"ValueError: no closest_dt, {radar_str}, {raddt.strftime('%Y%m%d-%H%M%S')}")
        return
    
    # Get truth/target data
    with open(target_file, 'r') as f:
        # list of lists of floats, or an empty list
        detects = [[float(x) for x in line.strip().split()] for line in f] 

    # Remap NEXRAD data and dealias velocity using pyart
    raddict = rad_utils.get_remapped_radar_data(file_path,
                                                raddt,
                                                target_tilt=0.5,
                                                fields=list(varnames.values()),
                                                x_limits=(-160000.0, 160000.0), # in m
                                                y_limits=(-160000.0, 160000.0), # in m
                                                grid_shape=(512, 512)
    )

    info = {}
    # Expand dims and remove nans
    for key,val in varnames.items():

        info[key] = np.expand_dims(np.nan_to_num(raddict[val], nan=-999) , axis=-1)


    outfile = raddt.strftime(opat)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    with tf.io.TFRecordWriter(outfile) as writer:
        example = create_example(info, detects)
        writer.write(example.SerializeToString())

    logging.info(outfile)

#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    # Drive the parallel processing

    # Patterns where the raw NEXRAD data live    
    ## 2011-2018
    datapatt1 = '/myrorss2/data/thea.sandmael/data/radar/%Y%m%d/{radar}/raw/{radar}*_V06'
    ## 2019-2024
    datapatt2 = '/myrorss2/work/thea.sandmael/radar/%Y%m%d/{radar}/raw/{radar}*_V06'
     
    # Output pattern for tfrecs
    outpatt = '/raid/jcintineo/torcnn/detection/tfrecs/%Y/%Y%m%d/{radar}_%Y%m%d-%H%M%S.tfrec'
    
    # Pattern for truth files.
    ## nontor truth files contain 0 points.
    ## tor truth files contain info for 1 or more points
    target_glob = f'/raid/jcintineo/torcnn/detection/truth_files/2024/20240515/K???_*.txt'

    target_files = np.sort(glob.glob(target_glob))

    # output varnames and pyart strings
    varnames = {'Velocity':'dealiased_velocity',
                'SpectrumWidth':'spectrum_width',
                'Reflectivity':'reflectivity',
                'RhoHV':'cross_correlation_ratio',
                'Zdr':'differential_reflectivity',
    }
    
    logger = logging.getLogger(__name__)
    logger.info(f"begin extract data and write TFRecords")
    
    # Common arguments to pass in
    # Do these varnames need to match those in the function?
    # I assume so. 
    collect_and_write_tfrec_args = dict(
        varnames=varnames,
        datapatt1=datapatt1,
        outpatt=outpatt,
        datapatt2=datapatt2,
        year_thresh=2019,
    )
    partial_collect_and_write_tfrec = functools.partial(collect_and_write_tfrec, **collect_and_write_tfrec_args)
    
    number_of_files = len(target_files)
    
    max_workers = 20 #min(gfs_columns_extract_workers, os.cpu_count())
    
    with tqdm(total=number_of_files) as pbar: # progress bar
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
    
            for target_file in target_files:
                
                futures.append(executor.submit(partial_collect_and_write_tfrec, target_file))
    
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing a row: {e}")
                    logger.error(traceback.format_exc())
                finally: 
                    pbar.update(1)

