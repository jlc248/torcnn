import sys,os
sys.path.insert(0, '../')
sys.path.append('../vda')
import utils
import rad_utils
import pyart
import numpy as np
import matplotlib.pyplot as plt
import glob
from datetime import datetime, timedelta
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import yaml

#------------------------------------------------------------------------------------------------------------
def load_config_to_dict(filepath):
    """
    Opens a text file with key-value pairs and returns a python dictionary.
    Handles nested dictionaries and lists automatically.
    """
    try:
        with open(filepath, 'r') as file:
            # Loader=yaml.SafeLoader ensures we don't execute arbitrary code
            config_dict = yaml.load(file, Loader=yaml.SafeLoader)
        return config_dict
    except FileNotFoundError:
        print(f"Error: The file at {filepath} was not found.")
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
    return None

#------------------------------------------------------------------------------------------------------------
def find_nexrad_binary(raddt, dpat):
    # Find closest file <= raddt
    all_files = glob.glob(raddt.strftime(dpat))
    
    if raddt.hour == 0: # check previous day if we're near 00Z
        all_files += glob.glob((raddt - timedelta(days=1)).strftime(dpat))
    all_files = np.sort(all_files)
                             #KVNX20240515_233356_V06 or KVNX20240515_233356_V06.gz or V07
    dts = [datetime.strptime(os.path.basename(ff).split('_V0')[0][4:], '%Y%m%d_%H%M%S') for ff in all_files]
    
    past_dts = [dt for dt in dts if dt <= raddt]

    if past_dts:
        closest_dt = max(past_dts)
        if (raddt - closest_dt).seconds > 360: # 6 min
            print("Not close enough to any past_dts")
            return None
        else:
            ind = dts.index(closest_dt)
            file_path = all_files[ind]
            return file_path
    else:
        print("Couldn't find any past_dts")
        return None

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

# Get byte-scaling info
bsinfo = utils.get_bsinfo()

# Patterns where the raw NEXRAD data live
## 2011-2018
datapatt1 = '/myrorss2/data/thea.sandmael/data/radar/%Y%m%d/{radar}/raw/{radar}*_V0*'
## 2019-2024
datapatt2 = '/myrorss2/work/thea.sandmael/radar/%Y%m%d/{radar}/raw/{radar}*_V0*'

dpat = datapatt1

varnames = {'Velocity':'dealiased_velocity',
            #'SpectrumWidth':'spectrum_width',
            'Reflectivity':'reflectivity',
            'RhoHV':'cross_correlation_ratio',
            #'Zdr':'differential_reflectivity',
}

# Open model
model_dir = '/raid/jcintineo/torcnn/detection/tests/test01'
model_file = f"{model_dir}/fit_conv_model.keras"
c = load_config_to_dict(f"{model_dir}/model_config.txt")
conv_model = load_model(model_file, compile=False)

#'reflectivity', 'cross_correlation_ratio', 'dealiased_velocity'])

truth_file = '/raid/jcintineo/torcnn/detection/truth_files_100km60min/2021/20210328/KDIX_20210328-230847.txt'
truth_file = '/raid/jcintineo/torcnn/detection/truth_files_100km60min/2013/20130520/KTLX_20130520-195544.txt'
with open(truth_file, 'r') as f:
    # list of lists of floats, or an empty list
    detects = [[float(x) for x in line.strip().split()] for line in f]

# Find binary NEXRAD file
raddt = datetime.strptime(os.path.basename(truth_file).split('_')[1], '%Y%m%d-%H%M%S.txt')
rad_name = os.path.basename(truth_file)[0:4]
dpat = dpat.replace('{radar}', rad_name)
file_path = find_nexrad_binary(raddt, dpat)

raddict = rad_utils.get_remapped_radar_data(file_path,
                                            raddt,
                                            target_tilt=0.5,
                                            fields=list(varnames.values()),
                                            x_limits=(-160000.0, 160000.0), # in m
                                            y_limits=(-160000.0, 160000.0), # in m
                                            grid_shape=(512, 512)
)

#fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
#ax[0].imshow(raddict['dealiased_velocity'], vmin=-30, vmax=30)
##ax[1].imshow(raddict['reflectivity'], vmin=0, vmax=60, cmap='plasma')
#ax[1].imshow(raddict['cross_correlation_ratio'], vmin=0.5, vmax=1, cmap='plasma')
#plt.show()
#sys.exit()

# Make input tensor
channel_list = []
for chan in c['channels']:
    data = np.expand_dims(
               utils.bytescale(
                   np.nan_to_num(raddict[varnames[chan]], nan=-999.),
                   bsinfo[chan]['vmin'],
                   bsinfo[chan]['vmax'],
                   min_byte_val=0,
                   max_byte_val=255
               ).astype(float)/255,
               axis=(0, -1)
           )

    channel_list.append(data)

# Combine all at once along the last axis
pred_tensor = np.concatenate(channel_list, axis=-1)
print(pred_tensor.shape)

## Check for NaNs in the input itself
#print(f"Input NaN count: {tf.reduce_sum(tf.cast(tf.math.is_nan(pred_tensor), tf.int32)).numpy()}")
#print(f"Input Max: {tf.reduce_max(pred_tensor).numpy()}")
#print(f"Input Min: {tf.reduce_min(pred_tensor).numpy()}")
#
## Run a single layer check if input is clean
#preds = conv_model(pred_tensor, training=False)
#if tf.reduce_any(tf.math.is_nan(preds)):
#    print("NaN detected in output!")
#sys.exit()

preds = conv_model.predict(pred_tensor)
print(preds.shape)

# Split the channels
obj_logits = preds[..., 0:1]
box_params = preds[..., 1:5]
class_logits = preds[..., 5:7]


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18,6))

# Apply the activations manually
objectness = tf.sigmoid(obj_logits).numpy()
ax[0].imshow(np.squeeze(objectness))
ax[1].imshow(raddict[varnames['Velocity']], vmin=-30, vmax=30, cmap='RdBu')
ax[2].imshow(raddict[varnames['Reflectivity']], vmin=0, vmax=60, cmap='plasma')
x = [d[1]*64 for d in detects]
y = [d[2]*64 for d in detects]

x8 = [_*8 for _ in x]
y8 = [_*8 for _ in y]
ax[0].scatter(x,y,color='red', marker='x', label='Ground Truth')
ax[1].scatter(x8,y8,color='red', marker='x', label='Ground Truth')
ax[2].scatter(x8,y8,color='red', marker='x', label='Ground Truth')
plt.show()
sys.exit()
probabilities = tf.nn.softmax(class_logits, axis=-1).numpy()

# Now 'objectness' is [0, 1] and 'probabilities' sum to 1.0
tvs_conf = probabilities[..., 0]
sev_conf = probabilities[..., 1]

print(np.min(objectness), np.max(objectness), len(np.where(objectness[0] >= 0.1)[0]))
print(np.min(tvs_conf), np.max(tvs_conf))
print(np.min(sev_conf), np.max(sev_conf))
