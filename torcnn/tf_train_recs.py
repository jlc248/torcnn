from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib
#matplotlib.use('Agg')
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
#os.environ["CUDA_VISIBLE_DEVICES"]="2";

import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import glob

import tensorflow as tf
tf.keras.backend.clear_session()
#from tensorflow import keras
#import tensorflow.keras.backend as K
#from tensorflow.keras import mixed_precision
#set the mixed precision policy
#mixed_precision.set_global_policy('mixed_float16')

import tf_models
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
import sys
import utils
import argparse
import logging
import logging_def
#from LRFinder import LRFinder

#gpus = tf.config.list_physical_devices('GPU')
#for gpu in gpus:
#  tf.config.experimental.set_memory_growth(gpu, True)

parser = argparse.ArgumentParser()
parser.add_argument('-m','--model',help="import this trained CNN model",type=str)
parser.add_argument('-c','--config_file',help="enter a config pkl file (usu. from a previous training session)",type=str)

args = parser.parse_args()


if not(args.config_file):
    import tf_config
    config = tf_config.tf_config()
else:
    config = pickle.load(open(args.config_file,'rb'))

#some global vars
INPUTS = config['inputs']
BINARIZE = config['binarize']
TARGETS= config['targets']
ntargets = len(TARGETS)
SCALAR_VARS = config['scalar_vars']
NGPU = config['ngpu']
BATCHSIZE = config['batchsize']
PS = config['ps']
try: STD = config['img_aug']['random_noise']
except KeyError: pass

byte_scaling_vals = config['byte_scaling_vals']

# get bsinfo for scalars
if SCALAR_VARS:
    bsinfo = utils.get_bsinfo()

##################################################################################################################
def tf_bytescale(data_arr, vmin, vmax, min_byte_val=0, max_byte_val=255):
    """
    Scales a TensorFlow tensor to a specified byte range.

    Args:
        data_arr (tf.Tensor): The input tensor to be scaled.
        vmin (tf.Tensor or float): The minimum value of the original data range.
        vmax (tf.Tensor or float): The maximum value of the original data range.
        min_byte_val (tf.Tensor or int, optional): The minimum byte value for the scaled output. Defaults to 0.
        max_byte_val (tf.Tensor or int, optional): The maximum byte value for the scaled output. Defaults to 255.

    Returns:
        tf.Tensor: The bytescaled tensor as tf.uint8.
    """
    # Ensure inputs are tensors for consistent operations.
    # Convert scalar Python floats/ints to TensorFlow constants if not already.
    vmin_tensor = tf.cast(vmin, tf.float32)
    vmax_tensor = tf.cast(vmax, tf.float32)
    min_byte_val_tensor = tf.cast(min_byte_val, tf.float32)
    max_byte_val_tensor = tf.cast(max_byte_val, tf.float32)
    data_tensor = tf.cast(data_arr, tf.float32)

    # Calculate the range of the target byte values.
    byte_range = max_byte_val_tensor - min_byte_val_tensor

    # The core scaling operation, all done using TensorFlow primitives
    # which are highly optimized for GPU/TPU.
    # The mathematical formula is applied to the entire tensor at once.
    scaled_data = ((data_tensor - vmin_tensor) / (vmax_tensor - vmin_tensor)) * byte_range + min_byte_val_tensor

    # Clip values to ensure they stay within the specified byte range.
    # `tf.clip_by_value` is the efficient TensorFlow equivalent of your
    # manual clipping with boolean indexing.
    clipped_data = tf.clip_by_value(scaled_data, clip_value_min=min_byte_val_tensor, clip_value_max=max_byte_val_tensor)

    # Round the values to the nearest integer.
    rounded_data = tf.round(clipped_data)

    # Cast to tf.uint8 as the final output.
    return tf.cast(rounded_data, tf.uint8)

##################################################################################################################
def parse_tfrecord_fn(example):

    feature_description = {}
   
    # Define the target features 
    for target in TARGETS:           
        feature_description[target] = tf.io.FixedLenFeature([], tf.int64)

    # Define scalar predictor features
    for scalar_var in SCALAR_VARS:
        feature_description[scalar_var] = tf.io.FixedLenFeature([], tf.float32)
    
    # Define the n-D predictor features
    for inp in INPUTS:
        for chan in inp:
            feature_description[chan] =  tf.io.FixedLenFeature([], tf.string)

    # Parse the single example
    features = tf.io.parse_single_example(example, feature_description)

    # Reshape the n-D inputs
    for inp in INPUTS:
        for chan in inp:
            features[chan] = tf.reshape(tf.io.parse_tensor(features[chan], tf.uint8), [PS[0], PS[1],1])

    return features
##################################################################################################################
def prepare_sample(features):

    # binarize targets and create sample_weight or mask
    for ii, target in enumerate(TARGETS):
        targetTmp = features[target]  
        if ii == 0:
            targetInt = targetTmp
        else:
            targetInt = tf.stack([targetInt, targetTmp])


    inputs = {}

    # These Inputs are already byte-scaled
    for ii,inp in enumerate(INPUTS):  #for each tf.layers.Input in the model construction
        for chIdx, varname in enumerate(inp):             #for each channel in the tf.layers.Input
            data = features[varname]
            if chIdx == 0:
                tensor = data
            else:
                tensor = tf.concat([tensor, data], axis=2)

        inputs[f'input_{ii}'] = tensor


    if SCALAR_VARS:
        # These need to be normalized/scaled
        for idx,scalar_var in enumerate(SCALAR_VARS): 
            scalarTmp = tf_bytescale(features[scalar_var], bsinfo[scalar_var]['vmin'], bsinfo[scalar_var]['vmax'])
            if idx == 0:
                scalars = scalarTmp
            else:
                scalars = tf.stack([scalars, scalarTmp])
        # Using ii from block above
        inputs[f'input_{ii+1}'] = scalars

    # Sample weight should be same shape as targetImage
    return inputs, targetInt #, sample_weight

#################################################################################################################################
def apply_augmentations(input_dict, target_img):
    """Apply data augmentations

      Args:
          x: input_dict, target_img

      Returns:
          augmented input_dict and target_img
      """

    #See https://benihime91.github.io/blog/deeplearning/tensorflow2.x/data_augmentation/image/2020/11/15/image-augmentation-tensorflow.html

    #randomly rotates an image by a factor of 90 degree
    rotate_prob = tf.random.uniform([], 0, 1.0, dtype=tf.float32)

    #We often have multiple TF/Keras Inputs in the model
    for key,inp in input_dict.items(): #N.B. for some reason, we can't put for-loops within the if/elif/else block.
        if rotate_prob > 0.66:
            inp = tf.image.rot90(inp, k=3); krot=3 # rotate 270º
        elif rotate_prob > 0.33:
            inp = tf.image.rot90(inp, k=2); krot=2 # rotate 180º
        else:
            inp = tf.image.rot90(inp, k=1); krot=1 # rotate 90º
        input_dict[key] = inp

    target_img = tf.image.rot90(target_img, k=krot)

    return input_dict, target_img

#################################################################################################################################
def apply_noise(input_dict, target_img):
    """Apply random Gaussian noise augmentation

      Args:
          x: input_dict, target_img

      Returns:
          augmented input_dict and target_img
    """

    #See https://stackoverflow.com/questions/41174769/additive-gaussian-noise-in-tensorflow
    #STD is global variable above
    for key,inp in input_dict.items():    #samples should already be scaled (i.e., mean around 0, std of 1).
        noise = tf.random.normal(shape=tf.shape(inp), mean=0.0, stddev=STD, dtype=tf.float32)
        input_dict[key] = inp + noise

    return input_dict, target_img
#################################################################################################################################

# Train: Create list of training datasets
print('\nBuilding training Dataset')

AUTOTUNE=tf.data.AUTOTUNE
BUFFER_SIZE = 16 * 1024**2 # 16MB

print(config['train_list'])

#train_filenames = tf.io.gfile.glob(config['train_list'])
# Use a list comprehension to glob both patterns, as the tf.io.gfile.glob was segfaulting
# tf.io.gfile.glob when using a variety of sources, like cloud and local.
# glob.glob will only work on local disk.
train_filenames = []
for pattern in config['train_list']:
    train_filenames.extend(glob.glob(pattern))

n_tsamples = len(train_filenames)
print('n_tsamples',n_tsamples)

train_ds = (tf.data.TFRecordDataset(train_filenames, num_parallel_reads=AUTOTUNE, buffer_size=BUFFER_SIZE)
           .shuffle(1024)
           .map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
           .map(prepare_sample, num_parallel_calls=AUTOTUNE)
           .batch(BATCHSIZE)
           .prefetch(AUTOTUNE)
)


# Augment data (optional)
if 'random_rotation' in config['img_aug']:
    aug_ds = train_ds.map(apply_augmentations, num_parallel_calls=AUTOTUNE)
    # Concatenate the original and augmented datasets (doubles dataset)
    train_ds = train_ds.concatenate(aug_ds)
    del aug_ds
if 'random_noise' in config['img_aug']: #original and concatenated augmentations are noised
    train_ds = train_ds.map(apply_noise, num_parallel_calls=AUTOTUNE)

# For quick_looks
#for inputs,labels in train_ds:
#    inp0 = inputs['input_0'].numpy()
#    inp1 = inputs['input_1'].numpy()
#    label = labels.numpy()
#    print(inp0.shape, inp1.shape, label.shape)
#    sys.exit()
#    fig,ax = plt.subplots(nrows=1,ncols=3)
#    for ii,axis in enumerate(ax.ravel()):
#        axis.axis('off')
#        if ii==0: axis.imshow(inp0[0,...,0])
#        if ii==1: axis.imshow(inp0[0,...,1])
#        if ii==2: axis.imshow(inp0[0,...,2])
#        print('scalars', inp1)
#        print('label', label)
#    plt.show()
#    sys.exit()


print('\nBuilding validation Dataset')

outdir = config['outdir']
if not args.model:
    os.makedirs(outdir)

# Change AutoShard policy. Not exactly sure what this does, but it at least gets rid of annoying warning messages.
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
train_ds = train_ds.with_options(options)

#val_filenames = tf.io.gfile.glob(config['val_list'])
# Use a list comprehension to glob both patterns, as the tf.io.gfile.glob was segfaulting
# tf.io.gfile.glob when using a variety of sources, like cloud and local.
# glob.glob will only work on local disk.
val_filenames = []
for pattern in config['val_list']:
    val_filenames.extend(glob.glob(pattern))
n_vsamples = len(val_filenames)
print('n_vsamples',n_vsamples)

val_ds = (tf.data.TFRecordDataset(val_filenames, num_parallel_reads=AUTOTUNE, buffer_size=BUFFER_SIZE)
           .map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
           .map(prepare_sample, num_parallel_calls=AUTOTUNE)
           .batch(BATCHSIZE)
           .prefetch(AUTOTUNE)
)
val_ds = val_ds.with_options(options)


# Print out config options
ofile = os.path.join(outdir,'model_config.txt')
of = open(ofile,'w')
for key, value in config.items():
    of.write(str(key) + ': ' + str(value) + '\n')
of.write(f"Number of training samples: {n_tsamples}\n")
of.write(f"Number of validation samples: {n_vsamples}\n")
of.close()
pickle.dump(config,open(os.path.join(outdir,'model_config.pkl'),'wb'))

#===================================================================================================================#

csvlogger = CSVLogger(f"{outdir}/log.csv", append=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=config['es_patience'], min_delta=0.00005)
mcp_save = ModelCheckpoint(os.path.join(outdir,'model-{epoch:02d}-{val_loss:03f}.h5'),save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', cooldown=config['rlr_cooldown'], verbose=1,# min_delta=0.00001,
                 factor=config['rlr_factor'], patience=config['rlr_patience'], mode='min')
#tboard = TensorBoard(log_dir='/home/jcintineo/pscnn/pscnn/tb_logs/',
#                     histogram_freq=1,
#                     write_images=True,
#                     profile_batch=(200,300)) #profile from batch X to batch Y
#lr_finder = LRFinder(min_lr=1e-7, max_lr=1e-1)
#callbacks = [lr_finder]
callbacks = [early_stopping, mcp_save, reduce_lr_loss, csvlogger] #, tboard]


if NGPU > 1:
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    with strategy.scope():
        if args.model:
            logging.info('Opening ' + args.model)
            metrics = tf_models.get_metrics(num_targets=ntargets)
            conv_model = load_model(args.model) #, custom_objects=CUSTOM_METRICS)
            conv_model.compile(loss=conv_model.loss, optimizer=conv_model.optimizer, metrics=metrics)
            print(conv_model.summary())
            initial_epoch = int(os.path.basename(args.model).split('-')[1])
        else:
            tf_model = getattr(tf_models,config['cnn'])
            conv_model = tf_model(config=config)
            initial_epoch = 0
else:
    if args.model:
        logging.info('Opening ' + args.model)
        metrics = tf_models.get_metrics(num_targets=ntargets)
        conv_model = load_model(args.model) #, custom_objects=CUSTOM_METRICS)
        conv_model.compile(loss=conv_model.loss, optimizer=conv_model.optimizer, metrics=metrics)
        print(conv_model.summary())
        initial_epoch = int(os.path.basename(args.model).split('-')[1])
    else:
        tf_model = getattr(tf_models,config['cnn'])
        conv_model = tf_model(config=config)
        initial_epoch = 0
#sys.exit()
#-----------------------------------------------------------------------------------------------------
print('input shapes:',config['input_tuples'])
logging.info('Fitting model...')

#FIT THE MODEL
history = conv_model.fit(
          x=train_ds,
          verbose=1,
          epochs=config['nepoch'],
          validation_data=val_ds,
          initial_epoch=initial_epoch,
          callbacks=callbacks)
#shuffle has no effect if generator OR a tf.data.Dataset

logging.info('Saving model and training history...')
#copy the model with the best val_loss
best_model = np.sort(glob.glob(f"{outdir}/model-*.h5"))[-1]
shutil.copy(best_model,f"{os.path.dirname(best_model)}/fit_conv_model.h5")

#load the best-val_loss model
conv_model = load_model(f"{outdir}/fit_conv_model.h5",compile=False) #False b/c no more training


pickle.dump(history.history,open(os.path.join(outdir,"training_history.pkl"),"wb"),protocol=4)
try: utils.training_history_figs(history.history,outdir)
except KeyError: pass #new run with incomplete history. Just skip


#save RAM
del train_ds, history


utils.add_scores_to_file(outdir)
