import matplotlib

#matplotlib.use("Agg")

import os
import numpy as np
import sys
import argparse
import glob
from datetime import datetime, timedelta, timezone
import time
import pickle
import matplotlib.pyplot as plt
import collections
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--indir",
    help="Directory containing trained CNN model and config.",
    type=str,
)
parser.add_argument(
    "-o",
    "--outdir",
    help="Output directory for model and figures. Default=args.indir + '/TEST/'",
)
parser.add_argument(
    "-m",
    "--model",
    help="Use this model instead of fit_conv_model.keras in --indir.",
    type=str,
)
parser.add_argument(
    "-f",
    "--filelist",
    help="Get a list of files from this pandas DataFrame for colated predictions with TORP.",
    type=str,
    default=[],
)
parser.add_argument(
    "-sd",
    "--save_data",
    help="Save the predictions, labels, and subclass information to a dataframe. \
         Default is to just save the metrics (model.evalute).",
    action="store_true",
)
parser.add_argument(
    "--mask",
    help="Set this to mask predictions.",
    action="store_true"
)
parser.add_argument(
    "-ng",
    "--ngpus",
    help="The number of GPUs to use. Default is to use zero.",
    type=int,
    default=0,
)
parser.add_argument(
    "-nc",
    "--ncpus",
    help="Limit TF to this number of CPUs. Default is to use all available.",
    type=int,
    default=-1,
)

args = parser.parse_args()

indir = args.indir

config = pickle.load(open(os.path.join(indir, "model_config.pkl"), "rb"))
model_file = args.model if (args.model) else f"{indir}/fit_conv_model.keras"

outdir = args.outdir if (args.outdir) else os.path.join(indir, "TEST")

if args.filelist:
    tmpDF = pd.read_pickle(args.filelist)
    filelist = list(tmpDF.tfrec)
else:
    filelist = []

ncpus = args.ncpus
ngpu = args.ngpus
mask = args.mask

PS = config["ps"]

INPUTS = config["inputs"]
TARGETS = config["targets"]
SCALAR_VARS = config["scalar_vars"] 
BSINFO = config["byte_scaling_vals"] # for scaling from 0 to 1 or -1 to +1 too
if ngpu == 0:
    batchsize = 256
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    batchsize = ngpu * 256
# else:
#  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#  os.environ["CUDA_VISIBLE_DEVICES"]="2" #hard-coded!
import keras
import tensorflow as tf
from tensorflow.keras.models import Model
import tf_metrics

optimizer = (
    tfa.optimizers.AdaBelief(learning_rate=0.0001)
    if (config["optimizer"].lower() == "adabelief")
    else "Adam"
)

if ncpus > 0:
    # limiting CPUs
    num_threads = 30
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = str(num_threads)
    tf.config.threading.set_inter_op_parallelism_threads(num_threads)
    tf.config.threading.set_intra_op_parallelism_threads(num_threads)

from tensorflow import keras

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ------------------------------------------------------------------------------------------------------------------------------------------------------
def get_metrics():

    metrics = ["accuracy"]

    custom_metrics = collections.OrderedDict()
    custom_metrics["loss"] = "foo"
    custom_metrics["accuracy"] = "foo"

    ntargets = len(TARGETS)

    for jj in range(ntargets):
        # AUPRC
        met_name = f'auprc_index{jj}'
        met = tf_metrics.AUC(name=met_name, curve='PR', index=jj)
        metrics.append(met)
        custom_metrics[met_name] = met 

        # Brier Score
        met_name = f'brier_score_index{jj}'
        met = tf_metrics.BrierScore(name=met_name, index=jj)
        metrics.append(met)
        custom_metrics[met_name] = met

        levs = np.arange(0.05,1,0.05)

        for ii, lev in enumerate(levs):
            # CSI
            met_name = f'csi{str(int(lev*100)).zfill(2)}_index{jj}'
            met = tf_metrics.csi(use_soft_discretization=False,
                                          hard_discretization_threshold=lev,
                                          name=met_name,
                                          index=jj
            )
            metrics.append(met)
            custom_metrics[met_name] = met

            # POD
            met_name = f'pod{str(int(lev*100)).zfill(2)}_index{jj}'            
            met = tf_metrics.pod(use_soft_discretization=False,
                                          hard_discretization_threshold=lev,
                                          name=met_name,
                                          index=jj
            )
            metrics.append(met)
            custom_metrics[met_name] = met

            # FAR
            met_name = f'far{str(int(lev*100)).zfill(2)}_index{jj}'
            met = tf_metrics.far(use_soft_discretization=False,
                                          hard_discretization_threshold=lev,
                                          name=met_name,
                                          index=jj
            )
            metrics.append(met)
            custom_metrics[met_name] = met

            if ii == 0:
               t0 = 0
            else:
               t0 = levs[ii-1]

            # Observed counts
            met_name = f"obsct{str(int(round(lev*100))).zfill(2)}_index{jj}"
            met = tf_metrics.ObsCt(
                threshold1=t0, threshold2=lev, name=met_name, index=jj
            )
            metrics.append(met)
            custom_metrics[met_name] = met

            # Forecast counts
            met_name = f"fcstct{str(int(round(lev*100))).zfill(2)}_index{jj}"
            met = tf_metrics.FcstCt(
                threshold1=t0, threshold2=lev, name=met_name, index=jj
            )
            metrics.append(met)
            custom_metrics[met_name] = met


    return custom_metrics, metrics

# ----------------------------------------------------------------------------------------------------------------
def normalize_channel(tensor, channel_name):
    info = BSINFO[channel_name]
    c_min = info['vmin']
    c_max = info['vmax']

    # Cast to float
    #tensor = tf.cast(tensor, tf.float32)

    # First, assume the uint8 (0-255) maps to the min/max range
    # Scaling uint8 to the physical units first:
    physical = (tensor / 255.0) * (c_max - c_min) + c_min

    if channel_name == 'Velocity':
        # Scale physical -80 to 80 -> -1.0 to 1.0
        # Formula: (val - center) / half_range
        normalized = physical / max(abs(c_min), abs(c_max))
    elif channel_name == 'AzShear' or channel_name == 'DivShear':
        # For shear, we do a Power Transform (x^0.5)
        # First, let's shift it to be purely positive for the transform
        shifted = physical + abs(c_min) # Now 0.0 to 0.05
        # Apply square root to "stretch" the low values
        stretched = tf.sqrt(shifted)
        # Final min-max scale to 0-1
        normalized = stretched / tf.sqrt(c_max - c_min)
    else:
        # Scale physical min to max -> 0.0 to 1.0
        # Formula: (val - min) / (max - min)
        normalized = (physical - c_min) / (c_max - c_min)

    return tf.clip_by_value(normalized, -1.0 if channel_name == 'Velocity' else 0.0, 1.0)

#---------------------------------------------------------------------------------------------------------------
# Moved feature_description outside for efficiency
feature_description = {}
for target in TARGETS:
    feature_description[target] = tf.io.FixedLenFeature([1], tf.int64)
for scalar_var in SCALAR_VARS:
    feature_description[scalar_var] = tf.io.FixedLenFeature([1], tf.float32)
for inp in INPUTS:
    for chan in inp:
        if chan != 'range_inv':
            feature_description[chan] = tf.io.FixedLenFeature([], tf.string)
if args.save_data:
    feature_description['pretorMinutes'] = tf.io.FixedLenFeature([], tf.int64)
    feature_description['magtornado'] = tf.io.FixedLenFeature([], tf.int64)
#---------------------------------------------------------------------------------------------------------------
def parse_and_prepare(example_proto, save_data=False):

    # Parse the record
    features = tf.io.parse_single_example(example_proto, feature_description)

    # Process Targets
    for ii, target in enumerate(TARGETS):
        targetTmp = features[target]
        if ii == 0:
            targetInt = targetTmp
        else:
            targetInt = tf.stack([targetInt, targetTmp])

    # Process Inputs
    processed_inputs = {}

    parsed_tensors = {}
    for inp in INPUTS:
        for chan in inp:
            if chan == 'range':
                parsed_tensors[chan] = tf.cast(tf.reshape(tf.io.parse_tensor(features[chan], tf.uint8), [PS[1], 1]), tf.float32)
            elif chan == 'range_inv':
                # Just a placeholder, logic handled below
                continue
            else:
                parsed_tensors[chan] = tf.cast(tf.reshape(tf.io.parse_tensor(features[chan], tf.uint8), [PS[0], PS[1], 1]), tf.float32)
            if chan != 'range_folded_mask' and chan != 'out_of_range_mask':
                parsed_tensors[chan] = normalize_channel(parsed_tensors[chan], chan)

    # Build 'radar' and 'coords' inputs
    # Assuming INPUTS[0] is radar and INPUTS[1] is coords
    for ii, inp_group in enumerate(INPUTS):
        group_tensors = []
        for varname in inp_group:
            if varname == 'range':
                data = parsed_tensors['range']
                data = tf.repeat(tf.expand_dims(data, axis=0), repeats=PS[0], axis=0)
            elif varname == 'range_inv':
                # Use the 'range' data for the inversion
                data = parsed_tensors['range']
                data = tf.repeat(tf.expand_dims(1.0 / data, axis=0), repeats=PS[0], axis=0)
            else:
                data = parsed_tensors[varname]
            group_tensors.append(data)

        # Concat along channel axis (axis=2)
        key = 'radar' if ii == 0 else 'coords'
        processed_inputs[key] = tf.concat(group_tensors, axis=-1)

    # Process Scalars
    if SCALAR_VARS:
        scalar_list = []
        for var in SCALAR_VARS:
            #val = tf_bytescale(features[var], BSINFO[var]['vmin'], BSINFO[var]['vmax'])
            c_min = BSINFO[var]['vmin']
            c_max = BSINFO[var]['vmax']
            # Normalize 0 to 1
            val = (features[var] - c_min) / (c_max - c_min)
            scalar_list.append(val)
        processed_inputs['scalars'] = tf.stack(scalar_list)

    if save_data:
        pretorMinutes = features['pretorMinutes']
        magtornado = features['magtornado']
        return processed_inputs, targetInt, pretorMinutes, magtornado
    else:
        return processed_inputs, targetInt

#---------------------------------------------------------------------------------------------------------------
################################################################################################################
################################################################################################################

# Load model and set up GPUs

if ngpu > 1:
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    with strategy.scope():
        custom_objs, metrics = get_metrics()
        conv_model = keras.models.load_model(model_file, custom_objects=custom_objs, compile=False)
        # If true, then the mask is being used via sample_weight. We need to use weighted_metrics
        if mask:
            conv_model.compile(loss=config["loss_fcn"], weighted_metrics=metrics)
        else:
            conv_model.compile(loss=config["loss_fcn"], metrics=metrics)
else:
    custom_objs, metrics = get_metrics()
    conv_model = keras.models.load_model(model_file, custom_objects=custom_objs, compile=False)
    # If true, then the mask is being used via sample_weight. We need to use weighted_metrics
    if mask:
        conv_model.compile(loss=config["loss_fcn"], weighted_metrics=metrics)
    else:
        conv_model.compile(loss=config["loss_fcn"], metrics=metrics)

print(conv_model.summary())

AUTOTUNE = tf.data.AUTOTUNE

if filelist:
    # this is a custom, ordered list of tfrecs
    val_lists = [filelist]
else:
    # Create the test dataset
    year = "2018"
    inroot = f"/work2/jcintineo/torcnn/tfrecs_combined/{year}*/"
    # subdirs: nontor, pretor_15, pretor_30, pretor_45, pretor_60, pretor_120, tor, spout
    subglobs = ["tornado", "wind", "hail", "nonsev", "pretor_15", "pretor_30"]
    
    # leave these empty if you only want one run
    #month_subdirs = ['01-02','03','04','05','06','07','08','09','10','11-12']
    month_subdirs = []
    
    #hour_subdirs = [str(i).zfill(2) for i in range(24)]
    hour_subdirs = []
    
    if len(month_subdirs) > 0:
        val_lists = []
        for sub in month_subdirs:
            if sub == "01-02":
                val_lists.append(f"{inroot}/{year}0[1-2]*/{subglob}/*tfrec")
            elif sub == "11-12":
                val_lists.append(f"{inroot}/{year}1[1-2]*/{subglob}/*tfrec")
            else:
                val_lists.append(f"{inroot}/{year}{sub}*/{subglob}/*tfrec")
    elif len(hour_subdirs) > 0:
        val_lists = []
        for sub in hour_subdirs:
            val_lists.append(f"{inroot}/{year}*/{subglob}/*-{sub}*.tfrec")
    else:
        val_lists = []
        for subglob in subglobs:
            val_lists.append(f"{inroot}/{subglob}/{subglob}*tfrec")  # this is your "one-run" case


#for ii, val_list in enumerate(val_lists):

if filelist:
    test_filenames = val_list
else:
    test_filenames = []
    for pattern in val_lists:
        test_filenames.extend(glob.glob(pattern))
   
# Get n_samples for sharded list
n_samples = 0
for tfile in test_filenames:
    n_samples += int(os.path.basename(tfile).split('__n')[1].split('.')[0])

print("\nNumber of samples:", n_samples, "\n")

try:
    assert(n_samples > 0)
except AssertionError:
    print(f'n_samples is {n_samples}')
    sys.exit(1)

# one-off to combine pretor with nontor from 2023
#test_filenames += glob.glob(f"/raid/jcintineo/torcnn/tfrecs/2023/2023*/nontor/*tfrec")
#n_samples = len(test_filenames)
#print("\nNumber of samples:", n_samples, "\n") 

# Create the TFRecordDataset
test_ds = tf.data.TFRecordDataset(test_filenames, num_parallel_reads=AUTOTUNE)

# Already handles the remainder since drop_remainder=False by default
if args.save_data:
    test_ds = (
        test_ds.map(lambda x: parse_and_prepare(x, save_data=args.save_data), num_parallel_calls=AUTOTUNE)
        .batch(batchsize)  # Don't batch if making numpy arrays
        .prefetch(AUTOTUNE)
    )
    features_ds = test_ds.map(lambda x, y, pt, mag: x)
    
else:
    test_ds = (
        test_ds.map(lambda x: parse_and_prepare(x, save_data=args.save_data), num_parallel_calls=AUTOTUNE)
        .batch(batchsize)  # Don't batch if making numpy arrays
        .prefetch(AUTOTUNE)
    )

# Disable AutoShard.
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = (
    tf.data.experimental.AutoShardPolicy.OFF
)
test_ds = test_ds.with_options(options)

if filelist:
    all_preds = np.array([])
    all_labels = np.array([])
    # for quick_looks (or used for debugging)
    for inputs,labels in test_ds:
        preds = conv_model.predict(inputs)
        all_preds = np.concatenate((all_preds, np.squeeze(preds)))
        all_labels =  np.concatenate((all_labels, np.squeeze(labels)))
    os.makedirs(outdir, exist_ok=True)
    np.save(f'{outdir}/predictions.npy', all_preds) 
    np.save(f'{outdir}/labels.npy', all_labels)
    print(len(all_preds), len(all_labels)) 
else:
    # Perform eval

    if args.save_data:
        # The model only sees the features (x)
        all_preds = conv_model.predict(features_ds, verbose=1)
        # Here we ignore the features (_) and grab the labels (y) and other info
        all_tornado = []
        all_pretorMinutes = []
        all_magtornado = []
        for _, y, z, a in test_ds.as_numpy_iterator():
            all_tornado.append(y)
            all_pretorMinutes.append(z)
            all_magtornado.append(a)
        # Convert lists of lists into single lists
        all_tornado = [item[0] for sublist in all_tornado for item in sublist]
        all_pretorMinutes = [item for sublist in all_pretorMinutes for item in sublist]
        all_magtornado = [item for sublist in all_magtornado for item in sublist]
        results_df = pd.DataFrame({
            'prob': all_preds.squeeze(),
            'tornado': all_tornado,
            'pretorMinutes': all_pretorMinutes,
            'magtornado': all_magtornado,
        })
        os.makedirs(outdir, exist_ok=True)
        results_df.to_csv(f"{outdir}/eval_data.csv", index=False)
        print(f"Saved {outdir}/eval_data.csv")
    else:

        eval_results = conv_model.evaluate(test_ds)
    
        cter = 0
        dict_results = collections.OrderedDict(
            {
                "n_gpu": ngpu,
                "batch_size": batchsize,
                "n_samples": n_samples,
                "val_lists": val_lists,
            }
        )
    
        for key in custom_objs:
            dict_results[key] = np.round(eval_results[cter], 5)
            cter += 1
    
        # Append outsubdir, if necessary
        if month_subdirs:
            final_outdir = f"{outdir}/month{month_subdirs[ii]}"
        elif hour_subdirs:
            final_outdir = f"{outdir}/hour{hour_subdirs[ii]}"
        else:
            final_outdir = outdir
    
        os.makedirs(final_outdir, exist_ok=True)
        pickle.dump(dict_results, open(f"{final_outdir}/eval_results.pkl", "wb"))
        print(f"Saved {final_outdir}/eval_results.pkl")
