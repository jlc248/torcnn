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
def parse_tfrecord_fn(example):

    feature_description = {}

    # Define the target features
    for target in TARGETS:
        feature_description[target] = tf.io.FixedLenFeature([1], tf.int64)

    # Define scalar predictor features
    for scalar_var in SCALAR_VARS:
        feature_description[scalar_var] = tf.io.FixedLenFeature([1], tf.float32)

    # Define the n-D predictor features
    for inp in INPUTS:
        for chan in inp:
            if chan != 'range_inv':
                feature_description[chan] =  tf.io.FixedLenFeature([], tf.string)

    # Parse the single example
    features = tf.io.parse_single_example(example, feature_description)

    # Reshape the n-D inputs
    for inp in INPUTS:
        for chan in inp:
            if chan == 'range':
                features[chan] = tf.reshape(tf.io.parse_tensor(features[chan], tf.uint8), [PS[1],1])
            elif chan == 'range_inv':
                # tf.identity copies the tensor ('range' must be before 'range_inv' in input_tuples[x])
                features[chan] = tf.identity(features['range'])
            else:
                features[chan] = tf.reshape(tf.io.parse_tensor(features[chan], tf.uint8), [PS[0], PS[1],1])

    return features

#---------------------------------------------------------------------------------------------------------------
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
    for ii,inp in enumerate(INPUTS):  #for each keras.Input in the model construction
        for chIdx, varname in enumerate(inp):             #for each channel in the tf.layers.Input

            data = tf.cast(features[varname], tf.float32)

            # Handle the coordinate features
            # Duplicate the 1D range vector (and range_inv) n_azimuths times
            if varname == 'range':
                data = tf.repeat(tf.expand_dims(data, axis=0), repeats=PS[0], axis=0)
            elif varname == 'range_inv':
                data = tf.repeat(tf.expand_dims(1.0 / data, axis=0), repeats=PS[0], axis=0)


            if chIdx == 0:
                tensor = data
            else:
                tensor = tf.concat([tensor, data], axis=2)

        if ii == 0:
            inputs['radar'] = tensor
        elif ii == 1:
            inputs['coords'] = tensor


    if SCALAR_VARS:
        # These need to be normalized/scaled
        for idx,scalar_var in enumerate(SCALAR_VARS):
            scalarTmp = tf_bytescale(features[scalar_var], bsinfo[scalar_var]['vmin'], bsinfo[scalar_var]['vmax'])
            if idx == 0:
                scalars = scalarTmp
            else:
                scalars = tf.stack([scalars, scalarTmp])
        # Using ii from block above
        inputs['scalars'] = scalars

    # Sample weight should be same shape as targetInt
    return inputs, targetInt #, sample_weight
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
    # Create the test dataset for running the permutation
    year = "2013"
    inroot = f"/raid/jcintineo/torcnn/tfrecs/{year}/"
    # subdirs: nontor, pretor_15, pretor_30, pretor_45, pretor_60, pretor_120, tor, spout
    subglob = "pretor_15"
    
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
        val_lists = [f"{inroot}/{year}*/{subglob}/*tfrec"]  # this is your "one-run" case


for ii, val_list in enumerate(val_lists):
    if filelist:
        test_filenames = val_list
    else:
        test_filenames = sorted(glob.glob(val_list))  # sort to ensure reproducibility
    n_samples = len(test_filenames)
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
    test_ds = (
        test_ds.map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
        .map(prepare_sample, num_parallel_calls=AUTOTUNE)
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
