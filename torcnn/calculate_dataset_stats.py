import tensorflow as tf
import numpy as np
from tqdm import tqdm
import glob
import utils

BSINFO = utils.get_bsinfo()
PS = (128, 256)

chans = ['Reflectivity', 'Velocity', 'RhoHV', 'SpectrumWidth', 'Zdr']

feature_description = {}
for chan in chans:
    feature_description[chan] = tf.io.FixedLenFeature([], tf.string)

#--------------------------------------------------------------------------------------------------------------------------------------
def normalize_channel(tensor, channel_name):
    info = BSINFO[channel_name]
    c_min = info['vmin']
    c_max = info['vmax']

    # First, assume the uint8 (0-255) maps to the min/max range
    # Scaling uint8 to the physical units first:
    physical = (tensor / 255.0) * (c_max - c_min) + c_min

    return physical
#--------------------------------------------------------------------------------------------------------------------------------------
def parse_and_prepare(example_proto):

    # Parse the record
    features = tf.io.parse_single_example(example_proto, feature_description)

    parsed_tensors = []

    for chan in chans:
        parsed_tensors.append(normalize_channel(tf.cast(tf.reshape(tf.io.parse_tensor(features[chan], tf.uint8), [PS[0], PS[1], 1]), tf.float32), chan))

    parsed_tensors = tf.concat(parsed_tensors, axis=-1)

    return parsed_tensors

#--------------------------------------------------------------------------------------------------------------------------------------
def calculate_dataset_stats(filenames, parse_func, batch_size=1024):
    """
    Calculates mean and std dev for Reflectivity, Velocity, CC, and SW.
    """
    # 1. Build a simple non-shuffled dataset
    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(parse_func, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Initialize accumulators (use float64 for precision with millions of samples)
    # Order: [Refl, Vel, CC, SW, Zdr]
    total_sum = tf.zeros((len(chans),), dtype=tf.float64)
    total_sq_sum = tf.zeros((len(chans),), dtype=tf.float64)
    total_count = 0.0

    print("Starting statistics calculation...")
    for batch in tqdm(ds):
        # Assuming batch is [Batch, Height, Width, len(chans)]
        images = tf.cast(batch, tf.float64)
        
        # Flatten pixels: [Batch * H * W, len(chans)]
        pixels = tf.reshape(images, (-1, len(chans)))
        
        # Batch Accumulation
        total_sum += tf.reduce_sum(pixels, axis=0)
        total_sq_sum += tf.reduce_sum(tf.square(pixels), axis=0)
        total_count += tf.cast(tf.shape(pixels)[0], tf.float64)

    # Calculate final stats
    means = total_sum / total_count
    
    # Variance = E[X^2] - (E[X])^2
    variances = (total_sq_sum / total_count) - tf.square(means)
    stds = tf.sqrt(variances)

    return means.numpy(), stds.numpy()

#======================================================================================================================================

# Takes 30 minutes or so to run

all_filenames = glob.glob("/work2/jcintineo/torcnn/tfrecs_combined/20*/*/*tfrec")
means, stds = calculate_dataset_stats(all_filenames, parse_and_prepare)

print(f"Means ({chans}): {means}")
print(f"Stds  ({chans}): {stds}")

#Means (['Reflectivity', 'Velocity', 'RhoHV', 'SpectrumWidth', 'Zdr']): [ 19.63109437 -36.69369631   0.67619126   2.12794626  -0.40182905]
#Stds  (['Reflectivity', 'Velocity', 'RhoHV', 'SpectrumWidth', 'Zdr']): [14.85550763 49.59021647  0.43495006  2.5681228   2.66428236]
