import os, sys
# No GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import pandas as pd
import glob
import pickle
import tqdm
import re
import numpy as np

BATCH_SIZE = 5000

config_file = '/work2/jcintineo/torcnn/tests/2011-19/test04/model_config.pkl'
test_config = pickle.load(open(config_file, 'rb'))
test = os.path.basename(os.path.dirname(config_file))

assert(bool(re.fullmatch(r'test\d{2}', test)))

tfrecord_path_patterns = test_config['val_list'] 

feature_description = {
    'Time': tf.io.FixedLenFeature([], tf.string), 
    'Radar': tf.io.FixedLenFeature([], tf.string),
    'obj_lat': tf.io.FixedLenFeature([], tf.float32),
    'obj_lon': tf.io.FixedLenFeature([], tf.float32),
}

def parse_function(example_proto):
    # Parse the input tf.train.Example proto using the dictionary above
    return tf.io.parse_single_example(example_proto, feature_description)

def get_keys_from_tfrecords(pattern_list):
    all_files = []
    for p in pattern_list:
        all_files.extend(glob.glob(p))

    if not all_files:
        raise FileNotFoundError("No files found matching the provided patterns.")

    print(f"Found {len(all_files)} total shards. Starting extraction...")

    # Create the dataset
    # Optimization: num_parallel_reads is CRITICAL for NFS
    dataset = tf.data.TFRecordDataset(
        all_files, 
        num_parallel_reads=tf.data.AUTOTUNE,
        buffer_size=100 * 1024 * 1024 # 100MB read buffer to smooth out network spikes
    )

    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batching before it hits Python reduces the "round-trip" overhead
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    times_list = []
    radars_list = []
    lats_list = []
    lons_list = []
    
    print("Reading TFRecords...")

    for batch in tqdm.tqdm(dataset.as_numpy_iterator(), desc="Batches"):
        lats_list.append(batch['obj_lat'])
        lons_list.append(batch['obj_lon'])

        # Vectorized byte decoding
        radars = batch['Radar']
        if isinstance(radars[0], bytes):
            radars = [r.decode('utf-8') for r in radars]
        radars_list.append(radars)

        times = batch['Time']
        if isinstance(times[0], bytes):
            times = [t.decode('utf-8') for t in times]
        times_list.append(times)

    # Create a unique set of keys to keep the merge efficient
    return pd.DataFrame({'Time': np.concatenate(times_list),
                         'Radar': np.concatenate(radars_list),
                         'Lat': np.concatenate(lats_list),
                         'Lon': np.concatenate(lons_list),
           }).drop_duplicates()


# Extract keys from TFRecords
tf_keys_df = get_keys_from_tfrecords(tfrecord_path_patterns)
print(len(tf_keys_df))

# Read DataFrame
df = pd.read_csv('/work2/jcintineo/TORP/combined_torp_rep_pretor/2018_combined.csv')

# Ensure precision isn't an issue
PRECISION = 4
tf_keys_df['Lat'] = tf_keys_df['Lat'].astype(np.float64).round(PRECISION)
tf_keys_df['Lon'] = tf_keys_df['Lon'].astype(np.float64).round(PRECISION)
df['Lat'] = df['Lat'].round(PRECISION)
df['Lon'] = df['Lon'].round(PRECISION)

# Clean up the CSV's "Unnamed" columns if they exist
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Filter the DataFrame
# We use an inner merge to keep only rows where (Time, Radar) exist in both
filtered_df = pd.merge(df, tf_keys_df, on=['Time', 'Radar', 'Lat', 'Lon'], how='inner')
#filtered_df = pd.merge(df, tf_keys_df, on=['Time', 'Radar'], how='inner')

print(f"Process complete. Resulting DataFrame has {len(filtered_df)} rows.")

outfile = f'/work2/jcintineo/TORP/combined_torp_rep_pretor/matches/val_{test}.csv'
filtered_df.to_csv(outfile, index=False)

print(f"Saved {outfile}.")

