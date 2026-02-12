import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import glob
import time
import numpy as np
from tqdm import tqdm

def count_records(file_pattern, batch_size=50000):
    """
    We run this once to get a count of records in a sharded dataset. This will be hard-coded in the config.
    """

    filenames = glob.glob(file_pattern)
    if not filenames:
        print("No files found!")
        return 0
        
    print(f"Found {len(filenames)} shards. Counting records...")
    
    # 1. Build the high-speed pipeline
    ds = tf.data.Dataset.from_tensor_slices(filenames)
    ds = ds.interleave(
        lambda x: tf.data.TFRecordDataset(x, num_parallel_reads=tf.data.AUTOTUNE),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )
    
    # 2. Batch the records just to speed up the Python-side loop
    # We only pull the 'count' of the batch to keep data transfer minimal
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    total_count = 0
    start_time = time.time()
    
    # 3. Use tqdm for the UI
    # We don't know the total yet (that's why we're counting!), 
    # so we use 'unit="rec"'
    pbar = tqdm(desc="Counting Records", unit=" rec", unit_scale=True)
    
    for batch in ds:
        # batch is a tensor of records; its size is the number of records
        current_batch_size = tf.shape(batch)[0].numpy()
        total_count += current_batch_size
        pbar.update(current_batch_size)
        
    pbar.close()
    
    duration = time.time() - start_time
    print(f"\n--- Done ---")
    print(f"Total Records: {total_count:,}")
    print(f"Time:          {duration:.2f}s")
    print(f"Speed:         {total_count/duration:,.0f} records/sec")
    
    return total_count


if __name__ == "__main__":
    # Point this to your sharded dataset 
    SHARD_PATH = "/work2/jcintineo/torcnn/tfrecs_shard/2023/*.tfrec"
    total = count_records(SHARD_PATH)
