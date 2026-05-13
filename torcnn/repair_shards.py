import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tqdm import tqdm
import glob
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def repair_single_shard(file_path, output_base):

    file_name = os.path.basename(file_path)
    subtype = os.path.basename(os.path.dirname(file_path))
    yyyymm = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
    
    output_dir = f"{output_base}/{yyyymm}/{subtype}" 

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, file_name)
    
    
    # Read the raw records without a schema
    raw_dataset = tf.data.TFRecordDataset(file_path)
    
    with tf.io.TFRecordWriter(output_path) as writer:
        for raw_record in raw_dataset:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            
            # REPAIR LOGIC: Force these keys to Int64List
            for key in ['hail', 'wind']:
                if key in example.features.feature:
                    feat = example.features.feature[key]
                    
                    # Extract the value regardless of which list it's in
                    if feat.HasField('float_list'):
                        val = int(feat.float_list.value[0])
                    elif feat.HasField('int64_list'):
                        val = int(feat.int64_list.value[0])
                    else:
                        continue # Or handle error
                    
                    # Overwrite with a clean Int64List
                    new_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=[val]))
                    example.features.feature[key].CopyFrom(new_feat)
            
            writer.write(example.SerializeToString())


def parallel_repair(input_pattern, output_dir, max_workers=20):
    shards = glob.glob(input_pattern)
    if not shards:
        print("No files found matching the pattern.")
        return

    print(f"Repairing {len(shards)} shards using {max_workers} workers...")
    
    # Use partial to fix the output_dir argument for the executor
    worker_func = partial(repair_single_shard, output_base=output_dir)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # list() forces the generator to evaluate and shows the progress bar
        list(tqdm(executor.map(worker_func, shards), total=len(shards)))


input_pattern = "/work2/jcintineo/torcnn/tfrecs_combined/201[1-9]??/*/*tfrec"
output_dir = "/work2/jcintineo/torcnn/tfrecs_TMP" 
assert(os.path.dirname(os.path.dirname(os.path.dirname(input_pattern))) != output_dir)
parallel_repair(input_pattern, output_dir, max_workers=80)
