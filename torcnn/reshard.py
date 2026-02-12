import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import glob
from tqdm import tqdm

# Settings
SAMPLES_PER_SHARD = 650  # get ~200MB shards
year=2012
ROOT_DIR = '/work2/jcintineo/torcnn/tfrecs'
OUTPUT_DIR = f'/work2/jcintineo/torcnn/tfrecs_shard/{year}'


def reshard_class_folders():
    # Find all class folders
    class_folders = glob.glob(os.path.join(ROOT_DIR, str(year), "2*", "*"))
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    total_records_read = 0
    total_records_written = 0

    print(f"Found {len(class_folders)} class folders. Starting re-sharding...")

    # Primary progress bar for folders
    for folder in tqdm(class_folders, desc="Overall Progress", unit="folder"):
        parts = folder.split(os.sep)
        class_name = parts[-1]
        date_name = parts[-2]
        
        tiny_files = glob.glob(os.path.join(folder, "*.tfrec"))
        if not tiny_files:
            continue
            
        # Create a dataset to read the raw bytes
        # Using num_parallel_reads here speeds up the consolidation process itself
        raw_dataset = tf.data.TFRecordDataset(tiny_files, num_parallel_reads=tf.data.AUTOTUNE)
        
        shard_idx = 0
        writer = None
        
        # Secondary progress bar for samples within the current folder
        for i, raw_record in enumerate(raw_dataset):
            if i % SAMPLES_PER_SHARD == 0:
                if writer:
                    writer.close()
                
                shard_path = os.path.join(
                    OUTPUT_DIR, 
                    f"{date_name}_{class_name}_shard_{shard_idx}.tfrec"
                )
                writer = tf.io.TFRecordWriter(shard_path)
                shard_idx += 1
            
            writer.write(raw_record.numpy())
            total_records_written += 1

        total_records_read += (i + 1) if 'i' in locals() else 0            
        if writer:
            writer.close()

if __name__ == "__main__":
    reshard_class_folders()
