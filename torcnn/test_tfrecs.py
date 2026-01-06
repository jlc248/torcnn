import tensorflow as tf
from tensorflow.python.lib.io import tf_record
import concurrent
import traceback
import logging
import functools
import glob
from tqdm import tqdm

def is_tfrecord_corrupted(file_path):
    """
    Checks a single TFRecord file for corruption.
    Returns True if corrupted, False otherwise.
    """
    try:
        for _ in tf_record.tf_record_iterator(file_path):
            pass
        return False  # No corruption found
    except tf.errors.DataLossError as e:
        print(f"ERROR: Corrupted record found in file: {file_path}")
        print(f"Error details: {e}")
        return True # Corruption detected
    except Exception as e:
        print(f"An unexpected error occurred for {file_path}: {e}")
        return True # Treat other read errors as corruption as well
################################################################################################################################
def find_corrupted_tfrecords(all_files):
    """
    Scans a directory for corrupted TFRecord files.
    """
    corrupted_files = []

    print(f"Scanning {len(all_files)} TFRecord files for corruption...")

    # Iterate through each file and check for corruption
    for i, file_path in enumerate(all_files, 1):
        if i % 1000 == 0:
            print(f"Checked {i}/{len(all_files)} files...")

        if is_tfrecord_corrupted(file_path):
            print(f"Corrupted file found: {file_path}")
            corrupted_files.append(file_path)

    print("\n--- Scan Complete ---")
    if corrupted_files:
        print("Found the following corrupted files:")
        for cf in corrupted_files:
            print(cf)
    else:
        print("No corrupted TFRecord files were found.")
#################################################################################################################################


if __name__ == "__main__":
    """
    Find which files are "corrupted" on the filesystem.
    """

    logger = logging.getLogger(__name__)
    logger.info(f"begin extract data and write TFRecords")

    import tf_config
    config = tf_config.tf_config()

    print(config['train_list'])

    train_filenames = []
    for pattern in config['train_list']:
        train_filenames.extend(glob.glob(pattern))

    partial_find_corruped_tf_records = functools.partial(find_corrupted_tfrecords) #, **find_corruped_tfrecords_args
    
    number_of_recs = len(train_filenames)
    
    max_workers = 100 #min(gfs_columns_extract_workers, os.cpu_count())
    
    with tqdm(total=number_of_recs) as pbar: # progress bar
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
    
            for rec in train_filenames:
    
                futures.append(executor.submit(partial_find_corruped_tf_records, rec))
    
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing a row: {e}")
                    logger.error(traceback.format_exc())
                finally:
                    pbar.update(1)
