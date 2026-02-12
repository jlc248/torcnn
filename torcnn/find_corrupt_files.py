import os
import time
import tensorflow as tf
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

# Critical for NFS multi-processing
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def check_file_with_timeout(file_path):
    """The actual work function performed in a separate process"""
    file_path = file_path.strip()
    try:
        # We check 5 records. This is enough to trigger most CRC errors.
        dataset = tf.data.TFRecordDataset(file_path, buffer_size=1024*256)
        for _ in dataset.take(5):
            pass
        return "GOOD"
    except (tf.errors.DataLossError, tf.errors.InvalidArgumentError):
        return "BAD"
    except Exception:
        return "OTHER_ERROR"

if __name__ == "__main__":
    input_list = "file_lists/2016-2024.txt"
    num_workers = 12  # Start conservative; if f/s is good, go to 16
    timeout_val = 20  # Seconds before we give up on a file
    
    if not os.path.exists(input_list):
        print(f"Error: {input_list} not found.")
        exit()

    with open(input_list, "r") as f:
        files = [line.strip() for line in f]

    total_files = len(files)
    print(f"Hardened Scan: {total_files} files | Workers: {num_workers} | Timeout: {timeout_val}s")
    start_time = time.time()
    stats = {"GOOD": 0, "BAD": 0, "TIMEOUT": 0, "OTHER_ERROR": 0}

    try: 
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_file = {executor.submit(check_file_with_timeout, f): f for f in files}
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                
                try:
                    status = future.result(timeout=timeout_val)
                    stats[status] += 1

                    if status != "GOOD":
                        with open("corrupted_files.txt", "a") as f_out:
                            f_out.write(f"{file_path} ({status})\n")

                except TimeoutError:
                    stats["TIMEOUT"] += 1
                    print(f"\n[!] TIMEOUT: {file_path}")
                    with open("corrupted_files.txt", "a") as f_out:
                        f_out.write(f"{file_path} (TIMEOUT)\n")
               
                processed = sum(stats.values()) 
                if processed % 10 == 0:
                    elapsed = time.time() - start_time
                    fps = processed / elapsed
                    rem = (total_files - processed) / fps
                    print(f"Done: {processed}/{total_files} | {fps:.1f} f/s | Bad: {stats['BAD']+stats['TIMEOUT']} | ETA: {int(rem//60)}m", end='\r') 

    except KeyboardInterrupt:
        print("\n\nScan interrupted by user.")

    # Final Summary Block
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*40)
    print(" SCAN SUMMARY")
    print("="*40)
    print(f"Total Files Scanned:  {sum(stats.values())}")
    print(f"Good Files:           {stats['GOOD']}")
    print(f"Corrupted (CRC):      {stats['BAD']}")
    print(f"Timeouts (NFS/IO):    {stats['TIMEOUT']}")
    print(f"Other Errors:         {stats['OTHER_ERROR']}")
    print("-" * 40)
    print(f"Total Time:           {int(duration // 3600)}h {int((duration % 3600) // 60)}m")
    print(f"Average Rate:         {sum(stats.values())/duration:.2f} files/sec")
    print("="*40)
