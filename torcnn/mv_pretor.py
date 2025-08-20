import os
import glob
import shutil
from multiprocessing import Pool
from tqdm import tqdm

# Define source and destination root directories
SOURCE_ROOT = "/raid/jcintineo/torcnn/tfrecs/"
DEST_ROOT = "/raid/jcintineo/torcnn/tfrecs_100km1hr/"
NUM_CPUS = 40  # Number of processes to use

def copy_directory(source_dir):
    """
    Copies 'pretor*' files from a source directory to the corresponding
    destination directory.
    """
    try:
        # Extract base year and day from the source_dir path
        parts = source_dir.split(os.sep)
        baseyear = parts[-2]
        baseday = parts[-1]

        destination_dir = os.path.join(DEST_ROOT, baseyear, baseday)

        # Create destination directory if it doesn't exist
        os.makedirs(destination_dir, exist_ok=True)

        # Find all dirs/files matching 'pretor*' in the source_dir
        items_to_copy = glob.glob(os.path.join(source_dir, "pretor*"))

        for item_path in items_to_copy:
            item_name = os.path.basename(item_path)
            destination_item = os.path.join(destination_dir, item_name)

            if os.path.isfile(item_path):
                # It's a file, use shutil.copy2
                shutil.copy2(item_path, destination_item)
            elif os.path.isdir(item_path):
                # It's a directory, use shutil.copytree
                # copytree requires the destination directory to NOT exist
                # The dirs_exist_ok=True argument handles this
                shutil.copytree(item_path, destination_item, dirs_exist_ok=True)
            else:
                # Handle other types of items if necessary (e.g., symlinks)
                print(f"Skipping non-file/non-directory item: {item_path}")       
 
        # Return success status and the directory path for reporting
        return True, source_dir
    except Exception as e:
        # Return failure status and the error message
        return False, f"Error copying {source_dir}: {e}"

if __name__ == "__main__":
    # Get all year directories
    year_dirs = sorted(glob.glob(os.path.join(SOURCE_ROOT, "2*")))

    all_date_dirs = []
    for year_dir in year_dirs:
        # Get all date directories within each year
        date_dirs = sorted(glob.glob(os.path.join(year_dir, "2*")))
        all_date_dirs.extend(date_dirs)

    print(f"Found {len(all_date_dirs)} date directories to process.")

    # Use multiprocessing Pool to distribute the copying tasks
    with Pool(processes=NUM_CPUS) as pool:
        # Using pool.imap_unordered for a dynamic progress bar
        # map() would wait for all results before displaying, imap_unordered shows progress as tasks complete
        results_iterator = pool.imap_unordered(copy_directory, all_date_dirs)
        
        # Wrap the iterator with tqdm to display a progress bar
        pbar = tqdm(results_iterator, total=len(all_date_dirs), desc="Copying directories", unit="dir")

        # Process the results and display the progress
        failed_copies = []
        for success, result_info in pbar:
            if not success:
                failed_copies.append(result_info)
    
    # Optional: Check for any failed copies
    if failed_copies:
        print("\n--- Failed Copies ---")
        for fail_info in failed_copies:
            print(fail_info)
    else:
        print("\nAll copying tasks completed successfully.")
