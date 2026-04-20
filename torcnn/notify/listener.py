import sys
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import fnmatch

# The pattern you are looking for
# This matches: any_path/Kxxx/TORPcsvShort/QC/any_file

#PATTERN = "*/K???/TORPcsvShort/QC/*"
PATTERN = "*/K???/TORPcsvShort/*"

class RadarHandler(FileSystemEventHandler):
    def on_closed(self, event):
        # 'on_closed' is equivalent to IN_CLOSE_WRITE
        if not event.is_directory:
            if fnmatch.fnmatch(event.src_path, PATTERN):
                print('New File: ' + event.src_path, flush=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <base_directory>")
        sys.exit(1)

    path = sys.argv[1]
    event_handler = RadarHandler()
    observer = Observer()
    
    # recursive=True handles all existing and NEW subdirectories automatically
    observer.schedule(event_handler, path, recursive=True)
    
    print(f"Monitoring {path} recursively...", file=sys.stderr)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
