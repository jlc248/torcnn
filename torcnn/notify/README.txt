Compile
- gcc watch_torp.c -o watch_torp

Run (wrap the path in quotes to prevent the shell from expanding the glob before the C program sees it)
- ./watch_torp "/sas8tb/localdata_MRMS/realtime/radar/K???/TORPcsv/"


Key Technical Details
- IN_CLOSE_WRITE: I chose this event instead of IN_CREATE or IN_MODIFY. If a file is being copied or written, IN_CREATE fires instantly when the file name appears, but before the data is actually there. IN_CLOSE_WRITE triggers only when the file descriptor is closed, meaning the writer is finished.

- glob(): This function handles the K??? pattern. It searches the filesystem and returns an array of all matching directory strings.

- Event Buffer: inotify returns a stream of bytes. We cast these into inotify_event structs. Because filenames vary in length, we increment the loop counter by EVENT_SIZE + event->len.

Limitations to Keep in Mind
- Static List: This script expands the K??? directories at startup. If you add a new KXXX directory while the script is running, it won't be automatically added to the watch list. You would need to restart the script or add logic to watch the parent radar/ directory for new directories.
