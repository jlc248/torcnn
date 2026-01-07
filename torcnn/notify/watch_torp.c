#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/inotify.h>
#include <unistd.h>
#include <glob.h>
#include <string.h>

#define EVENT_SIZE  (sizeof(struct inotify_event))
#define BUF_LEN     (1024 * (EVENT_SIZE + 16))
#define MAX_WATCHES 1024

// Structure to map watch descriptors to path names
typedef struct {
    int wd;
    char path[1024];
} Watch;

int main(int argc, char **argv) {
    // 1. Force line buffering for script compatibility
    setvbuf(stdout, NULL, _IOLBF, 0);

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <path_pattern>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int fd = inotify_init();
    if (fd < 0) { perror("inotify_init"); return 1; }

    glob_t glob_result;
    if (glob(argv[1], GLOB_TILDE, NULL, &glob_result) != 0) {
        fprintf(stderr, "Error resolving path pattern.\n");
        return 1;
    }

    Watch watches[MAX_WATCHES];
    size_t watch_count = 0;

    // 2. Add watches and store the paths
    for (size_t i = 0; i < glob_result.gl_pathc && i < MAX_WATCHES; i++) {
        int wd = inotify_add_watch(fd, glob_result.gl_pathv[i], IN_CLOSE_WRITE);
        if (wd != -1) {
            watches[watch_count].wd = wd;
            strncpy(watches[watch_count].path, glob_result.gl_pathv[i], 1023);
            watch_count++;
        }
    }

    printf("Listening for files in %zu directories...\n", watch_count);

    char buffer[BUF_LEN];
    while (1) {
        int length = read(fd, buffer, BUF_LEN);
        if (length < 0) { perror("read"); break; }

        int i = 0;
        while (i < length) {
            struct inotify_event *event = (struct inotify_event *) &buffer[i];
            if (event->len && (event->mask & IN_CLOSE_WRITE)) {
                
                // 3. Find the directory path associated with this watch descriptor
                char *dir_path = "unknown";
                for (size_t j = 0; j < watch_count; j++) {
                    if (watches[j].wd == event->wd) {
                        dir_path = watches[j].path;
                        break;
                    }
                }

                // 4. Print the full path (Dir + Filename)
                // Ensure there is a trailing slash handling
                size_t len = strlen(dir_path);
                if (len > 0 && dir_path[len-1] == '/') {
                    printf("New File: %s%s\n", dir_path, event->name);
                } else {
                    printf("New File: %s/%s\n", dir_path, event->name);
                }
            }
            i += EVENT_SIZE + event->len;
        }
    }

    globfree(&glob_result);
    close(fd);
    return 0;
}
