#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/inotify.h>
#include <unistd.h>
#include <glob.h>

#define EVENT_SIZE  (sizeof(struct inotify_event))
#define BUF_LEN     (1024 * (EVENT_SIZE + 16))

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <path_pattern>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int length, i = 0;
    int fd;
    char buffer[BUF_LEN];
    glob_t glob_result;

    // 1. Expand the path pattern (e.g., /path/to/K???)
    int return_value = glob(argv[1], GLOB_TILDE, NULL, &glob_result);
    if (return_value != 0) {
        fprintf(stderr, "Error: Could not resolve path pattern or no directories found.\n");
        return 1;
    }

    // 2. Initialize inotify
    fd = inotify_init();
    if (fd < 0) {
        perror("inotify_init");
    }

    // 3. Add watches for every matched directory
    printf("Watching %zu directories...\n", glob_result.gl_pathc);
    for (size_t i = 0; i < glob_result.gl_pathc; i++) {
        // IN_CLOSE_WRITE triggers when a file opened for writing is closed.
        // This ensures the file is finished being modified/created before printing.
        int watch = inotify_add_watch(fd, glob_result.gl_pathv[i], IN_CLOSE_WRITE);
        if (watch == -1) {
            printf("Could not watch: %s\n", glob_result.gl_pathv[i]);
        } else {
            printf("Watching: %s\n", glob_result.gl_pathv[i]);
        }
    }

    // 4. Read events loop
    while (1) {
        i = 0;
        length = read(fd, buffer, BUF_LEN);

        if (length < 0) {
            perror("read");
        }

        while (i < length) {
            struct inotify_event *event = (struct inotify_event *) &buffer[i];
            if (event->len) {
                if (event->mask & IN_CLOSE_WRITE) {
                    if (!(event->mask & IN_ISDIR)) {
                        printf("File Ready: %s\n", event->name);
                    }
                }
            }
            i += EVENT_SIZE + event->len;
        }
    }

    // Cleanup
    globfree(&glob_result);
    close(fd);
    return 0;
}
