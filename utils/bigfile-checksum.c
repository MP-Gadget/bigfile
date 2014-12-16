#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>

#include <unistd.h>
#include <signal.h>
#include "bigfile.h"

int main(int argc, char * argv[]) {
    size_t buffersize = 256 * 1024 * 1024;

    char * fmt = NULL;

    while(-1 != (opt = getopt(argc, argv, "B:"))) {
        switch(opt) {
            case 'B':
                sscanf(optarg, "%td", &buffersize);
                break;
            default:
                usage();
        }
    }
    signal(SIGUSR1, usr1);

    void * buffer = malloc(buffersize);
    unsigned int sum = 0;
    while(!feof(stdin)) {
        size_t readcount = fread(buffer, 1, buffersize, stdin);
        big_file_checksum(&sum, buffer, readcount);
    }
    unsigned int s = sum;
    unsigned int r = (s & 0xffff) + ((s & 0xffffffff) >> 16);
    unsigned int checksum = (r & 0xffff) + (r >> 16);
    printf("%u %u\n", checksum, sum);
    free(buffer);
    return 0;
}
