#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>

#include <unistd.h>
#include "bigfile.h"

int longfmt = 0;

static void usage() {
    fprintf(stderr, "usage: bigfile-set-attr [-t dtype] [-n nmemb] filepath block attr val \n");
    exit(1);
}

int main(int argc, char * argv[]) {
    BigFile bf = {0};
    BigBlock bb = {0};

    int opt;
    char * dtype = NULL;
    int nmemb = 0;
    
    while(-1 != (opt = getopt(argc, argv, "t:n:"))) {
        switch(opt) {
            case 't':
                dtype = optarg;
                break;
            case 'n':
                nmemb = atoi(optarg);
                break;
            default:
                usage();
        }
    }
    argv += optind - 1;
    if(argc - optind < 4) {
        usage();
    }

    if(0 != big_file_open(&bf, argv[1])) {
        fprintf(stderr, "failed to open file : %s %s\n", argv[1], big_file_get_error_message());
        exit(1);
    }
    if(0 != big_file_open_block(&bf, &bb, argv[2])) {
        fprintf(stderr, "failed to open block: %s %s\n", argv[2], big_file_get_error_message());
        exit(1);
    }
    int i; 
    BigBlockAttr * attr = big_block_lookup_attr(&bb, argv[3]);
    if(attr) {
        if(dtype == NULL) {
            dtype = attr->dtype;
        }
        if(nmemb == 0) {
            nmemb = attr->nmemb;
        }
    }
    if(nmemb == 0) nmemb = argc - optind + 1 - 4;
    if(nmemb != argc - optind + 1 - 4) {
        fprintf(stderr, "nmemb and number of arguments mismatch\n");
        exit(1);
    }
    char * data = malloc(dtype_itemsize(dtype) * nmemb);

    for(i = 4; i < argc - optind + 1; i ++) {
        char * p = data + (i - 4) * dtype_itemsize(dtype);
        dtype_parse(argv[i], dtype, p, NULL);
    }

    big_block_set_attr(&bb, argv[3], data, dtype, nmemb);

    big_block_close(&bb);
    big_file_close(&bf);
    return 0;
}
