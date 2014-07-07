#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <unistd.h>

#include "bigfile.h"
void usage() {
    fprintf(stderr, "usage: bigfile-copy [-n Nfile] filepath block newblock\n");
    exit(1);

}
int main(int argc, char * argv[]) {
    BigFile bf = {0};
    BigBlock bb = {0};
    BigBlock bbnew = {0};

    int ch;
    int Nfile = -1;
    while(-1 != (ch = getopt(argc, argv, "n:"))) {
        switch(ch) {
            case 'n':
                Nfile = atoi(optarg);
                break;
            default:
                usage();
        }
    }
    if(argc - optind + 1 != 4) {
        usage();
    }
    argv += optind - 1;
    if(0 != big_file_open(&bf, argv[1])) {
        fprintf(stderr, "failed to open: %s\n", big_file_get_error_message());
        exit(1);
    }
    if(0 != big_file_open_block(&bf, &bb, argv[2])) {
        fprintf(stderr, "failed to open: %s\n", big_file_get_error_message());
        exit(1);
    }
    if(Nfile == -1) {
        Nfile = bb.Nfile;
    }
    size_t newsize[Nfile];
    int i;
    for(i = 0; i < Nfile; i ++) {
        newsize[i] = (i + 1) * bb.size / Nfile
            - i * bb.size / Nfile;
    }

    if(0 != big_file_create_block(&bf, &bbnew, argv[3], bb.dtype, bb.nmemb, Nfile, newsize)) {
        fprintf(stderr, "failed to create temp: %s\n", big_file_get_error_message());
        exit(1);
    }

    if(bbnew.size != bb.size) {
        abort();
    }

    /* copy attrs */
    size_t nattr;
    BigBlockAttr * attrs = big_block_list_attrs(&bb, &nattr);
    for(i = 0; i < nattr; i ++) {
        BigBlockAttr * attr = &attrs[i];
        big_block_set_attr(&bbnew, attr->name, attr->data, attr->dtype, attr->nmemb);
    }
    
    /* copy data */
    size_t buffersize = 256 * 1024 * 1024;
    size_t chunksize = buffersize / (bb.nmemb * dtype_itemsize(bb.dtype));
    BigBlockPtr ptrnew;
    ptrdiff_t offset;
    BigArray array;

    for(offset = 0; offset < bb.size; ) {
        if(0 != big_block_read_simple(&bb, offset, chunksize, &array, NULL)) {
            fprintf(stderr, "failed to read original: %s\n", big_file_get_error_message());
            exit(1);
        }
        if(0 != big_block_seek(&bbnew, &ptrnew, offset)) {
            fprintf(stderr, "failed to seek new: %s\n", big_file_get_error_message());
            exit(1);
        }

        if(0 != big_block_write(&bbnew, &ptrnew, &array)) {
            fprintf(stderr, "failed to write new: %s\n", big_file_get_error_message());
            exit(1);
        }

        free(array.data);
        offset += array.dims[0];
    }
    if(0 != big_block_close(&bbnew)) {
        fprintf(stderr, "failed to close new: %s\n", big_file_get_error_message());
        exit(1);
    }
    big_block_close(&bb);
    big_file_close(&bf);
    return 0;
}
