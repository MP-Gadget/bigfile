#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>

#include "bigfile.h"

int main(int argc, char * argv[]) {
    BigFile bf = {0};
    BigBlock bb = {0};
    BigBlock bbnew = {0};

    if(argc < 4) {
        fprintf(stderr, "usage: bigfile-attrdump filepath block newblock newnfile\n");
        exit(1);
    }
    if(0 != big_file_open(&bf, argv[1])) {
        fprintf(stderr, "failed to open: %s\n", big_file_get_error_message());
        exit(1);
    }
    if(0 != big_file_open_block(&bf, &bb, argv[2])) {
        fprintf(stderr, "failed to open: %s\n", big_file_get_error_message());
        exit(1);
    }
    int newnfile = atoi(argv[4]);
    size_t newsize[newnfile];
    int i;
    for(i = 0; i < newnfile; i ++) {
        newsize[i] = (i + 1) * bb.size / newnfile 
            - i * bb.size / newnfile;
    }

    if(0 != big_file_create_block(&bf, &bbnew, argv[3], bb.dtype, bb.nmemb, newnfile, newsize)) {
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
    char * buffer = malloc(buffersize);
    size_t chunksize = buffersize / (bb.nmemb * dtype_itemsize(bb.dtype));
    BigBlockPtr ptr;
    BigBlockPtr ptrnew;
    ptrdiff_t offset;
    BigArray array;

    for(offset = 0; offset < bb.size; ) {
        size_t readsize = chunksize;
        if(readsize > bb.size - offset) {
            readsize = bb.size - offset;
        }
        size_t dims[2] = {readsize, bb.nmemb};
        ptrdiff_t strides[2] = {bb.nmemb * dtype_itemsize(bb.dtype), dtype_itemsize(bb.dtype)};

        big_array_init(&array, buffer, bb.dtype, 2, dims, strides);

        if(0 != big_block_seek(&bb, &ptr, offset)) {
            fprintf(stderr, "failed to seek original: %s\n", big_file_get_error_message());
            exit(1);
        }
        if(0 != big_block_seek(&bbnew, &ptrnew, offset)) {
            fprintf(stderr, "failed to seek new: %s\n", big_file_get_error_message());
            exit(1);
        }

        if(0 != big_block_read(&bb, &ptr, &array)) {
            fprintf(stderr, "failed to read original: %s\n", big_file_get_error_message());
            exit(1);
        }
        if(0 != big_block_write(&bbnew, &ptrnew, &array)) {
            fprintf(stderr, "failed to write new: %s\n", big_file_get_error_message());
            exit(1);
        }
        offset += chunksize;
    }
    if(0 != big_block_close(&bbnew)) {
        fprintf(stderr, "failed to close new: %s\n", big_file_get_error_message());
        exit(1);
    }
    big_block_close(&bb);
    big_file_close(&bf);
    return 0;
}
