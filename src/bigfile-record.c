#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <alloca.h>
#include <stdint.h>
#include <stddef.h>

#include "bigfile.h"
#include "bigfile-internal.h"

void
big_record_type_clear(BigRecordType * rtype)
{
    int i;
    int offset = 0;
    for(i = 0; rtype->nfield; i ++) {
        free(rtype->fields[i].name);
    }
    free(rtype->fields);
    rtype->nfield = 0;
    rtype->fields = NULL;
}

void
big_record_type_set(BigRecordType * rtype,
    int i,
    const char * name,
    const char * dtype,
    int nmemb)
{
    if(i >= rtype->nfield) {
        rtype->fields = realloc(rtype->fields,
            (i + 1) * sizeof(rtype->fields[0]));
        memset(&rtype->fields[rtype->nfield], 0,
            (i + 1 - rtype->nfield) * sizeof(rtype->fields[0]));
        rtype->nfield = i + 1;
    } else {
        if(rtype->fields[i].name) {
            free(rtype->fields[i].name);
        }
    }
    rtype->fields[i].name = _strdup(name);
    strncpy(rtype->fields[i].dtype, dtype, 8);
    rtype->fields[i].nmemb = nmemb;
}

void
big_record_type_complete(BigRecordType * rtype)
{
    int i;
    int offset = 0;
    for(i = 0; i < rtype->nfield; i ++) {
        if(rtype->fields[i].name == NULL) {
            fprintf(stderr, "Not all fields are filled.\n");
            abort();
        }
        rtype->fields[i].offset = offset;
        rtype->fields[i].elsize = big_file_dtype_itemsize(rtype->fields[i].dtype);
        offset += rtype->fields[i].elsize;
    }
    rtype->nfield = i;
    rtype->itemsize = offset;
}

void
big_record_set(const BigRecordType * rtype,
    void * buf,
    int i,
    const void * data)
{
    char * p = buf;
    memcpy(&p[rtype->fields[i].offset], data, rtype->fields[i].elsize);
}

void
big_record_get(const BigRecordType * rtype,
    const void * buf,
    int i,
    void * data) {
    const char * p = buf;
    memcpy(data, &p[rtype->fields[i].offset], rtype->fields[i].elsize);
}

int
big_record_view_field(const BigRecordType * rtype,
    int i,
    BigArray * array,
    size_t size,
    void * buf
)
{
    char * p = buf;
    size_t dims[2] = { size, rtype->fields[i].nmemb };
    ptrdiff_t strides[2] = { rtype->itemsize, rtype->fields[i].elsize };
    return big_array_init(array, &p[rtype->fields[i].offset],
                          rtype->fields[i].dtype,
                          2, dims, strides);
}

int
big_file_write_records(BigFile * bf,
    const BigRecordType * rtype,
    ptrdiff_t offset,
    size_t size,
    const void * buf)
{
    int i;
    for(i = 0; i < rtype->nfield; i ++) {
        BigArray array[1];
        BigBlock block[1];
        BigBlockPtr ptr = {0};

        /* rainwoodman: cast away the const. We don't really modify it.*/
        RAISEIF(0 != big_record_view_field(rtype, i, array, size, (void*) buf),
            ex_array,
            NULL);
        RAISEIF(0 != big_file_open_block(bf, block, rtype->fields[i].name),
            ex_open,
            NULL);
        RAISEIF(0 != big_block_seek(block, &ptr, offset),
            ex_seek,
            NULL);
        RAISEIF(0 != big_block_write(block, &ptr, array),
            ex_write,
            NULL);
        RAISEIF(0 != big_block_close(block),
            ex_close,
            NULL);
        ex_write:
        ex_seek:
            RAISEIF(0 != big_block_close(block),
            ex_close,
            NULL);
            return -1;
        ex_open:
        ex_close:
        ex_array:
            return -1;
    }
    return 0;
}


int
big_file_read_records(BigFile * bf,
    const BigRecordType * rtype,
    ptrdiff_t offset,
    size_t size,
    void * buf)
{
    int i;
    for(i = 0; i < rtype->nfield; i ++) {
        BigArray array[1];
        BigBlock block[1];
        BigBlockPtr ptr = {0};

        RAISEIF(0 != big_record_view_field(rtype, i, array, size, buf),
            ex_array,
            NULL);
        RAISEIF(0 != big_file_open_block(bf, block, rtype->fields[i].name),
            ex_open,
            NULL);
        RAISEIF(0 != big_block_seek(block, &ptr, offset),
            ex_seek,
            NULL);
        RAISEIF(0 != big_block_read(block, &ptr, array),
            ex_write,
            NULL);
        RAISEIF(0 != big_block_close(block),
            ex_close,
            NULL);
        ex_write:
        ex_seek:
            RAISEIF(0 != big_block_close(block),
            ex_close,
            NULL);
            return -1;
        ex_open:
        ex_close:
        ex_array:
            return -1;
    }
    return 0;
}
