#ifndef _BIGFILE_H_
#define _BIGFILE_H_
#include <stddef.h>

typedef struct BigFile {
    char * basename;
} BigFile;

typedef struct BigBlockAttr {
    int nmemb;
    char dtype[8];
    char * name;
    char * data;
} BigBlockAttr;

typedef struct BigBlockAttrSet {
    int dirty;
    char * attrbuf;
    size_t bufused;
    size_t bufsize;

    BigBlockAttr * attrlist;
    size_t listused;
    size_t listsize;
} BigBlockAttrSet;

typedef struct BigBlock {
    char dtype[8]; /* numpy style
                        dtype[0] <: little endian, > big endian, = machine endian
                        dtype[1] type char
                        dtype[2:] width in bytes
                    */
    int nmemb;  /* num of dtype typed elements per item */
    char * basename;
    size_t size;
    size_t * fsize; /* Nfile + 1, in units of elements */
    size_t * foffset; /* Nfile + 1, in units of elements */
    unsigned int * fchecksum; /* sysv sums of each file (unreduced) */
    int Nfile;
    BigBlockAttrSet attrset;
    int dirty;
} BigBlock;

typedef struct BigBlockPtr{
    int fileid;
    ptrdiff_t roffset; /* offset within a file */
    ptrdiff_t aoffset; /* abs offset */
} BigBlockPtr;

/**
 *
 * dtype stuff.
 * similar to numpy dtype descr.
 *
 * dtype[0] is endianness:
 *    < LE
 *    > BE
 *    = native
 * dtype[1] is kind
 *    'i'  int
 *    'f'  float
 *    'u'  unsigned int
 * dtype[2:] is the bytewidth
 *
 * dtype[0] can be omitted, in which case native is assumed.
 *
 */

typedef struct BigArray {
    int ndim;
    char dtype[8];
    ptrdiff_t dims[32];
    ptrdiff_t strides[32];
    size_t size;
    void * data;
} BigArray;

typedef struct BigArrayIter {
    ptrdiff_t pos[32];
    BigArray * array;
    int contiguous;
    void * dataptr;
} BigArrayIter;

int big_file_set_buffer_size(size_t bytes);
char * big_file_get_error_message(void);
void big_file_set_error_message(char * msg);
void big_file_checksum(unsigned int * sum, void * buf, size_t size);

/** Open a Bigfile: this stats the directory tree, but does not open the file.
 * It initialises the BigFile structure.
 * Arguments:
 * @param BigFile bf - pointer to uninitialised structure.
 * @param const char * basename - String containing directory to put snapshot in.*/
int big_file_open(BigFile * bf, const char * basename);

/** Create a Bigfile: this makes the directory tree and initialises the BigFile structure.
 * Arguments:
 * @param BigFile bf - pointer to uninitialised structure.
 * @param const char * basename - String containing directory to put snapshot in.*/
int big_file_create(BigFile * bf, const char * basename);
int big_file_list(BigFile * bf, char *** blocknames, int * Nblocks);
int big_file_open_block(BigFile * bf, BigBlock * block, const char * blockname);
int big_file_create_block(BigFile * bf, BigBlock * block, const char * blockname, const char * dtype, int nmemb, int Nfile, const size_t fsize[]);
int big_file_close(BigFile * bf);
int big_block_flush(BigBlock * block);
int big_file_mksubdir_r(const char * pathname, const char * subdir);

int big_block_open(BigBlock * bb, const char * basename);
int big_block_clear_checksum(BigBlock * bb);
int big_block_create(BigBlock * bb, const char * basename, const char * dtype, int nmemb, int Nfile, const size_t fsize[]);
int big_block_close(BigBlock * block);

/** Initialise BigBlockPtr to the place in the BigBlock offset elements from the beginning of the block.
 * This allows you to write into the BigBlock at a position other than the beginning.
 * @param offset - Position to seek to in units of the size of the array element, eg, 8 bytes for an i8. If offset == -1, seek to the end of the array.
 * All other offset < 0 seek array_size + 1 - offset.*/
int big_block_seek(BigBlock * bb, BigBlockPtr * ptr, ptrdiff_t offset);
int big_block_seek_rel(BigBlock * bb, BigBlockPtr * ptr, ptrdiff_t rel);
int big_block_eof(BigBlock * bb, BigBlockPtr * ptr);
int big_block_read(BigBlock * bb, BigBlockPtr * ptr, BigArray * array);
int big_block_read_simple(BigBlock * bb, ptrdiff_t start, ptrdiff_t size, BigArray * array, const char * dtype);

/** Write data stored in a BigArray to a BigBlock.
 * You cannot write beyond the end of the size of the block.
 * The value may be a (small) array.
 * Arguments:
 * @param block - pointer to opened BigBlock
 * @param ptr - Absolute position to write to in the file. Construct this with a call to big_block_seek.
 * @param array - BigArray containing the data which should be written.
 * @returns 0 if successful. */
int big_block_write(BigBlock * bb, BigBlockPtr * ptr, BigArray * array);
int big_block_remove_attr(BigBlock * block, const char * attrname);

/** Set an attribute on a BigBlock: attributes are plaintext key-value pairs stored in a special file in the Block directory.
 * The value may be a (small) array.
 * Arguments:
 * @param block - pointer to opened BigBlock
 * @param attrname - name of the attribute to store.
 * @param data - pointer to data to store
 * @param dtype - Type of data array in the format used by dtype.
 * @param nmemb - Number of members in the data array.
 * @returns 0 if successful. */
int big_block_set_attr(BigBlock * block, const char * attrname, const void * data, const char * dtype, int nmemb);

/** Get an attribute on a BigBlock: attributes are plaintext key-value pairs stored in a special file in the Block directory.
 * Attribute value is stored in the memory pointed to by data, so make sure it is big enough!
 * Arguments:
 * @param block - pointer to opened BigBlock
 * @param attrname - name of the attribute to store.
 * @param data - pointer to data in which to place attribute
 * @param dtype - Type of data array in the format used by dtype.
 * @param nmemb - Number of members to get. Must be equal to number of members
 * originally stored or an error is raised and -1 returned.*/
int big_block_get_attr(BigBlock * block, const char * attrname, void * data, const char * dtype, int nmemb);
BigBlockAttr * big_block_lookup_attr(BigBlock * block, const char * attrname);
BigBlockAttr * big_block_list_attrs(BigBlock * block, size_t * count);

int dtype_normalize(char * dst, const char * src);
void dtype_format(char * buffer, const char * dtype, const void * data, const char * flags);
void dtype_parse(const char * buffer, const char * dtype, void * data, const char * fmt);
int dtype_convert(BigArrayIter * dst, BigArrayIter * src, size_t nmemb);
int dtype_convert_simple(void * dst, const char * dstdtype, const void * src, const char * srcdtype, size_t nmemb);
int dtype_cmp(const char * dtype1, const char * dtype2);
char dtype_kind(const char * dtype);
int dtype_needswap(const char * dtype);
int dtype_itemsize(const char * dtype);

/** Create a BigArray from raw memory.
 * A BigArray is a checked, multidimensional, array format. It is used to provide metadata
 * about a raw void pointer to big_block_write. It is an implementation detail of this library, not stored on disc.
 * This routine constructs the BigArray.
 * Data is not copied, so the memory must not be freed before the BigArray is deallocated.
 * @params array - uninitialised BigArray.
 * @params buf - pointer to raw memory to be stored in the BigArray.
 * @params dtype - type of the data array as a string.
 * @params ndim - Number of dimensions to use in the BigArray.
 * Note that a BigArray can have up to 32 dimensions, but only 2 of them can be stored in a BigFile.
 * @params dims - list of integers containing the size of each dimension of the BigArray
 * @params strides - Integers containing the size of each element of the BigArray, for use with BigArrayIter.
 * This is a BigArray implementation of numpy strides, so see documentation here: http://www.scipy-lectures.org/advanced/advanced_numpy/#indexing-scheme-strides
 * For a multidimensional array with C ordering and N dimensions,
 * strides[i] is the length of the array in the ith dimension.
 * strides[N-1] is the size of a single element.
 * For BigArrays to be used in a BigFile,
 * strides[1] should be the size of one element, dtype_itemsize(dtype).
 * strides[0] should be one row: dims[1] * strides[1].
 * The array can be ordered in other ways. For example, if you have:
 *
 * struct P {  double pos[3];  double vel[3]; };
 *
 * You can use strides[1] = sizeof(double), strides[0] = sizeof(struct P).
 * Then set buf to &P[0].pos[0] to dump position or &P[0].vel[0] to dump velocity.
 */
int big_array_init(BigArray * array, void * buf, const char * dtype, int ndim, const size_t dims[], const ptrdiff_t strides[]);
int big_array_iter_init(BigArrayIter * iter, BigArray * array);
void big_array_iter_advance(BigArrayIter * iter);
#endif
