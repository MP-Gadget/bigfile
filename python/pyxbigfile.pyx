#cython: embedsignature=True
cimport numpy
from libc.stddef cimport ptrdiff_t
from libc.string cimport strcpy
from libc.stdlib cimport free
import numpy

cdef extern from "bigfile.c":
    struct CBigFile "BigFile":
        char * basename

    struct CBigBlock "BigBlock":
        char * dtype
        int nmemb
        char * basename
        size_t size
        int Nfile

    struct CBigBlockPtr "BigBlockPtr":
        pass

    struct CBigArray "BigArray":
        int ndim
        char * dtype
        ptrdiff_t * dims
        ptrdiff_t * strides
        size_t size
        void * data

    struct CBigBlockAttr "BigBlockAttr":
        int nmemb
        char dtype[8]
        char * name
        char * data

    char * big_file_get_error_message()
    void big_file_set_buffer_size(size_t bytes)
    int big_block_open(CBigBlock * bb, char * basename)
    int big_block_clear_checksum(CBigBlock * bb)
    int big_block_create(CBigBlock * bb, char * basename, char * dtype, int nmemb, int Nfile, size_t fsize[])
    int big_block_close(CBigBlock * block)
    int big_block_flush(CBigBlock * block)
    int big_block_seek(CBigBlock * bb, CBigBlockPtr * ptr, ptrdiff_t offset)
    int big_block_seek_rel(CBigBlock * bb, CBigBlockPtr * ptr, ptrdiff_t rel)
    int big_block_read(CBigBlock * bb, CBigBlockPtr * ptr, CBigArray * array)
    int big_block_write(CBigBlock * bb, CBigBlockPtr * ptr, CBigArray * array)
    int big_block_set_attr(CBigBlock * block, char * attrname, void * data, char * dtype, int nmemb)
    int big_block_get_attr(CBigBlock * block, char * attrname, void * data, char * dtype, int nmemb)
    CBigBlockAttr * big_block_lookup_attr(CBigBlock * block, char * attrname)
    CBigBlockAttr * big_block_list_attrs(CBigBlock * block, size_t * count)
    int big_array_init(CBigArray * array, void * buf, char * dtype, int ndim, size_t dims[], ptrdiff_t strides[])

    int big_file_open_block(CBigFile * bf, CBigBlock * block, char * blockname)
    int big_file_create_block(CBigFile * bf, CBigBlock * block, char * blockname, char * dtype, int nmemb, int Nfile, size_t fsize[])
    int big_file_open(CBigFile * bf, char * basename)
    int big_file_list(CBigFile * bf, char *** list, int * N)
    int big_file_create(CBigFile * bf, char * basename)
    int big_file_close(CBigFile * bf)

def set_buffer_size(bytes):
    big_file_set_buffer_size(bytes)

class BigFileError(Exception):
    def __init__(self, msg=None):
        if msg is None:
            msg = big_file_get_error_message()
        Exception.__init__(self, msg)

cdef class BigFile:
    cdef CBigFile bf
    cdef int closed

    def __cinit__(self):
        self.closed = True
    def __init__(self, filename, create=True):
        if create:
            if 0 != big_file_create(&self.bf, filename):
                raise BigFileError()
        else:
            if 0 != big_file_open(&self.bf, filename):
                raise BigFileError()
        self.closed = False

    def __dealloc__(self):
        if not self.closed:
            big_file_close(&self.bf)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    property blocks:
        def __get__(self):
            cdef char ** list
            cdef int N
            big_file_list(&self.bf, &list, &N)
            try:
                return [list[i] for i in range(N)]
            finally:
                for i in range(N):
                    free(list[i])
                free(list)

    def close(self):
        if 0 != big_file_close(&self.bf):
            raise BigFileError()
        self.closed = True

    def open(self, block):
        cdef BigBlock rt = BigBlock()
        if 0 != big_file_open_block(&self.bf, &rt.bb, block):
            raise BigFileError()
        rt.closed = False
        return rt

    def create(self, block, dtype=None, size=None, Nfile=1):
        cdef BigBlock rt = BigBlock()
        cdef numpy.ndarray fsize
        if dtype is None:
            if 0 != big_file_create_block(&self.bf, &rt.bb, block, 'i8',
                    1, 0, NULL):
                raise BigFileError()

        else:
            dtype = numpy.dtype(dtype)
            assert len(dtype.shape) <= 1
            if len(dtype.shape) == 0:
                items = 1
            else:
                items = dtype.shape[0]
            fsize = numpy.empty(dtype='intp', shape=Nfile)
            fsize[:] = (numpy.arange(Nfile) + 1) * size / Nfile \
                     - (numpy.arange(Nfile)) * size / Nfile
            if 0 != big_file_create_block(&self.bf, &rt.bb, block, dtype.base.str,
                    items, Nfile, <size_t*> fsize.data):
                raise BigFileError()
        rt.closed = False
        return rt

    def __getitem__(self, key):
        return self.open(key)

    def __contains__(self, key):
        return key in self.blocks

    def __iter__(self):
        return iter(self.blocks)

cdef class BigBlockAttrSet:
    cdef readonly BigBlock bb

    property keys:
        def __get__(self):
            if self.bb.closed:
                raise BigFileError("block closed")
            cdef size_t count
            cdef CBigBlockAttr * list
            list = big_block_list_attrs(&self.bb.bb, &count)
            return [list[i].name for i in range(count)]

    def __init__(self, BigBlock bb):
        self.bb = bb

    def __iter__(self):
        if self.bb.closed:
            raise BigFileError("block closed")
        return iter(self.keys)

    def __contains__(self, name):
        if self.bb.closed:
            raise BigFileError("block closed")
        cdef CBigBlockAttr * attr = big_block_lookup_attr(&self.bb.bb, name)
        if attr == NULL:
            return False
        return True

    def __getitem__(self, name):
        if self.bb.closed:
            raise BigFileError("block closed")
        cdef CBigBlockAttr * attr = big_block_lookup_attr(&self.bb.bb, name)
        if attr == NULL:
            raise KeyError("attr not found")
        cdef numpy.ndarray result = numpy.empty(attr[0].nmemb, attr[0].dtype)
        if(0 != big_block_get_attr(&self.bb.bb, name, result.data, attr[0].dtype,
            attr[0].nmemb)):
            raise BigFileError()
        return result

    def __setitem__(self, name, value):
        if self.bb.closed:
            raise BigFileError("block closed")
        cdef numpy.ndarray array = numpy.atleast_1d(value).ravel()

        if(0 != big_block_set_attr(&self.bb.bb, name, array.data, 
                array.dtype.base.str,
                array.shape[0])):
            raise BigFileError();

cdef class BigBlock:
    cdef CBigBlock bb
    cdef int closed

    property size:
        def __get__(self):
            return self.bb.size
    
    property dtype:
        def __get__(self):
            return numpy.dtype((self.bb.dtype, self.bb.nmemb))
    property attrs:
        def __get__(self):
            return BigBlockAttrSet(self)
    property Nfile:
        def __get__(self):
            return self.bb.Nfile

    def __cinit__(self):
        self.closed = True

    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    @staticmethod
    def open(filename):
        cdef BigBlock self = BigBlock()
        if 0 != big_block_open(&self.bb, filename):
            raise BigFileError()
        self.closed = False
        return self
    @staticmethod
    def create(filename, dtype=None, size=None, Nfile=1):
        cdef BigBlock self = BigBlock()
        cdef numpy.ndarray fsize

        if dtype is None:
            if 0 != big_block_create(&self.bb, filename, 'i8',
                    1, 0, NULL):
                raise BigFileError()
        else:
            dtype = numpy.dtype(dtype)
            assert len(dtype.shape) <= 1
            if len(dtype.shape) == 0:
                items = 1
            else:
                items = dtype.shape[0]
            fsize = numpy.empty(dtype='intp', shape=Nfile)
            fsize[:] = (numpy.arange(Nfile) + 1) * size / Nfile \
                     - (numpy.arange(Nfile)) * size / Nfile
            if 0 != big_block_create(&self.bb, filename, dtype.base.str,
                    items, Nfile, <size_t*> fsize.data):
                raise BigFileError()
        self.closed = False
        return self

    def clear_checksum(self):
        """ reset the checksum to zero for freshly overwriting the data set
        """
        big_block_clear_checksum(&self.bb)

    def write(self, start, numpy.ndarray buf):
        """ write at offset `start' a chunk of data inf buf.
           
            no checking is performed. assuming buf is of the correct dtype.
        """
        cdef CBigArray array
        cdef CBigBlockPtr ptr
        big_array_init(&array, buf.data, buf.dtype.str, 
                buf.ndim, 
                <size_t *> buf.shape,
                <ptrdiff_t *> buf.strides)
        if 0 != big_block_seek(&self.bb, &ptr, start):
            raise BigFileError()
        if 0 != big_block_write(&self.bb, &ptr, &array):
            raise BigFileError()

    def __getitem__(self, sl):
        """ returns a copy of data, sl can be a slice or a scalar
        """
        if isinstance(sl, slice):
            start, end, stop = sl.indices(self.size)
            if stop != 1:
                raise ValueError('must request a continous chunk')
            return self.read(start, end-start)
        elif numpy.isscalar(sl):
            sl = slice(sl, sl + 1)
            return self[sl][0]
        else:
            raise TypeError('Expecting a slice or a scalar, got a `%s`' %
                    str(type(sl)))

    def read(self, start, length, out=None):
        """ read from offset `start' a chunk of data of length `length', 
            into array `out'.

            out shall match length and self.dtype

            returns out, or a newly allocated array of out is None.
        """
        cdef numpy.ndarray result 
        cdef CBigArray array
        cdef CBigBlockPtr ptr
        cdef int i
        if length == -1:
            length = self.size - start
        if length + start > self.size:
            length = self.size - start
        if out is None:
            result = numpy.empty(dtype=self.dtype, shape=length)
        else:
            result = out
            if result.shape[0] != length:
                raise ValueError("output array length mismatches with the request")
            if result.dtype.base != self.dtype.base:
                raise ValueError("output array type mismatches with the block")
            
        big_array_init(&array, result.data, self.bb.dtype, 
                result.ndim, 
                <size_t *> result.shape,
                <ptrdiff_t *> result.strides)
        if 0 != big_block_seek(&self.bb, &ptr, start):
            raise BigFileError()
        if 0 != big_block_read(&self.bb, &ptr, &array):
            raise BigFileError()
        return result

    def flush(self):
        if not self.closed:
            if 0 != big_block_flush(&self.bb):
                raise BigFileError()

    def close(self):
        if not self.closed:
            if 0 != big_block_close(&self.bb):
                raise BigFileError()
            self.closed = True

    def __dealloc__(self):
        if not self.closed:
            big_block_close(&self.bb)

    def __repr__(self):
        return "<CBigBlock: %s dtype=%s, size=%d>" % (self.bb.basename,
                self.dtype, self.size)


