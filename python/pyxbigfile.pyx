#cython: embedsignature=True
cimport numpy
from libc.stddef cimport ptrdiff_t
from libc.string cimport strcpy
from libc.stdlib cimport free
import numpy

numpy.import_array()

try:
    basestring  # attempt to evaluate basestring
    def isstr(s):
        return isinstance(s, basestring)
except NameError:
    def isstr(s):
        return isinstance(s, str)


cdef extern from "bigfile.c":
    struct CBigFile "BigFile":
        char * basename

    struct CBigBlockAttrSet "BigBlockAttrSet":
        int dirty

    struct CBigBlock "BigBlock":
        char * dtype
        int nmemb
        char * basename
        size_t size
        int Nfile
        unsigned int * fchecksum; 
        int dirty
        CBigBlockAttrSet attrset;

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
    int big_block_remove_attr(CBigBlock * block, char * attrname)
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
    def __init__(self, filename, create=False):
        """ if create is True, create the file if it is nonexisting"""
        filename = filename.encode()
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

    property basename:
        def __get__(self):
            return '%s' % self.bf.basename

    def list_blocks(self):
        cdef char ** list
        cdef int N
        big_file_list(&self.bf, &list, &N)
        try:
            return sorted([str(list[i].decode()) for i in range(N)])
        finally:
            for i in range(N):
                free(list[i])
            free(list)
        return []

    def close(self):
        if 0 != big_file_close(&self.bf):
            raise BigFileError()
        self.closed = True

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
        name = name.encode()
        if self.bb.closed:
            raise BigFileError("block closed")
        cdef CBigBlockAttr * attr = big_block_lookup_attr(&self.bb.bb, name)
        if attr == NULL:
            return False
        return True

    def __getitem__(self, name):
        name = name.encode()
        if self.bb.closed:
            raise BigFileError("block closed")
        cdef CBigBlockAttr * attr = big_block_lookup_attr(&self.bb.bb, name)
        if attr == NULL:
            raise KeyError("attr not found")
        cdef numpy.ndarray result = numpy.empty(attr[0].nmemb, attr[0].dtype)
        if(0 != big_block_get_attr(&self.bb.bb, name, result.data, attr[0].dtype,
            attr[0].nmemb)):
            raise BigFileError()
        if attr[0].dtype[1] == 'S':
            return result.tostring().decode()
        return result

    def __delitem__(self, name):
        name = name.encode()
        if self.bb.closed:
            raise BigFileError("block closed")
        cdef CBigBlockAttr * attr = big_block_lookup_attr(&self.bb.bb, name)
        if attr == NULL:
            raise KeyError("attr not found")
        big_block_remove_attr(&self.bb.bb, name)

    def __setitem__(self, name, value):
        name = name.encode()
        if self.bb.closed:
            raise BigFileError("block closed")
        if isstr(value): 
            value = value.encode()
        cdef numpy.ndarray array = numpy.atleast_1d(value).ravel()
        if array.dtype.char == 'S':
            array = array.view(dtype='S1')
            dtype = 'S1'.encode()
        else:
            dtype = array.dtype.base.str.encode()
        if(0 != big_block_set_attr(&self.bb.bb, name, array.data, 
                dtype,
                array.shape[0])):
            raise BigFileError();

cdef class BigBlock:
    cdef CBigBlock bb
    cdef readonly int closed
    cdef public comm

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
        self.comm = None

    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    def open(self, BigFile f, blockname):
        blockname = blockname.encode()
        if 0 != big_file_open_block(&f.bf, &self.bb, blockname):
            raise BigFileError()
        self.closed = False

    def create(self, BigFile f, blockname, dtype=None, size=None, Nfile=1):
        blockname = blockname.encode()
        cdef numpy.ndarray fsize
        if dtype is None:
            if 0 != big_file_create_block(&f.bf, &self.bb, blockname, NULL,
                    0, 0, NULL):
                raise BigFileError()

        else:
            if Nfile < 0:
                raise ValueError("Cannot create negative number of files.")
            if Nfile == 0 and size != 0:
                raise ValueError("Cannot create zero files for non-zero number of items.")
            dtype = numpy.dtype(dtype)
            assert len(dtype.shape) <= 1
            if len(dtype.shape) == 0:
                items = 1
            else:
                items = dtype.shape[0]
            fsize = numpy.empty(dtype='intp', shape=Nfile)
            fsize[:] = (numpy.arange(Nfile) + 1) * size / Nfile \
                     - (numpy.arange(Nfile)) * size / Nfile
            if 0 != big_file_create_block(&f.bf, &self.bb, blockname, 
                    dtype.base.str.encode(),
                    items, Nfile, <size_t*> fsize.data):
                raise BigFileError()
        self.closed = False

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
        big_array_init(&array, buf.data, buf.dtype.str.encode(), 
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
            if result.dtype.base.itemsize != self.dtype.base.itemsize:
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

    def _flush(self):
        if self.closed: return
        if 0 != big_block_flush(&self.bb):
            raise BigFileError()

    def _MPI_flush(self):
        if self.closed: return
        comm = self.comm
        cdef unsigned int Nfile = self.bb.Nfile
        cdef unsigned int[:] fchecksum
        cdef unsigned int[:] fchecksum2

        dirty = self.bb.dirty
        dirty = any(comm.allgather(dirty))

        if Nfile > 0:
            fchecksum = <unsigned int[:Nfile]>self.bb.fchecksum
            fchecksum2 = fchecksum.copy()
            comm.Allreduce(fchecksum, fchecksum2)
            for i in range(Nfile):
                fchecksum[i] = fchecksum2[i]

        if comm.rank == 0: 
            self.bb.dirty = dirty
        else:
            self.bb.dirty = 0
            self.bb.attrset.dirty = 0

        if 0 != big_block_flush(&self.bb):
            raise BigFileError()
        comm.barrier()

    def _close(self):
        if self.closed: return
        if 0 != big_block_close(&self.bb):
            raise BigFileError()
        self.closed = True

    def _MPI_close(self):
        if self.closed: return
        self._MPI_flush()
        rt = big_block_close(&self.bb)
        self.closed = True
        if 0 != rt:
            raise BigFileError()
        comm = self.comm
        comm.barrier()

    def __dealloc__(self):
        if self.closed: return
        self.close()

    def __repr__(self):
        if self.closed:
            return "<CBigBlock: Closed>"

        return "<CBigBlock: %s dtype=%s, size=%d>" % (self.bb.basename,
                self.dtype, self.size)
