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

    struct CBigBlock "BigBlock":
        char * dtype
        int nmemb
        char * basename
        size_t size
        int Nfile
        unsigned int * fchecksum; 
        int dirty
        CBigAttrSet * attrset;

    struct CBigBlockPtr "BigBlockPtr":
        pass

    struct CBigArray "BigArray":
        int ndim
        char * dtype
        ptrdiff_t * dims
        ptrdiff_t * strides
        size_t size
        void * data

    struct CBigAttr "BigAttr":
        int nmemb
        char dtype[8]
        char * name
        char * data

    struct CBigAttrSet "BigAttrSet":
        pass

    char * big_file_get_error_message()
    void big_file_set_buffer_size(size_t bytes)
    int big_block_open(CBigBlock * bb, char * basename)
    int big_block_clear_checksum(CBigBlock * bb)
    int big_block_create(CBigBlock * bb, char * basename, char * dtype, int nmemb, int Nfile, size_t fsize[])
    int big_block_close(CBigBlock * block)
    int big_block_flush(CBigBlock * block)
    void big_block_set_dirty(CBigBlock * block, int value)
    void big_attrset_set_dirty(CBigAttrSet * attrset, int value)
    int big_block_seek(CBigBlock * bb, CBigBlockPtr * ptr, ptrdiff_t offset)
    int big_block_seek_rel(CBigBlock * bb, CBigBlockPtr * ptr, ptrdiff_t rel)
    int big_block_read(CBigBlock * bb, CBigBlockPtr * ptr, CBigArray * array)
    int big_block_write(CBigBlock * bb, CBigBlockPtr * ptr, CBigArray * array)
    int big_block_set_attr(CBigBlock * block, char * attrname, void * data, char * dtype, int nmemb)
    int big_block_remove_attr(CBigBlock * block, char * attrname)
    int big_block_get_attr(CBigBlock * block, char * attrname, void * data, char * dtype, int nmemb)
    CBigAttr * big_block_lookup_attr(CBigBlock * block, char * attrname)
    CBigAttr * big_block_list_attrs(CBigBlock * block, size_t * count)
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

class BigFileClosedError(Exception):
    def __init__(self, bigfile):
        Exception.__init__(self, "File is closed")

class BigBlockClosedError(Exception):
    def __init__(self, bigblock):
        Exception.__init__(self, "Block is closed")

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

    def _check_closed(self):
        if self.closed:
            raise BigFileClosedError(self)

    property basename:
        def __get__(self):
            self._check_closed();
            return '%s' % self.bf.basename.decode()

    def list_blocks(self):
        cdef char ** list
        cdef int N
        self._check_closed();
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

cdef class BigFileAttrSet:
    cdef readonly BigBlock bb

    def keys(self):
        self.bb._check_closed()
        cdef size_t count
        cdef CBigAttr * list
        list = big_block_list_attrs(&self.bb.bb, &count)
        return sorted([str(list[i].name.decode()) for i in range(count)])

    def __init__(self, BigBlock bb):
        self.bb = bb

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, name):
        name = name.encode()
        self.bb._check_closed()
        cdef CBigAttr * attr = big_block_lookup_attr(&self.bb.bb, name)
        if attr == NULL:
            return False
        return True

    def __getitem__(self, name):
        name = name.encode()
        self.bb._check_closed()

        cdef CBigAttr * attr = big_block_lookup_attr(&self.bb.bb, name)
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
        self.bb._check_closed()

        cdef CBigAttr * attr = big_block_lookup_attr(&self.bb.bb, name)
        if attr == NULL:
            raise KeyError("attr not found")
        big_block_remove_attr(&self.bb.bb, name)

    def __setitem__(self, name, value):
        name = name.encode()
        self.bb._check_closed()
        if isinstance(value, numpy.ndarray):
            if value.dtype.char == 'U':
                value = str(value).encode()

        if isstr(value): 
            value = value.encode()
        cdef numpy.ndarray buf = numpy.atleast_1d(value).ravel()
        if buf.dtype.char == 'S':
            buf = buf.view(dtype='S1')
            dtype = 'S1'.encode()
        else:
            dtype = buf.dtype.base.str.encode()
        print(name, value, dtype)
        if(0 != big_block_set_attr(&self.bb.bb, name, buf.data, 
                dtype,
                buf.shape[0])):
            raise BigFileError();

    def __repr__(self):
        t = ("<BigAttr (%s)>" %
            ','.join([ "%s=%s" %
                       (str(key), repr(self[key]))
                for key in self]))
        return t

cdef class BigBlock:
    cdef CBigBlock bb
    cdef readonly int closed
    cdef public comm

    property size:
        def __get__(self):
            self._check_closed()
            return self.bb.size
    
    property dtype:
        def __get__(self):
            self._check_closed()
            return numpy.dtype((self.bb.dtype, self.bb.nmemb))
    property attrs:
        def __get__(self):
            self._check_closed()
            return BigFileAttrSet(self)
    property Nfile:
        def __get__(self):
            self._check_closed()
            return self.bb.Nfile

    def _check_closed(self):
        if self.closed:
            raise BigBlockClosedError(self)

    def __cinit__(self):
        self.closed = True
        self.comm = None

    def __init__(self):
        pass

    def __enter__(self):
        self._check_closed()
        return self

    def __exit__(self, type, value, tb):
        self.close()

    def open(self, BigFile f, blockname):
        f._check_closed()
        blockname = blockname.encode()
        if 0 != big_file_open_block(&f.bf, &self.bb, blockname):
            raise BigFileError()
        self.closed = False

    def create(self, BigFile f, blockname, dtype=None, size=None, Nfile=1):
        f._check_closed()
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
        self._check_closed()
        big_block_clear_checksum(&self.bb)

    def write(self, start, numpy.ndarray buf):
        """ write at offset `start' a chunk of data inf buf.
           
            no checking is performed. assuming buf is of the correct dtype.
        """
        self._check_closed()
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
        self._check_closed()
        if isinstance(sl, slice):
            start, end, stop = sl.indices(self.size)
            if stop != 1:
                raise ValueError('must request a continous chunk')
            return self.read(start, end-start)
        elif sl is Ellipsis:
            return self[:]
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
        self._check_closed()
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

        dirty = any(comm.allgather(self.bb.dirty))

        if Nfile > 0:
            fchecksum = <unsigned int[:Nfile]>self.bb.fchecksum
            fchecksum2 = fchecksum.copy()
            comm.Allreduce(fchecksum, fchecksum2)
            for i in range(Nfile):
                fchecksum[i] = fchecksum2[i]

        if comm.rank == 0: 
            big_block_set_dirty(&self.bb, dirty);
        else:
            big_block_set_dirty(&self.bb, 0);
            big_attrset_set_dirty(self.bb.attrset, 0);

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
        if self.bb.dtype == b'####':
            return "<CBigBlock: %s>" % self.bb.basename

        return "<CBigBlock: %s dtype=%s, size=%d>" % (self.bb.basename,
                self.dtype, self.size)
