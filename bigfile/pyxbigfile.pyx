#cython: embedsignature=True

from cpython.ref cimport Py_INCREF, Py_DECREF
cimport numpy
from libc.stddef cimport ptrdiff_t
from libc.string cimport strcpy, memcpy, strdup
from libc.stdlib cimport free, malloc

import numpy
import os
import errno

numpy.import_array()

try:
    basestring  # attempt to evaluate basestring
    def isstr(s):
        return isinstance(s, basestring)
except NameError:
    def isstr(s):
        return isinstance(s, str)


cdef extern from "bigfile.h":
    ctypedef void * CBigFileStream "BigFileStream"

    struct CBigFileMethods "BigFileMethods":
      void * backend
      int (*mkdir)(void * backend, const char * dirname, char ** error)
      int (*dscan)(void * backend, const char * dirname, char *** names, char ** error)

      CBigFileStream (*fopen)(void * backend, const char * filename,
                    const char * mode,
                    int buffered,
                    char ** error)
      int (*fseek)(CBigFileStream stream, long offset, int whence, char ** error)
      size_t (*fread)(CBigFileStream stream, void *ptr, size_t size, char ** error)
      size_t (*freadall)(CBigFileStream stream, char ** buffer, char ** error)
      size_t (*fwrite)(CBigFileStream stream, const void *ptr, size_t size, char ** error)
      int (*fclose)(CBigFileStream handle)
        

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

    struct CBigRecordField "BigRecordField":
        char * name
        char * dtype
        int nmemb
        int elsize
        int offset

    struct CBigRecordType "BigRecordType":
        CBigRecordField * fields
        int nfield
        int itemsize

    char * big_file_get_error_message() nogil
    void big_file_set_buffer_size(size_t bytes) nogil
    int big_block_grow(CBigBlock * bb, int Nfilegrow, size_t fsize[]) nogil
    int big_block_close(CBigBlock * block) nogil
    void _big_block_close_internal(CBigBlock * block) nogil
    void * _big_block_pack(CBigBlock * block, size_t * n) nogil
    void _big_block_unpack(CBigBlock * block, void * buf) nogil

    int big_block_flush(CBigBlock * block) nogil
    int big_block_set_dirty(CBigBlock * block, int dirty) nogil
    int big_block_seek(CBigBlock * bb, CBigBlockPtr * ptr, ptrdiff_t offset) nogil
    int big_block_seek_rel(CBigBlock * bb, CBigBlockPtr * ptr, ptrdiff_t rel) nogil
    int big_block_read(CBigBlock * bb, CBigBlockPtr * ptr, CBigArray * array) nogil
    int big_block_write(CBigBlock * bb, CBigBlockPtr * ptr, CBigArray * array) nogil
    int big_block_set_attr(CBigBlock * block, char * attrname, void * data, char * dtype, int nmemb) nogil
    int big_block_remove_attr(CBigBlock * block, char * attrname) nogil
    int big_block_get_attr(CBigBlock * block, char * attrname, void * data, char * dtype, int nmemb) nogil
    void big_block_set_methods(CBigBlock * bb, CBigFileMethods * methods) nogil
    CBigAttr * big_block_lookup_attr(CBigBlock * block, char * attrname) nogil
    CBigAttr * big_block_list_attrs(CBigBlock * block, size_t * count) nogil
    int big_array_init(CBigArray * array, void * buf, char * dtype, int ndim, size_t dims[], ptrdiff_t strides[]) nogil

    int big_file_open_block(CBigFile * bf, CBigBlock * block, char * blockname) nogil
    int big_file_create_block(CBigFile * bf, CBigBlock * block, char * blockname, char * dtype, int nmemb, int Nfile, size_t fsize[]) nogil
    int big_file_open(CBigFile * bf, char * basename) nogil
    void big_file_set_methods(CBigFile * bf, CBigFileMethods * methods) nogil
    int big_file_list(CBigFile * bf, char *** list, int * N) nogil
    int big_file_create(CBigFile * bf, char * basename) nogil
    int big_file_close(CBigFile * bf) nogil

    void big_record_type_clear(CBigRecordType * rtype) nogil
    void big_record_type_set(CBigRecordType * rtype, int i, char * name, char * dtype, int nmemb) nogil
    void big_record_type_complete(CBigRecordType * rtype) nogil
    void big_record_set(CBigRecordType * rtype, void * buf, int ifield, void * data) nogil
    void big_record_get(CBigRecordType * rtype, void * buf, int ifield, void * data) nogil
    int big_record_view_field(CBigRecordType * rtype, int ifield,
        CBigArray * array, size_t size, void * buf) nogil
    int big_file_write_records(CBigFile * bf, CBigRecordType * rtype,
        ptrdiff_t offset, size_t size, void * buf) nogil
    int big_file_read_records(CBigFile * bf, CBigRecordType * rtype,
        ptrdiff_t offset, size_t size, void * buf) nogil
    int big_file_create_records(CBigFile * bf, CBigRecordType * rtype,
        char * mode, size_t Nfile, size_t * size_per_file) nogil

cdef extern from "bigfile-internal.h":
    pass


def set_buffer_size(bytes):
    big_file_set_buffer_size(bytes)

class Error(Exception):
    def __init__(self, msg=None):
        cdef char * errmsg = big_file_get_error_message()
        if errmsg == NULL:
            errmsg = "Unknown error (could have been swallowed due to poor threading support)"
        if msg is None:
            msg = errmsg
        Exception.__init__(self, msg)

class FileClosedError(Exception):
    def __init__(self, bigfile):
        Exception.__init__(self, "File is closed")

class ColumnClosedError(Exception):
    def __init__(self, bigblock):
        Exception.__init__(self, "Block is closed")

cdef class BigFilePyStream:
    cdef readonly object fobj
    def __init__(self, fobj):
        self.fobj = fobj

    @staticmethod
    cdef int _fclose(CBigFileStream stream) with gil:
        print("fclose called")
        cdef self = <BigFilePyStream>stream
        self.fobj.close()
        return 0
 
    @staticmethod
    cdef size_t _fread(CBigFileStream stream, void * buf, size_t size, char ** error) with gil:
        print("read called")
        cdef self = <BigFilePyStream>stream
        cdef numpy.ndarray data
        try:
            data = numpy.fromfile(self.fobj, count=size, dtype='u1')
        except Exception as e:
            error[0] = strdup(str(e).encode())
            return 0
        memcpy(buf, data.data, size)
        return size

    @staticmethod
    cdef size_t _fwrite(CBigFileStream stream, const void * buf, size_t size, char ** error) with gil:
        print("wrtie called")
        cdef self = <BigFilePyStream>stream
        try:
            numpy.array(<const char [:size:1]>buf).tofile(self.fobj)
        except Exception as e:
            error[0] = strdup(str(e).encode())
            return 0
        return size

    @staticmethod
    cdef size_t _freadall(CBigFileStream stream, char ** buffer, char ** error) with gil:
        print("freadall called")
        cdef self = <BigFilePyStream>stream
        cdef char * r
        try:
            self.fobj.seek(0)
            d = self.fobj.read()
            size = len(d)
            r = <char*>malloc(size + 1)
            memcpy(r, <char*>d, size)
            r[size] = 0
            buffer[0] = r
            return size
        except Exception as e:
            buffer[0] = NULL
            error[0] = strdup(str(e).encode())
            return 0

    @staticmethod
    cdef int _fseek(CBigFileStream stream, long offset, int whence, char ** error) with gil:
        print("fseek called")
        cdef self = <BigFilePyStream>stream
        try:
            self.fobj.seek(offset, whence)
            return 0   
        except Exception as e:
            error[0] = strdup(str(e).encode())
            return -1

# An unpickle function via __reduce__ is needed for Python 2;
# c.f. https://github.com/cython/cython/issues/2757
def _unpickle_object(kls, basekls, state):
    obj = basekls.__new__(kls)
    obj.__setstate__(state)
    return obj


cdef class BigFileBackend:
    cdef CBigFileMethods methods

    def __cinit__(self):
        self.methods.fopen = BigFileBackend._fopen
        self.methods.mkdir = BigFileBackend._mkdir
        self.methods.dscan = BigFileBackend._dscan
        self.methods.fclose = BigFilePyStream._fclose
        self.methods.fread = BigFilePyStream._fread
        self.methods.fwrite = BigFilePyStream._fwrite
        self.methods.freadall = BigFilePyStream._freadall
        self.methods.fseek = BigFilePyStream._fseek
        self.methods.backend = <void*> self
        print("backend is ", "%X" % <size_t> self.methods.backend, self)

    def __getstate__(self):
        return ()

    def __setstate__(self, state):
        pass

    def __reduce__(self):
        return _unpickle_object, (type(self), BigFileBackend, self.__getstate__(),)

    def open(self, filename, mode, buffering):
        raise NotImplementedError

    def mkdir(self, dirname):
        raise NotImplementedError

    def scandir(self, dirname):
        raise NotImplementedError

    @staticmethod
    cdef CBigFileStream _fopen(void * backend,
        const char * filename,
        const char * mode,
        int buffered,
        char ** error) with gil:
        cdef BigFileBackend self = <BigFileBackend> backend
        print("fopen self",  self)
        print("fopen filename = ", filename)
        print("fopen mode = ", mode)
        print("fopen backend = ",  "%X" % <size_t> backend)
        print("fopen self",  self)
        try:
            stream = BigFilePyStream(self.open(filename.decode(), mode.decode(), -1 if buffered else 0))
        except Exception as e:
            error[0] = strdup(str(e).encode())
            return NULL

        Py_INCREF(stream)
        return <void*>stream

    @staticmethod
    cdef int _mkdir(void * backend, const char * dirname, char ** error) with gil:
        cdef BigFileBackend self = <BigFileBackend>backend
        try:
            self.mkdir(dirname)
        except (IOError, OSError) as e:
            if e.errno == errno.EEXIST:
                return 0
        except Exception as e:
            print(e)
            error[0] = strdup(str(e).encode())
            print(error[0])
            return -1
        return 0 

    # FIXME: forward errors
    @staticmethod
    cdef int _dscan(void * backend, const char * dirname, char *** names, char **error) with gil:
        cdef BigFileBackend self = <BigFileBackend>backend
        r = self.scandir(dirname)

        names[0] = <char**>malloc(sizeof(char*) * len(r))
        for i, name in enumerate(r):
            names[0][i] = strdup(name)
        return len(r)

cdef class FileLowLevelAPI:
    cdef CBigFile bf
    cdef readonly object backend
    cdef int _deallocated

    def __cinit__(self):
        self._deallocated = True
        pass

    def __init__(self, filename, BigFileBackend backend, create=False):
        """ if create is True, create the file if it is nonexisting"""
        filename = filename.encode()
        cdef char * filenameptr = filename
        if create:
            with nogil:
                rt = big_file_create(&self.bf, filenameptr)
        else:
            with nogil:
                rt = big_file_open(&self.bf, filenameptr)
        if rt != 0:
            raise Error()

        self.backend = backend

        if backend is not None:
            big_file_set_methods(&self.bf, &backend.methods)
        else:
            big_file_set_methods(&self.bf, NULL)
        self._deallocated = False

    def __dealloc__(self):
        if not self._deallocated:
            big_file_close(&self.bf)
            self._deallocated = True

    def __reduce__(self):
        return _unpickle_object, (type(self), FileLowLevelAPI, self.__getstate__(),)

    def __getstate__(self):
        return (self.bf.basename.decode(), self._deallocated, self.backend)

    def __setstate__(self, state):
        cdef BigFileBackend backend
        filename, deallocated, backend = state
        filename = filename.encode()
        cdef char * filenameptr = filename

        self._deallocated = deallocated
        self.backend = backend
        if backend is not None:
            big_file_set_methods(&self.bf, &backend.methods)
        else:
            big_file_set_methods(&self.bf, NULL)
        if not deallocated:
            with nogil:
                rt = big_file_open(&self.bf, filenameptr)

    property basename:
        def __get__(self):
            return '%s' % self.bf.basename.decode()

    def list_blocks(self):
        cdef char ** list
        cdef int N
        with nogil:
            big_file_list(&self.bf, &list, &N)
        try:
            return sorted([str(list[i].decode()) for i in range(N)])
        finally:
            for i in range(N):
                free(list[i])
            free(list)
        return []

    def close(self):
        #never really need to close, since we are just freeing a few memory blocks
        pass

cdef class AttrSet:
    cdef readonly ColumnLowLevelAPI bb

    def keys(self):
        cdef size_t count
        cdef CBigAttr * list
        list = big_block_list_attrs(&self.bb.bb, &count)
        return sorted([str(list[i].name.decode()) for i in range(count)])

    def __init__(self, ColumnLowLevelAPI bb):
        self.bb = bb

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, name):
        name = name.encode()
        cdef CBigAttr * attr = big_block_lookup_attr(&self.bb.bb, name)
        if attr == NULL:
            return False
        return True

    def __getitem__(self, name):
        name = name.encode()

        cdef CBigAttr * attr = big_block_lookup_attr(&self.bb.bb, name)
        if attr == NULL:
            raise KeyError("attr not found")
        cdef numpy.ndarray result = numpy.empty(attr[0].nmemb, attr[0].dtype)
        if(0 != big_block_get_attr(&self.bb.bb, name, result.data, attr[0].dtype,
            attr[0].nmemb)):
            raise Error()
        if attr[0].dtype[1] == 'S':
            return [i.tostring().decode() for i in result]
        if attr[0].dtype[1] == 'a':
            return result.tostring().decode()
        return result

    def __delitem__(self, name):
        name = name.encode()

        cdef CBigAttr * attr = big_block_lookup_attr(&self.bb.bb, name)
        if attr == NULL:
            raise KeyError("attr not found")
        big_block_remove_attr(&self.bb.bb, name)

    def __setitem__(self, name, value):
        name = name.encode()



        if isstr(value):
            dtype = b'a1'
            value = numpy.array(str(value).encode()).ravel().view(dtype='S1').ravel()
        else:
            value = numpy.array(value).ravel()

            if value.dtype.char == 'U':
                value = numpy.array([i.encode() for i in value])

            if value.dtype.hasobject:
                raise ValueError("Attribute value of object type is not supported; serialize it first")

            dtype = value.dtype.base.str.encode()

        cdef numpy.ndarray buf = value

        if(0 != big_block_set_attr(&self.bb.bb, name, buf.data, 
                dtype,
                buf.shape[0])):
            raise Error();

    def __repr__(self):
        t = ("<BigAttr (%s)>" %
            ','.join([ "%s=%s" %
                       (str(key), repr(self[key]))
                for key in self]))
        return t

cdef class ColumnLowLevelAPI:
    cdef CBigBlock bb
    cdef public comm
    cdef int _deallocated
    cdef readonly object backend

    property size:
        def __get__(self):
            return self.bb.size

    property dtype:
        def __get__(self):
            # numpy no longer treats dtype = (x, 1) as dtype = x.
            # but bigfile relies on this.
            if self.bb.nmemb != 1:
                return numpy.dtype((self.bb.dtype, (self.bb.nmemb, )))
            else:
                return numpy.dtype(self.bb.dtype)
    property attrs:
        def __get__(self):
            return AttrSet(self)
    property Nfile:
        def __get__(self):
            return self.bb.Nfile

    def __cinit__(self):
        self.comm = None
        self._deallocated = True

    def __init__(self, BigFileBackend backend):
        self.backend = backend
        if backend is not None:
            big_block_set_methods(&self.bb, &backend.methods)
        else:
            big_block_set_methods(&self.bb, NULL)
        print("Column created with", self.backend)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.flush()

    def __reduce__(self):
        return _unpickle_object, (type(self), ColumnLowLevelAPI, self.__getstate__(),)

    def __getstate__(self):
        cdef void * buf
        cdef size_t n
        cdef numpy.ndarray container

        if not self._deallocated:
            buf = _big_block_pack(&self.bb, &n)
            container = numpy.zeros(n, dtype='uint8')
            memcpy(container.data, buf, n)
            free(buf)
        else:
            container = None

        return (container, self.comm, self._deallocated, self.backend)

    def __setstate__(self, state):
        cdef BigFileBackend backend
        buf, comm, deallocated, backend = state
        cdef numpy.ndarray container = buf

        self.comm = comm
        self._deallocated = deallocated

        self.backend = backend
        if backend is not None:
            big_block_set_methods(&self.bb, &backend.methods)
        else:
            big_block_set_methods(&self.bb, NULL)

        if not deallocated:
            _big_block_unpack(&self.bb, container.data)

    def open(self, FileLowLevelAPI f, blockname):
        blockname = blockname.encode()
        cdef char * blocknameptr = blockname
        with nogil:
            rt = big_file_open_block(&f.bf, &self.bb, blocknameptr)
        if rt != 0:
            raise Error()
        self._deallocated = False

    def create(self, FileLowLevelAPI f, blockname, dtype=None, size=None, numpy.intp_t Nfile=1):

        # need to hold the reference
        blockname = blockname.encode()

        cdef numpy.ndarray fsize
        cdef numpy.intp_t items
        cdef char * dtypeptr
        cdef char * blocknameptr

        blocknameptr = blockname
        if dtype is None:
            with nogil:
                rt = big_file_create_block(&f.bf, &self.bb, blocknameptr, NULL,
                        0, 0, NULL)
            if rt != 0:
                raise Error()
            print("CREATE called", self)
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
            fsize[:] = (numpy.arange(Nfile) + 1) * size // Nfile \
                     - (numpy.arange(Nfile)) * size // Nfile
            dtype2 = dtype.base.str.encode()
            dtypeptr = dtype2
            with nogil:
                rt = big_file_create_block(&f.bf, &self.bb, blocknameptr,
                    dtypeptr,
                    items, Nfile, <size_t*> fsize.data)
            if rt != 0:
                raise Error()

        self._deallocated = False

    def write(self, numpy.intp_t start, numpy.ndarray buf):
        """ write at offset `start' a chunk of data inf buf.

            no checking is performed. assuming buf is of the correct dtype.
        """
        cdef CBigArray array
        cdef CBigBlockPtr ptr

        big_array_init(&array, buf.data, buf.dtype.str.encode(), 
                buf.ndim, 
                <size_t *> buf.shape,
                <ptrdiff_t *> buf.strides)
        with nogil:
            rt = big_block_seek(&self.bb, &ptr, start)
        if rt != 0:
            raise Error()

        with nogil:
            rt = big_block_write(&self.bb, &ptr, &array)
        if rt != 0:
            raise Error()

    def append(self, numpy.ndarray buf, numpy.intp_t Nfile=1):
        """ Append new data at the end of the column.

            Note: this will flush the column to disk to ensure
            future opens of the column sees the grown size.

            All other opened refereneces to the column are no longer
            correct after this operation; they will not see the
            new size.

            This function is not concurrency friendly.
        """

        cdef numpy.ndarray fsize
        cdef numpy.intp_t size = len(buf)

        if Nfile < 0:
            raise ValueError("Cannot create negative number of files.")
        if Nfile == 0 and size != 0:
            raise ValueError("Cannot create zero files for non-zero number of items.")

        fsize = numpy.empty(dtype='intp', shape=Nfile)
        fsize[:] = (numpy.arange(Nfile) + 1) * size // Nfile \
                 - (numpy.arange(Nfile)) * size // Nfile

        # tail is the old end
        tail = self.bb.size

        with nogil:
            rt = big_block_grow(&self.bb, Nfile, <size_t*>fsize.data)
        if rt != 0:
            raise Error()

        # other opened columns are now stale.
        # flush to ensure bf['block'] gets the grown file.
        self.flush()

        return self.write(tail, buf)

    def read(self, numpy.intp_t start, numpy.intp_t length, out=None):
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

        with nogil:
            rt = big_block_seek(&self.bb, &ptr, start)
        if rt != 0:
            raise Error()

        with nogil:
            rt = big_block_read(&self.bb, &ptr, &array)
        if rt != 0:
            raise Error()
        return result

    def _flush(self):
        with nogil:
            rt = big_block_flush(&self.bb)
        if rt != 0:
            raise Error()

    def _MPI_broadcast(self, root, comm):

        cdef void * buf
        cdef size_t n
        cdef numpy.ndarray container

        if comm.rank == root:
            buf = _big_block_pack(&self.bb, &n)
            container = numpy.zeros(n, dtype='uint8')
            memcpy(container.data, buf, n)
        else:
            container = None

        container = comm.bcast(container, root=root)

        if comm.rank != root:
            _big_block_close_internal(&self.bb)
            _big_block_unpack(&self.bb, container.data)

        if comm.rank == root:
            free(buf)

    def _MPI_flush(self):
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

        cdef int rt

        if comm.rank == 0:
            big_block_set_dirty(&self.bb, dirty);
            with nogil:
                rt = big_block_flush(&self.bb)
        else:
            rt = 0

        self._MPI_broadcast(root=0, comm=comm)

        if any(comm.allgather(rt != 0)):
            raise Error()

        comm.barrier()

    def _close(self):
        # only need to flush; memory management is done at dealloc
        with nogil:
            rt = big_block_flush(&self.bb)
        if rt != 0:
            raise Error()

    def _MPI_close(self):
        # only need to flush; memory management is done at dealloc
        self._MPI_flush()

    def __dealloc__(self):
        if not self._deallocated:
            big_block_close(&self.bb)
            self._deallocated = True

    def __repr__(self):
        if self.bb.dtype == b'####':
            return "<CBigBlock: %s>" % (self.bb.basename or "<NULL>")

        try:
            return "<CBigBlock: %s dtype=%s, size=%d>" % (self.bb.basename,
                self.dtype, self.size)
        except:
            return "<CBigBlock is invalid>"
        

cdef class Dataset:
    cdef CBigRecordType rtype
    cdef readonly FileLowLevelAPI file
    cdef readonly size_t size
    cdef readonly tuple shape
    cdef readonly numpy.dtype dtype

    def __init__(self, file, dtype, size):
        self.file = file
        self.rtype.nfield = 0
        big_record_type_clear(&self.rtype)
        fields = []

        for i, name in enumerate(dtype.names):
            basedtype = dtype[name].base.str.encode()
            nmemb = int(numpy.prod(dtype[name].shape))

            big_record_type_set(&self.rtype, i,
                name.encode(),
                basedtype,
                nmemb,
            )
        big_record_type_complete(&self.rtype)

        self.size = size
        self.ndim = 1
        self.shape = (size, )

        dtype = []
        # No need to use offset, because numpy is also
        # compactly packed
        for i in range(self.rtype.nfield):
            if self.rtype.fields[i].nmemb == 1:
                shape = 1
            else:
                shape = (self.rtype.fields[i].nmemb, )
            dtype.append((
                self.rtype.fields[i].name.decode(),
                self.rtype.fields[i].dtype,
                shape)
            )
        self.dtype = numpy.dtype(dtype, align=False)
        assert self.dtype.itemsize == self.rtype.itemsize

    def read(self, numpy.intp_t start, numpy.intp_t length, numpy.ndarray out=None):
        if out is None:
            out = numpy.empty(length, self.dtype)
        with nogil:
            rt = big_file_read_records(&self.file.bf, &self.rtype, start, length, out.data)
        if rt != 0:
            raise Error()
        return out

    def _create_records(self, numpy.intp_t size, numpy.intp_t Nfile=1, char * mode=b"w+"):
        """ mode can be a+ or w+."""
        cdef numpy.ndarray fsize

        if Nfile < 0:
            raise ValueError("Cannot create negative number of files.")
        if Nfile == 0 and size != 0:
            raise ValueError("Cannot create zero files for non-zero number of items.")

        fsize = numpy.empty(dtype='intp', shape=Nfile)
        fsize[:] = (numpy.arange(Nfile) + 1) * size // Nfile \
                 - (numpy.arange(Nfile)) * size // Nfile

        with nogil:
            rt = big_file_create_records(&self.file.bf, &self.rtype, mode, Nfile, <size_t*>fsize.data)
        if rt != 0:
            raise Error()
        self.size = self.size + size

    def append(self, numpy.ndarray buf, numpy.intp_t Nfile=1):
        assert buf.dtype == self.dtype
        assert buf.ndim == 1
        tail = self.size
        self._create_records(len(buf), Nfile=Nfile, mode=b"a+")
        self.write(tail, buf)

    def write(self, numpy.intp_t start, numpy.ndarray buf):
        assert buf.dtype == self.dtype
        assert buf.ndim == 1
        with nogil:
            rt = big_file_write_records(&self.file.bf, &self.rtype, start, buf.shape[0], buf.data)
        if rt != 0:
            raise Error()

    def __dealloc__(self):
        big_record_type_clear(&self.rtype)
