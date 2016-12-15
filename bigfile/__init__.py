from .version import __version__

from .pyxbigfile import BigFileError, BigFileClosedError, BigBlockClosedError
from .pyxbigfile import BigBlock as BigBlockBase
from .pyxbigfile import BigFile as BigFileLowLevel
from .pyxbigfile import set_buffer_size
from . import bigfilempi

import os
import numpy

try:
    basestring  # attempt to evaluate basestring
    def isstr(s):
        return isinstance(s, basestring)
except NameError:
    def isstr(s):
        return isinstance(s, str)

def isstrlist(s):
    if not isinstance(s, list):
        return False
    return all([ isstr(ss) for ss in s])

class BigBlock(BigBlockBase):
    def flush(self):
        self._flush()
    def close(self):
        self._close()

class BigFileBase(BigFileLowLevel):
    def __init__(self, filename, create=False):
        BigFileLowLevel.__init__(self, filename, create)
        self._blocks = []

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    def __contains__(self, key):
        return key in self.blocks

    def __iter__(self):
        return iter(self.blocks)

    def __getitem__(self, key):
        if key.endswith('/'):
            return self.subfile(key)

        return self.open(key)


class BigFile(BigFileBase):

    def __init__(self, filename, create=False):
        BigFileBase.__init__(self, filename, create)
        del self._blocks

    @property
    def blocks(self):
        try:
            return self._blocks
        except AttributeError:
            self._blocks = self.list_blocks()
            return self._blocks

    def open(self, blockname):
        block = BigBlock()
        block.open(self, blockname)
        return block

    def create(self, blockname, dtype=None, size=None, Nfile=1):
        block = BigBlock()
        block.create(self, blockname, dtype, size, Nfile)
        self._blocks = self.list_blocks()
        return block

    def subfile(self, key):
        return BigFile(os.path.join(self.basename, key))

class BigBlockMPI(BigBlock):
    def __init__(self, comm):
        self.comm = comm
        BigBlock.__init__(self)

    def create(self, f, blockname, dtype=None, size=None, Nfile=1):
        if self.comm.rank == 0:
            super(BigBlockMPI, self).create(f, blockname, dtype, size, Nfile)
            super(BigBlockMPI, self).close()
        self.comm.barrier()
        self.open(f, blockname)

    def close(self):
        self._MPI_close()

    def flush(self):
        self._MPI_flush()

class BigFileMPI(BigFileBase):

    def __init__(self, comm, filename, create=False):
        self.comm = comm
        if self.comm.rank == 0:
            BigFileBase.__init__(self, filename, create)
            self.comm.barrier()
        if self.comm.rank != 0:
            self.comm.barrier()
            BigFileBase.__init__(self, filename, create=False)
        self.refresh()

    @property
    def blocks(self):
        return self._blocks

    def refresh(self):
        """ Refresh the list of blocks to the disk, collectively """
        if self.comm.rank == 0:
            self._blocks = self.list_blocks()
        else:
            self._blocks = None
        self._blocks = self.comm.bcast(self._blocks)

    def open(self, blockname):
        block = BigBlockMPI(self.comm)
        block.open(self, blockname)
        return block

    def subfile(self, key):
        return BigFileMPI(self.comm, os.path.join(self.basename, key))

    def create(self, blockname, dtype=None, size=None, Nfile=1):
        block = BigBlockMPI(self.comm)
        block.create(self, blockname, dtype, size, Nfile)
        self.refresh()
        return block

    def create_from_array(self, blockname, array, Nfile=None, memorylimit=1024 * 1024 * 256):
        """ create a block from array like objects
            The operation is well defined only if array is at most 2d.

            Parameters
            ----------
            array : array_like,
                array shall have a scalar dtype. 
            blockname : string
                name of the block
            Nfile : int or None
                number of physical files. if None, 32M items per file
                is used.
            memorylimit : int
                number of bytes to use for the buffering. relevant only if
                indexing on array returns a copy (e.g. IO or dask array)

        """
        size = self.comm.allreduce(len(array))

        # sane value -- 32 million items per physical file
        sizeperfile = 32 * 1024 * 1024

        if Nfile is None:
            Nfile = (size + sizeperfile - 1) // sizeperfile

        offset = sum(self.comm.allgather(len(array))[:self.comm.rank])
        dtype = numpy.dtype((array.dtype, array.shape[1:]))

        itemsize = dtype.itemsize
        # we will do some chunking

        # write memorylimit bytes at most (256M bytes)
        # round to 1024 items
        itemlimit = memorylimit // dtype.itemsize // 1024 * 1024

        with self.create(blockname, dtype, size, Nfile) as b:
            for i in range(0, len(array), itemlimit):
                b.write(offset + i, numpy.array(array[i:i+itemlimit]))

        return self.open(blockname)

class BigData:
    """ Accessing read-only subset of blocks from a bigfile.
    
        Parameters
        ----------
        file : BigFile

        blocks : list or None
            a list of blocks to use. If None is given, all blocks are used.

    """
    def __init__(self, file, blocks=None):
        if blocks is None:
            blocks = file.blocks

        self.blocknames = blocks
        self.blocks = dict([
            (block, file[block]) for block in self.blocknames])

        self.file = file
        dtype = []
        size = None
        for block in self.blocknames:
            bb = self.blocks[block]
            dtype.append((block, bb.dtype))
            if size is None: size = bb.size
            elif bb.size != size:
                raise BigFileError('Dataset length is inconsistent on %s' %block)

        self.size = size
        self.dtype = numpy.dtype(dtype)
        self.ndim = 1
        self.shape = (size, )

    def __getitem__(self, sl):
        if isinstance(sl, tuple):
            if len(sl) == 2:
                if isstr(sl[1]) or isstrlist(sl[1]):
                    # sl[0] shall be column name
                    sl = (sl[1], sl[0])
                col, sl = sl
                return self[col][sl]
            if len(sl) == 1:
                # Python 3? (a,) is sent in.
                return self[sl[0]]

        if isinstance(sl, slice):
            start, end, stop = sl.indices(self.size)
            assert stop == 1
            result = numpy.empty(end - start, dtype=self.dtype)
            for block in self.blocknames:
                result[block][:] = self.blocks[block][sl]
            return result
        elif sl is Ellipsis:
            return self[:]
        elif isstr(sl):
            return self.blocks[sl]
        elif isstrlist(sl):
            assert all([(col in self.blocks) for col in sl])
            return BigData(self.file, sl)
        elif numpy.isscalar(sl):
            sl = slice(sl, sl + 1)
            return self[sl][0]
        else:
            raise TypeError('Expecting a slice or a scalar, got a `%s`' %
                    str(type(sl)))


from numpy.testing import Tester
test = Tester().test
bench = Tester().bench
