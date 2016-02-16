from .pyxbigfile import BigFileError
from .pyxbigfile import BigBlock as BigBlockBase
from .pyxbigfile import BigFile as BigFileBase
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
class BigBlock(BigBlockBase):
    def flush(self):
        self._flush()
    def close(self):
        self._close()

class BigFile(BigFileBase):

    def open(self, blockname):
        block = BigBlock()
        block.open(self, blockname)
        return block

    def create(self, blockname, dtype=None, size=None, Nfile=1):
        block = BigBlock()
        block.create(self, blockname, dtype, size, Nfile)
        return block

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    def __getitem__(self, key):
        if key.endswith('/'):
            return BigFile(os.path.join(self.basename, key))

        return self.open(key)

    def __contains__(self, key):
        return key in self.blocks

    def __iter__(self):
        return iter(self.blocks)

class BigBlockMPI(BigBlock):
    def __init__(self, comm):
        self.comm = comm
        BigBlock.__init__(self)

    def create(self, f, blockname, dtype=None, size=None, Nfile=1):
        if self.comm.rank == 0:
            super(BigBlockMPI, self).create(f, blockname, dtype, size, Nfile)
        self.comm.barrier()
        self.open(f, blockname)

    def close(self):
        self._MPI_close()

    def flush(self):
        self._MPI_flush()

class BigFileMPI(BigFile):

    def __init__(self, comm, filename, create=False):
        self.comm = comm
        if self.comm.rank == 0:
            BigFile.__init__(self, filename, create)
        self.comm.barrier()
        if self.comm.rank != 0:
            BigFile.__init__(self, filename, create=False)

    def __getitem__(self, key):
        if key.endswith('/'):
            return BigFileMPI(comm, os.path.join(self.basename, key))

        return self.open(key)

    def open(self, blockname):
        block = BigBlockMPI(self.comm)
        block.open(self, blockname)
        return block

    def create(self, blockname, dtype=None, size=None, Nfile=1):
        block = BigBlockMPI(self.comm)
        block.create(self, blockname, dtype, size, Nfile)
        return block
 
    def create_from_array(self, blockname, array, Nfile=1):
        size = self.comm.allreduce(len(array))
        offset = sum(self.comm.allgather(len(array))[:self.comm.rank])
        with self.create(blockname, array.dtype, size, Nfile) as b:
            b.write(offset, array)
        return self.open(blockname) 

class BigData:
    def __init__(self, file, blocks):
        self.blocknames = blocks
        self.blocks = dict([
            (block, file[block]) for block in self.blocknames])

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

    def __getitem__(self, sl):
        if isinstance(sl, slice):
            start, end, stop = sl.indices(self.size)
            assert stop == 1
            result = numpy.empty(end - start, dtype=self.dtype)
            for block in self.blocknames:
                result[block][:] = self.blocks[block][sl]
            return result
        elif isstr(sl):
            return self.blocks[sl]
        elif numpy.isscalar(sl):
            sl = slice(sl, sl + 1)
            return self[sl][0]
        else:
            raise TypeError('Expecting a slice or a scalar, got a `%s`' %
                    str(type(sl)))


