from pyxbigfile import BigBlock
from pyxbigfile import BigFileError
from pyxbigfile import BigFile as BigFileBase
from pyxbigfile import set_buffer_size
import bigfilempi

import os
import numpy

try:
    basestring  # attempt to evaluate basestring
    def isstr(s):
        return isinstance(s, basestring)
except NameError:
    def isstr(s):
        return isinstance(s, str)

class BigFile(BigFileBase):
    def mpi_create(self, comm, block, dtype=None, size=None,
            Nfile=1):
        return bigfilempi.create_block(comm, self, block, dtype, size, Nfile)

    def mpi_create_from_data(self, comm, block, localdata, Nfile=1):
        bigfilempi.write_block(comm, self, block, localdata, Nfile)

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


