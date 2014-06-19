from pyxbigfile import PyBigBlock as BigBlock
from pyxbigfile import set_buffer_size

import numpy


def test():
    set_buffer_size(128);
    x = BigBlock.create('../test/bigblocktest.', Nfile=1, dtype=('i4', 3), size=128)
    print x
    data = numpy.arange(128 * 3).reshape(-1, 3)
    print 'size', x.size
    print 'dtype', x.dtype
    x.write(0, data)
    print 'written'
    print x.read(0, -1)

if __name__ == "__main__":
    test()
