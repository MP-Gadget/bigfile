
from bigfile import BigFile
from bigfile import BigBlock
import tempfile
import numpy
import shutil
from numpy.testing import assert_equal

dtypes = [
    'i4', 
    'u4', 
    'u8', 
    'f4', 
    'f8', 
    ('f4', 1),
    ('f4', 2), 
]

def test_create():
    fname = tempfile.mkdtemp()
    x = BigFile(fname, create=True)
    for d in dtypes:
        d = numpy.dtype(d)
        numpy.random.seed(1234)

        # test creating
        with x.create(d.str, Nfile=1, dtype=d, size=128) as b:
            shape = [ b.size ] + list(d.shape)
            data = numpy.random.uniform(99999, size=shape)
            b.write(0, data)

        with x[d.str] as b:
            assert_equal(b[:], data.astype(d.base))

        # test writing with an offset
        with x[d.str] as b:
            b.write(1, data[0:1])
            assert_equal(b[1:2], data[0:1].astype(d.base))

        # test writing beyond file length
        with x[d.str] as b:
            caught = False
            try:
                b.write(1, data)
            except:
                caught = True
            assert caught
    assert_equal(set(x.blocks), set([numpy.dtype(d).str for d in dtypes]))
    shutil.rmtree(fname)

def test_attr():
    fname = tempfile.mkdtemp()
    x = BigFile(fname, create=True)
    with x.create('header', Nfile=1, dtype='i8', size=0) as b:
        b.attrs['int'] = 128
        b.attrs['float'] = [128.0, 3, 4]
        b.attrs['string'] = 'abcdefg'

    with x.open('header') as b:
        assert_equal(b.attrs['int'], 128)
        assert_equal(b.attrs['float'], [128.0, 3, 4])
        assert_equal(b.attrs['string'],  'abcdefg')
        b.attrs['int'] = 30
        b.attrs['float'] = [3, 4]
        b.attrs['string'] = 'defg'

    with x.open('header') as b:
        assert_equal(b.attrs['int'], 30)
        assert_equal(b.attrs['float'], [3, 4])
        assert_equal(b.attrs['string'],  'defg')

    shutil.rmtree(fname)
