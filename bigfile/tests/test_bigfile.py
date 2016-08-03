
from bigfile import BigFile
from bigfile import BigFileMPI
from bigfile import BigData
from bigfile import BigFileClosedError

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
    ('complex64'), 
    ('complex128'), 
    ('complex128', 2), 
]

def test_create():
    fname = tempfile.mkdtemp()
    x = BigFile(fname, create=True)
    x.create('.')

    for d in dtypes:
        d = numpy.dtype(d)
        numpy.random.seed(1234)

        # test creating
        with x.create(d.str, Nfile=1, dtype=d, size=128) as b:
            data = numpy.random.uniform(100000, size=128*128).view(dtype=b.dtype.base).reshape([-1] + list(d.shape))[:b.size]
            b.write(0, data)

        with x[d.str] as b:
            assert_equal(b[:], data.astype(d.base))
            assert_equal(b[:],  b[...])

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
    import os
    os.system("ls -r %s" % fname)
    for b in x.blocks:
        assert b in x

    for b in x:
        assert b in x

    bd = BigData(x)
    assert set(bd.dtype.names) == set(x.blocks)
    d = bd[:]

    shutil.rmtree(fname)

def test_closed():
    fname = tempfile.mkdtemp()
    x = BigFile(fname, create=True)
    x.create('.')
    x.close()
    assert x.blocks == []
    try:
        h = x['.']
    except BigFileClosedError:
        pass
    try:
        x.refresh()
    except BigFileClosedError:
        pass

def test_attr():
    fname = tempfile.mkdtemp()
    x = BigFile(fname, create=True)
    with x.create('.', dtype=None) as b:
        b.attrs['int'] = 128
        b.attrs['float'] = [128.0, 3, 4]
        b.attrs['string'] = 'abcdefg'
        b.attrs['complex'] = 128 + 128J

    with x.open('.') as b:
        assert_equal(b.attrs['int'], 128)
        assert_equal(b.attrs['float'], [128.0, 3, 4])
        assert_equal(b.attrs['string'],  'abcdefg')
        assert_equal(b.attrs['complex'],  128 + 128J)
        b.attrs['int'] = 30
        b.attrs['float'] = [3, 4]
        b.attrs['string'] = 'defg'
        b.attrs['complex'] = 32 + 32J

    with x.open('.') as b:
        assert_equal(b.attrs['int'], 30)
        assert_equal(b.attrs['float'], [3, 4])
        assert_equal(b.attrs['string'],  'defg')
        assert_equal(b.attrs['complex'],  32 + 32J)

    shutil.rmtree(fname)

def test_mpi_create():
    from mpi4py import MPI
    if MPI.COMM_WORLD.rank == 0:
        fname = tempfile.mkdtemp()
        fname = MPI.COMM_WORLD.bcast(fname)
    else:
        fname = MPI.COMM_WORLD.bcast(None)
    x = BigFileMPI(MPI.COMM_WORLD, fname, create=True)
    for d in dtypes:
        d = numpy.dtype(d)
        numpy.random.seed(1234)

        # test creating
        with x.create(d.str, Nfile=1, dtype=d, size=128) as b:
            data = numpy.random.uniform(100000, size=128*128).view(dtype=b.dtype.base).reshape([-1] + list(d.shape))[:b.size]
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

    for b in x.blocks:
        assert b in x

    for b in x:
        assert b in x

    bd = BigData(x)
    assert set(bd.dtype.names) == set(x.blocks)
    d = bd[:]

    MPI.COMM_WORLD.barrier()
    if MPI.COMM_WORLD.rank == 0:
        shutil.rmtree(fname)

def test_mpi_attr():
    from mpi4py import MPI
    if MPI.COMM_WORLD.rank == 0:
        fname = tempfile.mkdtemp()
        fname = MPI.COMM_WORLD.bcast(fname)
    else:
        fname = MPI.COMM_WORLD.bcast(None)
    x = BigFileMPI(MPI.COMM_WORLD, fname, create=True)
    with x.create('.', dtype=None) as b:
        b.attrs['int'] = 128
        b.attrs['float'] = [128.0, 3, 4]
        b.attrs['string'] = 'abcdefg'

    with x.open('.') as b:
        assert_equal(b.attrs['int'], 128)
        assert_equal(b.attrs['float'], [128.0, 3, 4])
        assert_equal(b.attrs['string'],  'abcdefg')
        b.attrs['int'] = 30
        b.attrs['float'] = [3, 4]
        b.attrs['string'] = 'defg'

    with x.open('.') as b:
        assert_equal(b.attrs['int'], 30)
        assert_equal(b.attrs['float'], [3, 4])
        assert_equal(b.attrs['string'],  'defg')

    MPI.COMM_WORLD.barrier()
    if MPI.COMM_WORLD.rank == 0:
        shutil.rmtree(fname)
