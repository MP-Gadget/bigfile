from bigfile import BigFile
from bigfile import BigBlock
from bigfile import BigFileMPI
from bigfile import BigData
from bigfile import BigFileClosedError

import tempfile
import numpy
import shutil
from numpy.testing import assert_equal

dtypes = [
    '?', 
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

from mpi4py_test import MPIWorld, MPITest

@MPIWorld(NTask=1, required=1)
def test_create(comm):
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

@MPIWorld(NTask=1, required=1)
def test_fileattr(comm):
    import os.path
    fname = tempfile.mkdtemp()
    x = BigFile(fname, create=True)
    assert not os.path.exists(os.path.join(fname, 'attr-v2'))
    assert not os.path.exists(os.path.join(fname, '000000'))
    with x['.'] as bb:
        bb.attrs['value'] = 1234
        assert bb.attrs['value'] == 1234
    assert not os.path.exists(os.path.join(fname, 'header'))
    assert os.path.exists(os.path.join(fname, 'attr-v2'))

    shutil.rmtree(fname)

@MPIWorld(NTask=1, required=1)
def test_bigdata(comm):
    fname = tempfile.mkdtemp()
    x = BigFile(fname, create=True)
    x.create('.')

    for d in dtypes:
        dt = numpy.dtype(d)
        numpy.random.seed(1234)

        # test creating
        with x.create(str(d), Nfile=1, dtype=dt, size=128) as b:
            data = numpy.random.uniform(100000, size=128*128).view(dtype=b.dtype.base).reshape([-1] + list(dt.shape))[:b.size]
            b.write(0, data)

    bd = BigData(x)
    assert set(bd.dtype.names) == set(x.blocks)
    assert isinstance(bd[:], numpy.ndarray)
    assert isinstance(bd['f8'], BigBlock)
    assert_equal(len(bd['f8'].dtype), 0)
    # tuple of one item is the same as non-tuple
    assert isinstance(bd[('f8',)], BigBlock)
    assert_equal(len(bd[('f8',)].dtype), 0)

    assert isinstance(bd['f8', :10], numpy.ndarray)
    assert_equal(len(bd['f8', :10]), 10)
    assert_equal(len(bd['f8', :10].dtype), 0)
    assert_equal(len(bd[['f8',], :10].dtype), 1)

    # tuple of one item is the same as non-tuple
    assert_equal(len(bd[('f8',), :10].dtype), 0)
    assert isinstance(bd[:10, 'f8'], numpy.ndarray)
    assert isinstance(bd['f8'], BigBlock)
    assert isinstance(bd[['f8', 'f4'],], BigData)
    assert_equal(len(bd[['f8', 'f4'],].dtype), 2)
    assert isinstance(bd[['f8', 'f4'],:10], numpy.ndarray)

    shutil.rmtree(fname)

@MPIWorld(NTask=1, required=1)
def test_closed(comm):
    fname = tempfile.mkdtemp()
    x = BigFile(fname, create=True)
    x.create('.')
    x.close()
    assert x.blocks == []
    try:
        h = x['.']
    except BigFileClosedError:
        pass

@MPIWorld(NTask=1, required=1)
def test_attr(comm):
    fname = tempfile.mkdtemp()
    x = BigFile(fname, create=True)
    with x.create('.', dtype=None) as b:
        b.attrs['int'] = 128
        b.attrs['float'] = [128.0, 3, 4]
        b.attrs['string'] = 'abcdefg'
        b.attrs['complex'] = 128 + 128J
        b.attrs['bool'] = True
        b.attrs['arrayustring'] = numpy.array(u'unicode')
        b.attrs['arraysstring'] = numpy.array('str')

    with x.open('.') as b:
        assert_equal(b.attrs['int'], 128)
        assert_equal(b.attrs['float'], [128.0, 3, 4])
        assert_equal(b.attrs['string'],  'abcdefg')
        assert_equal(b.attrs['complex'],  128 + 128J)
        assert_equal(b.attrs['bool'],  True)
        b.attrs['int'] = 30
        b.attrs['float'] = [3, 4]
        b.attrs['string'] = 'defg'
        b.attrs['complex'] = 32 + 32J
        b.attrs['bool'] = False

    with x.open('.') as b:
        assert_equal(b.attrs['int'], 30)
        assert_equal(b.attrs['float'], [3, 4])
        assert_equal(b.attrs['string'],  'defg')
        assert_equal(b.attrs['complex'],  32 + 32J)
        assert_equal(b.attrs['bool'],  False)

    shutil.rmtree(fname)

@MPIWorld(NTask=[1, 2, 3, 4], required=1)
def test_mpi_create(comm):
    if comm.rank == 0:
        fname = tempfile.mkdtemp()
        fname = comm.bcast(fname)
    else:
        fname = comm.bcast(None)
    x = BigFileMPI(comm, fname, create=True)
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

    comm.barrier()
    if comm.rank == 0:
        shutil.rmtree(fname)

@MPIWorld(NTask=[1, 2, 3, 4], required=1)
def test_mpi_attr(comm):
    if comm.rank == 0:
        fname = tempfile.mkdtemp()
        fname = comm.bcast(fname)
    else:
        fname = comm.bcast(None)
    x = BigFileMPI(comm, fname, create=True)
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

    comm.barrier()
    if comm.rank == 0:
        shutil.rmtree(fname)

def test_version():
    import bigfile
    assert hasattr(bigfile, '__version__')

@MPITest(commsize=[1, 4])
def test_mpi_large(comm):
    if comm.rank == 0:
        fname = tempfile.mkdtemp()
        fname = comm.bcast(fname)
    else:
        fname = comm.bcast(None)
    x = BigFileMPI(comm, fname, create=True)

    size= 1024 * 1024
    for d in dtypes:
        d = numpy.dtype(d)
        numpy.random.seed(1234)

        # test creating with create_array; large enough for all types
        data = numpy.random.uniform(100000, size=4 * size).view(dtype=d.base).reshape([-1] + list(d.shape))[:size]
        data1 = comm.scatter(numpy.array_split(data, comm.size))

        with x.create_from_array(d.str, data1, memorylimit=1024 * 128) as b:
            pass

        with x[d.str] as b:
            assert_equal(b[:], data.astype(d.base))

    comm.barrier()
    if comm.rank == 0:
        shutil.rmtree(fname)
