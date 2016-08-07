
from bigfile import BigFile
from bigfile import BigFileMPI
from bigfile import BigData
from bigfile import BigFileClosedError

from mpi4py import MPI

def MPIWorld(NTask, required=[]):
    if not isinstance(NTask, (tuple, list)):
        NTask = (NTask,)

    if not isinstance(required, (tuple, list)):
        required = (required,)

    maxsize = max(required)
    if MPI.COMM_WORLD.size < maxsize:
        raise ValueError("Test Failed because the world is too small. Increase to mpirun -n %d" % maxsize)

    sizes = sorted(set(list(required) + list(NTask)))
    def dec(func):
        def wrapped(*args, **kwargs):
            for size in sizes:
                if MPI.COMM_WORLD.size < size: continue

                color = 0 if MPI.COMM_WORLD.rank < size else 1
                MPI.COMM_WORLD.barrier()
                comm = MPI.COMM_WORLD.Split(color)

                kwargs['comm'] = comm
                failed = 0
                msg = ""
                if color == 0:
                    assert comm.size == size
                    try:
                        func(*args, **kwargs)
                    except:
                        failed = 1
                        import traceback
                        msg = traceback.format_exc()
                gfailed = MPI.COMM_WORLD.allreduce(failed)
                if gfailed > 0:
                    msg = MPI.COMM_WORLD.allgather(msg)
                if failed: raise
                if gfailed > 0:
                    raise ValueError("Some ranks failed with %s" % "\n".join(msg))

        wrapped.__name__ = func.__name__
        return wrapped
    return dec


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
    try:
        x.refresh()
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
