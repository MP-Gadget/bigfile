import numpy
def create_block(comm, file, block, dtype=None, size=None, Nfile=1):
    if comm.rank == 0:
        b = file.create(block, dtype, size, Nfile)
        b.close()
    comm.barrier()
    return file.open(block)

def write_block(comm, file, block, data, Nfile=1):
    """ write distributed data set into file/block;
        currently the check sum is totally broken
    """
    # make it 2D
    # hope this won't copy!
    data = data.reshape(data.shape[0], -1)
    dtype = numpy.dtype((data.dtype.base,
            data.shape[-1]))

    size = comm.allreduce(data.shape[0])
    offset = numpy.array(comm.allgather(data.shape[0]), dtype='i8')
    offset = numpy.cumsum(offset)
    if comm.rank == 0:
        offset = 0
    else:
        offset = offset[comm.rank - 1]
    block = create_block(comm, file, block, dtype, size, Nfile)
    block.write(offset, data)
    block.close()
