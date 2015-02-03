from pyxbigfile import BigBlock
from pyxbigfile import BigFile as BigFileBase
from pyxbigfile import set_buffer_size
import bigfilempi

class BigFile(BigFileBase):
    def mpi_create(self, comm, block, dtype=None, size=None,
            Nfile=1):
        return bigfilempi.create_block(comm, self, block, dtype, size, Nfile)

    def mpi_create_from_data(self, comm, block, localdata, Nfile=1):
        bigfilempi.write_block(comm, self, block, localdata, Nfile)

