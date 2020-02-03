#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include <bigfile-mpi.h>

int
main(int argc, char * argv[])
{
    MPI_Init(&argc, &argv);

    BigRecordType rtype[1] = {{0}};
    big_record_type_clear(rtype);
    big_record_type_set(rtype, 0, "Position", "f8", 3);
    big_record_type_set(rtype, 1, "Velocity", "f4", 3);
    big_record_type_set(rtype, 2, "BHMass", "f4", 1);
    big_record_type_complete(rtype);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int nrank;
    MPI_Comm_size(MPI_COMM_WORLD, &nrank);

    BigFile file[1];

    big_file_mpi_create(file, "/tmp/example", MPI_COMM_WORLD);
    big_file_mpi_create_records(file, rtype, "w+", 2, (size_t[]){100, 100}, MPI_COMM_WORLD);

    int nmemb = 200 * (rank + 1) / nrank - 200 * (rank) / nrank;
    void * bufin = calloc(nmemb, rtype->itemsize);
    void * bufout = calloc(nmemb, rtype->itemsize);

    for(int i = 0; i < nmemb; i ++) {
        double pos[3] = {i, i * 10, i * 100};
        float vel[3] = {i + 1, i * 10 + 1, i * 100 + 1};
        float bhmass = i;
        big_record_set(rtype, bufout, i, 0, &pos);
        big_record_set(rtype, bufout, i, 1, &vel);
        big_record_set(rtype, bufout, i, 2, &bhmass);
    }

    big_file_mpi_write_records(file, rtype, 0, nmemb, bufout, nrank, MPI_COMM_WORLD);

    big_file_mpi_create_records(file, rtype, "a+", 1, (size_t[]){200}, MPI_COMM_WORLD);
    big_file_mpi_write_records(file, rtype, 200, nmemb, bufout, nrank, MPI_COMM_WORLD);

    /* verify */
    big_file_mpi_read_records(file, rtype, 0, nmemb, bufin, 1, MPI_COMM_WORLD);
    if(0 != memcmp(bufout, bufin, rtype->itemsize * nmemb)) {
        abort();
    }
    big_file_mpi_read_records(file, rtype, 200, nmemb, bufin, 1, MPI_COMM_WORLD);
    if(0 != memcmp(bufout, bufin, rtype->itemsize * nmemb)) {
        abort();
    }

    big_file_mpi_close(file, MPI_COMM_WORLD);
    big_record_type_clear(rtype);
    MPI_Finalize();
    return 0;
}
