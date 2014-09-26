CC=mpiicc

all: bigfile.a bigfile-mpi.a

bigfile.o: bigfile.c bigfile.h
	$(CC) -o $@ -c bigfile.c
bigfile.a: bigfile.o
	ar r $@ bigfile.o
bigfile-mpi.o: bigfile-mpi.c bigfile-mpi.h
	$(CC) -o $@ -c bigfile-mpi.c
bigfile-mpi.a: bigfile.o bigfile-mpi.o
	ar r $@ bigfile.o bigfile-mpi.o
