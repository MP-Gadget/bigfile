CC=mpiicc

all: libbigfile.a libbigfile-mpi.a

bigfile.o: bigfile.c bigfile.h
	$(CC) -o $@ -c bigfile.c
libbigfile.a: bigfile.o
	ar r $@ bigfile.o
bigfile-mpi.o: bigfile-mpi.c bigfile-mpi.h
	$(CC) -o $@ -c bigfile-mpi.c
libbigfile-mpi.a: bigfile.o bigfile-mpi.o
	ar r $@ bigfile.o bigfile-mpi.o
clean: 
	rm *.a *.o
