CC=mpiicc
PREFIX=/usr
all: libbigfile.a libbigfile-mpi.a

install: libbigfile.a libbigfile-mpi.a
	install libbigfile.a $(PREFIX)/lib/libbigfile.a
	install libbigfile-mpi.a $(PREFIX)/lib/libbigfile-mpi.a
	install bigfile-mpi.h $(PREFIX)/include/bigfile-mpi.h
	install bigfile.h $(PREFIX)/include/bigfile-mpi.h

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
