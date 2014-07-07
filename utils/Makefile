CC=mpiicc -g -O0
all: \
	bigfile-get-attr \
	bigfile-copy \
	bigfile-copy-mpi \
	bigfile-cat \
	bigfile-ls \
	$(NULL)

bigfile-get-attr: bigfile-get-attr.c ../src/bigfile.a
	$(CC) -o $@ $< ../src/bigfile.a -I../src
bigfile-copy: bigfile-copy.c ../src/bigfile.a
	$(CC) -o $@ $< ../src/bigfile.a -I../src
bigfile-copy-mpi: bigfile-copy-mpi.c ../src/bigfile.a ../src/bigfile-mpi.a
	$(CC) -o $@ $< ../src/bigfile.a ../src/bigfile-mpi.a -I../src
bigfile-cat: bigfile-cat.c ../src/bigfile.a
	$(CC) -o $@ $< ../src/bigfile.a -I../src
bigfile-ls: bigfile-ls.c ../src/bigfile.a
	$(CC) -o $@ $< ../src/bigfile.a -I../src
