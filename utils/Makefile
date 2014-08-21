CC=mpiicc -g -O0
all: \
	bigfile-get-attr \
	bigfile-set-attr \
	bigfile-copy \
	bigfile-copy-mpi \
	bigfile-sample-mpi \
	bigfile-cat \
	bigfile-create \
	bigfile-ls \
	$(NULL)

bigfile-get-attr: bigfile-get-attr.c ../src/bigfile.a
	$(CC) -o $@ $< ../src/bigfile.a -I../src
bigfile-set-attr: bigfile-set-attr.c ../src/bigfile.a
	$(CC) -o $@ $< ../src/bigfile.a -I../src
bigfile-copy: bigfile-copy.c ../src/bigfile.a
	$(CC) -o $@ $< ../src/bigfile.a -I../src
bigfile-copy-mpi: bigfile-copy-mpi.c ../src/bigfile.a ../src/bigfile-mpi.a
	$(CC) -o $@ $< ../src/bigfile.a ../src/bigfile-mpi.a -I../src
bigfile-sample-mpi: bigfile-sample-mpi.c ../src/bigfile.a ../src/bigfile-mpi.a
	$(CC) -o $@ $< ../src/bigfile.a ../src/bigfile-mpi.a -I../src -lgsl -lgslcblas -lm
bigfile-cat: bigfile-cat.c ../src/bigfile.a
	$(CC) -o $@ $< ../src/bigfile.a -I../src
bigfile-create: bigfile-create.c ../src/bigfile.a
	$(CC) -o $@ $< ../src/bigfile.a -I../src
bigfile-ls: bigfile-ls.c ../src/bigfile.a
	$(CC) -o $@ $< ../src/bigfile.a -I../src
