CC=mpiicc -g -O0
all: \
	bigfile-get-attr \
	bigfile-copy \
	bigfile-cat \
	bigfile-ls \
	$(NULL)

bigfile-get-attr: bigfile-get-attr.c ../src/bigfile.a
	$(CC) -o $@ $< ../src/bigfile.a -I../src
bigfile-copy: bigfile-copy.c ../src/bigfile.a
	$(CC) -o $@ $< ../src/bigfile.a -I../src
bigfile-cat: bigfile-cat.c ../src/bigfile.a
	$(CC) -o $@ $< ../src/bigfile.a -I../src
bigfile-ls: bigfile-ls.c ../src/bigfile.a
	$(CC) -o $@ $< ../src/bigfile.a -I../src
