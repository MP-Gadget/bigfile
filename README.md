bigfile
=======

scalable large data io for peta scale apps on bluewaters.

These Bigfiles are really big. 

Developed for the BlueTides simulation on BlueWaters at NCSA. 


It works: a snapshot file of BlueTides can be 45TB in size; 
we spend 10mins to dump a snapshot, and 5mins to read one.

Physically, a file is spread into many files, represented by a directory tree on the
Lustre files system.

Logically, a file consists of many blocks. A block is a two dimension table of a given data
type, of 'nmemb' columns and 'size' rows. Attributes can be attached to a block. Read/Write
operation automatically cast the buffer to requrested datatype. 

There is a python API;
There is a C API;
There is a C MPI API.

There are also tools to inspect these files:

bigfile-cat
bigfile-repartition
bigfile-ls
bigfile-get-attr

We originally plan
to use HDF5, but it does not integrate with our simulation software very well.
HDF5 does not provide a unified interface to access data spread into many files.


- Yu Feng
