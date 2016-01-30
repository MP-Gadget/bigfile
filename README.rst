bigfile
=======

A reproducible massively parallel IO library for large, hierarchical datasets.

:code:`bigfile` was originally developed for the BlueTides simulation 
on BlueWaters at NCSA. 

Build status
------------
.. image:: https://api.travis-ci.org/rainwoodman/bigfile.svg
    :alt: Build Status
    :target: https://travis-ci.org/rainwoodman/bigfile/

Description
-----------

A snapshot file of BlueTides can be 45TB in size; 
With the help of :code:`bigfile` it took 10 minutess 
to dump a snapshot from 81,000 MPI ranks, and 5mins to read one.

:code:`bigfile` provides a hierarchical structure of data columns via 
:code:`BigFile`, :code:`BigBlock` and :code:`BigData`. 

A :code:`BigBlock` block is striped into many physical files, represented by a directory tree on the Lustre files system. Because of this, the performance of bigfile is insulated from the configurations of the Lustre file system. 

A :code:`BigBlock` stores a two dimesional table of :code:`nmemb` columns and :code:`size` rows. Numerical type columns are supported.

Meta data (attributes) can be attached to a :code:`BigBlock`. Numerical attributes and string attributes are supported.

Type casting is performed on-the-fly if read/write operation requests a different data type than the file has stored.

Comparision with HDF5
---------------------

**Good**

- bigfile is simpler. The core library of bigfile consists of 2 source files, 2 header
  files, and 1 Makefile,  a total of less than 3000 lines of code, 
  easily maintained by one person or dropped into a project. 
  HDF5 is much more complicated.
- bigfile is reproducible. If the same data is written to the disk twice, the binary
  representation is guarenteed identicial. HDF5 keeps a time stamp.

- bigfile is comprehensible. The raw data on disk is stored as binary files
  that can be directly accessed by any application. The meta data (block 
  descriptions and attributes) is stored in plain text, and can be modified 
  with a text editor. HDF5 hides everything under the carpet. 

**Bad**

- bigfile is limited. The most typical usecase of bigfile is to store 
  large amount of precalculated data. The API favors a programming model 
  where data in memory is directly dumped to disk. There is no API for streaming.
  bigfile only support very simple data types, and composite data type 
  is simulated at the interface level. 
  HDF5 is much richer. 

API
---

Python, C and C/MPI are supported. Python/MPI is less efficient than we would like to be.

TODO
----

Document the API. At least the python API!

.. code:: python

    import bigfile

    f = bigfile.BigFile('PART_018')

    print (f.blocks)
    data = bigfile.BigData(f[0], ['Position', 'Velocity'])
    
    print (data.size)

    print data[10:30]

    
Install
-------

To install the Python binding

.. code:: bash

    pip install [--user] bigfile

to install the C API

.. code:: bash

    make install

Override CC MPICC, PREFIX as needed. Take a look at the Makefile is always recommended.

Shell
-----

We provide the following shell commands for file inspection:

- bigfile-cat
- bigfile-create
- bigfile-repartition
- bigfile-ls
- bigfile-get-attr
- bigfile-set-attr


 Yu Feng
