bigfile
=======

A Scalable BigData file format for PetaScale Applications

bigfile is originally developed for the BlueTides simulation 
on BlueWaters at NCSA. 

.. image:: https://api.travis-ci.org/rainwoodman/bigfile.svg
    :alt: Build Status
    :target: https://travis-ci.org/rainwoodman/bigfile/

These Bigfiles are really big.  
A snapshot file of BlueTides can be 45TB in size; 
With the help of :code:`bigfile` it took 10 minutess 
to dump a snapshot from 81000 MPI ranks, and 5mins to read one.

:code:`bigfile` provides a hierarchical structure of data columns (:code:`BigBlock`) and a uniform, continuous view of data in a column.

A :code:`BigBlock` block striped into many physical files, represented by a directory tree on the Lustre files system. Because of this, there is no need to tweak the striping of Lustre files.

A :code:`BigBlock` contains :code:`nmemb` columns and :code:`size` rows. 
Simple, numerical attributes can be attached to a block; 

Type casting is performed on-the-fly if read/write operation requests a different data type than the file.

bigfile feels similar with HDF5, but because bigfile has 
a much smaller set of functions, the source code is much simpler and the
API is cleaner.

API
---

Python, C and C/MPI are supported. The MPI support for Python does not support writing.

TODO
----

Document the API. At least the python API!

.. code:: python

    import bigfile

    f = bigfile.BigFile('PART_018')

    print (f.blocks)

    block = f['0/Position']
    
    print (block.size)

    print block[10:30]

    
Install
-------

To install the Python API

.. code:: bash

    python setup.py install --user

to install the C API

.. code:: bash

    make install

Override CC MPICC, PREFIX as needed.


Shell
-----

We provide the following shell commands for file inspection:

- bigfile-cat
- bigfile-repartition
- bigfile-ls
- bigfile-get-attr

 Yu Feng
