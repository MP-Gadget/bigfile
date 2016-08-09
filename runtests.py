from mpi4py_test import MPITester
import sys
import os.path

tester = MPITester(os.path.abspath(__file__), "bigfile")

tester.main(sys.argv)
