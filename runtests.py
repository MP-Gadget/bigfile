import sys
import os

from numpy.testing import Tester

# need an install to run these tests

from sys import argv

tester = Tester()
r = tester.test(extra_argv=['-w', 'tests'] + argv[1:])
if r.failures or r.errors:
    raise Exception("Test Failed")
