from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
extensions = [
        Extension("pyxbigfile", ["pyxbigfile.pyx"],
            include_dirs = ["../src"])]

setup(
    ext_modules = cythonize(extensions)
)

