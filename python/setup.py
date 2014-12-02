from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy
extensions = [
        Extension("bigfile.pyxbigfile", ["src/pyxbigfile.pyx"],
            include_dirs = ["../src", numpy.get_include()])]

setup(
    name="bigfile", version="0.1",
    author="Yu Feng",
    description="python binding of BigFile, a peta scale IO format",
    package_dir = {'bigfile': 'src'},
    install_requires=['cython', 'numpy'],
    packages= ['bigfile'],
    requires=['numpy'],
    ext_modules = cythonize(extensions)
)

