from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

extensions = [
        Extension("bigfile.pyxbigfile", ["python/pyxbigfile.pyx"],
            include_dirs = ["src/", numpy.get_include()])]

setup(
    name="bigfile", version="0.1.9",
    author="Yu Feng",
    author_email="rainwoodman@gmail.com",
    url="http://github.com/rainwoodman/bigfile",
    description="python binding of BigFile, a peta scale IO format",
    zip_safe = False,
    package_dir = {'bigfile': 'python'},
    install_requires=['cython', 'numpy'],
    packages= ['bigfile'],
    requires=['numpy'],
    ext_modules = cythonize(extensions)
)
