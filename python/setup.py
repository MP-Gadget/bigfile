from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
extensions = [
        Extension("bigfilepy.pyxbigfile", ["src/pyxbigfile.pyx"],
            include_dirs = ["../src"])]

setup(
    name="bigfilepy", version="0.1",
    author="Yu Feng",
    description="python binding of BigFile, a peta scale IO format",
    package_dir = {'bigfilepy': 'src'},
    install_requires=['cython', 'numpy'],
    packages= ['bigfilepy'],
    requires=['numpy'],
    ext_modules = cythonize(extensions)
)

