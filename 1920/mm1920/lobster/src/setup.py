import os
import sys
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

ext_modules=[
        Extension("*",
            ["*.pyx"],
            libraries=["m"],
            )
]
setup(
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()])

