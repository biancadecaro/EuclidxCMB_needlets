from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as nm
import os
import subprocess as sbp
import os.path as osp

# Recover the gcc compiler
GCCPATH_STRING = sbp.Popen(
    ['gcc', '-print-libgcc-file-name'],
    stdout=sbp.PIPE).communicate()[0]
GCCPATH = osp.normpath(osp.dirname(GCCPATH_STRING))

HOME = os.environ['HOME']

sources = ["cython_mylibc.pyx"]#, "../source/mylibc.c"]
setup(name="cython_mylibc",
      cmdclass={"build_ext": build_ext},
      ext_modules=[Extension("cython_mylibc", sources,
                             include_dirs=[nm.get_include(), "../include", '/usr/include','/usr/local/include','/usr/include/x86_64-linux-gnu/'],# HOME+"/fftw3/fftw-3.3.10-double/include", HOME+"/gsl/gsl-2.7.1/include"],
                             libraries=["mylibc","gsl","gslcblas","fftw3_omp","fftw3"],
                             library_dirs=["../", GCCPATH.decode("utf-8"),'/usr/local/lib', '/usr/lib/x86_64-linux-gnu'],# HOME+"/fftw3/fftw-3.3.10-double/lib", HOME+"/gsl/gsl-2.7.1/lib"],
                             extra_link_args=["-lgomp","-lm"],
                            )],
     )
