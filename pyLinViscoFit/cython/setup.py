from distutils.core import setup, Extension
from Cython.Build import cythonize

import numpy

ext_modules=[
    Extension("cE_relax_norm",
              ["cE_relax_norm.pyx"],
              extra_compile_args=["/openmp" ],
              extra_link_args=['/openmp']
              )
]

setup(
  name="cE_relax_norm",
  ext_modules=cythonize(ext_modules),
  include_dirs=[numpy.get_include()]
)