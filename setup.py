from distutils.core import setup, Extension
from Cython.Build import cythonize

import numpy

ext_modules=[
    Extension("npTest",
              ["npTest.pyx"],
              extra_compile_args=["/openmp" ],
              extra_link_args=['/openmp']
              )
]

setup(
  name="npTest",
  ext_modules=cythonize(ext_modules),
  include_dirs=[numpy.get_include()]
)





# from distutils.core import setup, Extension
# from Cython.Build import cythonize

# import numpy



# setup(
#     ext_modules=cythonize("npTest.pyx"),
#     include_dirs=[numpy.get_include()]
# )