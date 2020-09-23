from setuptools import Extension,setup
from Cython.Build import cythonize
import numpy as np
import os
print("If you don't have lapack and blas then need to install lapack and blas")
print("If not builded failed then use command to install some stuff")
print("To install lapack use command 'brew install lapack'")
print("To install openblas use command 'brew install openblas'")
os.system("apt-get install libblas-dev liblapack-dev")
setup(
    ext_modules=cythonize(Extension(
	  name="midasml2",
	  sources=[
              'pyx/midasml2.pyx',
              'src/wrapper1.cpp',
              'src/wrapper2.cpp',
              'src/fit_sgl.cpp',
              'src/ld_estim.cpp'
              ],
          language='c++',
	  extra_compile_args=['-w','-DARMA_DONT_USE_WRAPPER'],
          extra_link_args=['-framework Accelerate']
	  )
	),
	language='c++',
	include_dirs=[np.get_include(),'include/'],
)
