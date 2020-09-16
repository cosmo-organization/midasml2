from setuptools import Extension,setup
from Cython.Build import cythonize
import numpy as np
setup(
    ext_modules=cythonize(Extension(
	  name="midasml2",
	  sources=['pyx/midasml2.pyx','src/wrapper1.cpp','src/wrapper2.cpp','src/fit_sgl.cpp','src/ld_estim.cpp'],
	  language='c++',
	  extra_objects=["libs/blas_win64_MT.lib","libs/lapack_win64_MT.lib"]
	  )
	),
	language='c++',
	include_dirs=[np.get_include(),'include/'],
)