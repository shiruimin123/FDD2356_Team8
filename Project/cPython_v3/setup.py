from Cython.Build import cythonize
import numpy as np

from setuptools import Extension, setup


ext_modules = [
    Extension(
        "cythonfn",
        ["cythonfn.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    ext_modules=cythonize(ext_modules),
    include_dirs=[np.get_include()]
)

