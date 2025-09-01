from setuptools import setup, Extension
import numpy

# Define the C extension module
clinalg_module = Extension(
    'clinalg',
    sources=[
        'src/c_linalg/python_lm.c',
        'src/c_linalg/matrix.c'
    ],
    include_dirs=[
        'src/c_linalg',
        numpy.get_include()
    ],
    extra_compile_args=['-O3', '-std=c99'],
    extra_link_args=['-lm']
)

setup(
    name='clinalg',
    version='0.1.0',
    description='C-based linear algebra for high-performance regression',
    ext_modules=[clinalg_module],
    zip_safe=False,
)