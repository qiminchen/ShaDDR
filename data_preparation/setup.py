from distutils.core import setup
from Cython.Build import cythonize
#python setup.py build_ext --inplace
setup(name='cutils', ext_modules=cythonize("cutils.pyx"))
setup(name='get_point_cloud_cy', ext_modules=cythonize("get_point_cloud_cy.pyx"))
