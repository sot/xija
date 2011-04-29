from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension("core", ["core4.pyx"], include_dirs=[numpy.get_include()])]

setup(
  name = 'Xija core calc_model',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
