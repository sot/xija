# Licensed under a 3-clause BSD style license - see LICENSE.rst
from setuptools import setup, Extension
import os
from xija import __version__

long_description = """
Xija (pronounced "kiy - yuh", rhymes with Maya) is the thermal modeling
framework used Chandra thermal modeling:

* Modular and extensible modeling framework
* Single-step integration instead of analytic state-based solutions
* Model definition via Python code or static data structure
* Interactive and iterative model development and fitting
* Predictively model a node or use telemetry during development
* GUI interface for model development
* Matlab interface
"""

if os.name == "nt":
    link_args = ['/EXPORT:calc_model']
else:
    link_args = []

try:
    from testr.setup_helper import cmdclass
except ImportError:
    cmdclass = {}

core6_ext = Extension('xija.core', ['xija/core.c'],
                      extra_link_args=link_args)

entry_points = {'console_scripts': 'xija_gui_fit = xija.gui_fit:main'}

setup(name='xija',
      version=__version__,
      description='Thermal modeling framework for Chandra',
      long_description=long_description,
      author='Tom Aldcroft',
      author_email='taldcroft@cfa.harvard.edu',
      url='https://github.com/sot/xija',
      license='BSD',
      zip_safe=False,
      platforms=['any'],
      ext_modules=[core6_ext],
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Astronomy',
          'Topic :: Scientific/Engineering :: Physics',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 3',
          ],
      packages=['xija', 'xija.component', 'xija.tests'],
      package_data={'xija': ['libcore.so',
                             'component/earth_vis_grid_nside32.fits.gz'],
                    'xija.tests': ['*.npz', '*.json']},
      tests_require=['pytest'],
      cmdclass=cmdclass,
      entry_points=entry_points,
      )
