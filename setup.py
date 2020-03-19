# Licensed under a 3-clause BSD style license - see LICENSE.rst
from setuptools import setup, Extension
import os

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
    core1_ext = Extension('xija.core_1', ['xija/core_1.c'],
                      extra_link_args=['/EXPORT:calc_model_1'])
    core2_ext = Extension('xija.core_2', ['xija/core_2.c'],
                      extra_link_args=['/EXPORT:calc_model_2'])
else:
    core1_ext = Extension('xija.core_1', ['xija/core_1.c'])
    core2_ext = Extension('xija.core_2', ['xija/core_2.c'])

try:
    from testr.setup_helper import cmdclass
except ImportError:
    cmdclass = {}

entry_points = {'console_scripts': 'xija_gui_fit = xija.gui_fit.app:main'}

setup(name='xija',
      use_scm_version=True,
      setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
      description='Thermal modeling framework for Chandra',
      long_description=long_description,
      author='Tom Aldcroft',
      author_email='taldcroft@cfa.harvard.edu',
      url='https://github.com/sot/xija',
      license='BSD',
      zip_safe=False,
      platforms=['any'],
      ext_modules=[core1_ext, core2_ext],
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
      packages=['xija', 'xija.component', 'xija.tests', 'xija.gui_fit'],
      package_data={'xija': ['libcore.so',
                             'component/earth_vis_grid_nside32.fits.gz'],
                    'xija.tests': ['*.npz', '*.json'],
                    'xija.gui_fit': ['app_icon.png']},
      tests_require=['pytest'],
      cmdclass=cmdclass,
      entry_points=entry_points,
      )
