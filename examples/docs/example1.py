# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Example with a single node (ACA CCD temperature) with solar heating
(2 bins).
"""

import xija

name = __file__[:-3]

model = xija.XijaModel(name, start='2015:001', stop='2015:050')

model.add(xija.Node, 'aacccdpt')

model.add(xija.Pitch)

model.add(xija.Eclipse)

model.add(xija.SolarHeat,
          node='aacccdpt',
          pitch_comp='pitch',
          eclipse_comp='eclipse',
          P_pitches=[45, 180],
          Ps=[0.0, 0.0],
          ampl=0.0,
          epoch='2010:001',
         )

model.write('{}.json'.format(name))
