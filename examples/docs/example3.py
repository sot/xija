# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Example with a single node (ACA CCD temperature) with solar heating
(2 bins) and a heat sink.
"""
import xija

name = __file__[:-3]
model = xija.XijaModel(name, start='2015:001', stop='2015:050')

model.add(xija.Node,
          'aacccdpt',
         )

model.add(xija.Pitch,
         )
model.add(xija.Eclipse,
         )
model.add(xija.SolarHeat,
          u'aacccdpt',
          u'pitch',
          u'eclipse',
          [45, 70, 90, 115, 140, 180],
          [0.0] * 6,
          ampl=0.0,
          epoch='2010:001',
         )

model.add(xija.HeatSink,
          u'aacccdpt',
          tau=30.0,
          T=-16.0,
         )

model.write('{}.json'.format(name))
