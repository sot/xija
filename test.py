import numpy as np
import xija

mdl = xija.ThermalModel(start='2010:180', stop='2011:001')

node1 = mdl.add(xija.Node, 'tephin')
node2 = mdl.add(xija.Node, 'tcylaft6', predict=False)
coup12 = mdl.add(xija.Coupling, 'tephin', 'tcylaft6', tau=20)
#coup21 = mdl.add(xija.Coupling, 'tcylaft6', 'tephin', tau=40)
#pitch = mdl.add(xija.TelemData, 'aosares1')

#tephin_solar = mdl.add(xija.SolarHeat, 'tephin', pitch,
#                       Ps=[0.1, 0.5, 1.0, 1.5, 2.0],
#                       dPs=[0.01, 0.02, 0.03, 0.04, 0.05])

mdl.make()

mdl.calc()
