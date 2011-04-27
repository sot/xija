import numpy as np
import xija

mdl = xija.ThermalModel()

node1 = mdl.add(xija.Node, 'tephin')
node2 = mdl.add(xija.Node, 'tcylaft6')
coup1 = mdl.add(xija.Coupling, 'tephin', 'tcylaft6')
node2.predict = False

mdl.make_parvals()
