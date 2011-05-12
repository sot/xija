import os
import json
import numpy as np
import xija

mdl = xija.ThermalModel(start='2010:001', stop='2010:004')

tephin = mdl.add(xija.Node, 'tephin')
tcylaft6 = mdl.add(xija.Node, 'tcylaft6', predict=False)
coup_tephin_tcylaft6 = mdl.add(xija.Coupling, tephin, tcylaft6, tau=20)
aosares1 = mdl.add(xija.TelemData, 'aosares1')

tephin_solar = mdl.add(xija.SolarHeat, tephin, aosares1,
                       Ps=[0.1, 0.5, 1.0, 1.5, 2.0],
                       dPs=[0.01, 0.02, 0.03, 0.04, 0.05])

tephin_heatsink = mdl.add(xija.HeatSink, tephin, T=0.0, tau=20.0)

mdl.make()
mdl.write('test_write.json')

model_spec = json.load(open('test_write.json'))
mdl2 = xija.ThermalModel(start='2010:001', stop='2010:004', model_spec=model_spec)

os.unlink('test_write.json')
