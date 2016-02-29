import sys
import xija

model = xija.XijaModel(u'example4', start='2014:205:12:10:16.816', stop='2015:240:11:43:44.816', dt=328.0)

model.add(xija.Node,
          u'aacccdpt',
         )
model.add(xija.Pitch,
         )
model.add(xija.Eclipse,
         )

model.add(xija.Node,
          u'aca0',
          sigma=100000.0,
         )

model.add(xija.SolarHeat,
          u'aca0',
          u'pitch',
          u'eclipse',
          [45, 70, 90, 115, 140, 180],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          epoch=u'2015:040',
          ampl=0.0,
         )
model.add(xija.HeatSink,
          u'aca0',
          tau=30.0,
          T=-16.0,
         )

model.add(xija.Coupling,
          u'aacccdpt',
          u'aca0',
          tau=100.0,
         )

# Set solarheat__aca0 component parameters
comp = model.get_comp(u'solarheat__aca0')

par = comp.get_par(u'P_45')
par.update(dict(val=-0.00819254491543108, min=-10.0, max=10.0, fmt=u'{:.4g}', frozen=False))

par = comp.get_par(u'P_70')
par.update(dict(val=0.020523123184381896, min=-10.0, max=10.0, fmt=u'{:.4g}', frozen=False))

par = comp.get_par(u'P_90')
par.update(dict(val=0.021196139925064944, min=-10.0, max=10.0, fmt=u'{:.4g}', frozen=False))

par = comp.get_par(u'P_115')
par.update(dict(val=0.0191860284448866, min=-10.0, max=10.0, fmt=u'{:.4g}', frozen=False))

par = comp.get_par(u'P_140')
par.update(dict(val=0.005715691197120504, min=-10.0, max=10.0, fmt=u'{:.4g}', frozen=False))

par = comp.get_par(u'P_180')
par.update(dict(val=-0.05235661106194388, min=-10.0, max=10.0, fmt=u'{:.4g}', frozen=False))

par = comp.get_par(u'dP_45')
par.update(dict(val=0.0, min=-1.0, max=1.0, fmt=u'{:.4g}', frozen=True))

par = comp.get_par(u'dP_70')
par.update(dict(val=0.0, min=-1.0, max=1.0, fmt=u'{:.4g}', frozen=True))

par = comp.get_par(u'dP_90')
par.update(dict(val=0.0, min=-1.0, max=1.0, fmt=u'{:.4g}', frozen=True))

par = comp.get_par(u'dP_115')
par.update(dict(val=0.0, min=-1.0, max=1.0, fmt=u'{:.4g}', frozen=True))

par = comp.get_par(u'dP_140')
par.update(dict(val=0.0, min=-1.0, max=1.0, fmt=u'{:.4g}', frozen=True))

par = comp.get_par(u'dP_180')
par.update(dict(val=0.0, min=-1.0, max=1.0, fmt=u'{:.4g}', frozen=True))

par = comp.get_par(u'tau')
par.update(dict(val=1732.0, min=1000.0, max=3000.0, fmt=u'{:.4g}', frozen=True))

par = comp.get_par(u'ampl')
par.update(dict(val=0.0, min=-1.0, max=1.0, fmt=u'{:.4g}', frozen=True))

par = comp.get_par(u'bias')
par.update(dict(val=0.0, min=-1.0, max=1.0, fmt=u'{:.4g}', frozen=True))

# Set heatsink__aacccdpt component parameters
comp = model.get_comp(u'heatsink__aca0')

par = comp.get_par(u'T')
par.update(dict(val=-16.0, min=-100.0, max=100.0, fmt=u'{:.4g}', frozen=True))

par = comp.get_par(u'tau')
par.update(dict(val=176.65128930330445, min=2.0, max=200.0, fmt=u'{:.4g}', frozen=False))

if len(sys.argv) > 1:
    model.write(sys.argv[1])
