import sys
import xija

model = xija.XijaModel('4hfspat', start='2017:301:00:01:50.816', stop='2021:039:23:53:18.816', dt=328.0,
evolve_method=1, rk4=0)

model.add(xija.Node,
          '4hfspat',
         )
model.add(xija.Delay,
          '4hfspat',
          delay=0,
         )
model.add(xija.Pitch,
         )
model.add(xija.Eclipse,
         )
model.add(xija.SolarHeat,
          P_pitches=[45, 70, 90, 115, 140, 160, 180],
          Ps=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          ampl=0.0,
          eclipse_comp='eclipse',
          epoch='2021:001',
          node='4hfspat',
          pitch_comp='pitch',
         )
model.add(xija.HeatSink,
          T=-16.0,
          node='4hfspat',
          tau=30.0,
         )
model.add(xija.StepFunctionPower,
          P=0.07,
          id='_2iru',
          node='4hfspat',
          time='2018:283:14:00:00',
         )
model.add(xija.StepFunctionPower,
          P=-0.07,
          id='_1iru',
          node='4hfspat',
          time='2020:213:04:25:12',
         )
# Set delay__4hfspat component parameters
comp = model.get_comp('delay__4hfspat')

par = comp.get_par('delay')
par.update(dict(val=13.499999999999996, min=-40, max=40, fmt='{:.4g}', frozen=True))

# Set solarheat__4hfspat component parameters
comp = model.get_comp('solarheat__4hfspat')

par = comp.get_par('P_45')
par.update(dict(val=0.17233635027684308, min=-10.0, max=10.0, fmt='{:.4g}', frozen=True))

par = comp.get_par('P_70')
par.update(dict(val=0.23062326409902983, min=-10.0, max=10.0, fmt='{:.4g}', frozen=True))

par = comp.get_par('P_90')
par.update(dict(val=0.24228417253874154, min=-10.0, max=10.0, fmt='{:.4g}', frozen=True))

par = comp.get_par('P_115')
par.update(dict(val=0.23567973954542965, min=-10.0, max=10.0, fmt='{:.4g}', frozen=True))

par = comp.get_par('P_140')
par.update(dict(val=0.20035097429726695, min=-10.0, max=10.0, fmt='{:.4g}', frozen=True))

par = comp.get_par('P_180')
par.update(dict(val=0.09145374642612553, min=-10.0, max=10.0, fmt='{:.4g}', frozen=True))

par = comp.get_par('dP_45')
par.update(dict(val=0.020381767596966242, min=-1.0, max=1.0, fmt='{:.4g}', frozen=False))

par = comp.get_par('dP_70')
par.update(dict(val=0.014634511459946929, min=-1.0, max=1.0, fmt='{:.4g}', frozen=False))

par = comp.get_par('dP_90')
par.update(dict(val=0.020901514422723914, min=-1.0, max=1.0, fmt='{:.4g}', frozen=False))

par = comp.get_par('dP_115')
par.update(dict(val=0.01900222576906281, min=-1.0, max=1.0, fmt='{:.4g}', frozen=False))

par = comp.get_par('dP_140')
par.update(dict(val=0.01961773304332897, min=-1.0, max=1.0, fmt='{:.4g}', frozen=False))

par = comp.get_par('dP_180')
par.update(dict(val=0.025397352169416534, min=-1.0, max=1.0, fmt='{:.4g}', frozen=False))

par = comp.get_par('tau')
par.update(dict(val=1732.0, min=1000.0, max=3000.0, fmt='{:.4g}', frozen=True))

par = comp.get_par('ampl')
par.update(dict(val=0.00439453125, min=-1.0, max=1.0, fmt='{:.4g}', frozen=True))

par = comp.get_par('bias')
par.update(dict(val=0.0, min=-1.0, max=1.0, fmt='{:.4g}', frozen=True))

# Set heatsink__4hfspat component parameters
comp = model.get_comp('heatsink__4hfspat')

par = comp.get_par('T')
par.update(dict(val=-17.38862456311714, min=-100.0, max=100.0, fmt='{:.4g}', frozen=True))

par = comp.get_par('tau')
par.update(dict(val=224.16634713698585, min=2.0, max=300.0, fmt='{:.4g}', frozen=True))

# Set step_power_2iru__4hfspat component parameters
comp = model.get_comp('step_power_2iru__4hfspat')

par = comp.get_par('P')
par.update(dict(val=0.011488201458560068, min=-10.0, max=10.0, fmt='{:.4g}', frozen=True))

# Set step_power_1iru__4hfspat component parameters
comp = model.get_comp('step_power_1iru__4hfspat')

par = comp.get_par('P')
par.update(dict(val=-0.010011660309755826, min=-10.0, max=10.0, fmt='{:.4g}', frozen=True))

model.bad_times = []
if len(sys.argv) > 1:
    model.write(sys.argv[1])
