# Licensed under a 3-clause BSD style license - see LICENSE.rst
import xija
import numpy as np
import asciitable
from Chandra.Time import DateTime
from Ska.Matplotlib import plot_cxctime

pars = dict(acis150 =  28.029,
            acis50  =  54.192,
            acis90  =  26.975,
            c1      = 114.609,
            c2      =  11.362,
            hrci150 =  32.977,
            hrci50  =  38.543,
            hrci90  =  28.053,
            hrcs150 =  37.265,
            hrcs50  =  30.715,
            hrcs90  =  30.013,
            u01     =   6.036,
            u01quad =  -0.599,
            u12     =   8.451,
            )

u01 = pars['u01']
u12 = pars['u12']
c1 = pars['c1']
c2 = pars['c2']

P_pitches = [50, 90, 150]
P_vals = []
for instr in ('hrcs', 'hrci', 'acis'):
    for pitch in P_pitches:
        P_vals.append(pars['{0}{1}'.format(instr, pitch)])
P_vals = np.array(P_vals).reshape(3,3) * u01 / c1
P_vals = P_vals.tolist()

tau_e = c1 / u01
T_e = -128.0 * (1. / u01 + 1. / u12)
k = 1. / c2
tau12 = c1 / u12 
tau21 = c2 / u12 

mdl = xija.ThermalModel('psmc', start='2011:103:00:00:00.00', stop='2011:124:00:00:00')

pin1at = mdl.add(xija.Node, '1pin1at')
pdeaat = mdl.add(xija.Node, '1pdeaat')
pitch = mdl.add(xija.Pitch)
sim_z = mdl.add(xija.SimZ)

coup12 = mdl.add(xija.Coupling, pin1at, pdeaat, tau=tau12)
coup21 = mdl.add(xija.Coupling, pdeaat, pin1at, tau=tau21)
sol = mdl.add(xija.AcisPsmcSolarHeat, pin1at, pitch, sim_z, P_pitches=P_pitches, P_vals=P_vals)
heat = mdl.add(xija.HeatSink, pin1at, T=T_e, tau=tau_e)
# pow = mdl.add(xija.AcisPsmcPower, pdeaat, k=k)
fep_count = mdl.add(xija.CmdStatesData,
                    u'fep_count')
ccd_count = mdl.add(xija.CmdStatesData,
                    u'ccd_count')
vid_board = mdl.add(xija.CmdStatesData,
                    u'vid_board')
clocking = mdl.add(xija.CmdStatesData,
                   u'clocking')
pow = mdl.add(xija.AcisDpaStatePower, pdeaat, fep_count=fep_count,
              ccd_count=ccd_count, vid_board=vid_board, clocking=clocking)

mdl.make()
mdl.calc()
mdl.write('psmc_classic.json')

psmc = asciitable.read('models_dev/psmc/out_2010103_2010124/temperatures.dat')

figure(1)
clf()
plot_cxctime(pdeaat.times, pdeaat.mvals, 'b')
plot_cxctime(pdeaat.times, pdeaat.dvals, 'r')
plot_cxctime(psmc['time'], psmc['1pdeaat'], 'g')

figure(2)
plot_cxctime(pin1at.times, pin1at.mvals, 'b')
plot_cxctime(pin1at.times, pin1at.dvals, 'r')
plot_cxctime(psmc['time'], psmc['1pin1at'], 'g')

