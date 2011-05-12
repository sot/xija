"""
Replicate (mostly) the minusz TEPHIN node model.  (dPs set to zero though).

env PYTHONPATH=$PWD python minusz/minusz.py
"""

import xija
import json

P_pitches=[45, 60, 90, 120, 145, 170]
minusz = json.load(open('/proj/sot/ska/share/nmass/minusz/pars_minusz.json'))
sigmas = {'tephin': -10}

mdl = xija.ThermalModel(name='minusz')
nodes = {}
pitch = mdl.add(xija.Pitch)
for msid in minusz:
    pars = minusz[msid]
    Ps = [pars['pf_{0:03d}'.format(p)] for p in P_pitches]
    nodes[msid] = mdl.add(xija.Node, msid, sigma=sigmas.get(msid, -20))
    mdl.add(xija.SolarHeat, msid, pitch, P_pitches, Ps, ampl=pars['p_ampl'])
    mdl.add(xija.HeatSink, msid, T=pars['T_e'], tau=pars['tau_ext'])

for msid in minusz:
    pars = minusz[msid]
    coupled_nodes = [x for x in pars if x.startswith('tau_t')]
    for parname in coupled_nodes:
        mdl.add(xija.Coupling, msid, node2=parname[4:], tau=pars[parname])

mdl.make_parvals()
mdl.write('minusz/minusz.json')

# 128 ms for 180 days prediction (250 ms/year)
# Matches results from fit_nmass qualitatively well (visually compared
# residual plots).
