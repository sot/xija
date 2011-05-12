"""
Replicate (mostly) the minusz TEPHIN node model.  (dPs set to zero though).
"""

import json
import xija

P_pitches=[45, 60, 90, 120, 145, 170]
minusz = json.load(open('nmass/minusz/pars_minusz.json'))

mdl = xija.ThermalModel(start='2010:001', stop='2010:090')
nodes = {}
pitch = mdl.add(xija.Pitch)
for msid in minusz:
    pars = minusz[msid]
    Ps = [pars['pf_{0:03d}'.format(p)] for p in P_pitches]
    nodes[msid] = mdl.add(xija.Node, msid)
    mdl.add(xija.SolarHeat, msid, pitch, P_pitches, Ps, ampl=pars['p_ampl'])
    mdl.add(xija.HeatSink, msid, T=pars['T_e'], tau=pars['tau_ext'])

for msid in minusz:
    pars = minusz[msid]
    coupled_nodes = [x for x in pars if x.startswith('tau_t')]
    for parname in coupled_nodes:
        mdl.add(xija.Coupling, msid, node2=parname[4:], tau=pars[parname])

mdl.make()
mdl.calc()

# 128 ms for 180 days prediction (250 ms/year)
# Matches results from fit_nmass qualitatively well (visually compared
# residual plots).
