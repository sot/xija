"""
Replicate (mostly) the minusz TEPHIN node model.  (dPs set to zero though).
"""

import json
import numpy as np
import xija

P_pitches=[45, 60, 90, 120, 145, 170]
minusz = json.load(open('nmass/minusz/pars_minusz.json'))

mdl = xija.ThermalModel(start='2010:001', stop='2010:090')

nodes = {}

aosares1 = mdl.add(xija.TelemData, 'aosares1')
for msid in minusz:
    pars = minusz[msid]
    Ps = np.array([pars['pf_{0:03d}'.format(pitch)] for pitch in P_pitches])
    ampl = pars['p_ampl']
    T_ext = pars['T_e']
    tau_ext = pars['tau_ext']
    nodes[msid] = mdl.add(xija.Node, msid)
    mdl.add(xija.SolarHeat, msid, aosares1, P_pitches=P_pitches,
            Ps=Ps, ampl=ampl)
    mdl.add(xija.HeatSink, msid, T=T_ext, tau=tau_ext)

for msid in minusz:
    pars = minusz[msid]
    coupled_nodes = [x for x in pars if x.startswith('tau_t')]
    for parname in coupled_nodes:
        msid2 = parname[4:]
        mdl.add(xija.Coupling, msid, msid2, tau=pars[parname])

mdl.make()
mdl.calc()

# 128 ms for 180 days prediction (250 ms/year)
# Matches results from fit_nmass qualitatively well (visually compared
# residual plots).