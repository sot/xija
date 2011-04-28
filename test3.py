"""
Replicate (mostly) the minusz TEPHIN node model.  (dPs set to zero though).
"""

import numpy as np
import xija

mdl = xija.ThermalModel(start='2010:001', stop='2010:090')

tephin = mdl.add(xija.Node, 'tephin')
tcylaft6 = mdl.add(xija.Node, 'tcylaft6', predict=False)
tmzp_my = mdl.add(xija.Node, 'tmzp_my', predict=False)
coup__tephin__tcylaft6 = mdl.add(xija.Coupling, tephin, tcylaft6, tau=130.26)
coup__tephin__tmzp_my = mdl.add(xija.Coupling, tephin, tmzp_my, tau=105.91)
aosares1 = mdl.add(xija.TelemData, 'aosares1')
tephin_solar = mdl.add(xija.SolarHeat, tephin, aosares1,
                       P_pitches=[45, 60, 90, 120, 145, 170],
                       Ps=np.array([0.970, 1.42, 1.91, 1.92, 1.42, 0.69]),
                       ampl=0.0679)

tephin_heatsink = mdl.add(xija.HeatSink, tephin, T=0.0, tau=38.0)

pars_minusz = """
    "tephin": {
        "T_e": 0.0, 
        "p_ampl": 0.067919468805023628, 
        "pf_045": 0.96976110603706989, 
        "pf_060": 1.4221924089192244, 
        "pf_090": 1.9133045643725675, 
        "pf_120": 1.9183228014942686, 
        "pf_145": 1.4231235450884276, 
        "pf_170": 0.69415616876670061, 
        "pi_045": 0.74874818310176283, 
        "pi_060": 1.1062471631627799, 
        "pi_090": 1.4352599428318134, 
        "pi_120": 1.3754336146212318, 
        "pi_145": 0.92452578887819126, 
        "pi_170": 0.18535597796667336, 
        "tau_ext": 38.509613333455292, 
        "tau_sc": 1734.2973777080965, 
        "tau_tcylaft6": 130.26749285187424, 
        "tau_tmzp_my": 105.91197886711586
    }, 
"""

mdl.make()
mdl.calc()

# Plain core.pyx with no cython optimization. 1 loops, best of 3: 1.26 s per loop
