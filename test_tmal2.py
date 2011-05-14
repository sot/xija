import json
import xija


mdl = xija.ThermalModel(name='minusz', start='2011:001', stop='2011:002')
tephin = mdl.add(xija.Node, 'tephin')
tcylaft6 = mdl.add(xija.Node, 'tcylaft6')
coup12 = mdl.add(xija.Coupling, tephin, tcylaft6, tau=123.45)
coup21 = mdl.add(xija.Coupling, tcylaft6, tephin, tau=56.78)
pitch = mdl.add(xija.Pitch)
eclipse = mdl.add(xija.Eclipse)
tephin_solar = mdl.add(xija.SolarHeat, tephin, pitch, eclipse,
                       P_pitches=[45, 60, 90, 120, 145, 170],
                       Ps=[0.789, 0.789, 0.789, 0.789, 0.789, 0.789],
                       ampl=0.0)

tephin_heatsink = mdl.add(xija.HeatSink, tephin, T=-12.3, tau=34.5)

mdl.make()
mdl.calc()
