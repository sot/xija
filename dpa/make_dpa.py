import xija

pars =  dict(
                 acis105 =  35.433,
                 acis130 =  34.147,
                 acis150 =  44.147,
                 acis172 =  54.538,
                 acis45  =  23.801,
                 acis60  =  23.801,
                 acis90  =  24.672,
                 c1      =  39.887,
                 c2      =   3.290,
                 u01     =   5.449,
                 u12     =  22.517,
                 u01quad =  -0.599,
                )

u01 = pars['u01']
u12 = pars['u12']
c1 = pars['c1']
c2 = pars['c2']

P_pitches = [45, 60, 90, 105, 130, 150, 172]
P_vals = []
for instr in ('acis',):
    for pitch in P_pitches:
        P_vals.append(pars['{0}{1}'.format(instr, pitch)])
P_vals = np.array(P_vals) * u01 / c1

tau_e = c1 / u01
T_e = -128.0 * (1. / u01 + 1. / u12)
k = 1. / c2
tau12 = c1 / u12 
tau21 = c2 / u12 


mdl = xija.ThermalModel('dpa', start='2011:001', stop='2011:010')

node1 = mdl.add(xija.Node, '1dpamzt')
# node2 = mdl.add(xija.Node, '1dpamyt')
pitch = mdl.add(xija.Pitch)
eclipse = mdl.add(xija.Eclipse)
simz = mdl.add(xija.SimZ)

tau12 = 5
tau21 = 5
k = 0.14
T_e = 5.2
tau_e = 17.2
P_vals = np.array([0.58, 0.50, 0.41, 0.7, 1.0, 0.9, 0.79])
# mdl.add(xija.Coupling, node1, node2, tau=tau12)
# mdl.add(xija.Coupling, node2, node1, tau=tau21)
mdl.add(xija.SolarHeat, node1, pitch, eclipse, P_pitches=P_pitches, Ps=P_vals.tolist())
mdl.add(xija.HeatSink, node1, T=T_e, tau=tau_e)
mdl.add(xija.AcisDpaPower6, node1, k=k)

mdl.make()
mdl.write('dpa/dpa.json')


out = {'time': model.times,
       'data': model.comp['1dpamzt'].dvals,
       'model': model.comp['1dpamzt'].mvals,
       'pitch': model.comp['pitch'].dvals,
       'simz': model.comp['sim_z'].dvals,
       'power': model.comp['dpa__1dpamzt'].dvals,
       }
