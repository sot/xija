# Build the core.so module and put into the source directory.
# python setup.py build_ext --inplace

import xija

model = xija.ThermalModel('test', start='2011:001', stop='2011:005')
tephin = model.add(xija.Node, 'tephin')
model.add(xija.HeatSink, tephin, T=0.0, tau=200.0)
tephin.set_data(30.0)

model.make()
model.calc()

# plot(model.times - model.times[0], model.comp['tephin'].mvals)

mvals = model.comp['tephin'].mvals
assert len(mvals) == 1051
assert mvals[0] == 30.0
assert abs(mvals[500] - 8.75402) < 0.001
assert abs(mvals[1050] - 2.26212) < 0.001
