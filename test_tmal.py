import json
import xija


model_spec = json.load(open('minusz/minusz.json', 'r'))
mdl = xija.ThermalModel('minusz', start='2011:001', stop='2011:090', model_spec=model_spec)
mdl.make()
mdl.calc()
