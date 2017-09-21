# Licensed under a 3-clause BSD style license - see LICENSE.rst
import xija
import numpy as np
import matplotlib.pyplot as plt
from Ska.Matplotlib import pointpair

start = '2010:001'
stop = '2011:345'

msid = '1dpamzt'
model_spec = 'dpa.json'

model = xija.ThermalModel('dpa', start=start, stop=stop,
                          model_spec=model_spec)
model.make()
model.calc()

dpa = model.get_comp(msid)
resid = dpa.dvals - dpa.mvals

xscatter = np.random.uniform(-0.2, 0.2, size=len(dpa.dvals))
yscatter = np.random.uniform(-0.2, 0.2, size=len(dpa.dvals))
plt.clf()
plt.plot(dpa.dvals + xscatter, resid + yscatter, '.', ms=1.0, alpha=1)
plt.xlabel('{} telemetry (degC)'.format(msid.upper()))
plt.ylabel('Data - Model (degC)')
plt.title('Residual vs. Data ({} - {})'.format(start, stop))

bins = np.arange(6, 26.1, 2.0)
r1 = []
r99 = []
ns = []
xs = []
for x0, x1 in zip(bins[:-1], bins[1:]):
    ok = (dpa.dvals >= x0) & (dpa.dvals < x1)
    val1, val99 = np.percentile(resid[ok], [1, 99])
    xs.append((x0 + x1) / 2)
    r1.append(val1)
    r99.append(val99)
    ns.append(sum(ok))

xspp = pointpair(bins[:-1], bins[1:])
r1pp = pointpair(r1)
r99pp = pointpair(r99)

plt.plot(xspp, r1pp, '-r')
plt.plot(xspp, r99pp, '-r', label='1% and 99% limits')
plt.grid()
plt.ylim(-8, 14)
plt.xlim(5, 31)

plt.plot([5, 31], [3.5, 3.5], 'g--', alpha=1, label='+/- 3.5 degC')
plt.plot([5, 31], [-3.5, -3.5], 'g--', alpha=1)
for x, n, y in zip(xs, ns, r99):
    plt.text(x, max(y + 1, 5), 'N={}'.format(n),
         rotation='vertical', va='bottom', ha='center')

plt.legend(loc='upper right')

plt.savefig('dpa_resid_{}_{}.png'.format(start, stop))
