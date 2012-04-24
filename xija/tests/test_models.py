import os
import numpy as np

from .. import ThermalModel, __version__
from numpy import sin, cos, abs

print
print 'Version =', __version__

os.chdir(os.path.abspath(os.path.dirname(__file__)))


def test_dpa():
    mdl = ThermalModel('dpa', start='2012:001', stop='2012:007',
                            model_spec='dpa.json')
    times = (mdl.times - mdl.times[0]) / 10000.0
    mdl.comp['1dpamzt'].set_data(30.0)
    mdl.comp['sim_z'].set_data((100000 * sin(times)).astype(int))
    mdl.comp['pitch'].set_data(45.1 + 55 * (1 + sin(times / 2)))
    mdl.comp['eclipse'].set_data(cos(times) > 0.95)
    mdl.comp['fep_count'].set_data((abs(sin(times / 7.89)) * 6.5
                                    ).astype(int))
    mdl.comp['ccd_count'].set_data((abs(sin(times / 3.111)) * 6.5
                                    ).astype(int))
    mdl.comp['vid_board'].set_data((abs(sin(times / 2.5)) * 1.5
                                    ).astype(int))
    mdl.comp['clocking'].set_data((abs(cos(times) / 1.4) * 1.5
                                   ).astype(int))
    mdl.comp['dpa_power'].set_data(60)

    mdl.make()
    mdl.calc()
    dpa = mdl.comp['1dpamzt']
    if not os.path.exists('dpa.npz'):
        print 'Writing reference file dpa.npz'
        np.savez('dpa.npz', times=mdl.times, dvals=dpa.dvals,
                 mvals=dpa.mvals)

    regr = np.load('dpa.npz')
    assert np.allclose(mdl.times, regr['times'])
    assert np.allclose(dpa.dvals, regr['dvals'])
    assert np.allclose(dpa.mvals, regr['mvals'])
