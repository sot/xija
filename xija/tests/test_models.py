import os
import numpy as np
import pytest

from .. import ThermalModel, __version__
from numpy import sin, cos, abs

print
print 'Version =', __version__

os.chdir(os.path.abspath(os.path.dirname(__file__)))


def test_dpa_real():
    mdl = ThermalModel('dpa', start='2012:001', stop='2012:007',
                       model_spec='dpa.json')
    # Check that cmd_states database can be read.  Skip if not, probably
    # running test on a platform without access.
    try:
        mdl._get_cmd_states()
    except:
        pytest.skip('No commanded states access - '
                    'cannot run DPA model with real states')

    mdl.make()
    mdl.calc()
    dpa = mdl.comp['1dpamzt']
    reffile = 'dpa_real.npz'
    if not os.path.exists(reffile):
        print 'Writing reference file', reffile
        np.savez(reffile, times=mdl.times, dvals=dpa.dvals,
                 mvals=dpa.mvals)

    regr = np.load(reffile)
    assert np.allclose(mdl.times, regr['times'])
    assert np.allclose(dpa.dvals, regr['dvals'])
    assert np.allclose(dpa.mvals, regr['mvals'])


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


def test_minusz():
    mdl = ThermalModel('minusz', start='2012:001', stop='2012:004',
                            model_spec='minusz.json')
    times = (mdl.times - mdl.times[0]) / 10000.0
    msids = ('tephin', 'tcylaft6', 'tcylfmzm', 'tmzp_my', 'tfssbkt1')
    for msid in msids:
        mdl.comp[msid].set_data(10.0)
    mdl.comp['pitch'].set_data(45.1 + 55 * (1 + sin(times / 2)))
    mdl.comp['eclipse'].set_data(cos(times) > 0.95)

    mdl.make()
    mdl.calc()

    regrfile = 'minusz.npz'
    if not os.path.exists(regrfile):
        print 'Writing reference file', regrfile
        kwargs = {msid: mdl.comp[msid].mvals for msid in msids}
        np.savez(regrfile, times=mdl.times, **kwargs)

    regr = np.load(regrfile)
    assert np.allclose(mdl.times, regr['times'])
    for msid in msids:
        assert np.allclose(mdl.comp[msid].mvals, regr[msid])


def test_pftank2t():
    mdl = ThermalModel('pftank2t', start='2012:001', stop='2012:004',
                            model_spec='pftank2t.json')
    times = (mdl.times - mdl.times[0]) / 10000.0
    msids = ('pftank2t', 'pf0tank2t')
    for msid in msids:
        mdl.comp[msid].set_data(10.0)
    mdl.comp['pitch'].set_data(45.1 + 55 * (1 + sin(times / 2)))
    mdl.comp['eclipse'].set_data(cos(times) > 0.95)

    mdl.make()
    mdl.calc()

    regrfile = 'pftank2t.npz'
    if not os.path.exists(regrfile):
        print 'Writing reference file', regrfile
        kwargs = {msid: mdl.comp[msid].mvals for msid in msids}
        np.savez(regrfile, times=mdl.times, **kwargs)

    regr = np.load(regrfile)
    assert np.allclose(mdl.times, regr['times'])
    for msid in msids:
        assert np.allclose(mdl.comp[msid].mvals, regr[msid])
