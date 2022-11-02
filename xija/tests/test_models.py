# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import tempfile
import numpy as np
import pytest
from pathlib import Path

from xija import ThermalModel, Node, HeatSink, SolarHeat, Pitch, Eclipse, __version__
from xija.get_model_spec import get_xija_model_spec, get_xija_model_names
import xija
from numpy import sin, cos, abs

try:
    import Ska.Matplotlib  # noqa
    HAS_PLOTDATE = True
except ImportError:
    HAS_PLOTDATE = False

print()
print('Version =', __version__)

CURRDIR = os.path.abspath(os.path.dirname(__file__))


def abs_path(spec, as_path=False):
    path = os.path.join(CURRDIR, spec)
    if as_path:
        path = Path(path)
    return path


@pytest.mark.parametrize('as_path', [True, False])
def test_dpa_real(as_path):
    mdl = ThermalModel('dpa', start='2020:001:12:00:00', stop='2020:007:12:00:00',
                       model_spec=abs_path('dpa.json'))
    mdl._get_cmd_states()

    mdl.make()
    mdl.calc()
    dpa = mdl.comp['1dpamzt']
    reffile = abs_path('dpa_real.npz')
    if not os.path.exists(reffile):
        print('Writing reference file', reffile)
        np.savez(reffile, times=mdl.times, dvals=dpa.dvals,
                 mvals=dpa.mvals)

    regr = np.load(reffile)
    assert np.allclose(mdl.times, regr['times'])
    assert np.allclose(dpa.dvals, regr['dvals'])
    assert np.allclose(dpa.mvals, regr['mvals'])


def test_pitch_clip():
    """Pitch in this time range goes from 48.62 to 158.41.  dpa_clip.json
    has been modified so the solarheat pitch range is from 55 .. 153.
    Make sure the model still runs with no interpolation error.

    Parameters
    ----------

    Returns
    -------

    """
    mdl = ThermalModel('dpa', start='2012:001:12:00:00', stop='2012:007:12:00:00',
                       model_spec=abs_path('dpa_clip.json'))
    mdl._get_cmd_states()

    mdl.make()
    mdl.calc()


def test_pitch_range_clip():
    """Pitch in this time range goes from approximately 48.5 to 175.0 degrees.
    pftank2t.json is used as a placeholder to load a new thermal model, and
    should be able to be replaced with any other model. Make sure the pitch
    range stored in the model object does not get clipped to a narrower range.
    Make sure the pitch range stored does not include values outside of the 45
    to 180 degree range.

    Parameters
    ----------

    Returns
    -------

    """
    mdl = ThermalModel('tank', start='2019:120:12:00:00', stop='2019:122:12:00:00',
                       model_spec=abs_path('pftank2t.json'))

    mdl.comp['pf0tank2t'].set_data(20.0)
    mdl.make()
    mdl.calc()

    pitch = mdl.get_comp('pitch')
    assert np.any(pitch.mvals > 170)
    assert np.all((pitch.mvals > 45) & (pitch.mvals < 180))


def test_dpa_remove_pow():
    mdl = ThermalModel('dpa', start='2019:001:12:00:00', stop='2019:007:12:00:00',
                       model_spec=abs_path('dpa_remove_pow.json'))
    mdl._get_cmd_states()

    mdl.make()
    mdl.calc()
    dpa = mdl.comp['1dpamzt']
    reffile = abs_path('dpa_remove_pow.npz')
    if not os.path.exists(reffile):
        print('Writing reference file', reffile)
        np.savez(reffile, times=mdl.times, dvals=dpa.dvals,
                 mvals=dpa.mvals)

    regr = np.load(reffile)
    assert np.allclose(mdl.times, regr['times'])
    assert np.allclose(dpa.dvals, regr['dvals'])
    assert np.allclose(dpa.mvals, regr['mvals'])

def get_dpa_model():
    mdl = ThermalModel('dpa', start='2012:001:12:00:00', stop='2012:007:12:00:00',
                            model_spec=abs_path('dpa.json'))
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
    return mdl


def test_dpa():
    mdl = get_dpa_model()
    dpa = mdl.comp['1dpamzt']
    reffile = abs_path('dpa.npz')
    if not os.path.exists(reffile):
        print('Writing reference file dpa.npz')
        np.savez(reffile, times=mdl.times, dvals=dpa.dvals,
                 mvals=dpa.mvals)

    regr = np.load(reffile)
    assert np.allclose(mdl.times, regr['times'], rtol=0, atol=1e-3)
    assert np.allclose(dpa.dvals, regr['dvals'])
    assert np.allclose(dpa.mvals, regr['mvals'])


@pytest.mark.skipif('not HAS_PLOTDATE')
def test_plotdate():
    # Make sure model_plotdate property works
    mdl = get_dpa_model()
    mdl.comp['1dpamzt'].model_plotdate


def test_data_types():
    for data_type in (int, float, np.float32, np.float64, np.int32, np.int64, np.complex64):
        mdl = ThermalModel('dpa', start='2012:001:12:00:00', stop='2012:007:12:00:00',
                           model_spec=abs_path('dpa.json'))
        dpa = mdl.comp['1dpamzt']
        dpa.set_data(data_type(30.0))
        if data_type is np.complex64:
            with pytest.raises(ValueError):
                dpa.dvals
        else:
            dpa.dvals  # Property should evaluate OK


def test_minusz():
    mdl = ThermalModel('minusz', start='2012:001:12:00:00', stop='2012:004:12:00:00',
                       model_spec=abs_path('minusz.json'))
    times = (mdl.times - mdl.times[0]) / 10000.0
    msids = ('tephin', 'tcylaft6', 'tcylfmzm', 'tmzp_my', 'tfssbkt1')
    for msid in msids:
        mdl.comp[msid].set_data(10.0)
    mdl.comp['pitch'].set_data(45.1 + 55 * (1 + sin(times / 2)))
    mdl.comp['eclipse'].set_data(cos(times) > 0.95)

    mdl.make()
    mdl.calc()

    regrfile = abs_path('minusz.npz')
    if not os.path.exists(regrfile):
        print('Writing reference file', regrfile)
        kwargs = {msid: mdl.comp[msid].mvals for msid in msids}
        np.savez(regrfile, times=mdl.times, **kwargs)

    regr = np.load(regrfile)
    assert np.allclose(mdl.times, regr['times'])
    for msid in msids:
        assert np.allclose(mdl.comp[msid].mvals, regr[msid])


def test_pftank2t():
    mdl = ThermalModel('pftank2t', start='2012:001:12:00:00', stop='2012:004:12:00:00',
                       model_spec=abs_path('pftank2t.json'))
    times = (mdl.times - mdl.times[0]) / 10000.0
    msids = ('pftank2t', 'pf0tank2t')
    for msid in msids:
        mdl.comp[msid].set_data(10.0)
    mdl.comp['pitch'].set_data(45.1 + 55 * (1 + sin(times / 2)))
    mdl.comp['eclipse'].set_data(cos(times) > 0.95)

    mdl.make()
    mdl.calc()

    regrfile = abs_path('pftank2t.npz')
    if not os.path.exists(regrfile):
        print('Writing reference file', regrfile)
        kwargs = {msid: mdl.comp[msid].mvals for msid in msids}
        np.savez(regrfile, times=mdl.times, **kwargs)

    regr = np.load(regrfile)
    assert np.allclose(mdl.times, regr['times'])
    for msid in msids:
        assert np.allclose(mdl.comp[msid].mvals, regr[msid])

    # Test that setattr works for component parameter value by changing one
    # value and seeing that the model prediction changes substantially.
    mvals = mdl.comp['pftank2t'].mvals.copy()
    mdl.comp['solarheat__pf0tank2t'].P_60 = 5.0
    mdl.calc()
    mvals2 = mdl.comp['pftank2t'].mvals
    assert np.any(abs(mvals2 - mvals) > 5)


def test_multi_solar_heat_values():
    P_pitches = [45, 180]
    P_vals = [1.0, 1.0]
    P_pitches2 = [45, 65, 90, 140, 180]
    P_vals2 = [1.0, 1.0, 0.0, 1.0, 1.0]

    model = ThermalModel('test', start='2011:001:12:00:00', stop='2011:005:12:00:00')
    tephin = model.add(Node, 'tephin')
    tcylaft6 = model.add(Node, 'tcylaft6')
    pitch = model.add(Pitch)
    eclipse = model.add(Eclipse)

    model.add(HeatSink, tephin, T=0.0, tau=200.0)
    model.add(HeatSink, tcylaft6, T=0.0, tau=200.0)

    model.add(SolarHeat, tephin, pitch, eclipse, P_pitches, P_vals)
    model.add(SolarHeat, tcylaft6, pitch, eclipse, P_pitches2, P_vals2)

    tephin.set_data(30.0)
    tcylaft6.set_data(30.0)
    pitch.set_data(90.0)
    eclipse.set_data(False)

    model.make()
    model.calc()

    mvals = model.comp['tephin'].mvals
    mvals2 = model.comp['tcylaft6'].mvals
    assert len(mvals) == 1051
    assert mvals[0] == 30.0
    assert abs(mvals[500] - 157.4740) < 0.001
    assert abs(mvals[1050] - 196.4138) < 0.001

    assert len(mvals2) == 1051
    assert mvals2[0] == 30.0
    assert abs(mvals2[500] - 15.8338) < 0.001
    assert abs(mvals2[1050] - 11.4947) < 0.001

    # Make sure we can round-trip the model through a JSON file
    temp_name = tempfile.NamedTemporaryFile(delete=False).name
    model.write(temp_name)
    model2 = ThermalModel('test', model_spec=temp_name,
                          start='2011:001:12:00:00', stop='2011:005:12:00:00')
    os.unlink(temp_name)
    model2.get_comp('tephin').set_data(30.0)
    model2.get_comp('tcylaft6').set_data(30.0)
    model2.get_comp('pitch').set_data(90.0)
    model2.get_comp('eclipse').set_data(False)

    model2.make()
    model2.calc()

    mvals = model2.comp['tephin'].mvals
    mvals2 = model2.comp['tcylaft6'].mvals

    assert len(mvals) == 1051
    assert mvals[0] == 30.0
    assert abs(mvals[500] - 157.4740) < 0.001
    assert abs(mvals[1049] - 196.4138) < 0.001

    assert len(mvals2) == 1051
    assert mvals2[0] == 30.0
    assert abs(mvals2[500] - 15.8338) < 0.001
    assert abs(mvals2[1049] - 11.4947) < 0.001


def _get_model_for_dP_pitches(solar_class, msid, dP_pitches, dPs):
    model = xija.XijaModel(msid, start='2021:001', stop='2021:005')
    model.add(xija.Node, msid)
    model.add(xija.Pitch)
    model.add(xija.Eclipse)
    model.add(xija.SimZ)
    model.add(xija.Roll)
    model.add(xija.DetectorHousingHeater)
    model.add(solar_class,
              P_pitches=[45, 90, 135, 180],
              dP_pitches=dP_pitches,
              Ps=[0.0, 1.0, 1.0, 0.0],
              dPs=dPs,
              ampl=0.0,
              eclipse_comp='eclipse',
              epoch='2020:001',
              node=msid,
              pitch_comp='pitch',
              )
    model.add(xija.HeatSink,
              T=-20.0,
              node=msid,
              tau=30.0,
              )
    model.add(xija.StepFunctionPower,
              P=0.1,
              id='_1iru',
              node=msid,
              time='2021:003'
              )
    return model


@pytest.mark.parametrize('solar_class', [xija.SolarHeat,
                                         xija.SolarHeatHrc,
                                         xija.SolarHeatHrcMult,
                                         xija.SolarHeatHrcOpts,
                                         xija.SolarHeatMulplicative])
def test_fewer_dP_pitches(solar_class):
    """Test providing fewer dP_pitches than P_pitches for SolarHeat model
    """
    msid = '4hfspat'
    model1 = _get_model_for_dP_pitches(solar_class, msid, None, [0.0, 1.0, 2.0, 3.0])
    model2 = _get_model_for_dP_pitches(solar_class, msid, [45, 180], [0.0, 3.0])

    model1.make()
    model2.make()

    model1.calc()
    model2.calc()

    assert np.allclose(model1.comp[msid].mvals, model2.comp[msid].mvals)


def test_bad_times():
    """
    Test bad times handling for PM2THV1T model which has a large number of bad
    times defined, including in particular:

        ['2022:083:21:43:01.949', '2022:083:21:52:52.349'],
        ['2022:083:22:02:09.949', '2022:083:22:40:58.749'],
        ['2022:083:23:03:56.349', '2022:084:03:45:28.350'],
        ['2022:084:07:13:45.151', '2022:084:14:54:02.753'],
    """
    spec1 = get_xija_model_spec('pm2thv1t', version='3.40.2')[0]
    assert len(spec1['bad_times']) == 5876

    # Straddling two bad time intervals (model starts within first and ends
    # after second).
    mdl = xija.XijaModel(
        'test', '2022:083:22:30:00', '2022:084:04:00:00', model_spec=spec1,
    )
    assert mdl.bad_times == spec1["bad_times"]
    assert mdl.bad_times_indices == [[0, 2], [6, 58]]
    assert 58 < mdl.n_times

    # Within one bad time interval
    mdl = xija.XijaModel(
        'test', '2022:084:00:00:00', '2022:084:01:00:00', model_spec=spec1,
    )
    assert mdl.bad_times == spec1["bad_times"]
    assert mdl.bad_times_indices == [[0, len(mdl.times)]]

    # Within one bad time interval
    mdl = xija.XijaModel(
        'test', '2022:084:00:00:00', '2022:084:01:00:00', model_spec=spec1,
    )
    assert mdl.bad_times == spec1["bad_times"]
    assert mdl.bad_times_indices == [[0, len(mdl.times)]]

    # Within no bad time interval
    mdl = xija.XijaModel(
        'test', '2022:084:04:00:00', '2022:084:07:00:00', model_spec=spec1,
    )
    assert mdl.bad_times == spec1["bad_times"]
    assert mdl.bad_times_indices == []


model_names = get_xija_model_names()


@pytest.mark.parametrize('model_name', model_names)
def test_model_from_chandra_models(model_name):
    """
    Test that every model in the chandra_models repo can be run with xija.
    """
    from cheta import fetch
    model_spec, _ = get_xija_model_spec(model_name)
    mdl = xija.XijaModel(model_name, '2021:001', '2021:007', model_spec=model_spec)
    for comp in mdl.comps:
        if isinstance(comp, Node):
            if comp.msid.upper() not in fetch.content:
                comp.set_data(0.0)
    mdl.make()
    mdl.calc()
