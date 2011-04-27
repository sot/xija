"""
Next-generation thermal modeling framework for Chandra thermal modeling
"""

from odict import OrderedDict
import numpy as np

import Ska.Numpy
from Chandra.Time import DateTime
import Ska.engarchive.fetch_sci as fetch
import clogging

debug=1
if 'debug' in globals():
    from IPython.Debugger import Tracer
    pdb_settrace = Tracer()

logger = clogging.config_logger('xija', level=clogging.INFO)

class TelemSet(dict):
    """
    Dict of uniformly sampled 5min telemetry.
    """
    def __init__(self, start=None, stop=None, dt=328.8):
        self.dt = dt
        self.dt_ksec = self.dt / 1000.
        self.set_times(start, stop)

    def set_times(self, start, stop=None, dt=328.0):
        self.tstart = DateTime(start).secs
        self.tstop = DateTime(stop).secs
        self.datestart = DateTime(self.tstart).date
        self.datestop = DateTime(self.tstop).date

    def fetch(self, msid):
        msid = msid.lower()
        logger.info('Fetching msid: %s over %s to %s' % (msid, self.datestart, self.datestop))
        tlm = fetch.MSID(msid, self.tstart, self.tstop, stat='5min', filter_bad=True)
        if not self.keys():
            self.times = np.arange(tlm.times[0], tlm.times[-1], self.dt)
            self.ksecs = (self.times - self.times[0]) / 1000.
            self.n_times = len(self.times)

        self[msid] = Ska.Numpy.interpolate(tlm.vals, tlm.times, self.times, method='nearest')


def get_par_func(index):
    def _func(self):
        return self.parvals[index]
    return _func


def set_par_func(index):
    def _func(self, val):
        self.parvals[index] = val
    return _func


class ModelComponent(object):
    """ Model component base class"""
    def __init__(self, model, name):
        self.model = model
        self.name = name
        self.n_mvals = 0
        self.predict = False  # Predict values for this model component
        self.parnames = []
        self.parvals = []

    def __str__(self):
        return self.name

    n_parvals = property(lambda self: len(self.parvals))

    def add_par(self, name, val=None):
        setattr(ModelComponent, name,
                property(get_par_func(self.n_parvals),
                         set_par_func(self.n_parvals)))
        self.parnames.append(name)
        self.parvals.append(val)

class Node(ModelComponent):
    times = property(lambda self: self.model.tlms.times)

    def __init__(self, model, name, sigma=1.0, quant=None):
        ModelComponent.__init__(self, model, name)
        self.sigma = sigma
        self.quant = quant
        self.n_mvals = 1
        self.predict = True

    @property
    def dvals(self):
        if not hasattr(self, '_dvals'):
            self.model.tlms.fetch(self.name)
            self._dvals = self.model.tlms[self.name]
        return self._dvals


class Coupling(ModelComponent):
    """Couple two nodes together (one-way coupling)"""
    def __init__(self, model, node1, node2, tau):
        name = '{0}__{1}'.format(node1, node2)
        ModelComponent.__init__(self, model, name)
        self.node1 = self.model.comp[str(node1)]
        self.node2 = self.model.comp[str(node2)]
        self.add_par('tau', tau)
        self.add_par('tau2', 2)
        self.add_par('tau3', 3)

    # tau = property(get_par_func(0), set_par_func(0))

class HeatSink(ModelComponent):
    """Fixed temperature external heat bath"""
    def __init__(self, model, name, T):
        ModelComponent.__init__(self, model, name)
        self.add_par('T', T)

    @property
    def dvals(self):
        if not hasattr(self, '_dvals'):
            self._dvals = self.pars['T'].val * np.ones(self.model.tlms.n_times)
        return self._dvals


class PrecomputedHeatPower(ModelComponent):
    """Component that provides a static (precomputed) direct heat power input"""
    def __init__(self, model, name):
        ModelComponent.__init__(self, model, name)


class ActiveHeatPower(ModelComponent):
    """Component that provides active heat power input which depends on
    current or past computed model values"""
    def __init__(self, model, name):
        ModelComponent.__init__(self, model, name)


class SolarHeat(PrecomputedHeatPower):
    """Solar heating (pitch dependent)"""
    def __init__(self, model, name):
        ModelComponent.__init__(self, model, name)


class EarthHeat(PrecomputedHeatPower):
    """Earth heating of ACIS cold radiator (attitude, ephem dependent)"""
    def __init__(self, model, name):
        ModelComponent.__init__(self, model, name)

    
class AcisPower(PrecomputedHeatPower):
    """Heating from ACIS electronics (ACIS config dependent CCDs, FEPs etc)"""
    def __init__(self, model, name):
        ModelComponent.__init__(self, model, name)

    
class ProportialHeater(ActiveHeatPower):
    """Proportional heater (P = k * (T - T_set) for T > T_set)"""
    def __init__(self, model, name):
        ModelComponent.__init__(self, model, name)

        
class ThermostatHeater(ActiveHeatPower):
    """Thermostat heater (with configurable deadband)"""
    def __init__(self, model, name):
        ModelComponent.__init__(self, model, name)


class ThermalModel(object):
    def __init__(self, start='2011:115:00:00:00', stop='2011:115:01:00:00'):
        self.comp = OrderedDict()
        self.tlms = TelemSet(start, stop)
        
    def add(self, ComponentClass, *args, **kwargs):
        comp = ComponentClass(self, *args, **kwargs)
        self.comp[comp.name] = comp
        return comp

    comps = property(lambda self: self.comp.values())

    def make_parvals(self):
        """For components that have parameter values make a view into the
        global parameter values array.  But first count the total number of
        parameters and make that global param vals array ``self.parvals``.
        This must be manipulated only by reference (i.e. parvals[:] = ..)
        in order that the component views remain valid.
        """
        comps = [x for x in self.comps if x.n_parvals]
        n_parvals = sum(x.n_parvals for x in comps)
        i_parvals = np.cumsum([0] + [x.n_parvals for x in comps])
        self.parvals = np.zeros(n_parvals, dtype=np.float32)
        self.parnames = []
        for comp, i0, i1 in zip(comps, i_parvals[:-1], i_parvals[1:]):
            self.parnames.extend(comp.parnames)
            self.parvals[i0:i1] = comp.parvals  # copy existing values
            comp.parvals = self.parvals[i0:i1]  # make a local (view into global parvals
            comp.parvals_i = i0

    def make_mvals(self):
        """Initialize the global mvals (model values) array.  This is an
        N (rows) x n_times (cols) array that contains all data needed
        to compute the model prediction.  All rows are initialized to
        relevant data values (e.g. node temps, time-dependent power,
        external temperatures, etc).  In the model calculation some
        rows will be overwritten with predictions.
        """
        # Select components with data values, and from those select ones that get
        # predicted and those that do not get predicted
        comps = [x for x in self.comps if x.n_mvals]
        preds = [x for x in comps if x.predict]
        unpreds = [x for x in comps if not x.predict]

        # Register the location of component mvals in global mvals
        i = 0
        for comp in preds + unpreds:
            comp.mvals_i = i
            i += comp.n_mvals

        # Stack the input dvals.  This *copies* the data values.
        self.n_preds = len(preds)
        self.mvals = np.hstack(comp.dvals for comp in preds + unpreds)
        self.mvals.shape = (len(comps_dvals), -1)

    def make_mults(self):
        """"
        Iterate through Couplings to make mults: 2-d array containing rows
        in the format:
          [idx1, idx2]

        This provides a derivative coupling term in the form:
          d_mvals[idx1] /dt += U12 * mvals[idx2]
        """
        mults = []
        for comp in self.comps:
            if isinstance(comp, Coupling):
                i1 = comp.node1.mvals_i
                i2 = comp.node2.mvals_i
                mults.append((i1, i2))

        self.mults = np.array(mults)

    def make_heats(self):
        """"
        Iterate through Couplings to make heats.
        This provides a derivative heating term in the form:
          d_mvals[i1] /dt += mvals[i2]
        """
        heats = []
        for comp in self.vcomps:
            if isinstance(comp, PrecomputedHeatPower):
                i1 = comp.node1.mvals_i  # Node being heated
                i2 = comp.mvals_i        # mvals row with precomputed heat input
                heats.append((i1, i2))

        self.heats = np.array(heats)

    def calc(self, parvals, x=None):
        # Pre-allocate some arrays
        y = np.zeros(n_preds)
        k1 = np.zeros(n_preds)
        k2 = np.zeros(n_preds)
        d = np.zeros(n_preds)
        dt = self.model.tlms.dt_ksec * 2

        self.parvals[:] = parvals

        for comp in self.comps:
            comp.update()

        for j in np.arange(0, self.model.tlms.n_times-2, 2):
            # 2nd order Runge-Kutta (do 4th order later as needed)
            y[:] = mvals[:n_preds, j]
            k1[:] = dt * dT_dt(j, y, d)
            k2[:] = dt * dT_dt(j+1, y + k1 / 2.0, d)
            self.mvals[:n_preds, j+2] = y + k2

def dT_dt(j, y, d):
    d[:] = 0.0

    # Couplings with other nodes
    for i in xrange(len(mult_idxs)):
        i1 = mult_idxs[i, 0] 
        i2 = mult_idxs[i, 1]
        if i2 < n_preds:
            d[i1] += mult_vals[i] * y[i2]
        else:
            d[i1] += mult_vals[i] * mvals[i2, j]

    # Direct heat inputs (e.g. Solar, Earth)
    for i in xrange(len(head_idxs)):
        i1 = heats[i, 0] 
        i2 = heats[i, 1]
        d[i1] += mvals[i2, j]

    return d


