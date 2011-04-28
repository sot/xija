"""
Next-generation thermal modeling framework for Chandra thermal modeling
"""

from odict import OrderedDict
import numpy as np

import Ska.Numpy
from Chandra.Time import DateTime
import Ska.engarchive.fetch_sci as fetch
import clogging
import scipy.interpolate

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
    def __init__(self, model):
        self.model = model
        self.n_mvals = 0
        self.predict = False  # Predict values for this model component
        self.parnames = []
        self.parvals = []

    n_parvals = property(lambda self: len(self.parvals))

    def add_par(self, name, val=None):
        setattr(ModelComponent, name,
                property(get_par_func(self.n_parvals),
                         set_par_func(self.n_parvals)))
        self.parnames.append(name)
        self.parvals.append(val)

    def _set_mvals(self, vals):
        self.model.mvals[self.mvals_i, :] = vals
        
    def _get_mvals(self, vals):
        return self.model.mvals[self.mvals_i, :]
        
    mvals = property(_get_mvals, _set_mvals)

    @property
    def name(self):
        return self.__str__()

    def update(self):
        pass

class TelemData(ModelComponent):
    times = property(lambda self: self.model.tlms.times)

    def __init__(self, model, msid):
        ModelComponent.__init__(self, model)
        self.msid = msid
        self.n_mvals = 1
        self.predict = False

    @property
    def dvals(self):
        if not hasattr(self, '_dvals'):
            self.model.tlms.fetch(self.msid)
            self._dvals = self.model.tlms[self.msid]
        return self._dvals

    def __str__(self):
        return self.msid

class Node(TelemData):
    def __init__(self, model, msid, sigma=1.0, quant=None, predict=True):
        TelemData.__init__(self, model, msid)
        self.sigma = sigma
        self.quant = quant
        self.predict = predict


class Coupling(ModelComponent):
    """Couple two nodes together (one-way coupling)"""
    def __init__(self, model, node1, node2, tau):
        ModelComponent.__init__(self, model)
        self.node1 = self.model.get_comp(node1)
        self.node2 = self.model.get_comp(node2)
        self.add_par('tau', tau)

    def __str__(self):
        return 'coupling__{0}__{1}'.format(self.node1, self.node2)
        

class HeatSink(ModelComponent):
    """Fixed temperature external heat bath"""
    def __init__(self, model, node, T, tau):
        ModelComponent.__init__(self, model)
        self.add_par('T', T)
        self.add_par('tau', tau)
        self.node = self.model.get_comp(node)

    def __str__(self):
        return 'heatsink__{0}'.format(self.node)


class PrecomputedHeatPower(ModelComponent):
    """Component that provides a static (precomputed) direct heat power input"""
    pass


class ActiveHeatPower(ModelComponent):
    """Component that provides active heat power input which depends on
    current or past computed model values"""
    pass


class SolarHeat(PrecomputedHeatPower):
    """Solar heating (pitch dependent)"""

    def __init__(self, model, node, pitch_comp, P_pitches=None, Ps=None, dPs=None,
                 tau=1732.0, ampl=0.05, epoch='2011:001'):
        ModelComponent.__init__(self, model)
        self.node = self.model.get_comp(node)
        self.pitch_comp = pitch_comp

        if P_pitches is None:
            P_pitches = [45, 65, 90, 130, 180]
        self.P_pitches = np.array(P_pitches, dtype=np.float32)
        self.n_pitches = len(self.P_pitches)

        if Ps is None:
            Ps = np.ones_like(self.P_pitches)
        self.Ps = np.array(Ps, dtype=np.float32)

        if dPs is None:
            dPs = np.zeros_like(self.P_pitches)
        self.dPs = np.array(dPs, dtype=np.float32)

        self.epoch=epoch

        for pitch, power in zip(self.P_pitches, self.Ps):
            self.add_par('P_{0:.0f}'.format(float(pitch)), power)
        for pitch, dpower in zip(self.P_pitches, self.dPs):
            self.add_par('dP_{0:.0f}'.format(float(pitch)), dpower)
        self.add_par('tau', tau)
        self.add_par('ampl', ampl)
        self.n_mvals = 1

    @property
    def dvals(self):
        if not hasattr(self, 'pitches'):
            self.pitches = self.pitch_comp.dvals
        if not hasattr(self, 't_days'):
            self.t_days = (self.pitch_comp.times - DateTime(self.epoch).secs) / 86400.0
        if not hasattr(self, 't_phase'):
            time2000 = DateTime('2000:001:00:00:00').secs
            time2010 = DateTime('2010:001:00:00:00').secs
            secs_per_year = (time2010 - time2000) / 10.0
            t_year = (self.pitch_comp.times - time2000) / secs_per_year
            self.t_phase = t_year * 2 * np.pi

        Ps = self.parvals[0:self.n_pitches]
        dPs = self.parvals[self.n_pitches:2*self.n_pitches]
        Ps_interp = scipy.interpolate.interp1d(self.P_pitches, Ps, kind='cubic')
        dPs_interp = scipy.interpolate.interp1d(self.P_pitches, dPs, kind='cubic')
        P_vals = Ps_interp(self.pitches)
        dP_vals = dPs_interp(self.pitches)
        self._dvals = (P_vals + dP_vals * (1 - np.exp(-self.t_days / self.tau))
                       + self.ampl * np.cos(self.t_phase)).reshape(-1)
        return self._dvals

    def update(self):
        self.mvals = self.dvals

    def __str__(self):
        return 'solarheat__{0}'.format(self.node)


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

    def get_comp(self, name):
        """Get a model component.  Works with either a string or a component object"""
        return self.comp[str(name)]

    def make(self):
        self.make_parvals()
        self.make_mvals()
        self.make_mults()
        self.make_heats()
        self.make_heatsinks()

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
            self.parnames.extend(comp.name + '__' + x for x in comp.parnames)
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
        self.mvals.shape = (len(comps), -1)
        self.cvals = self.mvals[:, 0::2]

    def make_mults(self):
        """"
        Iterate through Couplings to make mults: 2-d array containing rows
        in the format:
          [idx1, idx2, parvals_i]

        This provides a derivative coupling term in the form:
          d_mvals[idx1] /dt += mvals[idx2] / parvals[parvals_i]
        """
        mults = []
        for comp in self.comps:
            if isinstance(comp, Coupling):
                i1 = comp.node1.mvals_i
                i2 = comp.node2.mvals_i
                mults.append((i1, i2, comp.parvals_i))

        self.mults = np.array(mults)

    def make_heats(self):
        """
        Iterate through PrecomputedHeatPower to make heats.
        This provides a derivative heating term in the form:
          d_mvals[i1] /dt += mvals[i2]
        """
        heats = []
        for comp in self.comps:
            if isinstance(comp, PrecomputedHeatPower):
                i1 = comp.node.mvals_i   # Node being heated
                i2 = comp.mvals_i        # mvals row with precomputed heat input
                heats.append((i1, i2))

        self.heats = np.array(heats)

    def make_heatsinks(self):
        """
        """
        heatsinks = []
        for comp in self.comps:
            if isinstance(comp, HeatSink):
                i1 = comp.node.mvals_i   # Node being heated
                T_parval_i = comp.parvals_i
                tau_parval_i = comp.parvals_i + 1
                heatsinks.append((i1, T_parval_i, tau_parval_i))

        self.heatsinks = np.array(heatsinks)

    def calc(self, parvals=None, x=None):
        n_preds = self.n_preds
        
        # Pre-allocate some arrays
        #y = np.zeros(n_preds)
        #k1 = np.zeros(n_preds)
        #k2 = np.zeros(n_preds)
        deriv = np.zeros(n_preds)
        dt = self.tlms.dt_ksec * 2

        if parvals is not None:
            self.parvals[:] = parvals

        for comp in self.comps:
            comp.update()

        mvals = self.mvals
        parvals = self.parvals
        mults = self.mults
        heats = self.heats
        heatsinks = self.heatsinks
        
        for j in np.arange(0, self.tlms.n_times-2, 2):
            # 2nd order Runge-Kutta (do 4th order later as needed)
            y = mvals[:n_preds, j]
            k1 = dt * dT_dt(j, y, deriv, n_preds, mvals, parvals, mults,
                            heats, heatsinks)
            k2 = dt * dT_dt(j+1, y + k1 / 2.0, deriv, n_preds, mvals, parvals, mults,
                            heats, heatsinks)
            self.mvals[:n_preds, j+2] = y + k2

def dT_dt(j, y, deriv, n_preds, mvals, parvals, mults, heats, heatsinks):
    deriv[:] = 0.0

    # Couplings with other nodes
    for i in xrange(len(mults)):
        i1 = mults[i, 0] 
        i2 = mults[i, 1]
        tau = parvals[mults[i, 2]]
        if i2 < n_preds and i1 < n_preds:
            deriv[i1] += (y[i2] - y[i1]) / tau
        else:
            deriv[i1] += (mvals[i2, j] - y[i1]) / tau

    # Direct heat inputs (e.g. Solar, Earth)
    for i in xrange(len(heats)):
        i1 = heats[i, 0]
        if i1 < n_preds:
            i2 = heats[i, 1]
            deriv[i1] += mvals[i2, j]

    # Couplings to heat sinks
    for i in xrange(len(heatsinks)):
        i1 = heatsinks[i, 0]
        if i1 < n_preds:
            T = parvals[heatsinks[i, 1]]
            tau = parvals[heatsinks[i, 2]]
            deriv[i1] += (T - mvals[i1, j]) / tau

    return deriv


