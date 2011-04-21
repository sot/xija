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

        self[msid] = Ska.Numpy.interpolate(tlm.vals, tlm.times, self.times, method='nearest')

def get_par_func(index):
    def _func(self):
        return self.pars[index]
    return _func

def set_par_func(index):
    def _func(self, val):
        self.pars[index] = val
    return _func

class ModelComponent(object):
    """ Model component base class"""
    def __init__(self):
        comps[self.name] = self


class Node(ModelComponent):
    times = property(lambda self: tlms.times)

    def __init__(self, name, sigma=1.0, quant=None):
        self.name = name
        self.sigma = sigma
        self.quant = quant
        self.mvals = None
        ModelComponent.__init__(self)

    @property
    def dvals(self):
        if not hasattr(self, '_dvals'):
            tlms.fetch(self.name)
            self._dvals = tlms[self.name]
        return self._dvals

    def __str__(self):
        return self.name


class Coupling(ModelComponent):
    """Couple two nodes together (one-way coupling)"""
    def __init__(self, node, node2, tau):
        self.name = '{0}__{1}'.format(node, node2)
        self.node = comps[str(node)]
        self.node2 = comps[str(node2)]
        self.pars = [tau]
        ModelComponent.__init__(self)

    tau = property(get_par_func(0), set_par_func(0))

class HeatSink(ModelComponent):
    """Fixed temperature external heat bath"""
    def __init__(self, name):
        self.name = name


class HeatPower(ModelComponent):
    """Component that provides a direct heat power input"""
    pass


class SolarHeat(HeatPower):
    """Solar heating (pitch dependent)"""
    def __init__(self, name, ):
        self.name = name


class EarthHeat(HeatPower):
    """Earth heating of ACIS cold radiator (attitude, ephem dependent)"""
    def __init__(self, name, ):
        self.name = name

    
class ProportialHeater(HeatPower):
    """Proportional heater (P = k * (T - T_set) for T > T_set)"""
    def __init__(self, name, ):
        self.name = name

        
class ThermostatHeater(HeatPower):
    """Thermostat heater (with configurable deadband)"""
    def __init__(self, name, ):
        self.name = name

        
class AcisPower(HeatPower):
    """Heating from ACIS electronics (ACIS config dependent CCDs, FEPs etc)"""
    def __init__(self, name, ):
        self.name = name
    
comps = OrderedDict()
msids = set()
tlms = TelemSet(start='2011:090')
