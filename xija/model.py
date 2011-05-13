"""
Next-generation thermal modeling framework for Chandra thermal modeling
"""

import json
from odict import OrderedDict
import numpy as np
import Ska.Numpy
from Chandra.Time import DateTime
import clogging
import scipy.interpolate

try:
    # Optional packages for model fitting or use on HEAD LAN
    import Ska.engarchive.fetch_sci as fetch
    import Chandra.cmd_states
    import Ska.DBI
except ImportError:
    pass

from .core import calc_model

if 'debug' in globals():
    from IPython.Debugger import Tracer
    pdb_settrace = Tracer()

logger = clogging.config_logger('xija', level=clogging.INFO)


class ModelComponent(object):
    """ Model component base class"""
    def __init__(self, model):
        self.model = model
        self.n_mvals = 0
        self.predict = False  # Predict values for this model component
        self.parnames = []
        self.parvals = []

    n_parvals = property(lambda self: len(self.parvals))
    times = property(lambda self: self.model.times)


    @staticmethod
    def get_par_func(index):
        def _func(self):
            return self.parvals[index]
        return _func

    @staticmethod
    def set_par_func(index):
        def _func(self, val):
            self.parvals[index] = val
        return _func

    def add_par(self, name, val=None):
        setattr(self.__class__, name,
                property(ModelComponent.get_par_func(self.n_parvals),
                         ModelComponent.set_par_func(self.n_parvals)))
        self.parnames.append(name)
        self.parvals.append(val)

    def _set_mvals(self, vals):
        self.model.mvals[self.mvals_i, :] = vals
        
    def _get_mvals(self):
        return self.model.mvals[self.mvals_i, :]
        
    mvals = property(_get_mvals, _set_mvals)

    @property
    def name(self):
        return self.__str__()

    def update(self):
        pass


# RENAME TelemData to Data or ??
class TelemData(ModelComponent):  
    times = property(lambda self: self.model.times)

    def __init__(self, model, msid, cmd_states_col=None, data=None):
        ModelComponent.__init__(self, model)
        self.msid = msid
        self.cmd_states_col = cmd_states_col or msid
        self.n_mvals = 1
        self.predict = False
        self.data = data

    def get_dvals_tlm(self):
        return self.model.fetch(self.msid)

    def get_dvals_cmd(self):
        return Chandra.cmd_states.interpolate_states(
            self.model.cmd_states[self.cmd_states_col], self.model.times)

    @property
    def dvals(self):
        if not hasattr(self, '_dvals'):
            if self.data is None:
                self._dvals = self.get_dvals_tlm()
            elif self.data == 'cmd':
                self._dvals = self.get_dvals_cmd()
            else:
                self._dvals = self.data
            
        return self._dvals

    def __str__(self):
        return self.msid


class Node(TelemData):
    def __init__(self, model, msid, data=None, sigma=-10, quant=None, predict=True):
        TelemData.__init__(self, model, msid, data)
        self._sigma = sigma
        self.quant = quant
        self.predict = predict

    @property
    def sigma(self):
        if self._sigma < 0:
            self._sigma = self.dvals.std() * (-self._sigma / 100.0)
        return self._sigma

    def calc_stat(self):
        return np.sum((self.dvals - self.mvals)**2 / self.sigma**2)

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


class Pitch(TelemData):
    def __init__(self, model, data=None):
        TelemData.__init__(self, model, 'aosares1', 'pitch', data)
    
    def __str__(self):
        return 'pitch'


class SimZ(TelemData):
    def __init__(self, model, data=None):
        TelemData.__init__(self, model, 'sim_z', 'simpos', data)
    
    @property
    def dvals(self):
        if not hasattr(self, '_dvals'):
            if self.data is None:
                self._dvals = np.rint(self.get_dvals_tlm() * -397.7225924607)
            elif self.data == 'cmd':
                self._dvals = self.get_dvals_cmd()
            else:
                self._dvals = self.data
            
        return self._dvals


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
                 tau=1732.0, ampl=0.05, epoch='2010:001'):
        ModelComponent.__init__(self, model)
        self.node = self.model.get_comp(node)
        self.pitch_comp = self.model.get_comp(pitch_comp)

        if P_pitches is None:
            P_pitches = [45, 65, 90, 130, 180]
        self.P_pitches = np.array(P_pitches, dtype=np.float)
        self.n_pitches = len(self.P_pitches)

        if Ps is None:
            Ps = np.ones_like(self.P_pitches)
        self.Ps = np.array(Ps, dtype=np.float)

        if dPs is None:
            dPs = np.zeros_like(self.P_pitches)
        self.dPs = np.array(dPs, dtype=np.float)

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
        self.P_vals = P_vals
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

    
class AcisPsmcSolarHeat(PrecomputedHeatPower):
    """Solar heating of PSMC box.  This is dependent on SIM-Z"""
    def __init__(self, model, node, pitch_comp, simz_comp, P_pitches=None, P_vals=None):
        ModelComponent.__init__(self, model)
        self.n_mvals = 1
        self.node = node
        self.pitch_comp = self.model.get_comp(pitch_comp)
        self.simz_comp = self.model.get_comp(simz_comp)
        self.P_pitches = np.array([50., 90., 150.] if (P_pitches is None) else P_pitches)
        self.simz_lims = ((-400000.0, -85000.0),  # HRC-S
                          (-85000.0, 0.0),        # HRC-I
                          (0.0, 400000.0))        # ACIS
        self.instr_names = ['hrcs', 'hrci', 'acis']
        for i, instr_name in enumerate(self.instr_names):
            for j, pitch in enumerate(self.P_pitches):
                self.add_par('P_{0}_{1:d}'.format(instr_name, pitch), P_vals[i, j])
        
    @property
    def dvals(self):
        if not hasattr(self, 'pitches'):
            self.pitches = self.pitch_comp.dvals
        if not hasattr(self, 'simzs'):
            self.simzs = self.simz_comp.dvals
            # Instrument 0=HRC-S 1=HRC-I 2=ACIS
            self.instrs = np.zeros(self.model.n_times, dtype=np.int8)
            for i, lims in enumerate(self.simz_lims):
                ok = (self.simzs > lims[0]) & (self.simzs <= lims[1])
                self.instrs[ok] = i

        # Interpolate power(pitch) for each instrument separately and make 2d stack
        n_p = len(self.P_pitches)
        heats = []
        for i in range(len(self.instr_names)):
            P_vals = self.parvals[i*n_p : (i+1)*n_p]
            heats.append(Ska.Numpy.interpolate(P_vals, self.P_pitches, self.pitches))
        self.heats = np.vstack(heats)

        # Now pick out the power(pitch) for the appropriate instrument at each time
        self._dvals = self.heats[self.instrs, np.arange(self.heats.shape[1])]
        return self._dvals

    def update(self):
        self.mvals = self.dvals

    def __str__(self):
        return 'psmc_solarheat__{0}'.format(self.node)
        

class AcisPsmcPower(PrecomputedHeatPower):
    """Heating from ACIS electronics (ACIS config dependent CCDs, FEPs etc)"""
    def __init__(self, model, node, k=1.0):
        ModelComponent.__init__(self, model)
        self.node = node
        self.k = k
        self.n_mvals = 1

    def __str__(self):
        return 'psmc__{0}'.format(self.node)

    @property
    def dvals(self):
        if not hasattr(self, '_dvals'):
            deav = self.model.fetch('1de28avo')
            deai = self.model.fetch('1deicacu')
            dpaav = self.model.fetch('1dp28avo')
            dpaai = self.model.fetch('1dpicacu')
            dpabv = self.model.fetch('1dp28bvo')
            dpabi = self.model.fetch('1dpicbcu')
            # maybe smooth? (already 5min telemetry, no need)
            self._dvals = self.k * (deav * deai + dpaav * dpaai + dpabv * dpabi)
        return self._dvals

    
class ProportialHeater(ActiveHeatPower):
    """Proportional heater (P = k * (T - T_set) for T > T_set)"""
    def __init__(self, model, name):
        ModelComponent.__init__(self, model, name)

        
class ThermostatHeater(ActiveHeatPower):
    """Thermostat heater (with configurable deadband)"""
    def __init__(self, model, name):
        ModelComponent.__init__(self, model, name)


class ThermalModel(object):
    def __init__(self, start='2011:115:00:00:00', stop='2011:115:01:00:00', dt=328.0,
                 model_spec=None, name='xijamod'):
        if model_spec:
            dt = model_spec['dt']
        self.name = name
        self.comp = OrderedDict()
        self.dt = dt
        self.dt_ksec = self.dt / 1000.
        self.times = self._eng_match_times(start, stop, dt)
        self.tstart = self.times[0]
        self.tstop = self.times[-1]
        self.ksecs = (self.times - self.tstart) / 1000.
        self.datestart = DateTime(self.tstart).date
        self.datestop = DateTime(self.tstop).date
        self.n_times = len(self.times)
        if model_spec:
            self._set_from_model_spec(model_spec)

    def _set_from_model_spec(self, model_spec):
        for comp in model_spec['comps']:
            ComponentClass = globals()[comp['class_name']]
            args = comp['init_args']
            kwargs = dict((str(k), v) for k, v in comp['init_kwargs'].items())
            self.add(ComponentClass, *args, **kwargs)
        self.make()
        parnames, parvals = zip(*model_spec['pars'])
        parnames = [str(x) for x in parnames]
        if self.parnames != parnames:
            raise ValueError('Model spec parnames do not match model: \n{0}\n{1}'.format(
                    parnames, self.parnames))
        self.parvals[:] = parvals

    @staticmethod
    def _eng_match_times(start, stop, dt):
        """Return an array of times between ``start`` and ``stop`` at ``dt`` sec
        intervals.  The times are roughly aligned (within 1 sec) to the timestamps
        in the '5min' (328 sec) Ska eng archive data.
        """
        time0 = 410270764.0
        i0 = int((DateTime(start).secs - time0) / dt) + 1
        i1 = int((DateTime(stop).secs - time0) / dt)
        return time0 + np.arange(i0, i1) * dt

    @property
    def pars(self):
        return dict((k, v) for k, v in zip(self.parnames, self.parvals))

    def fetch(self, msid):
        tpad = self.dt * 5
        datestart = DateTime(self.tstart - tpad).date
        datestop = DateTime(self.tstop + tpad).date
        logger.info('Fetching msid: %s over %s to %s' % (msid, datestart, datestop))
        tlm = fetch.MSID(msid, datestart, datestop, stat='5min', filter_bad=True)
        return Ska.Numpy.interpolate(tlm.vals, tlm.times, self.times, method='linear')

    def add(self, ComponentClass, *args, **kwargs):
        comp = ComponentClass(self, *args, **kwargs)
        # Store args and kwargs used to initialize object for later object
        # storage and re-creation
        comp.init_args = args
        comp.init_kwargs = kwargs
        self.comp[comp.name] = comp
        
        return comp

    comps = property(lambda self: self.comp.values())

    def get_comp(self, name):
        """Get a model component.  Works with either a string or a component object"""
        return self.comp[str(name)]

    def write(self, filename):
        """Write a full model specification for this model"""
        model_spec = dict(name=self.name,
                          comps=[],
                          pars=[(name, val) for name, val in zip(self.parnames, self.parvals)],
                          dt=self.dt,
                          tlm_code=None,
                          mval_names=[])
               
        stringify = lambda x: str(x) if isinstance(x, ModelComponent) else x
        for comp in self.comps:
            init_args = [stringify(x) for x in comp.init_args]
            init_kwargs = dict((k, stringify(v)) for k, v in comp.init_kwargs.items())
            model_spec['comps'].append(dict(class_name=comp.__class__.__name__,
                                            name=comp.name,
                                            init_args=init_args,
                                            init_kwargs=init_kwargs))
        with open(filename, 'w') as f:
            json.dump(model_spec, f, sort_keys=True, indent=4)

    def make(self):
        self.make_parvals()
        self.make_mvals()
        self.make_mults()
        self.make_heats()
        self.make_heatsinks()

    def get_parvals(self):
        if not hasattr(self, '_parvals'):
            self.make_parvals()

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
        self.parvals = np.zeros(n_parvals, dtype=np.float)
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
        """
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

    def calc(self):
        for comp in self.comps:
            comp.update()

        dt = self.dt_ksec * 2
        indexes = np.arange(0, self.n_times-2, 2)
        calc_model(indexes, dt, self.n_preds, self.mvals, self.parvals, self.mults,
                   self.heats, self.heatsinks)

    def calc_stat(self):
        self.calc()            # parvals already set with dummy_calc
        fit_stat = sum(comp.calc_stat() for comp in self.comps if comp.predict)
        return fit_stat

    def calc_staterror(self, data):
        return numpy.ones_like(data)
