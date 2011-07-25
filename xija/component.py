import numpy as np
from Chandra.Time import DateTime
import scipy.interpolate
import Ska.DBI
import Chandra.cmd_states
import Ska.Numpy
from Ska.Matplotlib import plot_cxctime

from . import tmal

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
    
    def plot_data_model(self, fig, ax):
        plot_cxctime(self.model.times, self.dvals, '-b', fig=fig, ax=ax)
        plot_cxctime(self.model.times, self.mvals, '-r', fig=fig, ax=ax)
        ax.grid()
        ax.set_title('{}: model (red) and data (blue)'.format(self.name))
        ax.set_ylabel('Temperature (degC)')

class Coupling(ModelComponent):
    """Couple two nodes together (one-way coupling)"""
    def __init__(self, model, node1, node2, tau):
        ModelComponent.__init__(self, model)
        self.node1 = self.model.get_comp(node1)
        self.node2 = self.model.get_comp(node2)
        self.add_par('tau', tau)

    def update(self):
        self.tmal_ints = (tmal.OPCODES['coupling'],
                          self.node1.mvals_i,  # y1 index
                          self.node2.mvals_i   # y2 index
                          )
        self.tmal_floats = (self.tau,)

    def __str__(self):
        return 'coupling__{0}__{1}'.format(self.node1, self.node2)


class HeatSink(ModelComponent):
    """Fixed temperature external heat bath"""
    def __init__(self, model, node, T, tau):
        ModelComponent.__init__(self, model)
        self.add_par('T', T)
        self.add_par('tau', tau)
        self.node = self.model.get_comp(node)

    def update(self):
        self.tmal_ints = (tmal.OPCODES['heatsink'],
                          self.node.mvals_i)  # dy1/dt index
        self.tmal_floats = (self.T,
                            self.tau)

    def __str__(self):
        return 'heatsink__{0}'.format(self.node)


class Pitch(TelemData):
    def __init__(self, model, data=None):
        TelemData.__init__(self, model, 'aosares1', 'pitch', data)
    
    def __str__(self):
        return 'pitch'


class Eclipse(TelemData):
    def __init__(self, model, data=None):
        TelemData.__init__(self, model, 'aoeclips', 'eclipse', data)
        self.n_mvals = 1
    
    @property
    def dvals(self):
        if not hasattr(self, '_dvals'):
            if self.data is None:
                aoeclips = self.model.fetch(self.msid, 'vals', 'nearest')
                self._dvals = aoeclips == 'ECL '
            elif self.data == 'cmd':
                raise NotImplementedError()
                #self._dvals = self.get_dvals_cmd()
            else:
                raise NotImplementedError()
                #self._dvals = self.data
        return self._dvals

    def update(self):
        self.mvals = np.where(self.dvals, 1, 0)

    def __str__(self):
        return 'eclipse'


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

    def update(self):
        self.mvals = self.dvals
        self.tmal_ints = (tmal.OPCODES['precomputed_heat'],
                           self.node.mvals_i,  # dy1/dt index
                           self.mvals_i,       # mvals row with precomputed heat input
                          )
        self.tmal_floats = ()


class ActiveHeatPower(ModelComponent):
    """Component that provides active heat power input which depends on
    current or past computed model values"""
    pass


class SolarHeat(PrecomputedHeatPower):
    """Solar heating (pitch dependent)"""
    def __init__(self, model, node, pitch_comp, eclipse_comp=None,
                 P_pitches=None, Ps=None, dPs=None,
                 tau=1732.0, ampl=0.05, bias=0.0, epoch='2010:001'):
        ModelComponent.__init__(self, model)
        self.node = self.model.get_comp(node)
        self.pitch_comp = self.model.get_comp(pitch_comp)
        self.eclipse_comp = (None if eclipse_comp is None
                             else self.model.get_comp(eclipse_comp))

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
        self.add_par('bias', bias)
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

        Ps = self.parvals[0:self.n_pitches] + self.bias
        dPs = self.parvals[self.n_pitches:2*self.n_pitches]
        Ps_interp = scipy.interpolate.interp1d(self.P_pitches, Ps, kind='cubic')
        dPs_interp = scipy.interpolate.interp1d(self.P_pitches, dPs, kind='cubic')
        P_vals = Ps_interp(self.pitches)
        dP_vals = dPs_interp(self.pitches)
        self.P_vals = P_vals
        self._dvals = (P_vals + dP_vals * (1 - np.exp(-self.t_days / self.tau))
                       + self.ampl * np.cos(self.t_phase)).reshape(-1)
        # Set power to 0.0 during eclipse (where eclipse_comp.dvals == True)
        if self.eclipse_comp is not None:
            self._dvals[self.eclipse_comp.dvals] = 0.0
        return self._dvals

    def __str__(self):
        return 'solarheat__{0}'.format(self.node)

    def plot_solar_heat(self, fig, ax):
        Ps = self.parvals[0:self.n_pitches] + self.bias
        Ps_interp = scipy.interpolate.interp1d(self.P_pitches, Ps, kind='cubic')
        # dPs = self.parvals[self.n_pitches:2*self.n_pitches]
        # dPs_interp = scipy.interpolate.interp1d(self.P_pitches, dPs, kind='cubic')
        pitches = np.linspace(self.P_pitches[0], self.P_pitches[-1], 100)
        P_vals = Ps_interp(pitches)
        ax.plot(self.P_pitches, Ps, 'or', markersize=3)
        ax.plot(pitches, P_vals, '-b')
        ax.set_title('{} solar heat input'.format(self.node.name))
        ax.set_xlim(40, 180)
        ax.grid()
        

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
        self.P_pitches = np.array([50., 90., 150.] if (P_pitches is None) else P_pitches,
                                  dtype=np.float)
        self.simz_lims = ((-400000.0, -85000.0),  # HRC-S
                          (-85000.0, 0.0),        # HRC-I
                          (0.0, 400000.0))        # ACIS
        self.instr_names = ['hrcs', 'hrci', 'acis']
        for i, instr_name in enumerate(self.instr_names):
            for j, pitch in enumerate(self.P_pitches):
                self.add_par('P_{0}_{1:d}'.format(instr_name, int(pitch)), P_vals[i, j])
        
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

    def __str__(self):
        return 'psmc_solarheat__{0}'.format(self.node)
        

class AcisPsmcPower(PrecomputedHeatPower):
    """Heating from ACIS electronics (ACIS config dependent CCDs, FEPs etc)"""
    def __init__(self, model, node, k=1.0):
        ModelComponent.__init__(self, model)
        self.node = self.model.get_comp(node)
        self.n_mvals = 1
        self.add_par('k', k)

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
            self._dvals = deav * deai + dpaav * dpaai + dpabv * dpabi
        return self._dvals

    def update(self):
        self.mvals = self.k * self.dvals
        self.tmal_ints = (tmal.OPCODES['precomputed_heat'],
                           self.node.mvals_i,  # dy1/dt index
                           self.mvals_i,       # mvals row with precomputed heat input
                          )
        self.tmal_floats = ()
    
    
class AcisDpaPower(PrecomputedHeatPower):
    """Heating from ACIS electronics (ACIS config dependent CCDs, FEPs etc)"""
    def __init__(self, model, node, k=1.0):
        ModelComponent.__init__(self, model)
        self.node = self.model.get_comp(node)
        self.add_par('k', k)
        self.n_mvals = 1

    def __str__(self):
        return 'dpa__{0}'.format(self.node)

    @property
    def dvals(self):
        if not hasattr(self, '_dvals'):
            dpaav = self.model.fetch('1dp28avo')
            dpaai = self.model.fetch('1dpicacu')
            dpabv = self.model.fetch('1dp28bvo')
            dpabi = self.model.fetch('1dpicbcu')
            # maybe smooth? (already 5min telemetry, no need)
            self._dvals = dpaav * dpaai + dpabv * dpabi
        return self._dvals

    def update(self):
        self.mvals = self.k * self.dvals / 10.0
        self.tmal_ints = (tmal.OPCODES['precomputed_heat'],
                           self.node.mvals_i,  # dy1/dt index
                           self.mvals_i,       # mvals row with precomputed heat input
                          )
        self.tmal_floats = ()
    
class AcisDpaPower6(PrecomputedHeatPower):
    """Heating from ACIS electronics (ACIS config dependent CCDs, FEPs etc)"""
    def __init__(self, model, node, k=1.0, dp611=0.0):
        ModelComponent.__init__(self, model)
        self.node = self.model.get_comp(node)
        self.add_par('k', k)
        self.add_par('dp611', dp611)
        self.n_mvals = 1

    def __str__(self):
        return 'dpa6__{0}'.format(self.node)

    @property
    def dvals(self):
        if not hasattr(self, '_dvals'):
            dpaav = self.model.fetch('1dp28avo')
            dpaai = self.model.fetch('1dpicacu')
            dpabv = self.model.fetch('1dp28bvo')
            dpabi = self.model.fetch('1dpicbcu')
            states = self.model.cmd_states
            self.mask611 = ((states['fep_count'] == 6) &
                            (states['clocking'] == 1) &
                            (states['vid_board'] == 1))

            self._dvals = dpaav * dpaai + dpabv * dpabi
        return self._dvals

    def update(self):
        self.mvals = self.k * self.dvals / 10.0
        self.mvals[self.mask611] += self.dp611
        self.tmal_ints = (tmal.OPCODES['precomputed_heat'],
                           self.node.mvals_i,  # dy1/dt index
                           self.mvals_i,       # mvals row with precomputed heat input
                          )
        self.tmal_floats = ()
    
class ProportialHeater(ActiveHeatPower):
    """Proportional heater (P = k * (T - T_set) for T > T_set)"""
    def __init__(self, model, name):
        ModelComponent.__init__(self, model, name)

        
class ThermostatHeater(ActiveHeatPower):
    """Thermostat heater (with configurable deadband)"""
    def __init__(self, model, name):
        ModelComponent.__init__(self, model, name)

