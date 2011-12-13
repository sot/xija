import re
import operator
import numpy as np
from itertools import izip

from Chandra.Time import DateTime
import scipy.interpolate
import Ska.Numpy
from Ska.Matplotlib import plot_cxctime, cxctime2plotdate

from . import tmal


class Param(dict):
    """Model component parameter.  Inherits from dict but adds attribute access
    for convenience."""
    def __init__(self, comp_name, name, val, min=-1e38, max=1e38,
                 fmt="{:.4g}", frozen=False):
        dict.__init__(self)
        self.comp_name = comp_name
        self.name = name
        self.val = val
        self.min = min
        self.max = max
        self.fmt = fmt
        self.frozen = frozen
        self.full_name = comp_name + '__' + name

    def __setattr__(self, attr, val):
        dict.__setitem__(self, attr, val)

    def __getattr__(self, attr):
        return dict.__getitem__(self, attr)


class ModelComponent(object):
    """ Model component base class"""
    def __init__(self, model):
        self.model = model
        self.n_mvals = 0
        self.predict = False  # Predict values for this model component
        self.pars = []

    n_parvals = property(lambda self: len(self.parvals))
    times = property(lambda self: self.model.times)

    @staticmethod
    def get_par_func(index):
        def _func(self):
            return self.pars[index].val
        return _func

    @staticmethod
    def set_par_func(index):
        def _func(self, val):
            self.pars[index].val = val
        return _func

    def add_par(self, name, val=None, min=-1e38, max=1e38, fmt="{:.4g}",
                frozen=False):
        setattr(self.__class__, name,
                property(ModelComponent.get_par_func(self.n_parvals),
                         ModelComponent.set_par_func(self.n_parvals)))
        self.pars.append(Param(self.name, name, val, min=min, max=max,
                               fmt=fmt, frozen=frozen))

    def _set_mvals(self, vals):
        self.model.mvals[self.mvals_i, :] = vals

    def _get_mvals(self):
        return self.model.mvals[self.mvals_i, :]

    mvals = property(_get_mvals, _set_mvals)

    @property
    def name(self):
        return self.__str__()

    @property
    def parvals(self):
        return np.array([par.val for par in self.pars])

    @property
    def parnames(self):
        return [par.name for par in self.pars]

    def update(self):
        pass

    def set_data(self, data, times=None):
        self.data = data
        if times is not None:
            self.data_times = times

    @property
    def dvals(self):
        if not hasattr(self, '_dvals'):
            if self.data is None:
                dvals = self.get_dvals_tlm()
            elif isinstance(self.data, np.ndarray):
                dvals = self.model.interpolate_data(self.data, self.data_times,
                                                    str(self))
            elif isinstance(self.data, (int, long, float, bool, basestring)):
                if isinstance(self.data, basestring):
                    dtype = 'S{}'.format(len(self.data))
                else:
                    dtype = type(self.data)
                dvals = np.empty(self.model.n_times, dtype=dtype)
                dvals[:] = self.data
            else:
                raise ValueError("Data value '{}' for '{}' component "
                                 "not allowed ".format(self.data, self))
            self._dvals = dvals
        return self._dvals


class Mask(ModelComponent):
    """Create object with a ``mask`` attribute corresponding to
      node.dvals op val
    where op is a binary operator in operator module that returns a np mask
      "ge": >=
      "gt": >
      "le": <=
      "lt": <
      "eq": ==
      "ne" !=
    """
    def __init__(self, model, node, op, val, min_=-1e38, max_=1e38):
        ModelComponent.__init__(self, model)
        # Usually do self.node = model.get_comp(node) right away.  But here
        # allow for a forward reference to a not-yet-existent node and check
        # only when self.mask is actually used. This allows for masking in a
        # node based on data for that same node.
        self.node = node
        self.op = op
        self.val = val
        self.model = model
        self.add_par('val', val, min=min_, max=max_, frozen=True)
        self.mask_val = None

    @property
    def mask(self):
        if not isinstance(self.node, ModelComponent):
            self.node = self.model.get_comp(self.node)
        # cache latest version of mask
        if self.val != self.mask_val:
            self.mask_val = self.val
            self._mask = getattr(operator, self.op)(self.node.dvals, self.val)
        return self._mask

    def __str__(self):
        return "mask__{}_{}".format(self.node, self.op)

    def plot_data__time(self, fig, ax):
        lines = ax.get_lines()
        if not lines:
            plot_cxctime(self.model.times,
                         np.where(self.mask, 1, 0),
                         '-b', fig=fig, ax=ax)
            ax.grid()
            ax.set_ylim(-0.1, 1.1)
            ax.set_title('{}: data'.format(self.name))


class TelemData(ModelComponent):
    times = property(lambda self: self.model.times)

    def __init__(self, model, msid):
        super(TelemData, self).__init__(model)
        self.msid = msid
        self.n_mvals = 1
        self.predict = False
        self.data = None
        self.data_times = None

    def get_dvals_tlm(self):
        return self.model.fetch(self.msid)

    def plot_data__time(self, fig, ax):
        lines = ax.get_lines()
        if not lines:
            self.model_plotdate = cxctime2plotdate(self.model.times)
            plot_cxctime(self.model.times, self.dvals, '-b', fig=fig, ax=ax)
            ax.grid()
            ax.set_title('{}: data'.format(self.name))
        else:
            lines[0].set_data(self.model_plotdate, self.dvals)

    def __str__(self):
        return self.msid


class CmdStatesData(TelemData):
    def get_dvals_tlm(self):
        return self.model.cmd_states[self.msid]


class Node(TelemData):
    def __init__(self, model, msid, sigma=-10, quant=None,
                 predict=True, mask=None):
        TelemData.__init__(self, model, msid)
        self._sigma = sigma
        self.quant = quant
        self.predict = predict
        self.mask = model.get_comp(mask)

    @property
    def sigma(self):
        if self._sigma < 0:
            self._sigma = self.dvals.std() * (-self._sigma / 100.0)
        return self._sigma

    def calc_stat(self):
        if self.mask is None:
            resid = self.dvals - self.mvals
        else:
            resid = self.dvals[self.mask.mask] - self.mvals[self.mask.mask]
        return np.sum(resid**2 / self.sigma**2)

    def plot_data__time(self, fig, ax):
        lines = ax.get_lines()
        if not lines:
            self.model_plotdate = cxctime2plotdate(self.model.times)
            plot_cxctime(self.model.times, self.dvals, '-b', fig=fig, ax=ax)
            plot_cxctime(self.model.times, self.mvals, '-r', fig=fig, ax=ax)
            ax.grid()
            ax.set_title('{}: model (red) and data (blue)'.format(self.name))
            ax.set_ylabel('Temperature (degC)')
        else:
            lines[0].set_data(self.model_plotdate, self.dvals)
            lines[1].set_data(self.model_plotdate, self.mvals)

    def plot_resid__time(self, fig, ax):
        lines = ax.get_lines()
        resids = self.dvals - self.mvals
        if self.mask:
            resids[~self.mask.mask] = np.nan

        if not lines:
            self.model_plotdate = cxctime2plotdate(self.model.times)
            plot_cxctime(self.model.times, resids, '-b', fig=fig, ax=ax)
            ax.grid()
            ax.set_title('{}: residuals (data - model)'.format(self.name))
            ax.set_ylabel('Temperature (degC)')
        else:
            lines[0].set_data(self.model_plotdate, resids)


class Coupling(ModelComponent):
    """Couple two nodes together (one-way coupling)"""
    def __init__(self, model, node1, node2, tau):
        ModelComponent.__init__(self, model)
        self.node1 = self.model.get_comp(node1)
        self.node2 = self.model.get_comp(node2)
        self.add_par('tau', tau, min=2.0, max=200.0)

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
    def __init__(self, model, node, T=0.0, tau=20.0):
        ModelComponent.__init__(self, model)
        self.node = self.model.get_comp(node)
        self.add_par('T', T, min=-100.0, max=100.0)
        self.add_par('tau', tau, min=2.0, max=200.0)

    def update(self):
        self.tmal_ints = (tmal.OPCODES['heatsink'],
                          self.node.mvals_i)  # dy1/dt index
        self.tmal_floats = (self.T,
                            self.tau)

    def __str__(self):
        return 'heatsink__{0}'.format(self.node)


class HeatSinkRef(ModelComponent):
    """Fixed temperature external heat bath, reparameterized so that varying
    tau does not affect the mean model temperature.  This requires an extra
    non-fitted parameter T_ref which corresponds to a reference temperature for
    the node.::

      dT/dt = U * (Te - T)
            = P + U* (T_ref - T)   # reparameterization

      P = U * (Te - T_ref)
      Te = P / U + T_ref

    In code below, "T" corresponds to "Te" above.  The "T" above is node.dvals.
    """
    def __init__(self, model, node, T=0.0, tau=20.0, T_ref=20.0):
        ModelComponent.__init__(self, model)
        self.node = self.model.get_comp(node)
        self.add_par('P', (T - T_ref) / tau, min=-10.0, max=10.0)
        self.add_par('tau', tau, min=2.0, max=200.0)
        self.add_par('T_ref', T_ref, min=-100, max=100)

    def update(self):
        self.tmal_ints = (tmal.OPCODES['heatsink'],
                          self.node.mvals_i)  # dy1/dt index
        self.tmal_floats = (self.P * self.tau + self.T_ref,
                            self.tau)

    def __str__(self):
        return 'heatsink__{0}'.format(self.node)


class Pitch(TelemData):
    def __init__(self, model):
        TelemData.__init__(self, model, 'aosares1')

    def __str__(self):
        return 'pitch'


class Eclipse(TelemData):
    def __init__(self, model):
        TelemData.__init__(self, model, 'aoeclips')
        self.n_mvals = 1
        self.fetch_attr = 'midvals'
        self.fetch_method = 'nearest'

    def get_dvals_tlm(self):
        aoeclips = self.model.fetch(self.msid, 'vals', 'nearest')
        return aoeclips == 'ECL '

    def update(self):
        self.mvals = np.where(self.dvals, 1, 0)

    def __str__(self):
        return 'eclipse'


class SimZ(TelemData):
    def __init__(self, model):
        TelemData.__init__(self, model, 'sim_z')

    def get_dvals_tlm(self):
        sim_z_mm = self.model.fetch(self.msid)
        return np.rint(sim_z_mm * -397.7225924607)


class PrecomputedHeatPower(ModelComponent):
    """Component that provides static (precomputed) direct heat power input"""

    def update(self):
        self.mvals = self.dvals
        self.tmal_ints = (tmal.OPCODES['precomputed_heat'],
                           self.node.mvals_i,  # dy1/dt index
                           self.mvals_i,
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
        self.eclipse_comp = self.model.get_comp(eclipse_comp)

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

        self.epoch = epoch

        for pitch, power in zip(self.P_pitches, self.Ps):
            self.add_par('P_{0:.0f}'.format(float(pitch)), power, min=-10.0,
                         max=10.0)
        for pitch, dpower in zip(self.P_pitches, self.dPs):
            self.add_par('dP_{0:.0f}'.format(float(pitch)), dpower, min=-1.0,
                         max=1.0)
        self.add_par('tau', tau, min=1000., max=3000.)
        self.add_par('ampl', ampl, min=-1.0, max=1.0)
        self.add_par('bias', bias, min=-1.0, max=1.0)
        self.n_mvals = 1

    @property
    def dvals(self):
        if not hasattr(self, 'pitches'):
            self.pitches = self.pitch_comp.dvals
        if not hasattr(self, 't_days'):
            self.t_days = (self.pitch_comp.times
                           - DateTime(self.epoch).secs) / 86400.0
        if not hasattr(self, 't_phase'):
            time2000 = DateTime('2000:001:00:00:00').secs
            time2010 = DateTime('2010:001:00:00:00').secs
            secs_per_year = (time2010 - time2000) / 10.0
            t_year = (self.pitch_comp.times - time2000) / secs_per_year
            self.t_phase = t_year * 2 * np.pi

        Ps = self.parvals[0:self.n_pitches] + self.bias
        dPs = self.parvals[self.n_pitches:2 * self.n_pitches]
        Ps_interp = scipy.interpolate.interp1d(self.P_pitches, Ps,
                                               kind='linear')
        dPs_interp = scipy.interpolate.interp1d(self.P_pitches, dPs,
                                                kind='linear')
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

    def plot_solar_heat__pitch(self, fig, ax):
        Ps = self.parvals[0:self.n_pitches] + self.bias
        Ps_interp = scipy.interpolate.interp1d(self.P_pitches, Ps,
                                               kind='linear')
        # dPs = self.parvals[self.n_pitches:2*self.n_pitches]
        # dPs_interp = scipy.interpolate.interp1d(self.P_pitches, dPs,
        #                                        kind='linear')
        pitches = np.linspace(self.P_pitches[0], self.P_pitches[-1], 100)
        P_vals = Ps_interp(pitches)
        lines = ax.get_lines()
        if lines:
            lines[0].set_data(self.P_pitches, Ps)
            lines[1].set_data(pitches, P_vals)
        else:
            ax.plot(self.P_pitches, Ps, 'or', markersize=3)
            ax.plot(pitches, P_vals, '-b')
            ax.set_title('{} solar heat input'.format(self.node.name))
            ax.set_xlim(40, 180)
            ax.grid()


class DpaSolarHeat(SolarHeat):
    """Solar heating (pitch dependent)"""
    def __init__(self, model, node, simz_comp, pitch_comp, eclipse_comp=None,
                 P_pitches=None, Ps=None, dPs=None,
                 tau=1732.0, ampl=0.05, bias=0.0, epoch='2010:001',
                 hrc_bias=0.0):
        SolarHeat.__init__(self, model, node, pitch_comp, eclipse_comp,
                           P_pitches, Ps, dPs, tau, ampl, bias, epoch)
        self.simz_comp = model.get_comp(simz_comp)
        self.add_par('hrc_bias', hrc_bias, min=-1.0, max=1.0)

    @property
    def dvals(self):
        if not hasattr(self, 'pitches'):
            self.pitches = self.pitch_comp.dvals
        if not hasattr(self, 't_days'):
            self.t_days = (self.pitch_comp.times
                           - DateTime(self.epoch).secs) / 86400.0
        if not hasattr(self, 't_phase'):
            time2000 = DateTime('2000:001:00:00:00').secs
            time2010 = DateTime('2010:001:00:00:00').secs
            secs_per_year = (time2010 - time2000) / 10.0
            t_year = (self.pitch_comp.times - time2000) / secs_per_year
            self.t_phase = t_year * 2 * np.pi
        if not hasattr(self, 'hrc_mask'):
            self.hrc_mask = self.simz_comp.dvals < 0

        Ps = self.parvals[0:self.n_pitches] + self.bias
        dPs = self.parvals[self.n_pitches:2 * self.n_pitches]
        Ps_interp = scipy.interpolate.interp1d(self.P_pitches, Ps,
                                               kind='linear')
        dPs_interp = scipy.interpolate.interp1d(self.P_pitches, dPs,
                                                kind='linear')
        P_vals = Ps_interp(self.pitches)
        dP_vals = dPs_interp(self.pitches)
        self.P_vals = P_vals
        self._dvals = (P_vals + dP_vals * (1 - np.exp(-self.t_days / self.tau))
                       + self.ampl * np.cos(self.t_phase)).reshape(-1)

        # Set power to 0.0 during eclipse (where eclipse_comp.dvals == True)
        if self.eclipse_comp is not None:
            self._dvals[self.eclipse_comp.dvals] = 0.0

        # Apply a constant bias offset to power for times when SIM-Z is at HRC
        self._dvals[self.hrc_mask] += self.hrc_bias

        return self._dvals

    def __str__(self):
        return 'solarheat__{0}'.format(self.node)


class EarthHeat(PrecomputedHeatPower):
    """Earth heating of ACIS cold radiator (attitude, ephem dependent)"""
    def __init__(self, model, name):
        ModelComponent.__init__(self, model, name)


class AcisPsmcSolarHeat(PrecomputedHeatPower):
    """Solar heating of PSMC box.  This is dependent on SIM-Z"""
    def __init__(self, model, node, pitch_comp, simz_comp, P_pitches=None,
                 P_vals=None):
        ModelComponent.__init__(self, model)
        self.n_mvals = 1
        self.node = node
        self.pitch_comp = self.model.get_comp(pitch_comp)
        self.simz_comp = self.model.get_comp(simz_comp)
        self.P_pitches = np.array([50., 90., 150.] if (P_pitches is None)
                                  else P_pitches, dtype=np.float)
        self.simz_lims = ((-400000.0, -85000.0),  # HRC-S
                          (-85000.0, 0.0),        # HRC-I
                          (0.0, 400000.0))        # ACIS
        self.instr_names = ['hrcs', 'hrci', 'acis']
        for i, instr_name in enumerate(self.instr_names):
            for j, pitch in enumerate(self.P_pitches):
                self.add_par('P_{0}_{1:d}'.format(instr_name, int(pitch)),
                             P_vals[i, j], min=-10.0, max=10.0)

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

        # Interpolate power(pitch) for each instrument separately and make 2d
        # stack
        n_p = len(self.P_pitches)
        heats = []
        for i in range(len(self.instr_names)):
            P_vals = self.parvals[i * n_p:(i + 1) * n_p]
            heats.append(Ska.Numpy.interpolate(P_vals, self.P_pitches,
                                               self.pitches))
        self.heats = np.vstack(heats)

        # Now pick out the power(pitch) for the appropriate instrument at each
        # time
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
        self.add_par('k', k, min=0.0, max=2.0)

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
                           self.mvals_i,  # mvals with precomputed heat input
                          )
        self.tmal_floats = ()


class AcisDpaPower(PrecomputedHeatPower):
    """Heating from ACIS electronics (ACIS config dependent CCDs, FEPs etc)"""
    def __init__(self, model, node, k=1.0):
        ModelComponent.__init__(self, model)
        self.node = self.model.get_comp(node)
        self.add_par('k', k, min=0.0, max=2.0)
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
                           self.mvals_i,  # mvals with precomputed heat input
                          )
        self.tmal_floats = ()

    def plot_data__time(self, fig, ax):
        lines = ax.get_lines()
        if lines:
            lines[0].set_data(self.model_plotdate, self.dvals)
        else:
            self.model_plotdate = cxctime2plotdate(self.model.times)
            plot_cxctime(self.model.times, self.dvals, '-b', fig=fig, ax=ax)
            ax.grid()
            ax.set_title('{}: data (blue)'.format(self.name))
            ax.set_ylabel('Power (W)')


class AcisDpaPowerClipped(PrecomputedHeatPower):
    """Heating from ACIS electronics (ACIS config dependent CCDs, FEPs etc)"""
    def __init__(self, model, node, k=1.0):
        ModelComponent.__init__(self, model)
        self.node = self.model.get_comp(node)
        self.add_par('k', k, min=0.0, max=2.0)
        self.add_par('minpwr', k, min=0.0, max=100.0)
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
        clipped_power = np.clip(self.dvals, self.minpwr, 1e38)
        self.mvals = self.k * clipped_power / 10.0
        self.tmal_ints = (tmal.OPCODES['precomputed_heat'],
                           self.node.mvals_i,  # dy1/dt index
                           self.mvals_i,  # mvals with precomputed heat input
                          )
        self.tmal_floats = ()

    def plot_data__time(self, fig, ax):
        lines = ax.get_lines()
        if lines:
            lines[0].set_data(self.model_plotdate, self.dvals)
        else:
            self.model_plotdate = cxctime2plotdate(self.model.times)
            plot_cxctime(self.model.times, self.dvals, '-b', fig=fig, ax=ax)
            ax.grid()
            ax.set_title('{}: data (blue)'.format(self.name))
            ax.set_ylabel('Power (W)')


class AcisDeaPower(PrecomputedHeatPower):
    """Heating from ACIS DEA"""
    def __init__(self, model, node, k=1.0):
        ModelComponent.__init__(self, model)
        self.node = self.model.get_comp(node)
        self.add_par('k', k, min=0.0, max=2.0)
        self.n_mvals = 1

    def __str__(self):
        return 'dea__{0}'.format(self.node)

    @property
    def dvals(self):
        if not hasattr(self, '_dvals'):
            deav = self.model.fetch('1de28avo')
            deai = self.model.fetch('1deicacu')
            self._dvals = deav * deai
        return self._dvals

    def update(self):
        self.mvals = self.k * self.dvals / 10.0
        self.tmal_ints = (tmal.OPCODES['precomputed_heat'],
                           self.node.mvals_i,  # dy1/dt index
                           self.mvals_i,
                          )
        self.tmal_floats = ()

    def plot_data__time(self, fig, ax):
        lines = ax.get_lines()
        if lines:
            lines[0].set_data(self.model_plotdate, self.dvals)
        else:
            self.model_plotdate = cxctime2plotdate(self.model.times)
            plot_cxctime(self.model.times, self.dvals, '-b', fig=fig, ax=ax)
            ax.grid()
            ax.set_title('{}: data (blue)'.format(self.name))
            ax.set_ylabel('Power (W)')


class AcisDpaStatePower(PrecomputedHeatPower):
    """Heating from ACIS electronics (ACIS config dependent CCDs, FEPs etc).
    Use commanded states and assign an effective power for each "unique" power
    state.  See dpa/NOTES.power.
    """
    def __init__(self, model, node, mult=1.0,
                 fep_count=None, ccd_count=None,
                 vid_board=None, clocking=None):
        super(AcisDpaStatePower, self).__init__(model)
        self.node = self.model.get_comp(node)
        self.fep_count = self.model.get_comp(fep_count)
        self.ccd_count = self.model.get_comp(ccd_count)
        self.vid_board = self.model.get_comp(vid_board)
        self.clocking = self.model.get_comp(clocking)
        self.add_par('pow_0xxx', 21.5, min=10, max=60)
        self.add_par('pow_1xxx', 29.2, min=15, max=60)
        self.add_par('pow_2x1x', 39.1, min=20, max=80)
        self.add_par('pow_3xx0', 55.9, min=20, max=100)
        self.add_par('pow_3xx1', 47.9, min=20, max=100)
        self.add_par('pow_4xxx', 57.0, min=20, max=120)
        self.add_par('pow_5xxx', 66.5, min=20, max=120)
        self.add_par('pow_66x0', 73.7, min=20, max=140)
        self.add_par('pow_6611', 76.5, min=20, max=140)
        self.add_par('pow_6xxx', 75.0, min=20, max=140)
        self.add_par('mult', mult, min=0.0, max=2.0)
        self.add_par('bias', 70, min=10, max=100)

        self.power_pars = [par for par in self.pars
                           if par.name.startswith('pow_')]
        self.n_mvals = 1
        self.data = None
        self.data_times = None

    def __str__(self):
        return 'dpa_power'

    @property
    def par_idxs(self):
        if not hasattr(self, '_par_idxs'):
            par_idxs = []
            # Make a regex corresponding to the last bit of each power
            # parameter name.  E.g. "pow_1xxx" => "1...".
            power_par_res = [par.name[4:].replace('x', '.')
                             for par in self.power_pars]
            for dpa_attrs in izip(self.fep_count.dvals,
                                  self.ccd_count.dvals,
                                  self.vid_board.dvals,
                                  self.clocking.dvals):
                for i, power_par_re in enumerate(power_par_res):
                    state_str = "{}{}{}{}".format(*dpa_attrs)
                    if re.match(power_par_re, state_str):
                        par_idxs.append(i)
                        break
                else:
                    raise ValueError('Error - no match for power state {}'
                                     .format(state_str))

            self._par_idxs = np.array(par_idxs)

        return self._par_idxs

    def get_dvals_tlm(self):
        """Model dvals is set to the telemetered power.  This is not actually
        used by the model, but is useful for diagnostics.
        """
        try:
            dvals = self.model.fetch('dp_dpa_power')
        except ValueError:
            dvals = np.zeros_like(self.model.times)
        return dvals

    def update(self):
        """Update the model prediction as a precomputed heat.  Make an array of
        the current power parameters, then slice that with self.par_idxs to
        generate the predicted power (based on the parameter specifying state
        power) at each time step.
        """
        power_parvals = np.array([par.val for par in self.power_pars])
        powers = power_parvals[self.par_idxs]
        self.mvals = self.mult / 100. * (powers - self.bias)
        self.tmal_ints = (tmal.OPCODES['precomputed_heat'],
                           self.node.mvals_i,  # dy1/dt index
                           self.mvals_i,
                          )
        self.tmal_floats = ()

    def plot_data__time(self, fig, ax):
        powers = self.mvals * 100. / self.mult + self.bias
        lines = ax.get_lines()
        if lines:
            lines[0].set_data(self.model_plotdate, self.dvals)
            lines[1].set_data(self.model_plotdate, powers)
        else:
            self.model_plotdate = cxctime2plotdate(self.model.times)
            plot_cxctime(self.model.times, self.dvals, '-b', fig=fig, ax=ax)
            plot_cxctime(self.model.times, powers, '-r', fig=fig, ax=ax)
            ax.grid()
            ax.set_title('{}: data (blue)'.format(self.name))
            ax.set_ylabel('Power (W)')


class AcisDpaPower6(PrecomputedHeatPower):
    """Heating from ACIS electronics (ACIS config dependent CCDs, FEPs etc)"""
    def __init__(self, model, node, k=1.0, dp611=0.0):
        ModelComponent.__init__(self, model)
        self.node = self.model.get_comp(node)
        self.add_par('k', k, min=0.0, max=2.0)
        self.add_par('dp611', dp611, min=-100.0, max=100.0)
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
                           self.mvals_i,
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
