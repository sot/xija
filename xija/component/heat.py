import re
from itertools import izip, count
import glob
import os

import numpy as np
import scipy.interpolate

try:
    import Ska.Numpy
    from Ska.Matplotlib import plot_cxctime
    from Chandra.Time import DateTime
except ImportError:
    pass

from .base import ModelComponent
from .. import tmal


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
    """Solar heating (pitch dependent)

    :param model: parent model
    :param node: node which is coupled to solar heat
    :param pitch_comp: solar Pitch component
    :param eclipse_comp: Eclipse component (optional)
    :param P_pitches: list of pitch values (default=[45, 65, 90, 130, 180])
    :param Ps: list of solar heating values (default=[1.0, ...])
    :param dPs: list of delta heating values (default=[0.0, ...])
    :param var_func: variability function ('exp' | 'linear')
    :param tau: variability timescale (days)
    :param ampl: ampl of annual sinusoidal heating variation
    :param bias: constant offset to all solar heating values
    :param epoch: reference date at which ``Ps`` values apply
    """
    def __init__(self, model, node, pitch_comp, eclipse_comp=None,
                 P_pitches=None, Ps=None, dPs=None, var_func='exp',
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
        self.var_func = getattr(self, var_func)

    @staticmethod
    def linear(days, k_inv):
        return days / k_inv

    @staticmethod
    def exp(days, tau):
        return 1 - np.exp(-days / tau)

    def dvals_post_hook(self):
        """Override this method to adjust self._dvals after main computation.
        """
        pass

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
        self._dvals = (P_vals + dP_vals * self.var_func(self.t_days, self.tau)
                       + self.ampl * np.cos(self.t_phase)).reshape(-1)
        # Set power to 0.0 during eclipse (where eclipse_comp.dvals == True)
        if self.eclipse_comp is not None:
            self._dvals[self.eclipse_comp.dvals] = 0.0

        # Allow for customization in SolarHeat subclasses
        self.dvals_post_hook()

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


class SolarHeatHrc(SolarHeat):
    """Solar heating (pitch and SIM-Z dependent)

    :param model: parent model
    :param node: node which is coupled to solar heat
    :param simz_comp: SimZ component
    :param pitch_comp: solar Pitch component
    :param eclipse_comp: Eclipse component (optional)
    :param P_pitches: list of pitch values (default=[45, 65, 90, 130, 180])
    :param Ps: list of solar heating values (default=[1.0, ...])
    :param dPs: list of delta heating values (default=[0.0, ...])
    :param var_func: variability function ('exp' | 'linear')
    :param tau: variability timescale (days)
    :param ampl: ampl of annual sinusoidal heating variation
    :param bias: constant offset to all solar heating values
    :param epoch: reference date at which ``Ps`` values apply
    :param hrc_bias: solar heating bias when SIM-Z < 0 (HRC)
    """
    def __init__(self, model, node, simz_comp, pitch_comp, eclipse_comp=None,
                 P_pitches=None, Ps=None, dPs=None, var_func='exp',
                 tau=1732.0, ampl=0.05, bias=0.0, epoch='2010:001',
                 hrc_bias=0.0):
        SolarHeat.__init__(self, model, node, pitch_comp, eclipse_comp,
                           P_pitches, Ps, dPs, var_func, tau, ampl, bias,
                           epoch)
        self.simz_comp = model.get_comp(simz_comp)
        self.add_par('hrc_bias', hrc_bias, min=-1.0, max=1.0)

    def dvals_post_hook(self):
        """Apply a bias power offset when SIM-Z is at HRC-S or HRC-I.
        """
        if not hasattr(self, 'hrc_mask'):
            self.hrc_mask = self.simz_comp.dvals < 0
        self._dvals[self.hrc_mask] += self.hrc_bias

# For back compatibility prior to Xija 0.2
DpaSolarHeat = SolarHeatHrc


class EarthHeat(PrecomputedHeatPower):
    """Earth heating of ACIS cold radiator (attitude, ephem dependent)"""
    def __init__(self, model, node,
                 orbitephem0_x, orbitephem0_y, orbitephem0_z,
                 aoattqt1, aoattqt2, aoattqt3, aoattqt4,
                 k=1.0):
        ModelComponent.__init__(self, model)
        self.node = self.model.get_comp(node)
        self.orbitephem0_x = self.model.get_comp(orbitephem0_x)
        self.orbitephem0_y = self.model.get_comp(orbitephem0_y)
        self.orbitephem0_z = self.model.get_comp(orbitephem0_z)
        self.aoattqt1 = self.model.get_comp(aoattqt1)
        self.aoattqt2 = self.model.get_comp(aoattqt2)
        self.aoattqt3 = self.model.get_comp(aoattqt3)
        self.aoattqt4 = self.model.get_comp(aoattqt4)
        self.n_mvals = 1
        self.add_par('k', k, min=0.0, max=2.0)

    @property
    def dvals(self):
        import Chandra.taco
        if not hasattr(self, '_dvals') and not self.get_cached():
            # Collect individual MSIDs for use in calc_earth_vis()
            ephem_xyzs = [getattr(self, 'orbitephem0_{}'.format(x))
                          for x in ('x', 'y', 'z')]
            aoattqt_1234s = [getattr(self, 'aoattqt{}'.format(x))
                             for x in range(1, 5)]
            ephems = np.array([x.dvals for x in ephem_xyzs]).transpose()
            q_atts = np.array([x.dvals for x in aoattqt_1234s]).transpose()

            self._dvals = np.empty(self.model.n_times, dtype=float)
            for i, ephem, q_att in izip(count(), ephems, q_atts):
                q_norm = np.sqrt(np.sum(q_att ** 2))
                if q_norm < 0.9:
                    print "Bad quaternion", i
                    q_att = np.array([0.0, 0.0, 0.0, 1.0])
                else:
                    q_att = q_att / q_norm
                _, illums, _ = Chandra.taco.calc_earth_vis(ephem, q_att)
                self._dvals[i] = illums.sum()

            self.put_cache()

        return self._dvals

    def put_cache(self):
        if os.path.exists('esa_cache'):
            cachefile = 'esa_cache/{}-{}.npz'.format(
                self.model.datestart, self.model.datestop)
            np.savez(cachefile, times=self.model.times,
                     dvals=self.dvals)

    def get_cached(self):
        """Find a cached version of the Earth solid angle values from
        file if possible.
        """
        dts = {}  # delta times for each matching file
        filenames = glob.glob('esa_cache/*.npz')
        for name in filenames:
            re_date = r'\d\d\d\d:\d\d\d:\d\d:\d\d:\d\d\.\d\d\d'
            re_cache_file = r'({})-({})'.format(re_date, re_date)
            m = re.search(re_cache_file, name)
            if m:
                f_datestart, f_datestop = m.groups()
                if (f_datestart <= self.model.datestart and
                    f_datestop >= self.model.datestop):
                    dts[name] = DateTime(f_datestop) - DateTime(f_datestart)
        if dts:
            cachefile = sorted(dts.items(), key=lambda x: x[1])[0][0]
            arrays = np.load(cachefile)
            self._dvals = self.model.interpolate_data(
                arrays['dvals'], arrays['times'], comp=self)
            return True
        else:
            return False

    def update(self):
        self.mvals = self.k * self.dvals
        self.tmal_ints = (tmal.OPCODES['precomputed_heat'],
                          self.node.mvals_i,  # dy1/dt index
                          self.mvals_i,  # mvals with precomputed heat input
                          )
        self.tmal_floats = ()

    def plot_data__time(self, fig, ax):
        lines = ax.get_lines()
        if not lines:
            plot_cxctime(self.model.times, self.dvals, '-b', fig=fig, ax=ax)
            ax.grid()
            ax.set_title('{}: data (blue)'.format(self.name))
            ax.set_ylabel('Illumination')
        else:
            lines[0].set_data(self.model_plotdate, self.dvals)

    def __str__(self):
        return 'earthheat__{0}'.format(self.node)


class AcisPsmcSolarHeat(PrecomputedHeatPower):
    """Solar heating of PSMC box.  This is dependent on SIM-Z"""
    def __init__(self, model, node, pitch_comp, simz_comp, P_pitches=None,
                 P_vals=None):
        ModelComponent.__init__(self, model)
        self.n_mvals = 1
        self.node = self.model.get_comp(node)
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
                             P_vals[i][j], min=-10.0, max=10.0)

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
        self.add_par('bias', 70.0, min=0.0, max=100.0)
        self.n_mvals = 1

    def __str__(self):
        return 'dpa__{0}'.format(self.node)

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
        self.mvals = self.k * (self.dvals - self.bias) / 10.0
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
        self.add_par('pow_2xxx', 39.1, min=20, max=80)
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
            # Make a regex corresponding to the last bit of each power
            # parameter name.  E.g. "pow_1xxx" => "1...".
            power_par_res = [par.name[4:].replace('x', '.')
                             for par in self.power_pars]

            par_idxs = np.zeros(6612, dtype=np.int) - 1
            for fep_count in range(0, 7):
                for ccd_count in range(0, 7):
                    for vid_board in range(0, 2):
                        for clocking in range(0, 2):
                            state = "{}{}{}{}".format(fep_count, ccd_count,
                                                      vid_board, clocking)
                            idx = int(state)
                            for i, power_par_re in enumerate(power_par_res):
                                if re.match(power_par_re, state):
                                    par_idxs[idx] = i
                                    break
                            else:
                                raise ValueError('No match for power state {}'
                                                 .format(state))

            idxs = (self.fep_count.dvals * 1000 + self.ccd_count.dvals * 100 +
                    self.vid_board.dvals * 10 + self.clocking.dvals)
            self._par_idxs = par_idxs[idxs]

            if self._par_idxs.min() < 0:
                raise ValueError('Fatal problem with par_idxs routine')

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
            plot_cxctime(self.model.times, self.dvals, '-b', fig=fig, ax=ax)
            plot_cxctime(self.model.times, powers, '-r', fig=fig, ax=ax)
            ax.grid()
            ax.set_title('{}: data (blue)'.format(self.name))
            ax.set_ylabel('Power (W)')


class PropHeater(PrecomputedHeatPower):
    """Proportional heater (P = k * (T_set - T) for T < T_set)."""
    def __init__(self, model, node, node_control=None, k=0.1, T_set=20.0):
        super(PropHeater, self).__init__(model)
        self.node = self.model.get_comp(node)
        self.node_control = (self.node if node_control is None
                             else self.model.get_comp(node_control))
        self.add_par('k', k, min=0.0, max=1.0)
        self.add_par('T_set', T_set, min=-50.0, max=100.0)
        self.n_mvals = 1

    def __str__(self):
        return 'prop_heat__{0}'.format(self.node)

    def get_dvals_tlm(self):
        """Return an array of zeros => no activation of the heater.
        """
        return np.zeros_like(self.model.times)

    def update(self):
        self.tmal_ints = (tmal.OPCODES['proportional_heater'],
                          self.node.mvals_i,  # dy1/dt index
                          self.node_control.mvals_i,
                          self.mvals_i
                          )
        self.tmal_floats = (self.T_set, self.k)

    def plot_data__time(self, fig, ax):
        lines = ax.get_lines()
        if lines:
            lines[0].set_data(self.model_plotdate, self.mvals)
        else:
            plot_cxctime(self.model.times, self.mvals, '-b', fig=fig, ax=ax)
            ax.grid()
            ax.set_title('{}: data (blue)'.format(self.name))
            ax.set_ylabel('Power')


class ThermostatHeater(ActiveHeatPower):
    """Thermostat heater (no deadband): heat = P for T < T_set).
    """
    def __init__(self, model, node, node_control=None, P=0.1, T_set=20.0):
        super(ThermostatHeater, self).__init__(model)
        self.node = self.model.get_comp(node)
        self.node_control = (self.node if node_control is None
                             else self.model.get_comp(node_control))
        self.add_par('P', P, min=0.0, max=1.0)
        self.add_par('T_set', T_set, min=-50.0, max=100.0)
        self.n_mvals = 1

    def __str__(self):
        return 'thermostat_heat__{0}'.format(self.node)

    def get_dvals_tlm(self):
        """Return an array of zeros => no activation of the heater.
        """
        return np.zeros_like(self.model.times)

    def update(self):
        self.tmal_ints = (tmal.OPCODES['thermostat_heater'],
                          self.node.mvals_i,  # dy1/dt index
                          self.node_control.mvals_i,
                          self.mvals_i
                          )
        self.tmal_floats = (self.T_set, self.P)

    def plot_data__time(self, fig, ax):
        lines = ax.get_lines()
        if lines:
            lines[0].set_data(self.model_plotdate, self.mvals)
        else:
            plot_cxctime(self.model.times, self.mvals, '-b', fig=fig, ax=ax)
            ax.grid()
            ax.set_title('{}: data (blue)'.format(self.name))
            ax.set_ylabel('Power')
