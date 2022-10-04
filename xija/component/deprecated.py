# Licensed under a 3-clause BSD style license - see LICENSE.rst

from xija import tmal
from xija.component.base import ModelComponent
from xija.component.heat import SolarHeat, PrecomputedHeatPower

import numpy as np

try:
    from Ska.Matplotlib import plot_cxctime
except ImportError:
    pass


class SolarHeatAcisCameraBody(SolarHeat):
    """Solar heating (pitch and SIM-Z dependent)

    Parameters
    ----------
    model :
        parent model
    node :
        node which is coupled to solar heat
    simz_comp :
        SimZ component
    pitch_comp :
        solar Pitch component
    eclipse_comp :
        Eclipse component (optional)
    P_pitches :
        list of pitch values (default=[45, 65, 90, 130, 180])
    Ps :
        list of solar heating values (default=[1.0, ...])
    dPs :
        list of delta heating values (default=[0.0, ...])
    var_func :
        variability function ('exp' | 'linear')
    tau :
        variability timescale (days)
    ampl :
        ampl of annual sinusoidal heating variation
    bias :
        constant offset to all solar heating values
    epoch :
        reference date at which ``Ps`` values apply
    dh_heater_comp :
        detector housing heater status (True = On)
    dh_heater_bias :
        bias power when DH heater is on
    dP_pitches :
        list of pitch values for dP (default=``P_pitches``)
    """
    def __init__(self, model, node, pitch_comp='pitch', eclipse_comp=None,
                 P_pitches=None, Ps=None, dPs=None, var_func='exp',
                 tau=1732.0, ampl=0.05, bias=0.0, epoch='2010:001:12:00:00',
                 dh_heater_comp='dh_heater', dh_heater_bias=0.0, dP_pitches=None):

        super().__init__(
            model, node, pitch_comp=pitch_comp, eclipse_comp=eclipse_comp,
            P_pitches=P_pitches, Ps=Ps, dPs=dPs, var_func=var_func,
            tau=tau, ampl=ampl, bias=bias, epoch=epoch, dP_pitches=dP_pitches)

        self.dh_heater_comp = model.get_comp(dh_heater_comp)
        self.add_par('dh_heater_bias', dh_heater_bias, min=-1.0, max=1.0)

    def dvals_post_hook(self):
        """Apply a bias power offset when detector housing heater is on"""
        self._dvals[self.dh_heater_comp.dvals] += self.dh_heater_bias


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

        Parameters
        ----------

        Returns
        -------

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
            plot_cxctime(self.model.times, self.dvals, '#386cb0', fig=fig, ax=ax)
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
            plot_cxctime(self.model.times, self.dvals, '#386cb0', fig=fig, ax=ax)
            ax.grid()
            ax.set_title('{}: data (blue)'.format(self.name))
            ax.set_ylabel('Power (W)')



