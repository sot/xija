import numpy as np

try:
    from Ska.Matplotlib import plot_cxctime
except ImportError:
    pass

from .base import ModelComponent
from .heat import PrecomputedHeatPower
from . import tmal


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
            plot_cxctime(self.model.times, self.dvals, '-b', fig=fig, ax=ax)
            ax.grid()
            ax.set_title('{}: data (blue)'.format(self.name))
            ax.set_ylabel('Power (W)')
