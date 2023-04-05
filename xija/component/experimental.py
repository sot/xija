# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np

try:
    from Ska.Matplotlib import plot_cxctime
except ImportError:
    pass

from . import tmal
from .base import ModelComponent
from .heat import PrecomputedHeatPower


class AcisDpaPower6(PrecomputedHeatPower):
    """Heating from ACIS electronics (ACIS config dependent CCDs, FEPs etc)"""

    def __init__(self, model, node, k=1.0, dp611=0.0):
        ModelComponent.__init__(self, model)
        self.node = self.model.get_comp(node)
        self.add_par("k", k, min=0.0, max=2.0)
        self.add_par("dp611", dp611, min=-100.0, max=100.0)
        self.n_mvals = 1

    def __str__(self):
        return "dpa6__{0}".format(self.node)

    @property
    def dvals(self):
        if not hasattr(self, "_dvals"):
            dpaav = self.model.fetch("1dp28avo")
            dpaai = self.model.fetch("1dpicacu")
            dpabv = self.model.fetch("1dp28bvo")
            dpabi = self.model.fetch("1dpicbcu")
            states = self.model.cmd_states
            self.mask611 = (
                (states["fep_count"] == 6)
                & (states["clocking"] == 1)
                & (states["vid_board"] == 1)
            )

            self._dvals = dpaav * dpaai + dpabv * dpabi
        return self._dvals

    def update(self):
        self.mvals = self.k * self.dvals / 10.0
        self.mvals[self.mask611] += self.dp611
        self.tmal_ints = (
            tmal.OPCODES["precomputed_heat"],
            self.node.mvals_i,  # dy1/dt index
            self.mvals_i,
        )
        self.tmal_floats = ()


class AcisDpaPowerClipped(PrecomputedHeatPower):
    """Heating from ACIS electronics (ACIS config dependent CCDs, FEPs etc)"""

    def __init__(self, model, node, k=1.0):
        ModelComponent.__init__(self, model)
        self.node = self.model.get_comp(node)
        self.add_par("k", k, min=0.0, max=2.0)
        self.add_par("minpwr", k, min=0.0, max=100.0)
        self.n_mvals = 1

    def __str__(self):
        return "dpa__{0}".format(self.node)

    @property
    def dvals(self):
        if not hasattr(self, "_dvals"):
            dpaav = self.model.fetch("1dp28avo")
            dpaai = self.model.fetch("1dpicacu")
            dpabv = self.model.fetch("1dp28bvo")
            dpabi = self.model.fetch("1dpicbcu")
            # maybe smooth? (already 5min telemetry, no need)
            self._dvals = dpaav * dpaai + dpabv * dpabi
        return self._dvals

    def update(self):
        clipped_power = np.clip(self.dvals, self.minpwr, 1e38)
        self.mvals = self.k * clipped_power / 10.0
        self.tmal_ints = (
            tmal.OPCODES["precomputed_heat"],
            self.node.mvals_i,  # dy1/dt index
            self.mvals_i,  # mvals with precomputed heat input
        )
        self.tmal_floats = ()

    def plot_data__time(self, fig, ax):
        lines = ax.get_lines()
        if lines:
            lines[0].set_data(self.model_plotdate, self.dvals)
        else:
            plot_cxctime(self.model.times, self.dvals, "-b", fig=fig, ax=ax)
            ax.grid()
            ax.set_title("{}: data (blue)".format(self.name))
            ax.set_ylabel("Power (W)")


class SolarHeatSimZ(SolarHeat):
    """Solar heating (pitch and SimZ dependent)"""

    def __init__(
        self,
        model,
        node,
        simz_comp,
        pitch_comp,
        eclipse_comp=None,
        P_pitches=None,
        Ps=None,
        dPs=None,
        var_func="exp",
        tau=1732.0,
        ampl=0.05,
        bias=0.0,
        epoch="2010:001:12:00:00",
        hrci_bias=0.0,
        hrcs_bias=0.0,
        acisi_bias=0.0,
    ):
        SolarHeat.__init__(
            self,
            model,
            node,
            pitch_comp,
            eclipse_comp,
            P_pitches,
            Ps,
            dPs,
            var_func,
            tau,
            ampl,
            bias,
            epoch,
        )
        self.simz_comp = model.get_comp(simz_comp)
        self.add_par("hrcs_bias", hrcs_bias, min=-1.0, max=1.0)
        self.add_par("hrci_bias", hrci_bias, min=-1.0, max=1.0)
        self.add_par("acisi_bias", acisi_bias, min=-1.0, max=1.0)

    @property
    def dvals(self):
        if not hasattr(self, "pitches"):
            self.pitches = self.pitch_comp.dvals
        if not hasattr(self, "t_days"):
            self.t_days = (self.pitch_comp.times - DateTime(self.epoch).secs) / 86400.0
        if not hasattr(self, "t_phase"):
            time2000 = DateTime("2000:001:00:00:00").secs
            time2010 = DateTime("2010:001:00:00:00").secs
            secs_per_year = (time2010 - time2000) / 10.0
            t_year = (self.pitch_comp.times - time2000) / secs_per_year
            self.t_phase = t_year * 2 * np.pi

        simz = self.simz_comp.dvals
        if not hasattr(self, "hrcs_mask"):
            self.hrcs_mask = simz < -75000
        if not hasattr(self, "hrci_mask"):
            self.hrci_mask = (simz >= -75000) & (simz < 0)
        if not hasattr(self, "acisi_mask"):
            self.acisi_mask = simz > 80000

        Ps = self.parvals[0 : self.n_pitches] + self.bias
        dPs = self.parvals[self.n_pitches : 2 * self.n_pitches]
        Ps_interp = scipy.interpolate.interp1d(self.P_pitches, Ps, kind="linear")
        dPs_interp = scipy.interpolate.interp1d(self.P_pitches, dPs, kind="linear")
        P_vals = Ps_interp(self.pitches)
        dP_vals = dPs_interp(self.pitches)
        self.P_vals = P_vals
        self._dvals = (
            P_vals
            + dP_vals * self.var_func(self.t_days, self.tau)
            + self.ampl * np.cos(self.t_phase)
        ).reshape(-1)

        # Set power to 0.0 during eclipse (where eclipse_comp.dvals == True)
        if self.eclipse_comp is not None:
            self._dvals[self.eclipse_comp.dvals] = 0.0

        # Apply a constant bias offset to power for times when SIM-Z is at HRC
        self._dvals[self.hrcs_mask] += self.hrcs_bias
        self._dvals[self.hrci_mask] += self.hrci_bias
        self._dvals[self.acisi_mask] += self.acisi_bias

        return self._dvals

    def __str__(self):
        return "solarheat__{0}".format(self.node)
