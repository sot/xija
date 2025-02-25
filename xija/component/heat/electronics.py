# Licensed under a 3-clause BSD style license - see LICENSE.rst

import re

import numpy as np

from xija import tmal
from xija.component.base import TelemData
from xija.component.heat.base import PrecomputedHeatPower

try:
    from Ska.Matplotlib import plot_cxctime
except ImportError:
    pass


class DetectorHousingHeater(TelemData):
    def __init__(self, model):
        TelemData.__init__(self, model, "1dahtbon")
        self.n_mvals = 1
        self.fetch_attr = "midvals"
        self.fetch_method = "nearest"

    def get_dvals_tlm(self):
        dahtbon = self.model.fetch(self.msid, "vals", "nearest")
        return dahtbon == "ON "

    def update(self):
        self.mvals = np.where(self.dvals, 1, 0)

    def __str__(self):
        return "dh_heater"


class AcisDpaStatePower(PrecomputedHeatPower):
    """Heating from ACIS electronics (ACIS config dependent CCDs, FEPs etc).
    Use commanded states and assign an effective power for each "unique" power
    state.  See dpa/NOTES.power.

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(
        self,
        model,
        node,
        mult=1.0,
        fep_count=None,
        ccd_count=None,
        vid_board=None,
        clocking=None,
        pow_states=None,
    ):
        super(AcisDpaStatePower, self).__init__(model)
        self.node = self.model.get_comp(node)
        self.fep_count = self.model.get_comp(fep_count)
        self.ccd_count = self.model.get_comp(ccd_count)
        self.vid_board = self.model.get_comp(vid_board)
        self.clocking = self.model.get_comp(clocking)
        if pow_states is None:
            pow_states = [
                "0xxx",
                "1xxx",
                "2xxx",
                "3xx0",
                "3xx1",
                "4xxx",
                "5xxx",
                "66x0",
                "6611",
                "6xxx",
            ]
        for ps in pow_states:
            self.add_par("pow_%s" % ps, 20, min=10, max=100)
        self.add_par("mult", mult, min=0.0, max=2.0)
        self.add_par("bias", 70, min=10, max=100)

        self.power_pars = [par for par in self.pars if par.name.startswith("pow_")]
        self.n_mvals = 1
        self.data = None
        self.data_times = None

    def __str__(self):
        return "dpa_power"

    @property
    def par_idxs(self):
        if not hasattr(self, "_par_idxs"):
            # Make a regex corresponding to the last bit of each power
            # parameter name.  E.g. "pow_1xxx" => "1...".
            power_par_res = [par.name[4:].replace("x", ".") for par in self.power_pars]

            par_idxs = np.zeros(6612, dtype=np.int_) - 1
            for fep_count in range(7):
                for ccd_count in range(7):
                    for vid_board in range(2):
                        for clocking in range(2):
                            state = "{}{}{}{}".format(
                                fep_count, ccd_count, vid_board, clocking
                            )
                            idx = int(state)
                            for i, power_par_re in enumerate(power_par_res):
                                if re.match(power_par_re, state):
                                    par_idxs[idx] = i
                                    break
                            else:
                                raise ValueError(
                                    "No match for power state {}".format(state)
                                )

            idxs = (
                self.fep_count.dvals * 1000
                + self.ccd_count.dvals * 100
                + self.vid_board.dvals * 10
                + self.clocking.dvals
            )
            self._par_idxs = par_idxs[idxs]

            if self._par_idxs.min() < 0:
                raise ValueError("Fatal problem with par_idxs routine")

        return self._par_idxs

    def get_dvals_tlm(self):
        """Model dvals is set to the telemetered power.  This is not actually
        used by the model, but is useful for diagnostics.

        Parameters
        ----------

        Returns
        -------

        """
        try:
            dvals = self.model.fetch("dp_dpa_power")
        except ValueError:
            dvals = np.zeros_like(self.model.times)
        return dvals

    def update(self):
        """Update the model prediction as a precomputed heat.  Make an array of
        the current power parameters, then slice that with self.par_idxs to
        generate the predicted power (based on the parameter specifying state
        power) at each time step.

        Parameters
        ----------

        Returns
        -------

        """
        power_parvals = np.array([par.val for par in self.power_pars])
        powers = power_parvals[self.par_idxs]
        self.mvals = self.mult / 100.0 * (powers - self.bias)
        self.tmal_ints = (
            tmal.OPCODES["precomputed_heat"],
            self.node.mvals_i,  # dy1/dt index
            self.mvals_i,
        )
        self.tmal_floats = ()

    def plot_data__time(self, fig, ax):
        powers = self.mvals * 100.0 / self.mult + self.bias
        lines = ax.get_lines()
        if lines:
            lines[0].set_data(self.model_plotdate, self.dvals)
            lines[1].set_data(self.model_plotdate, powers)
        else:
            plot_cxctime(
                self.model.times, powers, ls="-", color="#d92121", fig=fig, ax=ax
            )
            plot_cxctime(
                self.model.times, self.dvals, color="#386cb0", ls="-", fig=fig, ax=ax
            )
            ax.grid()
            ax.set_title("{}: model (red) and data (blue)".format(self.name))
            ax.set_ylabel("Power (W)")
