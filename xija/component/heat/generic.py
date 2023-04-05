# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from xija.component.heat.base import PrecomputedHeatPower, ActiveHeatPower

try:
    from Ska.Matplotlib import plot_cxctime
    from Chandra.Time import DateTime
except ImportError:
    pass

from xija import tmal


class PropHeater(PrecomputedHeatPower):
    """Proportional heater (P = k * (T_set - T) for T < T_set)."""

    def __init__(self, model, node, node_control=None, k=0.1, T_set=20.0):
        super(PropHeater, self).__init__(model)
        self.node = self.model.get_comp(node)
        self.node_control = (
            self.node if node_control is None else self.model.get_comp(node_control)
        )
        self.add_par("k", k, min=0.0, max=1.0)
        self.add_par("T_set", T_set, min=-50.0, max=100.0)
        self.n_mvals = 1

    def __str__(self):
        return "prop_heat__{0}".format(self.node)

    def get_dvals_tlm(self):
        """ """
        return np.zeros_like(self.model.times)

    def update(self):
        self.tmal_ints = (
            tmal.OPCODES["proportional_heater"],
            self.node.mvals_i,  # dy1/dt index
            self.node_control.mvals_i,
            self.mvals_i,
        )
        self.tmal_floats = (self.T_set, self.k)

    def plot_data__time(self, fig, ax):
        lines = ax.get_lines()
        if lines:
            lines[0].set_data(self.model_plotdate, self.mvals)
        else:
            plot_cxctime(self.model.times, self.mvals, "#386cb0", fig=fig, ax=ax)
            ax.grid()
            ax.set_title("{}: data (blue)".format(self.name))
            ax.set_ylabel("Power")


class ThermostatHeater(ActiveHeatPower):
    """Thermostat heater (no deadband): heat = P for T < T_set)."""

    def __init__(self, model, node, node_control=None, P=0.1, T_set=20.0):
        super(ThermostatHeater, self).__init__(model)
        self.node = self.model.get_comp(node)
        self.node_control = (
            self.node if node_control is None else self.model.get_comp(node_control)
        )
        self.add_par("P", P, min=0.0, max=1.0)
        self.add_par("T_set", T_set, min=-50.0, max=100.0)
        self.n_mvals = 1

    def __str__(self):
        return "thermostat_heat__{0}".format(self.node)

    def get_dvals_tlm(self):
        """ """
        return np.zeros_like(self.model.times)

    def update(self):
        self.tmal_ints = (
            tmal.OPCODES["thermostat_heater"],
            self.node.mvals_i,  # dy1/dt index
            self.node_control.mvals_i,
            self.mvals_i,
        )
        self.tmal_floats = (self.T_set, self.P)

    def plot_data__time(self, fig, ax):
        lines = ax.get_lines()
        if lines:
            lines[0].set_data(self.model_plotdate, self.mvals)
        else:
            plot_cxctime(self.model.times, self.mvals, "#386cb0", fig=fig, ax=ax)
            ax.grid()
            ax.set_title("{}: data (blue)".format(self.name))
            ax.set_ylabel("Power")


class StepFunctionPower(PrecomputedHeatPower):
    """A class that applies a constant heat power shift only
    after a certain point in time.  The shift is 0.0 before
    ``time`` and ``P`` after ``time``.

    Parameters
    ----------
    model :
        parent model object
    node :
        node name or object for which to apply shift
    time :
        time of step function shift
    P :
        size of shift in heat power (default=0.0)
    id :
        str, identifier to allow multiple steps (default='')

    Returns
    -------

    """

    def __init__(self, model, node, time, P=0.0, id=""):
        super(StepFunctionPower, self).__init__(model)
        self.time = DateTime(time).secs
        self.node = self.model.get_comp(node)
        self.n_mvals = 1
        self.id = id
        self.add_par("P", P, min=-10.0, max=10.0)

    def __str__(self):
        return f"step_power{self.id}__{self.node}"

    def get_dvals_tlm(self):
        """ """
        return np.zeros_like(self.model.times)

    def update(self):
        """Update the model prediction as a precomputed heat."""
        self.mvals = np.full_like(self.model.times, fill_value=self.P)
        idx0 = np.searchsorted(self.model.times, self.time)
        self.mvals[:idx0] = 0.0
        self.tmal_ints = (
            tmal.OPCODES["precomputed_heat"],
            self.node.mvals_i,  # dy1/dt index
            self.mvals_i,
        )
        self.tmal_floats = ()

    def plot_data__time(self, fig, ax):
        lines = ax.get_lines()
        if lines:
            lines[0].set_data(self.model_plotdate, self.mvals)
        else:
            plot_cxctime(self.model.times, self.mvals, "#386cb0", fig=fig, ax=ax)
            ax.grid()
            ax.set_title("{}: data (blue)".format(self.name))
            ax.set_ylabel("Power")


class MsidStatePower(PrecomputedHeatPower):
    """
    A class that applies a constant heat power shift only when the state of an
    MSID, ``state_msid``, matches a specified value, ``state_val``.  The shift
    is ``P`` when the ``state_val`` for ``state_msid`` is matched, otherwise it
    is 0.0.

    :param model: parent model object
    :param node: node name or object for which to apply shift
    :param state_msid: state MSID name
    :param state_val: value of ``state_msid`` to be matched
    :param P: size of shift in heat power (default=0.0)

    The name of this component is constructed by concatenating the state msid name
    and the state msid value with an underscore. For example, if the ``MsidStatePower``
    component defines a power value for when the COSSRBX values match the string,
    "ON ", the name of this component is ``cossrbx_on``.

    The ``dvals`` data stored for this component are a boolean type. To initialize
    this component, one would use the following syntax:
        model.comp['cossrbx_on'].set_data(True)

    """

    def __init__(self, model, node, state_msid, state_val, P=0.0):
        super(MsidStatePower, self).__init__(model)
        self.node = self.model.get_comp(node)
        self.state_msid = str(state_msid).lower()
        self.state_val = state_val
        self.state_val_str = str(state_val).lower().strip()
        self.n_mvals = 1
        self.add_par("P", P, min=-10.0, max=10.0)

    def __str__(self):
        return f"{self.state_msid}_{self.state_val_str}"

    def get_dvals_tlm(self):
        """
        Return an array of power values where the power is ``P`` when the
        ``state_val`` for ``state_msid`` is matched, otherwise it is 0.0.
        """
        dvals = self.model.fetch(self.state_msid, "vals", "nearest") == self.state_val
        return dvals

    def update(self):
        self.mvals = np.zeros_like(self.model.times)
        self.mvals[self.dvals] = self.P
        self.tmal_ints = (
            tmal.OPCODES["precomputed_heat"],
            self.node.mvals_i,  # dy1/dt index
            self.mvals_i,
        )
        self.tmal_floats = ()

    def plot_data__time(self, fig, ax):
        lines = ax.get_lines()
        if not lines:
            plot_cxctime(self.model.times, self.dvals, "#386cb0", fig=fig, ax=ax)
            ax.grid()
            ax.set_title(f"{self.name}: state match dvals (blue)")
            ax.set_ylabel(f"{self.state_msid.upper()} == {repr(self.state_val)}")
