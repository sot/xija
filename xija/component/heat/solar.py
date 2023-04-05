# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import scipy.interpolate

try:
    import Ska.Numpy
    from Chandra.Time import DateTime
    from Ska.Matplotlib import plot_cxctime
except ImportError:
    pass

from xija.component.base import ModelComponent
from xija.component.heat.base import PrecomputedHeatPower


class SolarHeatOffNomRoll(PrecomputedHeatPower):
    """Heating of a +Y or -Y face of a spacecraft component due to off-nominal roll.  The
    heating is proportional to the projection of the sun on body +Y axis (which is a value
    from -1 to 1).  There are two parameters ``P_plus_y`` and ``P_minus_y``.  For sun on
    the +Y side the ``P_plus_y`` parameter is used, and likewise for sun on -Y.  For
    example for +Y sun::

       heat = P_plus_y * sun_body_y

    The following reference has useful diagrams concerning off-nominal roll and
    projections: http://occweb.cfa.harvard.edu/twiki/pub/Aspect/WebHome/ROLLDEV3.pdf.

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(
        self,
        model,
        node,
        pitch_comp="pitch",
        roll_comp="roll",
        eclipse_comp=None,
        P_plus_y=0.0,
        P_minus_y=0.0,
    ):
        ModelComponent.__init__(self, model)
        self.node = self.model.get_comp(node)
        self.pitch_comp = self.model.get_comp(pitch_comp)
        self.roll_comp = self.model.get_comp(roll_comp)
        self.eclipse_comp = self.model.get_comp(eclipse_comp)

        self.add_par("P_plus_y", P_plus_y, min=-5.0, max=5.0)
        self.add_par("P_minus_y", P_minus_y, min=-5.0, max=5.0)
        self.n_mvals = 1

    @property
    def dvals(self):
        if not hasattr(self, "sun_body_y"):
            # Compute the projection of the sun vector on the body +Y axis.
            # Pitch and off-nominal roll (theta_S and d_phi in OFLS terminology)
            theta_S = np.radians(self.pitch_comp.dvals)
            d_phi = np.radians(self.roll_comp.dvals)
            self.sun_body_y = np.sin(theta_S) * np.sin(d_phi)
            self.plus_y = self.sun_body_y > 0

        self._dvals = (
            np.where(self.plus_y, self.P_plus_y, self.P_minus_y) * self.sun_body_y
        )

        # Set power to 0.0 during eclipse (where eclipse_comp.dvals == True)
        if self.eclipse_comp is not None:
            self._dvals[self.eclipse_comp.dvals] = 0.0

        return self._dvals

    def __str__(self):
        return "solarheat_off_nom_roll__{0}".format(self.node)


class SolarHeat(PrecomputedHeatPower):
    """Solar heating (pitch dependent)

    Parameters
    ----------
    model :
        parent model
    node :
        node which is coupled to solar heat
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
    dP_pitches :
        list of pitch values for dP (default=``P_pitches``)

    Returns
    -------

    """

    def __init__(
        self,
        model,
        node,
        pitch_comp="pitch",
        eclipse_comp=None,
        P_pitches=None,
        Ps=None,
        dPs=None,
        var_func="exp",
        tau=1732.0,
        ampl=0.05,
        bias=0.0,
        epoch="2010:001:12:00:00",
        dP_pitches=None,
    ):
        ModelComponent.__init__(self, model)
        self.node = self.model.get_comp(node)
        self.pitch_comp = self.model.get_comp(pitch_comp)
        self.eclipse_comp = self.model.get_comp(eclipse_comp)

        if P_pitches is None:
            P_pitches = [45, 65, 90, 130, 180]
        self.P_pitches = np.array(P_pitches, dtype=np.float64)
        self.n_pitches = len(self.P_pitches)

        if dP_pitches is None:
            dP_pitches = self.P_pitches
        self.dP_pitches = np.array(dP_pitches, dtype=np.float64)

        if (
            self.dP_pitches[0] != self.P_pitches[0]
            or self.dP_pitches[-1] != self.P_pitches[-1]
        ):
            raise ValueError("P_pitches and dP_pitches must span the same pitch range")

        if Ps is None:
            Ps = np.ones_like(self.P_pitches)
        self.Ps = np.array(Ps, dtype=np.float64)

        if dPs is None:
            dPs = np.zeros_like(self.dP_pitches)
        self.dPs = np.array(dPs, dtype=np.float64)

        self.epoch = epoch

        for pitch, power in zip(self.P_pitches, self.Ps):
            self.add_par("P_{0:.0f}".format(float(pitch)), power, min=-10.0, max=10.0)
        for pitch, dpower in zip(self.dP_pitches, self.dPs):
            self.add_par("dP_{0:.0f}".format(float(pitch)), dpower, min=-1.0, max=1.0)
        self.add_par("tau", tau, min=1000.0, max=3000.0)
        self.add_par("ampl", ampl, min=-1.0, max=1.0)
        self.add_par("bias", bias, min=-1.0, max=1.0)
        self.n_mvals = 1
        self.var_func = getattr(self, var_func)

    _t_phase = None

    @property
    def t_phase(self):
        if self._t_phase is None:
            time2000 = DateTime("2000:001:00:00:00").secs
            time2010 = DateTime("2010:001:00:00:00").secs
            secs_per_year = (time2010 - time2000) / 10.0
            t_year = (self.pitch_comp.times - time2000) / secs_per_year
            self._t_phase = t_year * 2 * np.pi
        return self._t_phase

    @t_phase.deleter
    def t_phase(self):
        self._t_phase = None

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, value):
        if hasattr(self, "_epoch"):
            if self.var_func is not self.linear:
                raise AttributeError("can only reset the epoch for var_func=linear")

            new_epoch = DateTime(value)
            epoch = DateTime(self.epoch)
            days = new_epoch - epoch

            # Don't make tiny updates to epoch
            if abs(days) < 10:
                return

            # Update the Ps params in place.  Note that self.Ps is basically for
            # setting the array size whereas the self.pars vals are the actual values
            # taken from the model spec file and used in fitting.
            Ps = self.parvals[0 : self.n_pitches]
            dPs = self.parvals[self.n_pitches : self.n_pitches + len(self.dP_pitches)]
            dPs_interp = np.interp(x=self.P_pitches, xp=self.dP_pitches, fp=dPs)

            Ps += dPs_interp * days / self.tau
            for par, P in zip(self.pars, Ps):
                par.val = P
                if P > par.max:
                    par.max = P
                elif P < par.min:
                    par.min = P

            print(
                "Updated model component {} epoch from {} to {}".format(
                    self, epoch.date[:8], new_epoch.date[:8]
                )
            )

            # In order to capture the new epoch when saving the model we need to
            # update ``init_kwargs`` since this isn't a formal model parameter
            self.init_kwargs["epoch"] = new_epoch.date[:8]

            # Delete these cached attributes which depend on epoch
            del self.t_days
            del self.t_phase

        self._epoch = value

    def dvals_post_hook(self):
        """Override this method to adjust self._dvals after main computation."""
        pass

    def _compute_dvals(self):
        vf = self.var_func(self.t_days, self.tau)
        return (
            self.P_vals + self.dP_vals * vf + self.ampl * np.cos(self.t_phase)
        ).reshape(-1)

    @property
    def dvals(self):
        if not hasattr(self, "pitches"):
            self.pitches = np.clip(
                self.pitch_comp.dvals, self.P_pitches[0], self.P_pitches[-1]
            )
        if not hasattr(self, "t_days"):
            self.t_days = (self.pitch_comp.times - DateTime(self.epoch).secs) / 86400.0

        Ps = self.parvals[0 : self.n_pitches] + self.bias
        dPs = self.parvals[self.n_pitches : self.n_pitches + len(self.dP_pitches)]

        Ps_interp = scipy.interpolate.interp1d(self.P_pitches, Ps, kind="linear")
        dPs_interp = scipy.interpolate.interp1d(self.dP_pitches, dPs, kind="linear")
        self.P_vals = Ps_interp(self.pitches)
        self.dP_vals = dPs_interp(self.pitches)

        self._dvals = self._compute_dvals()

        # Set power to 0.0 during eclipse (where eclipse_comp.dvals == True)
        if self.eclipse_comp is not None:
            self._dvals[self.eclipse_comp.dvals] = 0.0

        # Allow for customization in SolarHeat subclasses
        self.dvals_post_hook()

        return self._dvals

    def __str__(self):
        return "solarheat__{0}".format(self.node)

    def plot_solar_heat__pitch(self, fig, ax):
        Ps = self.parvals[0 : self.n_pitches] + self.bias
        Ps_interp = scipy.interpolate.interp1d(self.P_pitches, Ps, kind="linear")

        dPs = self.parvals[self.n_pitches : self.n_pitches + len(self.dP_pitches)]
        dPs_interp = scipy.interpolate.interp1d(self.dP_pitches, dPs, kind="linear")

        pitches = np.linspace(self.P_pitches[0], self.P_pitches[-1], 100)
        P_vals = Ps_interp(pitches)
        dP_vals = dPs_interp(pitches)

        lines = ax.get_lines()
        if lines:
            lines[0].set_data(self.P_pitches, Ps)
            lines[1].set_data(pitches, P_vals)
            lines[2].set_data(pitches, dP_vals + P_vals)
        else:
            ax.plot(self.P_pitches, Ps, "or", markersize=3)
            ax.plot(pitches, P_vals, "-b")
            ax.plot(pitches, dP_vals + P_vals, "-m")
            ax.set_title("{} solar heat input".format(self.node.name))
            ax.set_xlim(40, 180)
            ax.grid()


class SolarHeatMulplicative(SolarHeat):
    __doc__ = SolarHeat.__doc__

    def __init__(
        self,
        model,
        node,
        pitch_comp="pitch",
        eclipse_comp=None,
        P_pitches=None,
        Ps=None,
        dPs=None,
        var_func="exp",
        tau=1732.0,
        ampl=0.0334,
        bias=0.0,
        epoch="2010:001:12:00:00",
        dP_pitches=None,
    ):
        super().__init__(
            model,
            node,
            pitch_comp=pitch_comp,
            eclipse_comp=eclipse_comp,
            P_pitches=P_pitches,
            Ps=Ps,
            dPs=dPs,
            var_func=var_func,
            tau=tau,
            ampl=ampl,
            bias=bias,
            epoch=epoch,
            dP_pitches=dP_pitches,
        )

    def _compute_dvals(self):
        vf = self.var_func(self.t_days, self.tau)
        yv = 1.0 + self.ampl * np.cos(self.t_phase)
        return ((self.P_vals + self.dP_vals * vf) * yv).reshape(-1)


class SolarHeatHrc(SolarHeat):
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
    hrc_bias :
        solar heating bias when SIM-Z < 0 (HRC)
    dP_pitches :
        list of pitch values for dP (default=``P_pitches``)

    Returns
    -------

    """

    def __init__(
        self,
        model,
        node,
        simz_comp="sim_z",
        pitch_comp="pitch",
        eclipse_comp=None,
        P_pitches=None,
        Ps=None,
        dPs=None,
        var_func="exp",
        tau=1732.0,
        ampl=0.05,
        bias=0.0,
        epoch="2010:001:12:00:00",
        hrc_bias=0.0,
        dP_pitches=None,
    ):
        super().__init__(
            model,
            node,
            pitch_comp=pitch_comp,
            eclipse_comp=eclipse_comp,
            P_pitches=P_pitches,
            Ps=Ps,
            dPs=dPs,
            var_func=var_func,
            tau=tau,
            ampl=ampl,
            bias=bias,
            epoch=epoch,
            dP_pitches=dP_pitches,
        )

        self.simz_comp = model.get_comp(simz_comp)
        self.add_par("hrc_bias", hrc_bias, min=-1.0, max=1.0)

    def dvals_post_hook(self):
        """Apply a bias power offset when SIM-Z is at HRC-S or HRC-I."""
        if not hasattr(self, "hrc_mask"):
            self.hrc_mask = self.simz_comp.dvals < 0
        self._dvals[self.hrc_mask] += self.hrc_bias


class SolarHeatHrcOpts(SolarHeat):
    """Solar heating (pitch and SIM-Z dependent, two parameters for
    HRC-I and HRC-S)

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
    hrci_bias :
        solar heating bias when HRC-I is in the focal plane.
    hrcs_bias :
        solar heating bias when HRC-S is in the focal plane.
    dP_pitches :
        list of pitch values for dP (default=``P_pitches``)
    """

    def __init__(
        self,
        model,
        node,
        simz_comp="sim_z",
        pitch_comp="pitch",
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
        dP_pitches=None,
    ):
        super().__init__(
            model,
            node,
            pitch_comp=pitch_comp,
            eclipse_comp=eclipse_comp,
            P_pitches=P_pitches,
            Ps=Ps,
            dPs=dPs,
            var_func=var_func,
            tau=tau,
            ampl=ampl,
            bias=bias,
            epoch=epoch,
            dP_pitches=dP_pitches,
        )

        self.simz_comp = model.get_comp(simz_comp)
        self.add_par("hrci_bias", hrci_bias, min=-1.0, max=1.0)
        self.add_par("hrcs_bias", hrcs_bias, min=-1.0, max=1.0)

    def dvals_post_hook(self):
        """Apply a bias power offset when SIM-Z is at HRC-S or HRC-I."""
        if not hasattr(self, "hrci_mask"):
            self.hrci_mask = (self.simz_comp.dvals < 0) & (
                self.simz_comp.dvals > -86147
            )
        self._dvals[self.hrci_mask] += self.hrci_bias
        if not hasattr(self, "hrcs_mask"):
            self.hrcs_mask = self.simz_comp.dvals <= -86147
        self._dvals[self.hrcs_mask] += self.hrcs_bias


class SolarHeatHrcMult(SolarHeatHrcOpts, SolarHeatMulplicative):
    __doc__ = SolarHeatHrcOpts.__doc__

    def __init__(
        self,
        model,
        node,
        simz_comp="sim_z",
        pitch_comp="pitch",
        eclipse_comp=None,
        P_pitches=None,
        Ps=None,
        dPs=None,
        var_func="exp",
        tau=1732.0,
        ampl=0.0334,
        bias=0.0,
        epoch="2010:001:12:00:00",
        hrci_bias=0.0,
        hrcs_bias=0.0,
        dP_pitches=None,
    ):
        super().__init__(
            model,
            node,
            simz_comp=simz_comp,
            pitch_comp=pitch_comp,
            eclipse_comp=eclipse_comp,
            P_pitches=P_pitches,
            Ps=Ps,
            dPs=dPs,
            var_func=var_func,
            tau=tau,
            ampl=ampl,
            bias=bias,
            epoch=epoch,
            hrci_bias=hrci_bias,
            hrcs_bias=hrcs_bias,
            dP_pitches=dP_pitches,
        )


class SimZDepSolarHeat(PrecomputedHeatPower):
    """SIM-Z dependent solar heating"""

    simz_lims = None
    instr_names = None

    def __init__(
        self,
        model,
        node,
        pitch_comp="pitch",
        simz_comp="sim_z",
        dh_heater_comp="dh_heater",
        P_pitches=None,
        P_vals=None,
        dP_pitches=None,
        dPs=None,
        var_func="linear",
        tau=1732.0,
        ampl=0.05,
        epoch="2013:001:12:00:00",
        dh_heater=0.05,
    ):
        ModelComponent.__init__(self, model)
        self.n_mvals = 1
        self.node = self.model.get_comp(node)
        self.pitch_comp = self.model.get_comp(pitch_comp)
        self.simz_comp = self.model.get_comp(simz_comp)
        self.dh_heater_comp = self.model.get_comp(dh_heater_comp)
        self.P_pitches = np.array(
            [45.0, 55.0, 70.0, 90.0, 150.0] if (P_pitches is None) else P_pitches,
            dtype=float,
        )
        if dP_pitches is None:
            dP_pitches = self.P_pitches
        self.dP_pitches = np.array(dP_pitches, dtype=float)

        if (
            self.dP_pitches[0] != self.P_pitches[0]
            or self.dP_pitches[-1] != self.P_pitches[-1]
        ):
            raise ValueError("P_pitches and dP_pitches must span the same pitch range")

        self.n_p = len(self.P_pitches)
        self.n_dp = len(self.dP_pitches)
        self.n_instr = len(self.instr_names)
        if P_vals is None:
            P_vals = np.ones((self.n_instr, self.n_p))
        self.dPs = (
            np.zeros_like(self.dP_pitches)
            if dPs is None
            else np.array(dPs, dtype=np.float)
        )
        for i, instr_name in enumerate(self.instr_names):
            for j, pitch in enumerate(self.P_pitches):
                self.add_par(
                    "P_{0}_{1:d}".format(instr_name, int(pitch)),
                    P_vals[i][j],
                    min=-10.0,
                    max=10.0,
                )

        for j, pitch in enumerate(self.dPs):
            self.add_par(
                "dP_{0:d}".format(int(self.dP_pitches[j])),
                self.dPs[j],
                min=-1.0,
                max=1.0,
            )

        self.add_par("tau", tau, min=1000.0, max=3000.0)
        self.add_par("ampl", ampl, min=-1.0, max=1.0)
        self.add_par("dh_heater", dh_heater, min=-1.0, max=1.0)
        self.epoch = epoch
        self.var_func = getattr(self, var_func)

    @property
    def dvals(self):
        if not hasattr(self, "pitches"):
            self.pitches = np.clip(
                self.pitch_comp.dvals, self.P_pitches[0], self.P_pitches[-1]
            )

        if not hasattr(self, "simzs"):
            self.simzs = self.simz_comp.dvals
            self.instrs = np.zeros(self.model.n_times, dtype=np.int8)
            for i, lims in enumerate(self.simz_lims):
                ok = (self.simzs > lims[0]) & (self.simzs <= lims[1])
                self.instrs[ok] = i

        if not hasattr(self, "t_days"):
            self.t_days = (self.pitch_comp.times - DateTime(self.epoch).secs) / 86400.0
        if not hasattr(self, "t_phase"):
            time2000 = DateTime("2000:001:00:00:00").secs
            time2010 = DateTime("2010:001:00:00:00").secs
            secs_per_year = (time2010 - time2000) / 10.0
            t_year = (self.pitch_comp.times - time2000) / secs_per_year
            self.t_phase = t_year * 2 * np.pi

        # Interpolate power(pitch) for each instrument separately and make 2d
        # stack
        heats = []
        dPs = self.parvals[
            self.n_instr * self.n_p : (self.n_instr * self.n_p + self.n_dp)
        ]
        dP_vals = Ska.Numpy.interpolate(dPs, self.dP_pitches, self.pitches)
        d_heat = (
            dP_vals * self.var_func(self.t_days, self.tau)
            + self.ampl * np.cos(self.t_phase)
        ).ravel()

        for i in range(self.n_instr):
            P_vals = self.parvals[i * self.n_p : (i + 1) * self.n_p]
            heat = Ska.Numpy.interpolate(P_vals, self.P_pitches, self.pitches)
            heats.append(heat + d_heat)

        self.heats = np.vstack(heats)

        # Now pick out the power(pitch) for the appropriate instrument at each
        # time
        self._dvals = self.heats[self.instrs, np.arange(self.heats.shape[1])]

        # Increase heat power for times when detector housing heater is enabled
        self._dvals[self.dh_heater_comp.dvals] += self.dh_heater

        return self._dvals

    def plot_solar_heat__pitch(self, fig, ax):
        P_vals = {}
        for instr_name in self.instr_names:
            P_vals[instr_name] = []
            for pitch in self.P_pitches:
                P_vals[instr_name].append(
                    getattr(self, "P_{0}_{1:d}".format(instr_name, int(pitch)))
                )
        colors = ["b", "c", "r", "m"]
        lines = ax.get_lines()
        if lines:
            for i, instr_name in enumerate(self.instr_names):
                lines[i].set_data(self.P_pitches, P_vals[instr_name])
                # lines[i * 2 + 1].set_data(self.P_pitches, P_vals[instr_name], '-b')
        else:
            for i, instr_name in enumerate(self.instr_names):
                color = colors[i]
                ax.plot(
                    self.P_pitches,
                    P_vals[instr_name],
                    "o-{}".format(color),
                    markersize=5,
                    label=instr_name,
                )
            ax.set_title("{} solar heat input".format(self.node.name))
            ax.set_xlim(40, 180)
            ax.grid()
            ax.legend(loc="best")

    def __str__(self):
        return "simz_solarheat__{0}".format(self.node)


class AllSimZSolarHeat(SimZDepSolarHeat):
    """Solar Heating, SIM-Z for all four instruments"""

    simz_lims = (
        (-400000.0, -85000.0),  # HRC-S
        (-85000.0, 0.0),  # HRC-I
        (0.0, 83000.0),  # ACIS-S
        (83000.0, 400000.0),
    )  # ACIS-I
    instr_names = ["hrcs", "hrci", "aciss", "acisi"]

    def __str__(self):
        return f"all_simz_solarheat__{self.node}"


class AcisPsmcSolarHeat(AllSimZSolarHeat):
    """Solar heating of PSMC box.  This is dependent on SIM-Z"""

    def __str__(self):
        return f"psmc_solarheat__{self.node}"


class HrcISAcisSimZSolarHeat(SimZDepSolarHeat):
    """Solar Heating, SIM-Z for HRC-I/S, ACIS positions"""

    simz_lims = (
        (-400000.0, -85000.0),  # HRC-S
        (-85000.0, 0.0),  # HRC-I
        (0.0, 400000.0),
    )  # ACIS
    instr_names = ["hrcs", "hrci", "acis"]

    def __str__(self):
        return f"hrc_is_acis_simz_solarheat__{self.node}"


class AcisISHrcSimZSolarHeat(SimZDepSolarHeat):
    """Solar Heating, SIM-Z for ACIS-I/S, HRC positions"""

    simz_lims = (
        (-400000.0, 0.0),  # HRC
        (0.0, 83000.0),  # ACIS-S
        (83000.0, 400000.0),
    )  # ACIS-I
    instr_names = ["acisi", "aciss", "hrc"]

    def __str__(self):
        return f"hrc_acis_is_simz_solarheat__{self.node}"
