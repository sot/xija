# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function

import re
from itertools import count
import glob
import os
from six.moves import zip
from pathlib import Path

from numba import jit
import numpy as np
import scipy.interpolate
import astropy.units as u
from astropy.io import fits

try:
    import Ska.Numpy
    from Ska.Matplotlib import plot_cxctime
    from Chandra.Time import DateTime
except ImportError:
    pass

from .base import ModelComponent, TelemData
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

    @staticmethod
    def linear(days, k_inv):
        return days / k_inv

    @staticmethod
    def exp(days, tau):
        return 1 - np.exp(-days / tau)


class ActiveHeatPower(ModelComponent):
    """Component that provides active heat power input which depends on
    current or past computed model values

    Parameters
    ----------

    Returns
    -------

    """
    pass


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

    def __init__(self, model, node, pitch_comp, roll_comp, eclipse_comp=None,
                 P_plus_y=0.0, P_minus_y=0.0):
        ModelComponent.__init__(self, model)
        self.node = self.model.get_comp(node)
        self.pitch_comp = self.model.get_comp(pitch_comp)
        self.roll_comp = self.model.get_comp(roll_comp)
        self.eclipse_comp = self.model.get_comp(eclipse_comp)

        self.add_par('P_plus_y', P_plus_y, min=-5.0, max=5.0)
        self.add_par('P_minus_y', P_minus_y, min=-5.0, max=5.0)
        self.n_mvals = 1

    @property
    def dvals(self):
        if not hasattr(self, 'sun_body_y'):
            # Compute the projection of the sun vector on the body +Y axis.
            # Pitch and off-nominal roll (theta_S and d_phi in OFLS terminology)
            theta_S = np.radians(self.pitch_comp.dvals)
            d_phi = np.radians(self.roll_comp.dvals)
            self.sun_body_y = np.sin(theta_S) * np.sin(d_phi)
            self.plus_y = self.sun_body_y > 0

        self._dvals = np.where(self.plus_y, self.P_plus_y, self.P_minus_y) * self.sun_body_y

        # Set power to 0.0 during eclipse (where eclipse_comp.dvals == True)
        if self.eclipse_comp is not None:
            self._dvals[self.eclipse_comp.dvals] = 0.0

        return self._dvals

    def __str__(self):
        return 'solarheat_off_nom_roll__{0}'.format(self.node)


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

    Returns
    -------

    """
    def __init__(self, model, node, pitch_comp, eclipse_comp=None,
                 P_pitches=None, Ps=None, dPs=None, var_func='exp',
                 tau=1732.0, ampl=0.05, bias=0.0, epoch='2010:001:12:00:00'):
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

    _t_phase = None

    @property
    def t_phase(self):
        if self._t_phase is None:
            time2000 = DateTime('2000:001:00:00:00').secs
            time2010 = DateTime('2010:001:00:00:00').secs
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
        if hasattr(self, '_epoch'):
            if self.var_func is not self.linear:
                raise AttributeError('can only reset the epoch for var_func=linear')

            new_epoch = DateTime(value)
            epoch = DateTime(self.epoch)
            days = new_epoch - epoch

            # Don't make tiny updates to epoch
            if abs(days) < 10:
                return

            # Update the Ps params in place.  Note that self.Ps is basically for
            # setting the array size whereas the self.pars vals are the actual values
            # taken from the model spec file and used in fitting.
            Ps = self.parvals[0:self.n_pitches]
            dPs = self.parvals[self.n_pitches:2 * self.n_pitches]
            Ps += dPs * days / self.tau
            for par, P in zip(self.pars, Ps):
                par.val = P
                if P > par.max:
                    par.max = P
                elif P < par.min:
                    par.min = P

            print('Updated model component {} epoch from {} to {}'
                  .format(self, epoch.date[:8], new_epoch.date[:8]))

            # In order to capture the new epoch when saving the model we need to
            # update ``init_kwargs`` since this isn't a formal model parameter
            self.init_kwargs['epoch'] = new_epoch.date[:8]

            # Delete these cached attributes which depend on epoch
            del self.t_days
            del self.t_phase

        self._epoch = value

    def dvals_post_hook(self):
        """Override this method to adjust self._dvals after main computation."""
        pass

    def _compute_dvals(self):
        vf = self.var_func(self.t_days, self.tau)
        return (self.P_vals + self.dP_vals*vf +
                self.ampl * np.cos(self.t_phase)).reshape(-1)

    @property
    def dvals(self):
        if not hasattr(self, 'pitches'):
            self.pitches = np.clip(self.pitch_comp.dvals, self.P_pitches[0], self.P_pitches[-1])
        if not hasattr(self, 't_days'):
            self.t_days = (self.pitch_comp.times
                           - DateTime(self.epoch).secs) / 86400.0

        Ps = self.parvals[0:self.n_pitches] + self.bias
        dPs = self.parvals[self.n_pitches:2 * self.n_pitches]
        Ps_interp = scipy.interpolate.interp1d(self.P_pitches, Ps,
                                               kind='linear')
        dPs_interp = scipy.interpolate.interp1d(self.P_pitches, dPs,
                                                kind='linear')
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
        return 'solarheat__{0}'.format(self.node)

    def plot_solar_heat__pitch(self, fig, ax):
        Ps = self.parvals[0:self.n_pitches] + self.bias
        Ps_interp = scipy.interpolate.interp1d(self.P_pitches, Ps,
                                               kind='linear')

        dPs = self.parvals[self.n_pitches:2 * self.n_pitches]
        dPs_interp = scipy.interpolate.interp1d(self.P_pitches, dPs,
                                                kind='linear')

        pitches = np.linspace(self.P_pitches[0], self.P_pitches[-1], 100)
        P_vals = Ps_interp(pitches)
        dP_vals = dPs_interp(pitches)

        lines = ax.get_lines()
        if lines:
            lines[0].set_data(self.P_pitches, Ps)
            lines[1].set_data(pitches, P_vals)
            lines[2].set_data(self.P_pitches, dPs + Ps)
            lines[3].set_data(pitches, dP_vals + P_vals)
        else:
            ax.plot(self.P_pitches, Ps, 'or', markersize=3)
            ax.plot(pitches, P_vals, '-b')
            ax.plot(self.P_pitches, dPs + Ps, 'om', markersize=3)
            ax.plot(pitches, dP_vals + P_vals, '-m')
            ax.set_title('{} solar heat input'.format(self.node.name))
            ax.set_xlim(40, 180)
            ax.grid()


class SolarHeatMulplicative(SolarHeat):
    def __init__(self, model, node, pitch_comp, eclipse_comp=None,
                 P_pitches=None, Ps=None, dPs=None, var_func='exp',
                 tau=1732.0, ampl=0.0334, bias=0.0, epoch='2010:001:12:00:00'):
        super(SolarHeatMulplicative, self).__init__(
            model, node, pitch_comp, eclipse_comp=eclipse_comp,
            P_pitches=P_pitches, Ps=Ps, dPs=dPs, var_func=var_func,
            tau=tau, ampl=ampl, bias=bias, epoch=epoch)

    def _compute_dvals(self):
        vf = self.var_func(self.t_days, self.tau)
        yv = (1.0 + self.ampl*np.cos(self.t_phase))
        return ((self.P_vals+self.dP_vals*vf)*yv).reshape(-1)


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

    Returns
    -------

    """
    def __init__(self, model, node, pitch_comp, eclipse_comp=None,
                 P_pitches=None, Ps=None, dPs=None, var_func='exp',
                 tau=1732.0, ampl=0.05, bias=0.0, epoch='2010:001:12:00:00',
                 dh_heater_comp=None, dh_heater_bias=0.0):

        super(SolarHeatAcisCameraBody, self).__init__(
            model, node, pitch_comp, eclipse_comp=eclipse_comp,
            P_pitches=P_pitches, Ps=Ps, dPs=dPs, var_func=var_func,
            tau=tau, ampl=ampl, bias=bias, epoch=epoch)

        self.dh_heater_comp = model.get_comp(dh_heater_comp)
        self.add_par('dh_heater_bias', dh_heater_bias, min=-1.0, max=1.0)

    def dvals_post_hook(self):
        """Apply a bias power offset when detector housing heater is on"""
        self._dvals[self.dh_heater_comp.dvals] += self.dh_heater_bias


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

    Returns
    -------

    """
    def __init__(self, model, node, simz_comp, pitch_comp, eclipse_comp=None,
                 P_pitches=None, Ps=None, dPs=None, var_func='exp',
                 tau=1732.0, ampl=0.05, bias=0.0, epoch='2010:001:12:00:00',
                 hrc_bias=0.0):
        SolarHeat.__init__(self, model, node, pitch_comp, eclipse_comp,
                           P_pitches, Ps, dPs, var_func, tau, ampl, bias,
                           epoch)
        self.simz_comp = model.get_comp(simz_comp)
        self.add_par('hrc_bias', hrc_bias, min=-1.0, max=1.0)

    def dvals_post_hook(self):
        """Apply a bias power offset when SIM-Z is at HRC-S or HRC-I."""
        if not hasattr(self, 'hrc_mask'):
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

    Returns
    -------

    """
    def __init__(self, model, node, simz_comp, pitch_comp, eclipse_comp=None,
                 P_pitches=None, Ps=None, dPs=None, var_func='exp',
                 tau=1732.0, ampl=0.05, bias=0.0, epoch='2010:001:12:00:00',
                 hrci_bias=0.0, hrcs_bias=0.0):
        SolarHeat.__init__(self, model, node, pitch_comp, eclipse_comp,
                           P_pitches, Ps, dPs, var_func, tau, ampl, bias,
                           epoch)
        self.simz_comp = model.get_comp(simz_comp)
        self.add_par('hrci_bias', hrci_bias, min=-1.0, max=1.0)
        self.add_par('hrcs_bias', hrcs_bias, min=-1.0, max=1.0)

    def dvals_post_hook(self):
        """Apply a bias power offset when SIM-Z is at HRC-S or HRC-I."""
        if not hasattr(self, 'hrci_mask'):
            self.hrci_mask = (self.simz_comp.dvals < 0) & \
                             (self.simz_comp.dvals > -86147)
        self._dvals[self.hrci_mask] += self.hrci_bias
        if not hasattr(self, 'hrcs_mask'):
            self.hrcs_mask = self.simz_comp.dvals <= -86147
        self._dvals[self.hrcs_mask] += self.hrcs_bias


class SolarHeatHrcMult(SolarHeatHrcOpts, SolarHeatMulplicative):
    def __init__(self, model, node, simz_comp, pitch_comp, eclipse_comp=None,
                 P_pitches=None, Ps=None, dPs=None, var_func='exp',
                 tau=1732.0, ampl=0.0334, bias=0.0, epoch='2010:001:12:00:00',
                 hrci_bias=0.0, hrcs_bias=0.0):
        super(SolarHeatHrcMult, self).__init__(
            model, node, simz_comp, pitch_comp, eclipse_comp=eclipse_comp,
            P_pitches=P_pitches, Ps=Ps, dPs=dPs, var_func=var_func, tau=tau,
            ampl=ampl, bias=bias, epoch=epoch, hrci_bias=hrci_bias,
            hrcs_bias=hrcs_bias)

# For back compatibility prior to Xija 0.2
DpaSolarHeat = SolarHeatHrc


class EarthHeat(PrecomputedHeatPower):
    """Earth heating of ACIS cold radiator (attitude, ephem dependent)"""

    use_earth_vis_grid = True
    earth_vis_grid_path = Path(__file__).parent / 'earth_vis_grid_nside32.fits.gz'

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

    def calc_earth_vis_from_grid(self, ephems, q_atts):
        import astropy_healpix
        healpix = astropy_healpix.HEALPix(nside=32, order='nested')

        if not hasattr(self, 'earth_vis_grid'):
            with fits.open(self.earth_vis_grid_path) as hdus:
                hdu = hdus[0]
                hdr = hdu.header
                vis_grid = hdu.data / hdr['scale']  # 12288 x 100
                self.__class__.earth_vis_grid = vis_grid

                alts = np.logspace(np.log10(hdr['alt_min']),
                                   np.log10(hdr['alt_max']),
                                   hdr['n_alt'])
                self.__class__.log_earth_vis_dists = np.log(hdr['earthrad'] + alts)

        ephems = ephems.astype(np.float64)
        dists, lons, lats = get_dists_lons_lats(ephems, q_atts)

        hp_idxs = healpix.lonlat_to_healpix(lons * u.rad, lats * u.rad)

        # Linearly interpolate distances for appropriate healpix pixels.
        # Code borrowed a bit from Ska.Numpy.Numpy._interpolate_vectorized.
        xin = self.log_earth_vis_dists
        xout = np.log(dists)
        idxs = np.searchsorted(xin, xout)

        # Extrapolation is linear.  This should never happen in this application
        # because of how the grid is defined.
        idxs = idxs.clip(1, len(xin) - 1)

        x0 = xin[idxs - 1]
        x1 = xin[idxs]

        # Note here the "fancy-indexing" which is indexing into a 2-d array
        # with two 1-d arrays.
        y0 = self.earth_vis_grid[hp_idxs, idxs - 1]
        y1 = self.earth_vis_grid[hp_idxs, idxs]

        self._dvals[:] = (xout - x0) / (x1 - x0) * (y1 - y0) + y0

    def calc_earth_vis_from_taco(self, ephems, q_atts):
        import acis_taco

        for i, ephem, q_att in zip(count(), ephems, q_atts):
            _, illums, _ = acis_taco.calc_earth_vis(ephem, q_att)
            self._dvals[i] = illums.sum()

    @property
    def dvals(self):
        if not hasattr(self, '_dvals') and not self.get_cached():

            # Collect individual MSIDs for use in calc_earth_vis()
            ephem_xyzs = [getattr(self, 'orbitephem0_{}'.format(x))
                          for x in ('x', 'y', 'z')]
            aoattqt_1234s = [getattr(self, 'aoattqt{}'.format(x))
                             for x in range(1, 5)]
            # Note: the copy() here is so that the array becomes contiguous in
            # memory and allows numba to run faster (and avoids NumbaPerformanceWarning:
            # np.dot() is faster on contiguous arrays).
            ephems = np.array([x.dvals for x in ephem_xyzs]).transpose().copy()
            q_atts = np.array([x.dvals for x in aoattqt_1234s]).transpose()

            # Q_atts can have occasional bad values, maybe because the
            # Ska eng 5-min "midvals" are not lined up, but I'm not quite sure.
            # TODO: this (legacy) solution isn't great.  Investigate what's
            # really happening.
            q_norm = np.sqrt(np.sum(q_atts ** 2, axis=1))
            bad = np.abs(q_norm - 1.0) > 0.1
            if np.any(bad):
                print(f"Replacing bad midval quaternions with [1,0,0,0] at times "
                      f"{self.model.times[bad]}")
                q_atts[bad, :] = [0.0, 0.0, 0.0, 1.0]
            q_atts[~bad, :] = q_atts[~bad, :] / q_norm[~bad, np.newaxis]

            # Finally initialize dvals and update in-place appropriately
            self._dvals = np.empty(self.model.n_times, dtype=float)
            if self.use_earth_vis_grid:
                self.calc_earth_vis_from_grid(ephems, q_atts)
            else:
                self.calc_earth_vis_from_taco(ephems, q_atts)

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

        Parameters
        ----------

        Returns
        -------

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
            plot_cxctime(self.model.times, self.dvals, ls='-',
                         color='#386cb0', fig=fig, ax=ax)
            ax.grid()
            ax.set_title('{}: data (blue)'.format(self.name))
            ax.set_ylabel('Illumination (sr)')
        else:
            lines[0].set_data(self.model_plotdate, self.dvals)

    def __str__(self):
        return 'earthheat__{0}'.format(self.node)


class DetectorHousingHeater(TelemData):
    def __init__(self, model):
        TelemData.__init__(self, model, '1dahtbon')
        self.n_mvals = 1
        self.fetch_attr = 'midvals'
        self.fetch_method = 'nearest'

    def get_dvals_tlm(self):
        dahtbon = self.model.fetch(self.msid, 'vals', 'nearest')
        return dahtbon == 'ON '

    def update(self):
        self.mvals = np.where(self.dvals, 1, 0)

    def __str__(self):
        return 'dh_heater'


class AcisPsmcSolarHeat(PrecomputedHeatPower):
    """Solar heating of PSMC box.  This is dependent on SIM-Z"""
    def __init__(self, model, node, pitch_comp, simz_comp, dh_heater_comp, P_pitches=None,
                 P_vals=None, dPs=None, var_func='linear',
                 tau=1732.0, ampl=0.05, epoch='2013:001:12:00:00', dh_heater=0.05):
        ModelComponent.__init__(self, model)
        self.n_mvals = 1
        self.node = self.model.get_comp(node)
        self.pitch_comp = self.model.get_comp(pitch_comp)
        self.simz_comp = self.model.get_comp(simz_comp)
        self.dh_heater_comp = self.model.get_comp(dh_heater_comp)
        self.P_pitches = np.array([45., 55., 70., 90., 150.] if (P_pitches is None)
                                  else P_pitches, dtype=np.float)
        self.dPs = np.zeros_like(self.P_pitches) if dPs is None else np.array(dPs, dtype=np.float)
        self.simz_lims = ((-400000.0, -85000.0),  # HRC-S
                          (-85000.0, 0.0),        # HRC-I
                          (0.0, 83000.0),         # ACIS-S
                          (83000.0, 400000.0))    # ACIS-I

        self.instr_names = ['hrcs', 'hrci', 'aciss', 'acisi']
        for i, instr_name in enumerate(self.instr_names):
            for j, pitch in enumerate(self.P_pitches):
                self.add_par('P_{0}_{1:d}'.format(instr_name, int(pitch)),
                             P_vals[i][j], min=-10.0, max=10.0)

        for j, pitch in enumerate(self.dPs):
            self.add_par('dP_{0:d}'.format(int(self.P_pitches[j])),
                         self.dPs[j], min=-1.0, max=1.0)

        self.add_par('tau', tau, min=1000., max=3000.)
        self.add_par('ampl', ampl, min=-1.0, max=1.0)
        self.add_par('dh_heater', dh_heater, min=-1.0, max=1.0)
        self.epoch = epoch
        self.var_func = getattr(self, var_func)

    @property
    def dvals(self):
        if not hasattr(self, 'pitches'):
            self.pitches = np.clip(self.pitch_comp.dvals, self.P_pitches[0], self.P_pitches[-1])

        if not hasattr(self, 'simzs'):
            self.simzs = self.simz_comp.dvals
            # Instrument 0=HRC-S 1=HRC-I 2=ACIS
            self.instrs = np.zeros(self.model.n_times, dtype=np.int8)
            for i, lims in enumerate(self.simz_lims):
                ok = (self.simzs > lims[0]) & (self.simzs <= lims[1])
                self.instrs[ok] = i

        if not hasattr(self, 't_days'):
            self.t_days = (self.pitch_comp.times
                           - DateTime(self.epoch).secs) / 86400.0
        if not hasattr(self, 't_phase'):
            time2000 = DateTime('2000:001:00:00:00').secs
            time2010 = DateTime('2010:001:00:00:00').secs
            secs_per_year = (time2010 - time2000) / 10.0
            t_year = (self.pitch_comp.times - time2000) / secs_per_year
            self.t_phase = t_year * 2 * np.pi

        # Interpolate power(pitch) for each instrument separately and make 2d
        # stack
        n_p = len(self.P_pitches)
        n_instr = len(self.instr_names)
        heats = []
        dPs = self.parvals[n_instr * n_p:(n_instr + 1) * n_p]
        dP_vals = Ska.Numpy.interpolate(dPs, self.P_pitches, self.pitches)
        d_heat = (dP_vals * self.var_func(self.t_days, self.tau)
                  + self.ampl * np.cos(self.t_phase)).ravel()

        for i in range(n_instr):
            P_vals = self.parvals[i * n_p:(i + 1) * n_p]
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
        self.instr_names = ['hrcs', 'hrci', 'aciss', 'acisi']
        for instr_name in self.instr_names:
            P_vals[instr_name] = []
            for pitch in self.P_pitches:
                P_vals[instr_name].append(getattr(self, 'P_{0}_{1:d}'
                                                  .format(instr_name, int(pitch))))
        colors = ['b', 'c', 'r', 'm']
        lines = ax.get_lines()
        if lines:
            for i, instr_name in enumerate(self.instr_names):
                lines[i].set_data(self.P_pitches, P_vals[instr_name])
                # lines[i * 2 + 1].set_data(self.P_pitches, P_vals[instr_name], '-b')
        else:
            for instr_name, color in zip(self.instr_names, colors):
                ax.plot(self.P_pitches, P_vals[instr_name], 'o-{}'.format(color), markersize=5,
                        label=instr_name)
            ax.set_title('{} solar heat input'.format(self.node.name))
            ax.set_xlim(40, 180)
            ax.grid()
            ax.legend(loc='best')

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


class AcisDpaStatePower(PrecomputedHeatPower):
    """Heating from ACIS electronics (ACIS config dependent CCDs, FEPs etc).
    Use commanded states and assign an effective power for each "unique" power
    state.  See dpa/NOTES.power.

    Parameters
    ----------

    Returns
    -------

    """
    def __init__(self, model, node, mult=1.0,
                 fep_count=None, ccd_count=None,
                 vid_board=None, clocking=None,
                 pow_states=None):
        super(AcisDpaStatePower, self).__init__(model)
        self.node = self.model.get_comp(node)
        self.fep_count = self.model.get_comp(fep_count)
        self.ccd_count = self.model.get_comp(ccd_count)
        self.vid_board = self.model.get_comp(vid_board)
        self.clocking = self.model.get_comp(clocking)
        if pow_states is None:
            pow_states = ["0xxx", "1xxx", "2xxx", "3xx0",
                          "3xx1", "4xxx", "5xxx", "66x0",
                          "6611", "6xxx"]
        for ps in pow_states:
            self.add_par('pow_%s' % ps, 20, min=10, max=100)
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
            plot_cxctime(self.model.times, powers, ls='-',
                         color='#d92121', fig=fig, ax=ax)
            plot_cxctime(self.model.times, self.dvals,
                         color='#386cb0', ls='-', fig=fig, ax=ax)
            ax.grid()
            ax.set_title('{}: model (red) and data (blue)'.format(self.name))
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
        """ """
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
            plot_cxctime(self.model.times, self.mvals, '#386cb0', fig=fig, ax=ax)
            ax.grid()
            ax.set_title('{}: data (blue)'.format(self.name))
            ax.set_ylabel('Power')


class ThermostatHeater(ActiveHeatPower):
    """Thermostat heater (no deadband): heat = P for T < T_set)."""
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
        """ """
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
            plot_cxctime(self.model.times, self.mvals, '#386cb0', fig=fig, ax=ax)
            ax.grid()
            ax.set_title('{}: data (blue)'.format(self.name))
            ax.set_ylabel('Power')


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
    def __init__(self, model, node, time, P=0.0, id=''):
        super(StepFunctionPower, self).__init__(model)
        self.time = DateTime(time).secs
        self.node = self.model.get_comp(node)
        self.n_mvals = 1
        self.id = id
        self.add_par('P', P, min=-10.0, max=10.0)

    def __str__(self):
        return f'step_power{self.id}__{self.node}'

    def get_dvals_tlm(self):
        """ """
        return np.zeros_like(self.model.times)

    def update(self):
        """Update the model prediction as a precomputed heat."""
        self.mvals = np.full_like(self.model.times, fill_value=self.P)
        idx0 = np.searchsorted(self.model.times, self.time)
        self.mvals[:idx0] = 0.0
        self.tmal_ints = (tmal.OPCODES['precomputed_heat'],
                          self.node.mvals_i,  # dy1/dt index
                          self.mvals_i,
                          )
        self.tmal_floats = ()

    def plot_data__time(self, fig, ax):
        lines = ax.get_lines()
        if lines:
            lines[0].set_data(self.model_plotdate, self.mvals)
        else:
            plot_cxctime(self.model.times, self.mvals, '#386cb0', fig=fig, ax=ax)
            ax.grid()
            ax.set_title('{}: data (blue)'.format(self.name))
            ax.set_ylabel('Power')


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
        self.add_par('P', P, min=-10.0, max=10.0)

    def __str__(self):
        return f'{self.state_msid}_{self.state_val_str}'

    def get_dvals_tlm(self):
        """
        Return an array of power values where the power is ``P`` when the
        ``state_val`` for ``state_msid`` is matched, otherwise it is 0.0.
        """
        dvals = self.model.fetch(self.state_msid, 'vals', 'nearest') == self.state_val
        return dvals

    def update(self):
        self.mvals = np.zeros_like(self.model.times)
        self.mvals[self.dvals] = self.P
        self.tmal_ints = (tmal.OPCODES['precomputed_heat'],
                          self.node.mvals_i,  # dy1/dt index
                          self.mvals_i,
                          )
        self.tmal_floats = ()

    def plot_data__time(self, fig, ax):
        lines = ax.get_lines()
        if not lines:
            plot_cxctime(self.model.times, self.dvals, '#386cb0', fig=fig, ax=ax)
            ax.grid()
            ax.set_title(f'{self.name}: state match dvals (blue)')
            ax.set_ylabel(f'{self.state_msid.upper()} == {repr(self.state_val)}')


@jit(nopython=True)
def quat_to_transform_transpose(q):
    """

    Parameters
    ----------
    q :
        

    Returns
    -------
    type
        

    """
    x, y, z, w = q
    xx2 = 2 * x * x
    yy2 = 2 * y * y
    zz2 = 2 * z * z
    xy2 = 2 * x * y
    wz2 = 2 * w * z
    zx2 = 2 * z * x
    wy2 = 2 * w * y
    yz2 = 2 * y * z
    wx2 = 2 * w * x

    rmat = np.empty((3, 3), np.float64)
    rmat[0, 0] = 1. - yy2 - zz2
    rmat[1, 0] = xy2 - wz2
    rmat[2, 0] = zx2 + wy2
    rmat[0, 1] = xy2 + wz2
    rmat[1, 1] = 1. - xx2 - zz2
    rmat[2, 1] = yz2 - wx2
    rmat[0, 2] = zx2 - wy2
    rmat[1, 2] = yz2 + wx2
    rmat[2, 2] = 1. - xx2 - yy2

    return rmat


@jit(nopython=True)
def get_dists_lons_lats(ephems, q_atts):
    n_vals = len(ephems)
    lons = np.empty(n_vals, dtype=np.float64)
    lats = np.empty(n_vals, dtype=np.float64)
    dists = np.empty(n_vals, dtype=np.float64)

    ephems = -ephems  # swap to Chandra-body-centric

    for ii in range(n_vals):
        ephem = ephems[ii]
        q_att = q_atts[ii]
        # Earth vector in Chandra body coords (p_earth_body)
        xform = quat_to_transform_transpose(q_att)
        peb = np.dot(xform, ephem)

        # Convert cartesian to spherical coords
        s = np.hypot(peb[0], peb[1])
        dists[ii] = np.hypot(s, peb[2])
        lons[ii] = np.arctan2(peb[1], peb[0])
        lats[ii] = np.arctan2(peb[2], s)

    return dists, lons, lats
