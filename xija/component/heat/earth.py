# Licensed under a 3-clause BSD style license - see LICENSE.rst

from xija import tmal
from xija.component.base import ModelComponent
from xija.component.heat.base import PrecomputedHeatPower

from astropy.io import fits
import numpy as np
from itertools import count
from pathlib import Path
import os
import astropy.units as u
import glob
from numba import jit
import re

try:
    from Ska.Matplotlib import plot_cxctime
    from Chandra.Time import DateTime
except ImportError:
    pass

__all__ = ["EarthHeat"]


class EarthHeat(PrecomputedHeatPower):
    """Earth heating of ACIS cold radiator (attitude, ephem dependent)"""

    use_earth_vis_grid = True
    earth_vis_grid_path = Path(__file__).parent / "earth_vis_grid_nside32.fits.gz"

    def __init__(
        self,
        model,
        node,
        orbitephem0_x,
        orbitephem0_y,
        orbitephem0_z,
        aoattqt1,
        aoattqt2,
        aoattqt3,
        aoattqt4,
        solarephem0_x=None,
        solarephem0_y=None,
        solarephem0_z=None,
        k=1.0,
        k2=None,
    ):
        ModelComponent.__init__(self, model)
        self.node = self.model.get_comp(node)
        self.orbitephem0_x = self.model.get_comp(orbitephem0_x)
        self.orbitephem0_y = self.model.get_comp(orbitephem0_y)
        self.orbitephem0_z = self.model.get_comp(orbitephem0_z)
        self.aoattqt1 = self.model.get_comp(aoattqt1)
        self.aoattqt2 = self.model.get_comp(aoattqt2)
        self.aoattqt3 = self.model.get_comp(aoattqt3)
        self.aoattqt4 = self.model.get_comp(aoattqt4)
        if solarephem0_x is None:
            self.solarephem0_x = None
        else:
            self.solarephem0_x = self.model.get_comp(solarephem0_x)
        if solarephem0_y is None:
            self.solarephem0_y = None
        else:
            self.solarephem0_y = self.model.get_comp(solarephem0_y)
        if solarephem0_z is None:
            self.solarephem0_z = None
        else:
            self.solarephem0_z = self.model.get_comp(solarephem0_z)
        self.use_earth_phase = np.all(
            [getattr(self, f"solarephem0_{ax}") is not None for ax in "xyz"]
        )
        self.n_mvals = 1
        self.add_par("k", k, min=0.0, max=2.0)
        if k2 is not None:
            self.add_par("k2", k2, min=0.0, max=2.0)
        self.earth_phase = 1.0

    def calc_earth_vis_from_grid(self, ephems, q_atts):
        import astropy_healpix

        healpix = astropy_healpix.HEALPix(nside=32, order="nested")

        if not hasattr(self, "earth_vis_grid"):
            with fits.open(self.earth_vis_grid_path) as hdus:
                hdu = hdus[0]
                hdr = hdu.header
                vis_grid = hdu.data / hdr["scale"]  # 12288 x 100
                self.__class__.earth_vis_grid = vis_grid

                alts = np.logspace(
                    np.log10(hdr["alt_min"]), np.log10(hdr["alt_max"]), hdr["n_alt"]
                )
                self.__class__.log_earth_vis_dists = np.log(hdr["earthrad"] + alts)

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
        if not hasattr(self, "_dvals") and not self.get_cached():
            # Collect individual MSIDs for use in calc_earth_vis()
            ephem_xyzs = [
                getattr(self, "orbitephem0_{}".format(x)) for x in ("x", "y", "z")
            ]
            aoattqt_1234s = [getattr(self, "aoattqt{}".format(x)) for x in range(1, 5)]
            # Note: the copy() here is so that the array becomes contiguous in
            # memory and allows numba to run faster (and avoids NumbaPerformanceWarning:
            # np.dot() is faster on contiguous arrays).
            ephems = np.array([x.dvals for x in ephem_xyzs]).transpose().copy()
            q_atts = np.array([x.dvals for x in aoattqt_1234s]).transpose()

            # Q_atts can have occasional bad values, maybe because the
            # Ska eng 5-min "midvals" are not lined up, but I'm not quite sure.
            # TODO: this (legacy) solution isn't great.  Investigate what's
            # really happening.
            q_norm = np.sqrt(np.sum(q_atts**2, axis=1))
            bad = np.abs(q_norm - 1.0) > 0.1
            if np.any(bad):
                print(
                    "Replacing bad midval quaternions with [1,0,0,0] at times "
                    f"{self.model.times[bad]}"
                )
                q_atts[bad, :] = [0.0, 0.0, 0.0, 1.0]
            q_atts[~bad, :] = q_atts[~bad, :] / q_norm[~bad, np.newaxis]

            # Finally initialize dvals and update in-place appropriately
            self._dvals = np.empty(self.model.n_times, dtype=float)
            if self.use_earth_vis_grid:
                self.calc_earth_vis_from_grid(ephems, q_atts)
            else:
                self.calc_earth_vis_from_taco(ephems, q_atts)

            # This next bit optionally checks to see if the solar ephemeris
            # was passed in, and if it was it computes the fraction of the
            # Earth's surface that is illuminated by the Sun. Originally
            # discussed at:
            # https://occweb.cfa.harvard.edu/twiki/bin/view/ThermalWG/MeetingNotes2022x03x03
            solar_xyzs = [getattr(self, f"solarephem0_{x}") for x in "xyz"]

            if self.use_earth_phase:
                solars = np.array([x.dvals for x in solar_xyzs]).transpose().copy()

                cos = np.sum(ephems * solars, axis=1)
                es = np.sum(ephems * ephems, axis=1) * np.sum(solars * solars, axis=1)
                cos /= np.sqrt(es)
                self.earth_phase = 0.5 * (1.0 + cos)

            self.put_cache()

        return self._dvals

    def put_cache(self):
        if os.path.exists("esa_cache"):
            cachefile = "esa_cache/{}-{}.npz".format(
                self.model.datestart, self.model.datestop
            )
            np.savez(cachefile, times=self.model.times, dvals=self.dvals)

    def get_cached(self):
        """Find a cached version of the Earth solid angle values from
        file if possible.

        Parameters
        ----------

        Returns
        -------

        """
        dts = {}  # delta times for each matching file
        filenames = glob.glob("esa_cache/*.npz")
        for name in filenames:
            re_date = r"\d\d\d\d:\d\d\d:\d\d:\d\d:\d\d\.\d\d\d"
            re_cache_file = r"({})-({})".format(re_date, re_date)
            m = re.search(re_cache_file, name)
            if m:
                f_datestart, f_datestop = m.groups()
                if (
                    f_datestart <= self.model.datestart
                    and f_datestop >= self.model.datestop
                ):
                    dts[name] = DateTime(f_datestop) - DateTime(f_datestart)
        if dts:
            cachefile = sorted(dts.items(), key=lambda x: x[1])[0][0]
            arrays = np.load(cachefile)
            self._dvals = self.model.interpolate_data(
                arrays["dvals"], arrays["times"], comp=self
            )
            return True
        else:
            return False

    def update(self):
        self.mvals = self.k * self.dvals
        if self.use_earth_phase:
            self.mvals += self.k2 * self.earth_phase * self.dvals
        self.tmal_ints = (
            tmal.OPCODES["precomputed_heat"],
            self.node.mvals_i,  # dy1/dt index
            self.mvals_i,  # mvals with precomputed heat input
        )
        self.tmal_floats = ()

    def plot_earth_phase__time(self, fig, ax):
        lines = ax.get_lines()
        if not lines:
            plot_cxctime(
                self.model.times,
                self.earth_phase,
                ls="-",
                color="#386cb0",
                fig=fig,
                ax=ax,
            )
            ax.grid()
            ax.set_title("Earth Phase")
            ax.set_ylabel("Earth Phase")
        else:
            lines[0].set_data(self.model_plotdate, self.dvals)

    def plot_data__time(self, fig, ax):
        lines = ax.get_lines()
        if not lines:
            plot_cxctime(
                self.model.times, self.dvals, ls="-", color="#386cb0", fig=fig, ax=ax
            )
            ax.grid()
            ax.set_title("{}: data (blue)".format(self.name))
            ax.set_ylabel("Illumination (sr)")
        else:
            lines[0].set_data(self.model_plotdate, self.dvals)

    def __str__(self):
        return "earthheat__{0}".format(self.node)


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
    rmat[0, 0] = 1.0 - yy2 - zz2
    rmat[1, 0] = xy2 - wz2
    rmat[2, 0] = zx2 + wy2
    rmat[0, 1] = xy2 + wz2
    rmat[1, 1] = 1.0 - xx2 - zz2
    rmat[2, 1] = yz2 - wx2
    rmat[0, 2] = zx2 - wy2
    rmat[1, 2] = yz2 + wx2
    rmat[2, 2] = 1.0 - xx2 - yy2

    return rmat
