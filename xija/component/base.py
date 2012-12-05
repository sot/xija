import numpy as np

try:
    from Ska.Matplotlib import plot_cxctime, cxctime2plotdate
except ImportError:
    pass

from .. import tmal


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
    """Model component base class"""
    def __init__(self, model):
        self.model = model
        self.n_mvals = 0
        self.predict = False  # Predict values for this model component
        self.pars = []
        self.data = None
        self.data_times = None
        self.model_plotdate = cxctime2plotdate(self.model.times)

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

    def get_par(self, name):
        for par in self.pars:
            if par.name == name:
                return par
        else:
            raise ValueError('No par named "{}" in {}',
                             self.__class__.__name__)

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

    def get_dvals_tlm(self):
        return np.zeros_like(self.model.times)

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


class TelemData(ModelComponent):
    times = property(lambda self: self.model.times)

    def __init__(self, model, msid, mval=True, data=None,
                 fetch_attr='vals'):
        super(TelemData, self).__init__(model)
        self.msid = msid
        self.n_mvals = 1 if mval else 0
        self.predict = False
        self.data = data
        self.data_times = None
        self.fetch_attr = fetch_attr

    def get_dvals_tlm(self):
        return self.model.fetch(self.msid, attr=self.fetch_attr)

    def plot_data__time(self, fig, ax):
        lines = ax.get_lines()
        if not lines:
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
    """Time-series dataset for prediction.

    If the ``sigma`` value is negative then sigma is computed from the
    node data values as the specified percent of the data standard
    deviation.  The default ``sigma`` value is -10, so this implies
    using a sigma of 10% of the data standard deviation.  If ``sigma``
    is set to 0 then the fit statistic is set to 0.0 for this node.

    :param model: parent model
    :param msid: MSID for telemetry data
    :param name: component name (default=``msid``)
    :param sigma: sigma value used in chi^2 fit statistic
    :param quant: use quantized stats (not currently implemented)
    :param predict: compute prediction for this node (default=True)
    :param mask: Mask component for masking values from fit statistic
    :param data: Node data (None or a single value)
    """
    def __init__(self, model, msid, sigma=-10, quant=None,
                 predict=True, mask=None, name=None, data=None,
                 fetch_attr='vals'):
        TelemData.__init__(self, model, msid, data=data,
                           fetch_attr=fetch_attr)
        self._sigma = sigma
        self.quant = quant
        self.predict = predict
        self.mask = model.get_comp(mask)
        self._name = name or msid

    def __str__(self):
        return self._name

    @property
    def randx(self):
        """Random X-offset for plotting which is a uniform distribution
        with width = self.quant or 1.0
        """
        if not hasattr(self, '_randx'):
            dx = self.quant or 1.0
            self._randx = np.random.uniform(low=-dx / 2.0, high=dx / 2.0,
                                            size=self.model.n_times)
        return self._randx

    @property
    def sigma(self):
        if self._sigma < 0:
            self._sigma = self.dvals.std() * (-self._sigma / 100.0)
        return self._sigma

    def calc_stat(self):
        if self.sigma == 0:
            return 0.0
        if self.mask is None:
            resid = self.dvals - self.mvals
        else:
            resid = self.dvals[self.mask.mask] - self.mvals[self.mask.mask]
        return np.sum(resid ** 2 / self.sigma ** 2)

    def plot_data__time(self, fig, ax):
        lines = ax.get_lines()
        if not lines:
            plot_cxctime(self.model.times, self.dvals, '-b', fig=fig, ax=ax)
            plot_cxctime(self.model.times, self.mvals, '-r', fig=fig, ax=ax)
            ax.grid()
            ax.set_title('{}: model (red) and data (blue)'.format(self.name))
            ax.set_ylabel('Temperature (degC)')
        else:
            lines[0].set_ydata(self.dvals)
            lines[1].set_ydata(self.mvals)

    def plot_resid__time(self, fig, ax):
        lines = ax.get_lines()
        resids = self.dvals - self.mvals
        if self.mask:
            resids[~self.mask.mask] = np.nan

        if not lines:
            plot_cxctime(self.model.times, resids, '-b', fig=fig, ax=ax)
            ax.grid()
            ax.set_title('{}: residuals (data - model)'.format(self.name))
            ax.set_ylabel('Temperature (degC)')
        else:
            lines[0].set_ydata(resids)

    def plot_resid__data(self, fig, ax):
        lines = ax.get_lines()
        resids = self.dvals - self.mvals
        if self.mask:
            resids[~self.mask.mask] = np.nan

        if not lines:
            ax.plot(self.dvals + self.randx, resids, ',b', mew=0.0)
            ax.grid()
            ax.set_title('{}: residuals (data - model) vs data'.format(self.name))
            ax.set_ylabel('Temperature (degC)')
        else:
            lines[0].set_ydata(resids)


class Coupling(ModelComponent):
    """\
    First-order coupling between Nodes `node1` and `node2`
    ::

      dy1/dt = -(y1 - y2) / tau
    """
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

    def get_dvals_tlm(self):
        vals = self.model.fetch(self.msid, attr=self.fetch_attr)
        # Pitch values outside of 45 to 180 are not possible.  Normally
        # this is geniune bad data that gets sent down in safe mode when
        # the spacecraft is at normal sun.  So set these values to 90.
        bad = (vals >= 180.0) | (vals <= 45.0)
        vals[bad] = 90.0
        # Thermal models typically calibrated between 45 to 170 degrees
        # so clip values to that range.
        vals.clip(45.001, 169.999, out=vals)
        return vals

    def __str__(self):
        return 'pitch'


class AcisFPtemp(Node):
    """Make a wrapper around MSID FPTEMP_11 because that currently comes from
    the eng_archive in K instead of C.
    """
    def __init__(self, model, mask=None):
        Node.__init__(self, model, 'fptemp_11', mask=mask)

    def get_dvals_tlm(self):
        fptemp = self.model.fetch(self.msid, 'vals', 'nearest')
        return fptemp - 273.15

    def __str__(self):
        return 'fptemp'


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
