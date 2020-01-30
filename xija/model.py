# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Xija - framework to model complex time-series data using a network of
coupled nodes with pluggable model components that define the node
interactions.
"""
from __future__ import print_function

import os
import json
import ctypes
from collections import OrderedDict
from six.moves import cStringIO as StringIO
import six

import numpy as np

from . import component
from . import tmal

try:
    # Optional packages for model fitting or use on HEAD LAN
    from Chandra.Time import DateTime
    from astropy.io import ascii
    import Ska.Numpy
    import Chandra.cmd_states
    import Ska.DBI
except ImportError:
    pass

import pyyaks.context as pyc
from .files import files as xija_files
from . import clogging

# HDF5 version of commanded states table
H5FILE = '/proj/sot/ska/data/cmd_states/cmd_states.h5'

src = pyc.CONTEXT['src'] if 'src' in pyc.CONTEXT else pyc.ContextDict('src')
files = (pyc.CONTEXT['file'] if 'file' in pyc.CONTEXT else
         pyc.ContextDict('files', basedir=os.getcwd()))
files.update(xija_files)

if 'debug' in globals():
    from IPython.Debugger import Tracer
    pdb_settrace = Tracer()

logger = clogging.config_logger('xija', level=clogging.INFO)

DEFAULT_DT = 328.0
dt_factors = np.array([1.0, 0.5, 0.25, 0.2, 0.125, 0.1, 0.05, 0.025])

#int calc_model(int n_times, int n_preds, int n_tmals, double dt,
#               double **mvals, int **tmal_ints, double **tmal_floats)


def convert_type_star_star(array, ctype_type):
    f4ptr = ctypes.POINTER(ctype_type)
    return (f4ptr * len(array))(*[row.ctypes.data_as(f4ptr) for row in array])


class FetchError(Exception):
    pass


class XijaModel(object):
    """Xija model class to encapsulate all ModelComponents and provide the
    infrastructure to define and evaluate models.

    The parameters ``name``, ``start``, and ``stop`` are determined as follows:

    - If a model specification is provided then that sets the default values
      for keywords that are not supplied to the class init call.
    - ``evolve_method = 1`` uses the original ODE solver which treats every
      two steps as a full RK2 step.
    - ``evolve_method = 2`` uses the new ODE solver which treats every step
      as a full RK2 step, and optionally allows for RK4 if ``rk4 = 1``.  
    - Otherwise defaults are: ``name='xijamodel'``, ``start = stop - 45 days``,
      ``stop = NOW - 30 days``, ``dt = 328 secs``, ``evolve_method = 1``, 
      ``rk4 = 0``

    :param name: model name
    :param start: model start time (any DateTime format)
    :param stop: model stop time (any DateTime format)
    :param dt: delta time step (default=328 sec)
    :param model_spec: model specification (None | filename | dict)
    :param cmd_states: commanded states input (None | structured array)
    :param evolve_method: choose method to evolve ODE (None | 1 or 2, default 1)
    :param rk4: use 4th-order Runge-Kutta to evolve ODE, only works with
           evolve_method == 2 (None | 0 or 1, default 0)
    """
    def __init__(self, name=None, start=None, stop=None, dt=None,
                 model_spec=None, cmd_states=None, evolve_method=None,
                 rk4=None):

        # If model_spec supplied as a string then read model spec as a dict
        if isinstance(model_spec, six.string_types):
            model_spec = json.load(open(model_spec, 'r'))
        # If a model_spec is now available (dict) then use as kwarg defaults
        if model_spec:
            stop = stop or model_spec['datestop']
            start = start or model_spec['datestart']
            name = name or model_spec['name']
            dt = dt or model_spec['dt']
            evolve_method = evolve_method or model_spec.get('evolve_method', None)
            rk4 = rk4 or model_spec.get('rk4', None)

        if stop is None:
            stop = DateTime() - 30
        if start is None:
            start = DateTime(stop) - 45
        if name is None:
            name = 'xijamodel'
        if dt is None:
            dt = DEFAULT_DT
        if evolve_method is None:
            evolve_method = 1
        if rk4 is None:
            rk4 = 0

        self.name = name
        self.comp = OrderedDict()
        self.dt = self._get_allowed_timestep(dt)
        self.dt_ksec = self.dt / 1000.
        self.times = self._eng_match_times(start, stop)
        self.tstart = self.times[0]
        self.tstop = self.times[-1]
        self.ksecs = (self.times - self.tstart) / 1000.
        self.datestart = DateTime(self.tstart).date
        self.datestop = DateTime(self.tstop).date
        self.n_times = len(self.times)
        self.evolve_method = evolve_method
        self.rk4 = rk4

        try:
            self.bad_times = model_spec['bad_times']
        except:
            pass
        else:
            self.bad_times_indices = []
            for t0, t1 in self.bad_times:
                t0, t1 = DateTime([t0, t1]).secs
                i0, i1 = np.searchsorted(self.times, [t0, t1])
                if i1 > i0:
                    self.bad_times_indices.append((i0, i1))

        self.pars = []
        if model_spec:
            self._set_from_model_spec(model_spec)
        self.cmd_states = cmd_states

    def _get_allowed_timestep(self, dt):
        """
        This method ensures that only certain timesteps are chosen,
        which are integer multiples of 8.2 and where 328.0/dt is an
        integer. 
        """
        if dt > DEFAULT_DT:
            logger.warning("dt = %g s greater than upper "
                           "limit of %g s! " % (dt, DEFAULT_DT) +
                           "Setting dt = %g s." % DEFAULT_DT)
            return DEFAULT_DT
        dt_factor = dt / DEFAULT_DT
        idx = np.argmin(np.abs(dt_factor-dt_factors))
        dt = DEFAULT_DT*dt_factors[idx]
        logger.debug("Using dt = %g s." % dt)
        return dt

    def _set_from_model_spec(self, model_spec):
        for comp in model_spec['comps']:
            ComponentClass = getattr(component, comp['class_name'])
            args = comp['init_args']
            kwargs = dict((str(k), v) for k, v in comp['init_kwargs'].items())
            self.add(ComponentClass, *args, **kwargs)

        pars = model_spec['pars']
        if len(pars) != len(self.pars):
            raise ValueError('Number of spec pars does not match model: \n'
                             '{0}\n{1}'.format(len(pars), len(self.pars)))
        for par, specpar in zip(self.pars, pars):
            for attr in specpar:
                setattr(par, attr, specpar[attr])

    def inherit_from_model_spec(self, inherit_spec):
        """Inherit parameter values from any like-named parameters within the
        inherit_spec model specification.  This is useful for making a new
        variation of an existing model.
        """
        try:
            inherit_spec = json.load(open(inherit_spec, 'r'))
        except:
            pass

        inherit_pars = {par['full_name']: par for par in inherit_spec['pars']}
        for par in self.pars:
            if par.full_name in inherit_pars:
                logger.info("Inheriting par {}".format(par.full_name))
                par.val = inherit_pars[par.full_name]['val']
                par.min = inherit_pars[par.full_name]['min']
                par.max = inherit_pars[par.full_name]['max']
                par.frozen = inherit_pars[par.full_name]['frozen']
                par.fmt = inherit_pars[par.full_name]['fmt']

    def _eng_match_times(self, start, stop):
        """Return an array of times between ``start`` and ``stop`` at ``dt``
        sec intervals.  The times are roughly aligned (within 1 sec) to the
        timestamps in the '5min' (328 sec) Ska eng archive data.
        """
        time0 = 410270764.0
        i0 = int((DateTime(start).secs - time0) / self.dt) + 1
        i1 = int((DateTime(stop).secs - time0) / self.dt)
        return time0 + np.arange(i0, i1) * self.dt

    def _get_cmd_states(self):
        if not hasattr(self, '_cmd_states'):
            logger.info('Reading commanded states DB over %s to %s' %
                        (self.datestart, self.datestop))
            for dbi in ('hdf5', 'sybase'):
                try:
                    states = Chandra.cmd_states.fetch_states(
                        self.datestart, self.datestop, dbi=dbi)
                    break
                except IOError as err:
                    logger.info('Warning: ' + str(err))
                    pass
            else:
                # Both hdf5 and sybase failed
                raise IOError('Could not read commanded states from '
                              'HDF5 or sybase tables')

            self._cmd_states = Chandra.cmd_states.interpolate_states(
                states, self.times)

        return self._cmd_states

    def _set_cmd_states(self, states):
        """Set the states that define component data inputs.

        :param states: numpy structured array
        """
        if states is not None:
            if (states[0]['tstart'] >= self.times[0] or
                states[-1]['tstop'] <= self.times[-1]):
                raise ValueError('cmd_states time range too small:\n'
                                 '{} : {} versus {} : {}'.format(
                        states[0]['tstart'], states[-1]['tstop'],
                        self.times[0], self.times[-1]))

            indexes = np.searchsorted(states['tstop'], self.times)
            self._cmd_states = states[indexes]

    cmd_states = property(_get_cmd_states, _set_cmd_states)
    """test cmdstats"""

    def fetch(self, msid, attr='vals', method='linear'):
        """Get data from the Chandra engineering archive."""
        tpad = DEFAULT_DT*5.0
        datestart = DateTime(self.tstart - tpad).date
        datestop = DateTime(self.tstop + tpad).date
        logger.info('Fetching msid: %s over %s to %s' %
                    (msid, datestart, datestop))
        try:
            import Ska.engarchive.fetch_sci as fetch
            tlm = fetch.MSID(msid, datestart, datestop, stat='5min')
            tlm.filter_bad_times()
        except ImportError:
            raise ValueError('Ska.engarchive.fetch not available')
        if tlm.times[0] > self.tstart or tlm.times[-1] < self.tstop:
            raise ValueError('Fetched telemetry does not span model start and '
                             'stop times for {}'.format(msid))
        vals = Ska.Numpy.interpolate(getattr(tlm, attr), tlm.times,
                                     self.times, method=method)
        return vals

    def interpolate_data(self, data, times, comp=None):
        """Interpolate supplied ``data`` values at the model times using
        nearest-neighbor or state value interpolation.

        The ``times`` arg can be either a 1-d or 2-d ndarray.  If 1-d,
        then ``data`` is interpreted as a set of values at the specified
        ``times``.  If 2-d then ``data`` is interpreted as a set of binned
        state values with ``tstarts = times[0, :]`` and
        ``tstops = times[1, :]``.
        """
        if times is None:
            if len(data) != self.n_times:
                raise ValueError('Data length not equal to model times'
                                 ' for {} component'.format(comp))
            return data

        if len(data) != times.shape[-1]:
            raise ValueError('Data length not equal to data times'
                             ' for {} component'.format(comp))

        if times.ndim == 1:  # Data value specification
            vals = Ska.Numpy.interpolate(data, times, self.times,
                                         method='nearest')
        elif times.ndim == 2:  # State-value specification
            tstarts = times[0]
            tstops = times[1]
            if self.times[0] < tstarts[0] or self.times[-1] > tstops[-1]:
                raise ValueError('Model times extend outside the state value'
                                 ' data_times for component {}'.format(comp))
            indexes = np.searchsorted(tstops, self.times)
            vals = data[indexes]
        else:
            raise ValueError('data_times for {} has {} dimensions, '
                             ' must be either 1 or 2'.format(comp, times.ndim))
        return vals

    def add(self, ComponentClass, *args, **kwargs):
        """Add a new component to the model"""
        comp = ComponentClass(self, *args, **kwargs)
        # Store args and kwargs used to initialize object for later object
        # storage and re-creation
        comp.init_args = args
        comp.init_kwargs = kwargs
        self.comp[comp.name] = comp
        for par in comp.pars:
            self.pars.append(par)

        return comp

    comps = property(lambda self: list(self.comp.values()))
    """List of model components"""

    def get_comp(self, name):
        """Get a model component.  Works with either a string or a component
        object"""
        return None if name is None else self.comp[str(name)]

    @property
    def model_spec(self):
        """Generate a full model specification data structure for this
        model"""
        model_spec = dict(name=self.name,
                          comps=[],
                          dt=self.dt,
                          datestart=self.datestart,
                          datestop=self.datestop,
                          tlm_code=None,
                          mval_names=[],
                          evolve_method=self.evolve_method,
                          rk4=self.rk4)

        if hasattr(self, 'bad_times'):
            model_spec['bad_times'] = self.bad_times

        model_spec['pars'] = [dict(par) for par in self.pars]

        stringfy = lambda x: (str(x) if isinstance(x, component.ModelComponent)
                              else x)
        for comp in self.comps:
            init_args = [stringfy(x) for x in comp.init_args]
            init_kwargs = dict((k, stringfy(v))
                               for k, v in comp.init_kwargs.items())
            model_spec['comps'].append(dict(class_name=comp.__class__.__name__,
                                            name=comp.name,
                                            init_args=init_args,
                                            init_kwargs=init_kwargs))
        return model_spec

    def write_vals(self, filename):
        """Write dvals and mvals for each model component (as applicable) to an
        ascii table file.  Some component have neither (couplings), some have
        just dvals (TelemData), others have both (Node, AcisDpaPower).
        Everything is guaranteed to be time synced, so write a single time
        column.
        """
        colvals = OrderedDict(time=self.times)
        for comp in self.comps:
            if hasattr(comp, 'dvals'):
                colvals[comp.name + '_data'] = comp.dvals
            if hasattr(comp, 'mvals') and comp.predict:
                colvals[comp.name + '_model'] = comp.mvals

        ascii.write(colvals, filename, names=list(colvals.keys()))

    def write(self, filename, model_spec=None):
        """Write the model specification as JSON or Python to a file.

        If the file name ends with ".py" then the output will the Python
        code to create the model (using ``get_model_code()``), otherwise
        the JSON model specification will be written.

        :param filename: output filename
        :param model_spec: model spec structure (optional)
        """
        if model_spec is None:
            model_spec = self.model_spec

        with open(filename, 'w') as f:
            if filename.endswith('.py'):
                f.write(self.get_model_code())
            else:
                json.dump(model_spec, f, sort_keys=True, indent=4)

    def get_model_code(self):
        """Return Python code that will create the current model.

        This is useful during model development as a way to derive from and
        modify existing models while retaining good parameter values.

        :returns: string of Python code
        """
        out = StringIO()
        ms = self.model_spec

        model_call = "model = xija.XijaModel({}, start={}, stop={}, dt={},\n"
        model_call += "evolve_method={} rk4={}\n"

        print("import sys", file=out)
        print("import xija\n", file=out)
        print(model_call.format(repr(ms['name']), repr(ms['datestart']),
                                repr(ms['datestop']), repr(ms['dt']),
                                repr(ms['evolve_method']), repr(ms['rk4'])), file=out)

        for comp in ms['comps']:
            args = [repr(x) for x in comp['init_args']]
            kwargs = ['{}={}'.format(k, repr(v))
                      for k, v in comp['init_kwargs'].items()]
            print('model.add(xija.{},'.format(comp['class_name']), file=out)
            for arg in args:
                print('          {},'.format(arg), file=out)
            for kwarg in kwargs:
                print('          {},'.format(kwarg), file=out)
            print('         )', file=out)

        parattrs = ('val', 'min', 'max', 'fmt', 'frozen')
        last_comp_name = None
        for par in ms['pars']:
            comp_name = par['comp_name']
            if comp_name != last_comp_name:
                print('# Set {} component parameters'.format(comp_name), file=out)
                print('comp = model.get_comp({})\n' \
                    .format(repr(comp_name)), file=out)
            print('par = comp.get_par({})'.format(repr(par['name'])), file=out)
            par_upds = ['{}={}'.format(attr, repr(par[attr]))
                        for attr in parattrs]
            print('par.update(dict({}))\n'.format(', '.join(par_upds)), file=out)
            last_comp_name = comp_name

        if hasattr(self, 'bad_times'):
            print("model.bad_times = {}".format(repr(self.bad_times)), file=out)

        print("if len(sys.argv) > 1:", file=out)
        print("    model.write(sys.argv[1])", file=out)

        return out.getvalue()

    def _get_parvals(self):
        """Return a (read-only) tuple of parameter values."""
        return tuple(par.val for par in self.pars)

    def _set_parvals(self, vals):
        """Set the full list of parameter values.  No provision is made for
        setting individual elements or slicing (use self.pars directly in this
        case)."""
        if len(vals) != len(self.pars):
            raise ValueError('Length mismatch setting parvals {} vs {}'.format(
                    len(self.pars), len(vals)))
        for par, val in zip(self.pars, vals):
            par.val = val

    parvals = property(_get_parvals, _set_parvals)

    @property
    def parnames(self):
        """Return a tuple of all model parameter names"""
        return tuple(par.full_name for par in self.pars)

    def make(self):
        """Call self.make_mvals and self.make_tmal to prepare for model evaluation
        once all model components have been added."""
        self.make_mvals()
        self.make_tmal()

    def make_mvals(self):
        """Initialize the global mvals (model values) array.  This is an
        N (rows) x n_times (cols) array that contains all data needed
        to compute the model prediction.  All rows are initialized to
        relevant data values (e.g. node temps, time-dependent power,
        external temperatures, etc).  In the model calculation some
        rows will be overwritten with predictions.
        """
        # Select components with data values, and from those select ones that
        # get predicted and those that do not get predicted
        comps = [x for x in self.comps if x.n_mvals]
        preds = [x for x in comps if x.predict]
        unpreds = [x for x in comps if not x.predict]

        # Register the location of component mvals in global mvals
        i = 0
        for comp in preds + unpreds:
            comp.mvals_i = i
            i += comp.n_mvals

        # Stack the input dvals.  This *copies* the data values.
        self.n_preds = len(preds)
        self.mvals = np.hstack([comp.dvals for comp in preds + unpreds])
        self.mvals.shape = (len(comps), -1)  # why doesn't this use vstack?
        self.cvals = self.mvals[:, 0::2]

    def make_tmal(self):
        """ Make the TMAL "code" using components that generate TMAL
        statements"""
        for comp in self.comps:
            comp.update()
        tmal_comps = [x for x in self.comps if hasattr(x, 'tmal_ints')]
        self.tmal_ints = np.zeros((len(tmal_comps), tmal.N_INTS),
                                  dtype=np.int32)
        self.tmal_floats = np.zeros((len(tmal_comps), tmal.N_FLOATS),
                                    dtype=np.float)
        for i, comp in enumerate(tmal_comps):
            self.tmal_ints[i, 0:len(comp.tmal_ints)] = comp.tmal_ints
            self.tmal_floats[i, 0:len(comp.tmal_floats)] = comp.tmal_floats

    def calc(self):
        """Calculate the model.  The results appear in the self.mvals array."""
        self.make_tmal()
        # int calc_model(int n_times, int n_preds, int n_tmals, float dt,
        #                float **mvals, int **tmal_ints, float **tmal_floats)

        mvals = convert_type_star_star(self.mvals, ctypes.c_double)
        tmal_ints = convert_type_star_star(self.tmal_ints, ctypes.c_int)
        tmal_floats = convert_type_star_star(self.tmal_floats, ctypes.c_double)

        if self.evolve_method == 1:
            dt = self.dt_ksec * 2
            self.core.calc_model(self.n_times, self.n_preds,
                                 len(self.tmal_ints), dt, mvals,
                                 tmal_ints, tmal_floats)
        elif self.evolve_method == 2:
            dt = self.dt_ksec
            self.core_new.calc_model_new(self.rk4, self.n_times, self.n_preds,
                                         len(self.tmal_ints), dt, mvals,
                                         tmal_ints, tmal_floats)

        # hackish fix to ensure last value is computed
        self.mvals[:, -1] = self.mvals[:, -2]

    def calc_stat(self):
        """Calculate model fit statistic as the sum of component fit stats"""
        self.calc()            # parvals already set with dummy_calc
        fit_stat = sum(comp.calc_stat() for comp in self.comps if comp.predict)
        return fit_stat

    def calc_staterror(self, data):
        """Calculate model fit statistic error (dummy array for Sherpa use)"""
        return np.ones_like(data)

    @property
    def date_range(self):
        """Return formatted date range string"""
        return '%s_%s' % (DateTime(self.tstart).greta[:7],
                          DateTime(self.tstop).greta[:7])

    @property
    def core(self):
        """Lazy-load the "core" ctypes shared object libary that does the
        low-level model calculation via the C "calc_model" routine.  Only
        load once by setting/returning a class attribute.
        """
        if not hasattr(XijaModel, '_core'):
            loader_path = os.path.abspath(os.path.dirname(__file__))
            _core = np.ctypeslib.load_library('core', loader_path)
            _core.calc_model.restype = ctypes.c_int
            _core.calc_model.argtypes = [
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_double,
                ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
                ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
                ctypes.POINTER(ctypes.POINTER(ctypes.c_double))
                ]
            XijaModel._core = _core
        return XijaModel._core

    @property
    def core_new(self):
        """Lazy-load the "core_new" ctypes shared object libary that does the
        low-level model calculation via the C "calc_model_new" routine.  Only
        load once by setting/returning a class attribute.
        """
        if not hasattr(XijaModel, '_core_new'):
            loader_path = os.path.abspath(os.path.dirname(__file__))
            _core_new = np.ctypeslib.load_library('core_new', loader_path)
            _core_new.calc_model_new.restype = ctypes.c_int
            _core_new.calc_model_new.argtypes = [
                ctypes.c_int, ctypes.c_int, ctypes.c_int, 
                ctypes.c_int, ctypes.c_double,
                ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
                ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
                ctypes.POINTER(ctypes.POINTER(ctypes.c_double))
            ]
            XijaModel._core_new = _core_new
        return XijaModel._core_new

ThermalModel = XijaModel
