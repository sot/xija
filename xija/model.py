"""
Next-generation thermal modeling framework for Chandra thermal modeling
"""

import os
import json
import ctypes
from collections import OrderedDict

import numpy as np
import Ska.Numpy
from Chandra.Time import DateTime
import asciitable

from . import component
from . import tmal

try:
    # Optional packages for model fitting or use on HEAD LAN
    import Ska.engarchive.fetch_sci as fetch
    import Chandra.cmd_states
    import Ska.DBI
except ImportError:
    pass

import pyyaks.context as pyc
from .files import files as xija_files
from . import clogging

src = pyc.CONTEXT['src'] if 'src' in pyc.CONTEXT else pyc.ContextDict('src')
files = (pyc.CONTEXT['file'] if 'file' in pyc.CONTEXT else
         pyc.ContextDict('files', basedir=os.getcwd()))
files.update(xija_files)

if 'debug' in globals():
    from IPython.Debugger import Tracer
    pdb_settrace = Tracer()

logger = clogging.config_logger('xija', level=clogging.INFO)

#int calc_model(int n_times, int n_preds, int n_tmals, double dt,
#               double **mvals, int **tmal_ints, double **tmal_floats)


def convert_type_star_star(array, ctype_type):
    f4ptr = ctypes.POINTER(ctype_type)
    return (f4ptr * len(array))(*[row.ctypes.data_as(f4ptr) for row in array])


class FetchError(Exception):
    pass


class ThermalModel(object):
    def __init__(self, name, start=None, stop=None, dt=328.0, model_spec=None,
                 cmd_states=None):
        if stop is None:
            stop = DateTime().secs - 30 * 86400
        if start is None:
            start = DateTime(stop).secs - 45 * 86400
        self.name = name
        self.comp = OrderedDict()
        self.dt = dt
        self.dt_ksec = self.dt / 1000.
        self.times = self._eng_match_times(start, stop, dt)
        self.tstart = self.times[0]
        self.tstop = self.times[-1]
        self.ksecs = (self.times - self.tstart) / 1000.
        self.datestart = DateTime(self.tstart).date
        self.datestop = DateTime(self.tstop).date
        self.n_times = len(self.times)
        self.pars = []
        if model_spec:
            self._set_from_model_spec(model_spec)
        self.cmd_states = cmd_states

    def _set_from_model_spec(self, model_spec):
        try:
            model_spec = json.load(open(model_spec, 'r'))
        except:
            pass

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

    @staticmethod
    def _eng_match_times(start, stop, dt):
        """Return an array of times between ``start`` and ``stop`` at ``dt``
        sec intervals.  The times are roughly aligned (within 1 sec) to the
        timestamps in the '5min' (328 sec) Ska eng archive data.
        """
        time0 = 410270764.0
        i0 = int((DateTime(start).secs - time0) / dt) + 1
        i1 = int((DateTime(stop).secs - time0) / dt)
        return time0 + np.arange(i0, i1) * dt

    def _get_cmd_states(self):
        if not hasattr(self, '_cmd_states'):
            logger.info('Reading commanded states DB over %s to %s' %
                        (self.datestart, self.datestop))
            try:
                db = Ska.DBI.DBI(dbi='sybase', database='aca', user='aca_read')
            except Exception as err:
                raise RuntimeError(
                    'Unable to connect to sybase cmd_states: {}'.format(err))
            states = db.fetchall("""select * from cmd_states
                                 where tstop >= {0} and tstart <= {1}
                                 """.format(self.tstart, self.tstop))
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

    def fetch(self, msid, attr='vals', method='linear'):
        tpad = self.dt * 5
        datestart = DateTime(self.tstart - tpad).date
        datestop = DateTime(self.tstop + tpad).date
        logger.info('Fetching msid: %s over %s to %s' %
                    (msid, datestart, datestop))
        try:
            tlm = fetch.MSID(msid, datestart, datestop, stat='5min')
        except NameError:
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
        comp = ComponentClass(self, *args, **kwargs)
        # Store args and kwargs used to initialize object for later object
        # storage and re-creation
        comp.init_args = args
        comp.init_kwargs = kwargs
        self.comp[comp.name] = comp
        for par in comp.pars:
            self.pars.append(par)

        return comp

    comps = property(lambda self: self.comp.values())

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
                          mval_names=[])

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
            print 'HEJ', comp.name, comp.predict, type(comp)
            if hasattr(comp, 'dvals'):
                colvals[comp.name + '_data'] = comp.dvals
            if hasattr(comp, 'mvals') and comp.predict:
                colvals[comp.name + '_model'] = comp.mvals

        asciitable.write(colvals, filename, names=colvals.keys())

    def write(self, filename, model_spec=None):
        """Write the model specification as JSON to a file
        """
        if model_spec is None:
            model_spec = self.model_spec

        with open(filename, 'w') as f:
            json.dump(model_spec, f, sort_keys=True, indent=4)

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
        return tuple(par.full_name for par in self.pars)

    def make(self):
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
        self.mvals = np.hstack(comp.dvals for comp in preds + unpreds)
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
        self.make_tmal()
        # int calc_model(int n_times, int n_preds, int n_tmals, float dt,
        #                float **mvals, int **tmal_ints, float **tmal_floats)

        dt = self.dt_ksec * 2
        mvals = convert_type_star_star(self.mvals, ctypes.c_double)
        tmal_ints = convert_type_star_star(self.tmal_ints, ctypes.c_int)
        tmal_floats = convert_type_star_star(self.tmal_floats, ctypes.c_double)

        self.core.calc_model(self.n_times, self.n_preds, len(self.tmal_ints),
                             dt, mvals, tmal_ints, tmal_floats)

        # hackish fix to ensure last value is computed
        self.mvals[:, -1] = self.mvals[:, -2]

    def calc_stat(self):
        self.calc()            # parvals already set with dummy_calc
        fit_stat = sum(comp.calc_stat() for comp in self.comps if comp.predict)
        return fit_stat

    def calc_staterror(self, data):
        return np.ones_like(data)

    @property
    def date_range(self):
        return '%s_%s' % (DateTime(self.tstart).greta[:7],
                          DateTime(self.tstop).greta[:7])

    def plot_fit_resids(self, savefig=False, names=None):
        import matplotlib.pyplot as plt
        from Ska.Matplotlib import plot_cxctime
        src['model'] = self.name
        if 'outdir' not in src:
            src['outdir'] = '.'

        self.calc()
        times = self.times
        comps = (comp for comp in self.comps
                 if comp.predict and isinstance(comp, component.Node))

        for i, comp in enumerate(comps):
            if names and comp.name not in names:
                continue
            plt.figure(i + 10, figsize=(10, 5))
            plt.clf()

            plt.subplot(2, 1, 1)
            plot_cxctime(times, comp.dvals, '-b')
            plot_cxctime(times, comp.mvals, '-r')
            plt.title(comp.name.upper() + ' ' + self.date_range)
            plt.ylabel('degC')
            plt.grid()

            resid = (comp.dvals - comp.mvals)
            plt.subplot(2, 1, 2)
            plot_cxctime(times, resid, '-m')
            ylim = np.max(np.abs(resid)) * 1.1
            if ylim < 6:
                ylim = 6
            plt.ylim(-ylim, ylim)
            plt.ylabel('degC')
            plt.grid()

            plt.subplots_adjust(bottom=0.1, top=0.93, hspace=0.15)
            if savefig:
                src['date_range'] = self.date_range
                src['msid'] = comp.name
                plt.savefig(files['fit_resid.png'].abs)

    @property
    def core(self):
        """Lazy-load the "core" ctypes shared object libary that does the
        low-level model calculation via the C "calc_model" routine.  Only
        load once by setting/returning a class attribute.
        """
        if not hasattr(ThermalModel, '_core'):
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
            ThermalModel._core = _core
        return ThermalModel._core
