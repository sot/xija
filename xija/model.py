"""
Next-generation thermal modeling framework for Chandra thermal modeling
"""

import os
import json
from odict import OrderedDict
import numpy as np
import Ska.Numpy
from Chandra.Time import DateTime
import clogging

from . import component

try:
    # Optional packages for model fitting or use on HEAD LAN
    import Ska.engarchive.fetch_sci as fetch
    import Chandra.cmd_states
    import Ska.DBI
except ImportError:
    pass

import pyyaks.context as pyc
from .core import calc_model
from .files import files as xija_files

src = pyc.CONTEXT['src'] if 'src' in pyc.CONTEXT else pyc.ContextDict('src')
files = (pyc.CONTEXT['file'] if 'file' in pyc.CONTEXT else
         pyc.ContextDict('files', basedir=os.getcwd()))
files.update(xija_files)

if 'debug' in globals():
    from IPython.Debugger import Tracer
    pdb_settrace = Tracer()

logger = clogging.config_logger('xija', level=clogging.INFO)


class ThermalModel(object):
    def __init__(self, start='2011:115:00:00:00', stop='2011:115:01:00:00', dt=328.0,
                 model_spec=None, name='xijamod'):
        if model_spec:
            dt = model_spec['dt']
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
        if model_spec:
            self._set_from_model_spec(model_spec)

    def _set_from_model_spec(self, model_spec):
        for comp in model_spec['comps']:
            ComponentClass = globals()[comp['class_name']]
            args = comp['init_args']
            kwargs = dict((str(k), v) for k, v in comp['init_kwargs'].items())
            self.add(ComponentClass, *args, **kwargs)

        self.make_pars()
        parnames, parvals = zip(*model_spec['pars'])
        parnames = [str(x) for x in parnames]
        if self.parnames != parnames:
            raise ValueError('Model spec parnames do not match model: \n{0}\n{1}'.format(
                    parnames, self.parnames))
        self.parvals[:] = parvals

    @staticmethod
    def _eng_match_times(start, stop, dt):
        """Return an array of times between ``start`` and ``stop`` at ``dt`` sec
        intervals.  The times are roughly aligned (within 1 sec) to the timestamps
        in the '5min' (328 sec) Ska eng archive data.
        """
        time0 = 410270764.0
        i0 = int((DateTime(start).secs - time0) / dt) + 1
        i1 = int((DateTime(stop).secs - time0) / dt)
        return time0 + np.arange(i0, i1) * dt

    @property
    def pars(self):
        return dict((k, v) for k, v in zip(self.parnames, self.parvals))

    def fetch(self, msid, attr='means', method='linear'):
        tpad = self.dt * 5
        datestart = DateTime(self.tstart - tpad).date
        datestop = DateTime(self.tstop + tpad).date
        logger.info('Fetching msid: %s over %s to %s' % (msid, datestart, datestop))
        tlm = fetch.MSID(msid, datestart, datestop, stat='5min', filter_bad=True)
        vals = Ska.Numpy.interpolate(getattr(tlm, attr), tlm.times, self.times, method=method)
        return vals

    def add(self, ComponentClass, *args, **kwargs):
        comp = ComponentClass(self, *args, **kwargs)
        # Store args and kwargs used to initialize object for later object
        # storage and re-creation
        comp.init_args = args
        comp.init_kwargs = kwargs
        self.comp[comp.name] = comp
        
        return comp

    comps = property(lambda self: self.comp.values())

    def get_comp(self, name):
        """Get a model component.  Works with either a string or a component object"""
        return self.comp[str(name)]

    def write(self, filename):
        """Write a full model specification for this model"""
        model_spec = dict(name=self.name,
                          comps=[],
                          pars=[(name, val) for name, val in zip(self.parnames, self.parvals)],
                          dt=self.dt,
                          tlm_code=None,
                          mval_names=[])
               
        stringify = lambda x: str(x) if isinstance(x, component.ModelComponent) else x
        for comp in self.comps:
            init_args = [stringify(x) for x in comp.init_args]
            init_kwargs = dict((k, stringify(v)) for k, v in comp.init_kwargs.items())
            model_spec['comps'].append(dict(class_name=comp.__class__.__name__,
                                            name=comp.name,
                                            init_args=init_args,
                                            init_kwargs=init_kwargs))
        with open(filename, 'w') as f:
            json.dump(model_spec, f, sort_keys=True, indent=4)

    def _get_parvals(self):
        """For components that have parameter values make a view into the
        global parameter values array.  But first count the total number of
        parameters and make that global param vals array ``self.parvals``.
        """
        if not hasattr(self, '_parvals'):
            self.make_pars()
        return self._parvals
    
    def _set_parvals(self, vals):
        self.parvals[:] = vals
        
    parvals = property(_get_parvals, _set_parvals)

    @property
    def parnames(self):
        if not hasattr(self, '_parnames'):
            self.make_pars()
        return self._parnames

    def make(self):
        self.make_pars()
        self.make_mvals()
        self.make_mults()
        self.make_heats()
        self.make_heatsinks()

    def make_pars(self):
        if not hasattr(self, '_parvals'):
            comps = [x for x in self.comps if x.n_parvals]
            n_parvals = sum(x.n_parvals for x in comps)
            i_parvals = np.cumsum([0] + [x.n_parvals for x in comps])
            self._parvals = np.zeros(n_parvals, dtype=np.float)
            self._parnames = []
            for comp, i0, i1 in zip(comps, i_parvals[:-1], i_parvals[1:]):
                self._parnames.extend(comp.name + '__' + x for x in comp.parnames)
                self._parvals[i0:i1] = comp.parvals  # copy existing values

                # This hidden interaction is not really nice...
                comp.parvals = self._parvals[i0:i1]  # make a local (view into global parvals
                comp.parvals_i = i0

    def make_mvals(self):
        """Initialize the global mvals (model values) array.  This is an
        N (rows) x n_times (cols) array that contains all data needed
        to compute the model prediction.  All rows are initialized to
        relevant data values (e.g. node temps, time-dependent power,
        external temperatures, etc).  In the model calculation some
        rows will be overwritten with predictions.
        """
        # Select components with data values, and from those select ones that get
        # predicted and those that do not get predicted
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
        self.mvals.shape = (len(comps), -1)
        self.cvals = self.mvals[:, 0::2]

    def make_mults(self):
        """
        Iterate through Couplings to make mults: 2-d array containing rows
        in the format:
          [idx1, idx2, parvals_i]

        This provides a derivative coupling term in the form:
          d_mvals[idx1] /dt += mvals[idx2] / parvals[parvals_i]
        """
        mults = []
        for comp in self.comps:
            if isinstance(comp, component.Coupling):
                i1 = comp.node1.mvals_i
                i2 = comp.node2.mvals_i
                mults.append((i1, i2, comp.parvals_i))

        self.mults = np.array(mults)

    def make_heats(self):
        """
        Iterate through PrecomputedHeatPower to make heats.
        This provides a derivative heating term in the form:
          d_mvals[i1] /dt += mvals[i2]
        """
        heats = []
        for comp in self.comps:
            if isinstance(comp, component.PrecomputedHeatPower):
                i1 = comp.node.mvals_i   # Node being heated
                i2 = comp.mvals_i        # mvals row with precomputed heat input
                heats.append((i1, i2))

        self.heats = np.array(heats)

    def make_heatsinks(self):
        """
        """
        heatsinks = []
        for comp in self.comps:
            if isinstance(comp, component.HeatSink):
                i1 = comp.node.mvals_i   # Node being heated
                T_parval_i = comp.parvals_i
                tau_parval_i = comp.parvals_i + 1
                heatsinks.append((i1, T_parval_i, tau_parval_i))

        self.heatsinks = np.array(heatsinks)

    def calc(self):
        for comp in self.comps:
            comp.update()

        dt = self.dt_ksec * 2
        indexes = np.arange(0, self.n_times-2, 2)
        calc_model(indexes, dt, self.n_preds, self.mvals, self.parvals, self.mults,
                   self.heats, self.heatsinks)

    def calc_stat(self):
        self.calc()            # parvals already set with dummy_calc
        fit_stat = sum(comp.calc_stat() for comp in self.comps if comp.predict)
        return fit_stat

    def calc_staterror(self, data):
        return numpy.ones_like(data)

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
            plt.figure(i+10, figsize=(10,5))
            plt.clf()

            plt.subplot(2, 1, 1)
            plot_cxctime(times, comp.dvals, '-b')
            plot_cxctime(times, comp.mvals, '-r')
            plt.title(comp.name.upper() + ' ' + self.date_range)
            plt.ylabel('degC')
            plt.grid()

            resid = (comp.dvals - comp.mvals) * 1.8
            plt.subplot(2, 1, 2)
            plot_cxctime(times, resid, '-m')
            ylim = np.max(np.abs(resid)) * 1.1
            if ylim < 6:
                ylim = 6
            plt.ylim(-ylim, ylim)
            plt.ylabel('degF')
            plt.grid()

            plt.subplots_adjust(bottom=0.1, top=0.93, hspace=0.15)
            if savefig:
                src['date_range'] = self.date_range
                src['msid'] = comp.name
                plt.savefig(files['fit_resid.png'].abs)
    
