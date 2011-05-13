#!/usr/bin/env python

import os
import sys
import time
import re
import json
import shutil
import logging
import glob

import numpy as np
import sherpa.ui as ui
from mpi4py import MPI
from Chandra.Time import DateTime
import pyyaks.context

import clogging   # get rid of this or something
import xija

freeze_pars = ('.*',
               '.*__T_e',
               '.*__pi_.*',
               '.*__pf_.*',
               '.*__tau_.*',
               '.*__tau_sc',
               '.*__p_ampl',
               )

#thaw_pars = (#'tcylaft6__pf_045',
             # 'tmzp_my__pf_.*',
             #'tcylfmzm__pf_.*',
             # '.*__p_ampl',
             #'.*__tau_sc',
             #'tcylfmzm__tau_t.*',
             #'.*__tau_ext',
             #'tau_tmzp_cnt',
             #)

class CalcModel(object):
    def __init__(self, model, comm=None, outdir='', pardir=''):
        self.model = model
        self.comm = comm
        if self.comm:
            comm.bcast(dict(cmd='init', model=model.name, tstart=model.tstart, tstop=model.tstop,
                            outdir=model.outdir, pardir=model.pardir),  
                       root=MPI.ROOT)

    def __call__(self, parvals, x):
        fit_logger.info('Calculating params:')
        for parname, parval, newparval in zip(self.model.parnames, self.model.parvals, parvals):
            if parval != newparval:
                fit_logger.info('  {0}: {1}'.format(parname, newparval))
        self.model.parvals[:] = parvals

        if self.comm:
            self.comm.bcast(dict(cmd='calc_model', parvals=parvals), root=MPI.ROOT)

        return np.ones_like(x)


class CalcStat(object):
    def __init__(self, model, comm=None):
        self.model = model
        self.comm = comm
        self.cache_fit_stat = {}
        
    def __call__(self, data, model, staterror=None, syserror=None, weight=None):
        parvals_key = tuple('%.4e' % x for x in self.model.parvals)
        try:
            fit_stat = self.cache_fit_stat[parvals_key]
            logger.debug('nmass_model: Cache hit %s' % str(parvals_key))
        except KeyError:
            if self.comm:
                msg = dict(cmd='calc_stat', data=data, model=model, staterror=staterror,
                           syserror=syserror, weight=weight)
                comm.bcast(msg, root=MPI.ROOT)
                fit_stat = np.array(0.0, 'd')
                comm.Reduce(None, [fit_stat, MPI.DOUBLE], op=MPI.SUM, root=MPI.ROOT)
                fit_stat = fit_stat.tolist() / comm.Get_size()
            else:
                fit_stat = self.model.calc_stat()

        fit_logger.info('Fit statistic: %.4f' % fit_stat)
        self.cache_fit_stat[parvals_key] = fit_stat
        
        return fit_stat, np.ones(1)

def fit_model(model,
             comm=None,
             method='simplex',
             config=None,
             nofit=None,
             freeze_pars=freeze_pars,
             thaw_pars=[],
             ):

    dummy_data = np.zeros(1)
    dummy_times = np.arange(1)
    ui.load_arrays(1, dummy_times, dummy_data)

    ui.set_method(method)
    ui.get_method().config.update(config or sherpa_configs.get(method, {}))

    ui.load_user_model(CalcModel(model, comm), 'xijamod')
    ui.add_user_pars('xijamod', model.parnames)
    ui.set_model(1, 'xijamod')

    fit_parnames = set()
    print 'thawpars=', thaw_pars
    for parname, parval in zip(model.parnames, model.parvals):
        getattr(xijamod, parname).val = parval
        fit_parnames.add(parname)
        if any([re.match(x + '$', parname) for x in freeze_pars]):
            print 'Freezing', parname
            ui.freeze(getattr(xijamod, parname))
            fit_parnames.remove(parname)
        if any([re.match(x + '$', parname) for x in thaw_pars]):
            print 'Thawing', parname
            ui.thaw(getattr(xijamod, parname))
            fit_parnames.add(parname)

    ui.load_user_stat('xijastat', CalcStat(model, comm), lambda x: np.ones_like(x))
    ui.set_stat(xijastat)

    if fit_parnames and not nofit:
        ui.fit(1)
    else:
        model.calc()

def make_out_dir():
    out_dir = files['out_dir'].abs
    out_dirs = glob.glob(out_dir + '.*')
    out_dirs_search = re.compile(r'{0}\.(\d+)'.format(out_dir)).search
    out_indexes = [int(out_dirs_search(x).group(1)) for x in out_dirs if out_dirs_search(x)]
    if out_indexes:
        n_outs = max(out_indexes)
    else:
        n_outs = 0
    if os.path.exists(out_dir):
        os.rename(out_dir, out_dir + '.%d' % (n_outs + 1))
    os.makedirs(out_dir)

def get_options():
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--days",
                      type='float',
                      default=90,
                      help="Number of days in fit interval (default=90")
    parser.add_option("--stop",
                      help="Stop time of fit interval (default=NOW - 7 days)")
    parser.add_option("--method",
                      default='simplex',
                      help="Fit method (default=simple)")
    parser.add_option("--model",
                      default='minusz',
                      help="Model to predict (default=minusz)")
    parser.add_option("--thaw-pars",
                      default='solarheat__tmzp_my__P_60',
                      help="List of parameters (space-separated) to thaw (default=none)")
    parser.add_option("--nproc",
                      default=0,
                      type='int',
                      help="Number of processors (default=1)")
    parser.add_option("--ftol",
                      default=1e-4,
                      type='float',
                      help="ftol convergence parameter (default=1e-4")
    parser.add_option("--outdir",
                      default='fit',
                      help="Output directory within <msid> root (default=fit)")
    parser.add_option("--pardir",
                      help="Directory containing model params file (default=<model>)")
    parser.add_option("--quiet",
                      default=False,
                      action='store_true',
                      help="Suppress screen output")
    parser.add_option("--nofit",
                      action="store_true",
                      help="Do not fit (default=False)")
    (opt, args) = parser.parse_args()
    return (opt, args)

# Default configurations for fit methods
sherpa_configs = dict(
    simplex = dict(ftol=1e-3,
                   finalsimplex=0,   # converge based only on length of simplex
                   maxfev=1000),
    )

opt, args = get_options()

src = pyyaks.context.ContextDict('src')
files = pyyaks.context.ContextDict('files', basedir=os.getcwd())
files.update(xija.files)

fit_logger = clogging.config_logger('fit', level=clogging.INFO)
sherpa_logger = logging.getLogger("sherpa")
loggers = (fit_logger, sherpa_logger)
if opt.quiet:
    for logger in loggers:
        for h in logger.handlers:
            logger.removeHandler(h)

# Use supplied stop time or NOW - 7 days (truncated to nearest day)
stop = opt.stop or DateTime(DateTime().secs - 7 * 86400).date[:8]
start = DateTime(DateTime(stop).secs - opt.days * 86400).date[:8]

src['model'] = opt.model
src['outdir'] = opt.outdir
src['pardir'] = opt.pardir or opt.model

model_spec = json.load(open(files['model_spec.json'].abs, 'r'))
model = xija.ThermalModel(start, stop, name=opt.model, model_spec=model_spec)
model.outdir = src['outdir'].val
model.pardir = src['pardir'].val

thaw_pars = opt.thaw_pars.split()

make_out_dir()

hdlr = logging.FileHandler(files['fit_log'].abs, 'w')
hdlr.setLevel(logging.INFO)
hdlr.setFormatter(logging.Formatter('%(message)s'))
for logger in loggers:
    logger.addHandler(hdlr)

fit_logger.info('Running: ' + ' '.join(sys.argv))
fit_logger.info('Start: {0}'.format(start))
fit_logger.info('Stop: {0}'.format(stop))
fit_logger.info('Options: {0}'.format(opt))
fit_logger.info('')
fit_logger.info('Start time: {0}'.format(time.ctime()))

if opt.nproc:
    fit_logger.info('fit_nmass: Spawning processes')
    comm = MPI.COMM_SELF.Spawn(sys.executable, args=['worker.py'], maxprocs=opt.nproc)
    fit_logger.info('fit_nmass: Finished spawning processes')
    fit_logger.info('fit_nmass: comm.Get_size() = {0}'.format(comm.Get_size()))
else:
    comm = None

try:
    fit_model(model, comm=comm, method=opt.method, nofit=opt.nofit, thaw_pars=thaw_pars)
    model.write(files['fit_spec.json'].abs)

    kwargs = {'savefig': True}
    if comm:
        pass
        #comm.bcast(dict(cmd='model', func='plot_fit_resid', kwargs=kwargs), root=MPI.ROOT)
    else:
        pass
        # model.plot_fit_resid(**kwargs)

    if comm:
        pass
        #comm.bcast(dict(cmd='model', func='plot_hist', kwargs=kwargs), root=MPI.ROOT)
    else:
        if 0:
            model.plot_hist(select='all', **kwargs)
            model.plot_hist(select='hot', **kwargs)
            model.write_fit_data()
            model.write_html_report()

finally:
    if comm:
        comm.bcast({'cmd': 'stop'}, root=MPI.ROOT)
        semaphore = np.array(0.0, 'd')
        comm.Reduce(None, [semaphore, MPI.DOUBLE], op=MPI.SUM, root=MPI.ROOT)

        time.sleep(1)
        comm.Disconnect()

    fit_logger.info('Stop time: {0}'.format(time.ctime()))

