import sys
import os
import multiprocessing
import time
import pygtk
pygtk.require('2.0')
import gtk

import re
import json
import shutil
import logging
import glob

# Matplotlib setup: use Agg backend for command-line (non-interactive) operation
import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')

import numpy as np
import sherpa.ui as ui
from Chandra.Time import DateTime
import pyyaks.context as pyc

import xija
import clogging   # get rid of this or something

class FitTerminated(Exception):
    pass

class CalcModel(object):
    def __init__(self, model):
        self.model = model

    def __call__(self, parvals, x):
        """This is the Sherpa calc_model function, but in this case calc_model does not
        actually calculate anything but instead just stores the desired paramters.  This
        allows for multiprocessing where only the fit statistic gets passed between nodes.
        """
        fit_logger.info('Calculating params:')
        for parname, parval, newparval in zip(self.model.parnames, self.model.parvals, parvals):
            if parval != newparval:
                fit_logger.info('  {0}: {1}'.format(parname, newparval))
        self.model.parvals[:] = parvals

        time.sleep(1)
        return np.ones_like(x)


class CalcStat(object):
    def __init__(self, model, pipe):
        self.pipe = pipe
        self.model = model
        self.cache_fit_stat = {}
        self.min_fit_stat = None
        
    def __call__(self, _data, _model, staterror=None, syserror=None, weight=None):
        """Calculate fit statistic for the xija model.  The args _data and _model
        are sent by Sherpa but they are fictitious -- the real data and model are
        stored in the xija model self.model.
        """
        parvals_key = tuple('%.4e' % x for x in self.model.parvals)
        try:
            fit_stat = self.cache_fit_stat[parvals_key]
            fit_logger.info('nmass_model: Cache hit %s' % str(parvals_key))
        except KeyError:
            fit_stat = self.model.calc_stat()
            fit_logger.info('pars={}'.format('\n'.join(
                    ["{}={}".format(x,y) for (x,y) in zip(self.model.parnames, self.model.parvals)])))
        fit_logger.info('Fit statistic: %.4f' % fit_stat)
        self.cache_fit_stat[parvals_key] = fit_stat
        fit_logger.info('Done with cache')
        
        if self.min_fit_stat is None or fit_stat < self.min_fit_stat:
            self.min_fit_stat = fit_stat
            self.min_parvals = self.model.parvals.copy()

        fit_logger.info('Done with min_fit_stat')

        while self.pipe.poll():
            pipe_val = self.pipe.recv()
            if pipe_val == 'get_pars':
                self.pipe.send(self.min_parvals)
            elif pipe_val == 'terminate':
                raise FitTerminated('terminated')

        fit_logger.info('Returning from fit_stat')
        return fit_stat, np.ones(1)


class FitWorker(object):
    def __init__(self):
        self.parent_pipe, self.child_pipe = multiprocessing.Pipe()

    def start(self, *args):
        """Start a Sherpa fit process as a spawned (non-blocking) process.
        """
        self.fit_process = multiprocessing.Process(target=self.fit)
        self.fit_process.start()
        logging.debug('Fit started')

    def terminate(self, *args):
        """Terminate a Sherpa fit process in a controlled way by sending a
        message.  Get the final parameter values if possible.
        """
        self.parent_pipe.send('get_pars')
        if self.parent_pipe.poll(1.0):
            pars = self.parent_pipe.recv()
            logging.debug('Got pars {}'.format(pars))
        else:
            pars = None
            logging.debug('Could not get pars prior to terminate')

        self.parent_pipe.send('terminate')
        self.fit_process.join()
        logging.debug('Fit terminated')

    def freeze_or_thaw_params(self):
        """Go through each model parameter and either freeze or thaw it.
        Return a list of the thawed (fitted) parameters.
        """
        fit_parnames = set()
        for parname, parval in zip(model.parnames, model.parvals):
            getattr(xijamod, parname).val = parval
            fit_parnames.add(parname)
            if any([re.match(x + '$', parname) for x in freeze_pars]):
                fit_logger.info('Freezing ' + parname)
                ui.freeze(getattr(xijamod, parname))
                fit_parnames.remove(parname)
            if any([re.match(x + '$', parname) for x in thaw_pars]):
                fit_logger.info('Thawing ' + parname)
                ui.thaw(getattr(xijamod, parname))
                fit_parnames.add(parname)
                if 'tau' in parname:
                    getattr(xijamod, parname).min = 0.1
        return fit_parnames

    def fit(self):
        comm = None
        method = 'simplex'
        config = None
        nofit = None

        dummy_data = np.zeros(1)
        dummy_times = np.arange(1)
        ui.load_arrays(1, dummy_times, dummy_data)

        ui.set_method(method)
        ui.get_method().config.update(config or sherpa_configs.get(method, {}))

        xijamod = CalcModel(model)
        ui.load_user_model(xijamod, 'xijamod')
        ui.add_user_pars('xijamod', model.parnames)
        ui.set_model(1, 'xijamod')

        calc_stat = CalcStat(model, self.child_pipe)
        ui.load_user_stat('xijastat', calc_stat, lambda x: np.ones_like(x))
        ui.set_stat(xijastat)

        fit_parnames = self.freeze_or_thaw_params()
        
        if fit_parnames and not nofit:
            try:
                ui.fit(1)
            except FitTerminated as e:
                logging.debug('Got FitTerminated exception {}'.format(e))
                
        model.calc()

class MainWindow(object):
    # This is a callback function. The data arguments are ignored
    # in this example. More on callbacks below.
    def destroy(self, widget, data=None):
        print "destroy signal occurred"
        gtk.main_quit()

    def __init__(self):
        # create a new window
        self.window = gtk.Window(gtk.WINDOW_TOPLEVEL)
        self.window.connect("destroy", self.destroy)
    
        # Sets the border width of the window.
        self.window.set_border_width(10)
        self.main_box = gtk.HBox()
        self.window.add(self.main_box)

        self.button = gtk.Button("Fit now")
        self.button.connect("clicked", fit_worker.start)
        self.main_box.pack_start(self.button)
    
        self.button2 = gtk.Button("Stop!")
        self.button2.connect("clicked", fit_worker.terminate)
        self.main_box.pack_start(self.button2)

        # and the window
        self.window.show_all()

    def main(self):
        # All PyGTK applications must have a gtk.main(). Control ends here
        # and waits for an event to occur (like a key press or mouse event).
        gtk.main()

# If the program is run directly or passed as an argument to the python
# interpreter then create a HelloWorld instance and show it

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
                      default=30,
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
                      default='solarheat__tephin__P_.*',
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

opt, args = get_options()

fit_worker = FitWorker()

# Default configurations for fit methods
sherpa_configs = dict(
    simplex = dict(ftol=1e-3,
                   finalsimplex=0,   # converge based only on length of simplex
                   maxfev=1000),
    )

src = pyc.CONTEXT['src'] if 'src' in pyc.CONTEXT else pyc.ContextDict('src')
files = (pyc.CONTEXT['file'] if 'file' in pyc.CONTEXT else
         pyc.ContextDict('files', basedir=os.getcwd()))
files.update(xija.files)

fit_logger = clogging.config_logger('fit', level=clogging.INFO,
                                    format='[%(levelname)s] (%(processName)-10s) %(message)s')
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
model = xija.ThermalModel(opt.model, start, stop, model_spec=model_spec)
model.make()   

model.outdir = src['outdir'].val
model.pardir = src['pardir'].val

thaw_pars = opt.thaw_pars.split()
freeze_pars = ('.*',
               '.*__T_e',
               '.*__pi_.*',
               '.*__pf_.*',
               '.*__tau_.*',
               '.*__tau_sc',
               '.*__p_ampl',
               )

# make_out_dir()

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

main_window = MainWindow()
main_window.main()
