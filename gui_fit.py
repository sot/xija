#!/usr/bin/env python

import sys
import os
import multiprocessing
import time
import pygtk
pygtk.require('2.0')
import gtk
import gobject
from itertools import count

import re
import json
import shutil
import logging
import glob

from matplotlib.figure import Figure
from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas
from matplotlib.backends.backend_gtkagg import NavigationToolbar2GTKAgg as NavigationToolbar

import numpy as np
import sherpa.ui as ui
from Chandra.Time import DateTime
import pyyaks.context as pyc

import xija
import clogging   # get rid of this or something

# from xija.fit import (FitTerminated, CalcModel, CalcStat, FitWorker)

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

        return np.ones_like(x)


class CalcStat(object):
    def __init__(self, model, pipe):
        self.pipe = pipe
        self.model = model
        self.cache_fit_stat = {}
        self.min_fit_stat = None
        self.min_par_vals = self.model.parvals
        
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

        fit_logger.info('Fit statistic: %.4f' % fit_stat)
        self.cache_fit_stat[parvals_key] = fit_stat
        
        if self.min_fit_stat is None or fit_stat < self.min_fit_stat:
            self.min_fit_stat = fit_stat
            self.min_parvals = self.model.parvals.copy()

        self.message = {'status': 'fitting',
                        'time': time.time(),
                        'parvals': self.model.parvals,
                        'fit_stat': fit_stat,
                        'min_parvals': self.min_parvals,
                        'min_fit_stat': self.min_fit_stat}
        self.pipe.send(self.message)

        while self.pipe.poll():
            pipe_val = self.pipe.recv()
            if pipe_val == 'terminate':
                self.model.parvals = self.min_parvals
                raise FitTerminated('terminated')

        return fit_stat, np.ones(1)


class FitWorker(object):
    def __init__(self, model, freeze_pars, thaw_pars):
        self.model = model
        self.parent_pipe, self.child_pipe = multiprocessing.Pipe()
        self.freeze_pars = freeze_pars
        self.thaw_pars = thaw_pars

    def start(self, widget=None):
        """Start a Sherpa fit process as a spawned (non-blocking) process.
        """
        self.fit_process = multiprocessing.Process(target=self.fit)
        self.fit_process.start()
        logging.debug('Fit started')

    def terminate(self, widget=None):
        """Terminate a Sherpa fit process in a controlled way by sending a
        message.  Get the final parameter values if possible.
        """
        self.parent_pipe.send('terminate')
        self.fit_process.join()
        logging.debug('Fit terminated')

    def freeze_or_thaw_params(self, xijamod):
        """Go through each model parameter and either freeze or thaw it.
        Return a list of the thawed (fitted) parameters.
        """
        fit_parnames = set()
        for parname, parval in zip(self.model.parnames, self.model.parvals):
            getattr(xijamod, parname).val = parval
            fit_parnames.add(parname)
            if any([re.match(x + '$', parname) for x in self.freeze_pars]):
                fit_logger.info('Freezing ' + parname)
                ui.freeze(getattr(xijamod, parname))
                fit_parnames.remove(parname)
            if any([re.match(x + '$', parname) for x in self.thaw_pars]):
                fit_logger.info('Thawing ' + parname)
                ui.thaw(getattr(xijamod, parname))
                fit_parnames.add(parname)
                if 'tau' in parname:
                    getattr(xijamod, parname).min = 0.1
        return fit_parnames

    def fit(self):
        method = 'simplex'

        dummy_data = np.zeros(1)
        dummy_times = np.arange(1)
        ui.load_arrays(1, dummy_times, dummy_data)

        ui.set_method(method)
        ui.get_method().config.update(sherpa_configs.get(method, {}))

        ui.load_user_model(CalcModel(self.model), 'xijamod')  # sets global xijamod
        ui.add_user_pars('xijamod', self.model.parnames)
        ui.set_model(1, 'xijamod')

        calc_stat = CalcStat(self.model, self.child_pipe)
        ui.load_user_stat('xijastat', calc_stat, lambda x: np.ones_like(x))
        ui.set_stat(xijastat)

        # If any params are thawed then do the fit
        if self.freeze_or_thaw_params(xijamod):
            try:
                ui.fit(1)
                calc_stat.message['status'] = 'finished'
                logging.debug('Fit finished normally')
            except FitTerminated as err:
                calc_stat.message['status'] = 'terminated'
                logging.debug('Got FitTerminated exception {}'.format(err))

        self.child_pipe.send(calc_stat.message)


class WidgetTable(dict):
    def __init__(self, n_rows, n_cols=None, colnames=None, show_header=False):
        if n_cols is None and colnames is None:
            raise ValueError('WidgetTable needs either n_cols or colnames')
        if colnames:
            self.colnames = colnames
            self.n_cols = len(colnames)
        else:
            self.n_cols = n_cols
            self.colnames = ['col{}'.format(i+1) for i in range(n_cols)]
        self.n_rows = n_rows
        self.show_header = show_header

        self.table = gtk.Table(rows=self.n_rows, columns=self.n_cols)

        self.box = gtk.ScrolledWindow()
        self.box.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
        self.box.add_with_viewport(self.table)

        dict.__init__(self)
            
    def __getitem__(self, rowcol):
        """Get widget at location (row, col) where ``col`` can be specified as
        either a numeric index or the column name.

        [Could do a nicer API that mimics np.array access, column wise by name,
        row-wise by since numeric index, or element by (row, col).  But maybe
        this isn't needed now].
        """
        row, col = rowcol
        if col in self.colnames:
            col = self.colnames.index(col)
        return dict.__getitem__(self, (row, col))

    def __setitem__(self, rowcol, widget):
        row, col = rowcol
        if rowcol in self:
            self.table.remove(self[rowcol])
        dict.__setitem__(self, rowcol, widget)
        if self.show_header:
            row += 1
        self.table.attach(widget, col, col + 1, row, row + 1)
        widget.show()


class Panel(object):
    def __init__(self, orient='h', homogeneous=False, spacing=0):
        Box = gtk.HBox if orient == 'h' else gtk.VBox
        self.box = Box(homogeneous, spacing)
        self.orient = orient

    def pack_start(self, child, expand=True, fill=True, padding=0):
        if isinstance(child, Panel):
            child = child.box
        return self.box.pack_start(child, expand, fill, padding)
        
    def pack_end(self, child, expand=True, fill=True, padding=0):
        if isinstance(child, Panel):
            child = child.box
        return self.pack_start(child, expand, fill, padding)
        

class PlotsPanel(Panel):
    def __init__(self, fit_worker, main_window):
        Panel.__init__(self, orient='v')
        self.model = fit_worker.model
        self.main_window = main_window
        self.plot_panels = []

    def add_plot_panel(self, plot_name):
        plot_panel = PlotPanel(plot_name, self)
        self.pack_start(plot_panel.box)
        self.plot_panels.append(plot_panel)
        self.main_window.show_all()

    def delete_plot_panel(self, widget, plot_name):
        plot_panels = []
        for plot_panel in self.plot_panels:
            if plot_panel.plot_name == plot_name:
                self.box.remove(plot_panel.box)
            else:
                plot_panels.append(plot_panel)

    def update(self, widget=None):
        for plot_panel in self.plot_panels:
            plot_panel.update()


class PlotPanel(Panel):
    def __init__(self, plot_name, plots_panel):
        Panel.__init__(self, orient='v')
        self.model = plots_panel.model

        self.plot_name = plot_name
        comp_name, plot_method = plot_name.split() # E.g. "tephin fit_resid"
        self.comp = [comp for comp in self.model.comps if comp.name == comp_name][0]
        self.plot_method = plot_method

        fig = Figure()
        canvas = FigureCanvas(fig)  # a gtk.DrawingArea
        toolbar = NavigationToolbar(canvas, plots_panel.main_window)
        delete_plot_button = gtk.Button('Delete')
        delete_plot_button.connect('clicked', plots_panel.delete_plot_panel, plot_name)

        toolbar_box = gtk.HBox() 
        toolbar_box.pack_start(toolbar)
        toolbar_box.pack_end(delete_plot_button, False, False, 0)
        self.pack_start(canvas)
        self.pack_start(toolbar_box, False, False)

        self.fig = fig
        self.ax = fig.add_subplot(1, 1, 1)
        self.canvas = canvas
        self.canvas.show()

    def update(self):
        self.model.calc()
        self.ax.cla()     # just for testing now
        plot_func = getattr(self.comp, 'plot_' + self.plot_method)
        plot_func(fig=self.fig, ax=self.ax)
        self.canvas.draw()


class ParamsPanel(Panel):
    def __init__(self, fit_worker):
        Panel.__init__(self, orient='v')
        self.fit_worker = fit_worker
        model = fit_worker.model
        params_table = WidgetTable(n_rows=len(model.parvals),
                                   n_cols=2,
                                   colnames=['parname', 'parval'],
                                   show_header=True)
        for row, parname, parval in zip(count(), model.parnames, model.parvals):
            params_table[row, 0] = gtk.Label(parname)
            params_table[row, 0].set_alignment(0, 0.5)
            params_table[row, 1] = gtk.Label(str(parval))
            params_table[row, 1].set_alignment(0, 0.5)
        self.pack_start(params_table.box, padding=0)
        self.params_table = params_table

    def update(self, fit_worker):
        model = fit_worker.model
        for row, parval in enumerate(model.parvals):
            self.params_table[row, 1].set_text(str(parval))


class ConsolePanel(Panel):
    def __init__(self, fit_worker):
        Panel.__init__(self, orient='v')
        self.fit_worker = fit_worker
        self.pack_start(gtk.Label('console_panel'), False, False, 0)


class ControlButtonsPanel(Panel):
    def __init__(self, fit_worker):
        Panel.__init__(self, orient='h')
        self.fit_worker = fit_worker
        
        self.fit_button = gtk.Button("Fit")
        self.stop_button = gtk.Button("Stop")
        self.add_plot_button = self.make_add_plot_button()
        self.quit_button = gtk.Button('Quit')

        self.pack_start(self.fit_button, False, False, 0)
        self.pack_start(self.stop_button, False, False, 0)
        self.pack_start(self.add_plot_button, False, False, 0)
        self.pack_start(self.quit_button, False, False, 0)

    def make_add_plot_button(self):
        apb = gtk.combo_box_new_text()
        apb.append_text('Add plot...')

        plot_names = ('{} {}'.format(comp.name, attr[5:])
                      for comp in self.fit_worker.model.comps
                      for attr in dir(comp)
                      if attr.startswith('plot_'))

        for plot_name in plot_names:
            apb.append_text(plot_name)

        apb.set_active(0)
        return apb


class MainLeftPanel(Panel):
    def __init__(self, fit_worker, main_window):
        Panel.__init__(self, orient='v')
        self.control_buttons_panel = ControlButtonsPanel(fit_worker)
        self.plots_panel = PlotsPanel(fit_worker, main_window)
        self.pack_start(self.control_buttons_panel, False, False, 0)
        self.pack_start(self.plots_panel)


class MainRightPanel(Panel):
    def __init__(self, fit_worker):
        Panel.__init__(self, orient='v')
        self.params_panel = ParamsPanel(fit_worker)
        self.console_panel = ConsolePanel(fit_worker)

        self.pack_start(self.params_panel)
        self.pack_start(self.console_panel, False, False, 0)


class MainWindow(object):
    # This is a callback function. The data arguments are ignored
    # in this example. More on callbacks below.
    def __init__(self, fit_worker):
        self.fit_worker = fit_worker
        # create a new window
        self.window = gtk.Window(gtk.WINDOW_TOPLEVEL)
        self.window.connect("destroy", self.destroy)
        self.window.set_default_size(1400, 800)
    
        # Sets the border width of the window.
        self.window.set_border_width(10)
        self.main_box = Panel(orient='h')
        self.window.add(self.main_box.box)

        self.main_left_panel = MainLeftPanel(fit_worker, self.window)
        self.main_right_panel = MainRightPanel(fit_worker)
        self.main_box.pack_start(self.main_left_panel)
        self.main_box.pack_start(self.main_right_panel)

        mlp = self.main_left_panel
        cbp = mlp.control_buttons_panel
        cbp.fit_button.connect("clicked", fit_worker.start)
        cbp.fit_button.connect("clicked", lambda widget:
                                  gobject.timeout_add(200, self.fit_monitor))
        cbp.stop_button.connect("clicked", fit_worker.terminate)
        cbp.quit_button.connect('clicked', self.destroy)
        cbp.add_plot_button.connect('changed', self.add_plot)

        # and the window
        self.window.show_all()

    def main(self):
        # All PyGTK applications must have a gtk.main(). Control ends here
        # and waits for an event to occur (like a key press or mouse event).
        gtk.main()

    def destroy(self, widget, data=None):
        gtk.main_quit()

    def add_plot(self, widget):
        model = widget.get_model()
        index = widget.get_active()
        pp = self.main_left_panel.plots_panel
        if index:
            print "Add plot", model[index][0]
            pp.add_plot_panel(model[index][0])
            pp.update()
        widget.set_active(0)
            
    def fit_monitor(self):
        fit_worker = self.fit_worker
        msg = None
        while fit_worker.parent_pipe.poll():
            # Keep reading messages until there are no more or until getting
            # a message indicating fit is stopped.
            msg = fit_worker.parent_pipe.recv()
            fit_stopped = msg['status'] in ('terminated', 'finished')
            if fit_stopped:
                fit_worker.fit_process.join()
                break

        if msg:
            # Update the fit_worker model parameters and then the corresponding
            # params table widget. 
            fit_worker.model.parvals = msg['parvals']
            self.main_right_panel.params_panel.update(fit_worker)
            self.main_left_panel.plots_panel.update()
            return not fit_stopped
        else:
            return True


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
                      default=15,
                      help="Number of days in fit interval (default=90")
    parser.add_option("--stop",
                      default="2010:180",
                      help="Stop time of fit interval (default=NOW - 7 days)")
    parser.add_option("--method",
                      default='simplex',
                      help="Fit method (default=simple)")
    parser.add_option("--model",
                      default='minusz',
                      help="Model to predict (default=minusz)")
    parser.add_option("--thaw-pars",
                      default='solarheat__t.*__P_.*',
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

# Default configurations for fit methods
sherpa_configs = dict(
    simplex = dict(ftol=1e-3,
                   finalsimplex=0,   # converge based only on length of simplex
                   maxfev=1000),
    )
thaw_pars = opt.thaw_pars.split()
freeze_pars = ('.*',
               '.*__T_e',
               '.*__pi_.*',
               '.*__pf_.*',
               '.*__tau_.*',
               '.*__tau_sc',
               '.*__p_ampl',
               )
fit_worker = FitWorker(model, freeze_pars, thaw_pars)

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

main_window = MainWindow(fit_worker)
main_window.main()
