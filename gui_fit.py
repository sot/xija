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
import argparse
import fnmatch

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
        self.model.parvals = parvals

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
            self.min_parvals = self.model.parvals

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
    def __init__(self, model, method='simplex'):
        self.model = model
        self.method = method
        self.parent_pipe, self.child_pipe = multiprocessing.Pipe()

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

    def fit(self):
        dummy_data = np.zeros(1)
        dummy_times = np.arange(1)
        ui.load_arrays(1, dummy_times, dummy_data)

        ui.set_method(self.method)
        ui.get_method().config.update(sherpa_configs.get(self.method, {}))

        ui.load_user_model(CalcModel(self.model), 'xijamod')  # sets global xijamod
        ui.add_user_pars('xijamod', self.model.parnames)
        ui.set_model(1, 'xijamod')

        calc_stat = CalcStat(self.model, self.child_pipe)
        ui.load_user_stat('xijastat', calc_stat, lambda x: np.ones_like(x))
        ui.set_stat(xijastat)

        # Set frozen, min, and max attributes for each xijamod parameter
        for par in self.model.pars:
            xijamod_par = getattr(xijamod, par.full_name)
            xijamod_par.val = par.val
            xijamod_par.frozen = par.frozen
            xijamod_par.min = par.min
            xijamod_par.max = par.max

        if any(not par.frozen for par in self.model.pars):
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
        self.sharex = {}        # Shared x-axes keyed by x-axis type

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
        self.plot_panels = plot_panels

    def update(self, widget=None):
        self.model.calc()
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

        # Add shared x-axes for plot methods matching <yaxis_type>__<xaxis_type>.
        # First such plot has sharex=None, subsequent ones use the first axis.
        try:
            xaxis_type = plot_method.split('__')[1]
        except IndexError:
            self.ax = fig.add_subplot(111)
        else:
            self.ax = fig.add_subplot(111, sharex=plots_panel.sharex.get(xaxis_type))
            plots_panel.sharex.setdefault(xaxis_type, self.ax)
            
        self.canvas = canvas
        self.canvas.show()

    def update(self):
        plot_func = getattr(self.comp, 'plot_' + self.plot_method)
        plot_func(fig=self.fig, ax=self.ax)
        self.canvas.draw()


class ParamsPanel(Panel):
    def __init__(self, fit_worker, plots_panel):
        Panel.__init__(self, orient='v')
        self.fit_worker = fit_worker
        self.plots_panel = plots_panel
        model = fit_worker.model
        params_table = WidgetTable(n_rows=len(model.pars),
                                   n_cols=6,
                                   colnames=['name', 'val', 'thawed', 'min', 'max'],
                                   show_header=True)
        self.adj_handlers = {}
        for row, par in zip(count(), model.pars):
            # Thawed (i.e. fit the parameter)
            frozen = params_table[row, 0] = gtk.CheckButton()
            frozen.set_active(not par.frozen)
            frozen.connect('toggled', self.frozen_toggled, par)

            # par full name
            params_table[row, 1] = gtk.Label(par.full_name)
            params_table[row, 1].set_alignment(0, 0.5)

            # Value
            params_table[row, 2] = gtk.Label(par.fmt.format(par.val))
            params_table[row, 2].set_alignment(0, 0.5)

            # Slider
            incr = (par.max - par.min) / 100.0
            adj = gtk.Adjustment(par.val, par.min, par.max, incr, incr, 0.0)
            slider = gtk.HScale(adj)
            slider.set_update_policy(gtk.UPDATE_CONTINUOUS)
            slider.set_draw_value(False)
            params_table[row, 4] = slider
            handler = adj.connect('value_changed', self.slider_changed, row)
            self.adj_handlers[row] = handler

            # Min of slider
            entry = params_table[row, 3] = gtk.Entry()
            entry.set_text(par.fmt.format(par.min))
            entry.set_width_chars(4)
            entry.connect('activate', self.minmax_changed, adj, par, 'min')

            # Max of slider
            entry = params_table[row, 5] = gtk.Entry()
            entry.set_text(par.fmt.format(par.max))
            entry.set_width_chars(6)
            entry.connect('activate', self.minmax_changed, adj, par, 'max')

        self.pack_start(params_table.box, True, True, padding=10)
        self.params_table = params_table

    def frozen_toggled(self, widget, par):
        par.frozen = not widget.get_active()

    def minmax_changed(self, widget, adj, par, minmax):
        """Min or max Entry box value changed.  Update the slider (adj)
        limits and the par min/max accordingly.
        """
        try:
            val = float(widget.get_text())
        except ValueError:
            pass
        else:
            (adj.set_lower if minmax == 'min' else adj.set_upper)(val)
            incr = (adj.get_upper() - adj.get_lower()) / 100.0
            adj.set_step_increment(incr)
            adj.set_page_increment(incr)
            setattr(par, minmax, val)

    def slider_changed(self, widget, row):
        parval = widget.value  # widget is an adjustment
        par = self.fit_worker.model.pars[row]
        par.val = parval
        self.params_table[row, 2].set_text(par.fmt.format(parval))
        self.plots_panel.update()
        
    def update(self, fit_worker):
        model = fit_worker.model
        for row, par in enumerate(model.pars):
            val_label = self.params_table[row, 2]
            par_val_text = par.fmt.format(par.val)
            if val_label.get_text() != par_val_text:
                val_label.set_text(par_val_text)
                # Change the slider value but block the signal to update the plot
                adj = self.params_table[row, 4].get_adjustment()
                adj.handler_block(self.adj_handlers[row])
                adj.set_value(par.val)
                adj.handler_unblock(self.adj_handlers[row])


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
        self.save_button = gtk.Button("Save")
        self.add_plot_button = self.make_add_plot_button()
        self.quit_button = gtk.Button('Quit')
        self.command_entry = gtk.Entry()
        self.command_entry.set_width_chars(10)
        self.command_entry.set_text('')
        self.command_panel = Panel()
        self.command_panel.pack_start(gtk.Label('Command:'), False, False, 0)
        self.command_panel.pack_start(self.command_entry, False, False, 0)

        self.pack_start(self.fit_button, False, False, 0)
        self.pack_start(self.stop_button, False, False, 0)
        self.pack_start(self.save_button, False, False, 0)
        self.pack_start(self.add_plot_button, False, False, 0)
        self.pack_start(self.command_panel, False, False, 0)
        self.pack_start(self.quit_button, False, False, 0)

    def make_add_plot_button(self):
        apb = gtk.combo_box_new_text()
        apb.append_text('Add plot...')

        plot_names = ['{} {}'.format(comp.name, attr[5:])
                      for comp in self.fit_worker.model.comps
                      for attr in dir(comp)
                      if attr.startswith('plot_')]

        self.plot_names = plot_names
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
    def __init__(self, fit_worker, plots_panel):
        Panel.__init__(self, orient='v')
        self.params_panel = ParamsPanel(fit_worker, plots_panel)
        self.console_panel = ConsolePanel(fit_worker)

        self.pack_start(self.params_panel)
        self.pack_start(self.console_panel, False)


class MainWindow(object):
    # This is a callback function. The data arguments are ignored
    # in this example. More on callbacks below.
    def __init__(self, fit_worker):
        self.fit_worker = fit_worker

        # create a new window
        self.window = gtk.Window(gtk.WINDOW_TOPLEVEL)
        self.window.connect("destroy", self.destroy)
        self.window.set_default_size(*gui_config.get('size', (1400, 800)))
    
        # Sets the border width of the window.
        self.window.set_border_width(10)
        self.main_box = Panel(orient='h')
        self.window.add(self.main_box.box)

        self.main_left_panel = MainLeftPanel(fit_worker, self.window)
        mlp = self.main_left_panel
        self.main_right_panel = MainRightPanel(fit_worker, mlp.plots_panel)
        self.main_box.pack_start(self.main_left_panel)
        self.main_box.pack_start(self.main_right_panel)

        cbp = mlp.control_buttons_panel
        cbp.fit_button.connect("clicked", fit_worker.start)
        cbp.fit_button.connect("clicked", self.fit_monitor)
        cbp.stop_button.connect("clicked", fit_worker.terminate)
        cbp.save_button.connect("clicked", self.save_model_file)
        cbp.quit_button.connect('clicked', self.destroy)
        cbp.add_plot_button.connect('changed', self.add_plot)
        cbp.command_entry.connect('activate', self.command_activated)

        # Add plots from previous Save
        for plot_name in gui_config.get('plot_names', []):
            try:
                plot_index = cbp.plot_names.index(plot_name) + 1
                cbp.add_plot_button.set_active(plot_index)
                print "Adding plot {} {}".format(plot_name, plot_index)
                time.sleep(0.05)  # is it needed?
            except ValueError:
                print "ERROR: Unexpected plot_name {}".format(plot_name)

        # Show everything finally
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
            
    def fit_monitor(self, widget=None):
        fit_worker = self.fit_worker
        msg = None
        fit_stopped = False
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

        # If fit has not stopped then set another timeout 200 msec from now
        if not fit_stopped:
            gobject.timeout_add(200, self.fit_monitor)
            
        # Terminate the current timeout
        return False

    def command_activated(self, widget):
        """Respond to a command like "freeze solarheat*dP*" submitted via the
        command entry box.  The first word is either "freeze" or "thaw" (with
        possibility for other commands later) and the subsequent args are
        space-delimited parameter globs using the UNIX file-globbing syntax.
        This then sets the corresponding params_table checkbuttons.
        """
        command = widget.get_text().strip()
        vals = command.split()
        cmd = vals[0]  # currently freeze or thaw
        if cmd not in ('freeze', 'thaw') or len(vals) <= 1:
            # dialog box..
            print "ERROR: bad command: {}".format(command)
            return
        par_regexes = [fnmatch.translate(x) for x in vals[1:]]

        params_table = self.main_right_panel.params_panel.params_table
        for row, par in enumerate(self.fit_worker.model.pars):
            for par_regex in par_regexes:
                if re.match(par_regex, par.full_name):
                     checkbutton = params_table[row, 0]
                     checkbutton.set_active(cmd == 'thaw')
        widget.set_text('')

    def save_model_file(self, widget):
        chooser = gtk.FileChooserDialog(title=None,action=gtk.FILE_CHOOSER_ACTION_SAVE,
                                        buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                                                 gtk.STOCK_SAVE, gtk.RESPONSE_OK))
        chooser.set_default_response(gtk.RESPONSE_OK)
        chooser.set_current_name(gui_config['filename'])
        filter = gtk.FileFilter()
        filter.set_name("Model files")
        filter.add_pattern("*.json")
        chooser.add_filter(filter)

        filter = gtk.FileFilter()
        filter.set_name("All files")
        filter.add_pattern("*")
        chooser.add_filter(filter)
        
        response = chooser.run()
        filename = chooser.get_filename()
        chooser.destroy()

        if response == gtk.RESPONSE_OK:
            model_spec = self.fit_worker.model.model_spec
            gui_config['plot_names'] = [x.plot_name
                                        for x in self.main_left_panel.plots_panel.plot_panels]
            gui_config['size'] = self.window.get_size()
            model_spec['gui_config'] = gui_config
            try:
                self.fit_worker.model.write(filename, model_spec)
                gui_config['filename'] = filename
            except IOError:
                print "Error writing {}".format(filename)
                # Raise a dialog box here.
        

def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename",
                      help="Model file")
    parser.add_argument("--days",
                      type=float,
                      default=15,
                      help="Number of days in fit interval (default=90")
    parser.add_argument("--stop",
                      help="Stop time of fit interval (default=model values)")
    parser.add_argument("--nproc",
                      default=0,
                      type=int,
                      help="Number of processors (default=1)")
    parser.add_argument("--fit-method",
                        default="simplex",
                        help="Sherpa fit method (simplex|moncar|levmar)")
    parser.add_argument("--inherit-from",
                      help="Inherit par values from model spec file")
    parser.add_argument("--quiet",
                      default=False,
                      action='store_true',
                      help="Suppress screen output")

    return parser.parse_args()

opt = get_options()

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

model_spec = json.load(open(opt.filename, 'r'))
src['model'] = model_spec['name']

# Use supplied stop time and days OR use model_spec values if stop not supplied
if opt.stop:
    start = DateTime(DateTime(opt.stop).secs - opt.days * 86400).date[:8]
    stop = opt.stop
else:
    start = model_spec['datestart']
    stop = model_spec['datestop']

model = xija.ThermalModel(model_spec['name'], start, stop, model_spec=model_spec)
model.make()

if opt.inherit_from:
    inherit_spec = json.load(open(opt.inherit_from, 'r'))
    inherit_pars = {par['full_name']: par for par in inherit_spec['pars']}
    for par in model.pars:
        if par.full_name in inherit_pars:
            print "Inheriting par {}".format(par.full_name)
            par.val = inherit_pars[par.full_name]['val']
            par.min = inherit_pars[par.full_name]['min']
            par.max = inherit_pars[par.full_name]['max']
            par.frozen = inherit_pars[par.full_name]['frozen']
            par.fmt = inherit_pars[par.full_name]['fmt']

gui_config = model_spec.get('gui_config', {})
gui_config['filename'] = os.path.abspath(opt.filename)

# Default configurations for fit methods
sherpa_configs = dict(
    simplex = dict(ftol=1e-3,
                   finalsimplex=0,   # converge based only on length of simplex
                   maxfev=1000),
    )

fit_worker = FitWorker(model, opt.fit_method)

main_window = MainWindow(fit_worker)
main_window.main()
