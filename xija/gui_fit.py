#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import print_function

import sys
import os
import ast
import multiprocessing
import time
import functools

from PyQt5 import QtGui, QtCore, QtWidgets

from itertools import count
import argparse
import fnmatch

import re
import json
import logging

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import numpy as np

from Chandra.Time import DateTime
import pyyaks.context as pyc
import pyyaks.logger

try:
    import acis_taco as taco
except ImportError:
    import Chandra.taco as taco
import xija
import sherpa.ui as ui

logging.basicConfig(level=logging.DEBUG)
logging.debug('Importing gui_fit from {}'.format(__file__))

# Globals for fitting, set in main
MODEL = None
POOL = None
PARENT_PIPE = None
CHILD_PIPE = None
FIT_WORKER = None
CALC_STAT = None

# from xija.fit import (FitTerminated, CalcModel, CalcStat, FitWorker)


fit_logger = pyyaks.logger.get_logger(name='fit', level=logging.INFO,
                                      format='[%(levelname)s] (%(processName)-10s) %(message)s')

# Default configurations for fit methods
sherpa_configs = dict(
    simplex = dict(ftol=1e-3,
                   finalsimplex=0,   # converge based only on length of simplex
                   maxfev=1000),
    )
gui_config = {}


class FitTerminated(Exception):
    pass


def sherpa_xija_model(parvals, x):
    """This is the Sherpa calc_model function, but in this case calc_model does not
    actually calculate anything but instead just stores the desired paramters.  This
    allows for multiprocessing where only the fit statistic gets passed between nodes.
    """
    print('fit call: {}'.format(x))
    logging.info('Calculating params:')
    for parname, parval, newparval in zip(MODEL.parnames, MODEL.parvals, parvals):
        if parval != newparval:
            fit_logger.info('  {0}: {1}'.format(parname, newparval))
    MODEL.parvals = parvals

    return np.ones_like(x)


class CalcStat(object):
    def __init__(self):
        self.cache_fit_stat = {}
        self.min_fit_stat = None
        self.min_par_vals = MODEL.parvals

    def __call__(self, _data, _model, staterror=None, syserror=None, weight=None):
        """Calculate fit statistic for the xija model.  The args _data and _model
        are sent by Sherpa but they are fictitious -- the real data and model are
        stored in the xija model MODEL.
        """
        # parvals_key = tuple('%.4e' % x for x in MODEL.parvals)
        try:
            raise KeyError
            # fit_stat = self.cache_fit_stat[parvals_key]
            # fit_logger.info('nmass_model: Cache hit %s' % str(parvals_key))
        except KeyError:
            fit_stat = MODEL.calc_stat()

        fit_logger.info('Fit statistic: %.4f' % fit_stat)
        # self.cache_fit_stat[parvals_key] = fit_stat

        if self.min_fit_stat is None or fit_stat < self.min_fit_stat:
            self.min_fit_stat = fit_stat
            self.min_parvals = MODEL.parvals

        self.message = {'status': 'fitting',
                        'time': time.time(),
                        'parvals': MODEL.parvals,
                        'fit_stat': fit_stat,
                        'min_parvals': self.min_parvals,
                        'min_fit_stat': self.min_fit_stat}

        CHILD_PIPE.send(self.message)

        while CHILD_PIPE.poll():
            pipe_val = CHILD_PIPE.recv()
            if pipe_val == 'terminate':
                MODEL.parvals = self.min_parvals
                raise FitTerminated('terminated')

        return fit_stat, np.ones(1)


class FitWorker(object):
    def start(self, widget=None):
        """Start a Sherpa fit process as a spawned (non-blocking) process.
        """
        self.fit_process = POOL.apply_async(sherpa_fit, [()],
                                            callback=sherpa_fit_finished)
        logging.debug('Fit started')

    def terminate(self, widget=None):
        """Terminate a Sherpa fit process in a controlled way by sending a
        message.  Get the final parameter values if possible.
        """
        PARENT_PIPE.send('terminate')
        self.fit_process.wait()
        logging.debug('Fit terminated')


def sherpa_fit_finished(result):
    logging.debug('Finished fit {}'.format(result))


def sherpa_fit(*args):
    print('Here in global_fit{}'.format(args))

    # Set frozen, min, and max attributes for each xijamod parameter
    for par in MODEL.pars:
        xijamod_par = getattr(xijamod, par.full_name)
        xijamod_par.val = par.val
        xijamod_par.frozen = par.frozen
        xijamod_par.min = par.min
        xijamod_par.max = par.max

    if any(not par.frozen for par in MODEL.pars):
        try:
            ui.fit(1)
            CALC_STAT.message['status'] = 'finished'
            logging.debug('Fit finished normally')
        except FitTerminated as err:
            CALC_STAT.message['status'] = 'terminated'
            logging.debug('Got FitTerminated exception {}'.format(err))

    CHILD_PIPE.send(CALC_STAT.message)


class WidgetTable(dict):
    def __init__(self, n_rows, n_cols=None, colnames=None, show_header=False, colwidths=None):
        if n_cols is None and colnames is None:
            raise ValueError('WidgetTable needs either n_cols or colnames')
        if colnames:
            self.colnames = colnames
            self.n_cols = len(colnames)
        else:
            self.n_cols = n_cols
            self.colnames = ['col{}'.format(i + 1) for i in range(n_cols)]
        self.n_rows = n_rows
        self.show_header = show_header

        self.table = QtWidgets.QTableWidget(self.n_rows, self.n_cols)

        if show_header and colnames:
            self.table.setHorizontalHeaderLabels(colnames)

        if colwidths:
            for col, width in colwidths.items():
                self.table.setColumnWidth(col, width)

        # self.box.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
        # self.box.setWidget(self.table)

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
        dict.__setitem__(self, rowcol, widget)
        self.table.setCellWidget(row, col, widget)
        # widget.show()


class Panel(object):
    def __init__(self, orient='h', homogeneous=False, spacing=0):
        Box = QtWidgets.QHBoxLayout if orient == 'h' else QtWidgets.QVBoxLayout
        self.box = Box()  # homogeneous, spacing ??
        self.orient = orient

    def add_stretch(self, value):
        self.box.addStretch(value)

    def pack_start(self, child, expand=True, fill=True, padding=0):
        if isinstance(child, Panel):
            child = child.box
        if isinstance(child, QtWidgets.QBoxLayout):
            out = self.box.addLayout(child)
        else:
            out = self.box.addWidget(child)  # , expand, fill, padding
        return out

    def pack_end(self, child, expand=True, fill=True, padding=0):
        return self.pack_start(child, expand, fill, padding)


def clearLayout(layout):
    """
    From http://stackoverflow.com/questions/9374063/pyqt4-remove-widgets-and-layout-as-well
    """
    if layout is not None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                clearLayout(item.layout())


class MplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure()  # figsize=(width, height), dpi=dpi)

        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)
        self.updateGeometry()


class PlotBox(QtWidgets.QVBoxLayout):
    def __init__(self, plot_name, plots_box):
        super(PlotBox, self).__init__()

        comp_name, plot_method = plot_name.split()  # E.g. "tephin fit_resid"
        self.comp = MODEL.comp[comp_name]
        self.plot_method = plot_method

        self.plot_name = plot_name

        canvas = MplCanvas(parent=None)
        toolbar = NavigationToolbar(canvas, parent=None)

        delete_plot_button = QtWidgets.QPushButton('Delete')
        delete_plot_button.clicked.connect(
            functools.partial(plots_box.delete_plot_box, plot_name))

        toolbar_box = QtWidgets.QHBoxLayout()
        toolbar_box.addWidget(toolbar)
        toolbar_box.addStretch(1)
        toolbar_box.addWidget(delete_plot_button)

        self.addWidget(canvas)
        self.addLayout(toolbar_box)

        self.fig = canvas.fig
        # Add shared x-axes for plot methods matching <yaxis_type>__<xaxis_type>.
        # First such plot has sharex=None, subsequent ones use the first axis.
        try:
            xaxis_type = plot_method.split('__')[1]
        except IndexError:
            self.ax = self.fig.add_subplot(111)
        else:
            sharex = plots_box.sharex.get(xaxis_type)
            self.ax = self.fig.add_subplot(111, sharex=sharex)
            if sharex is not None:
                self.ax.autoscale(enable=False, axis='x')
            plots_box.sharex.setdefault(xaxis_type, self.ax)

        self.canvas = canvas
        self.canvas.show()

    def update(self):
        plot_func = getattr(self.comp, 'plot_' + self.plot_method)
        plot_func(fig=self.fig, ax=self.ax)
        self.canvas.draw()


class PlotsBox(QtWidgets.QVBoxLayout):
    def __init__(self, main_window):
        super(QtWidgets.QVBoxLayout, self).__init__()
        self.main_window = main_window
        self.sharex = {}        # Shared x-axes keyed by x-axis type

    def add_plot_box(self, plot_name):
        plot_name = str(plot_name)
        print('adding plot ', plot_name)
        plot_box = PlotBox(plot_name, self)
        self.addLayout(plot_box)
        plot_box.update()

    def delete_plot_box(self, plot_name):
        for plot_box in self.findChildren(PlotBox):
            print(plot_box.plot_name)
            if plot_box.plot_name == plot_name:
                self.removeItem(plot_box)
                clearLayout(plot_box)
        self.update()

    def update_plots(self):
        cbp = self.main_window.main_left_panel.control_buttons_panel
        cbp.update_status.setText(' BUSY... ')
        MODEL.calc()
        for plot_box in self.findChildren(PlotBox):
            print('Updating plot {}'.format(plot_box.plot_name))
            plot_box.update()
        cbp.update_status.setText('')


class ParamsPanel(Panel):
    def __init__(self, plots_panel):
        Panel.__init__(self, orient='v')
        self.plots_panel = plots_panel

        params_table = WidgetTable(n_rows=len(MODEL.pars),
                                   colnames=['fit', 'name', 'val', 'thawed', 'min', 'max'],
                                   colwidths={0: 30, 1: 250},
                                   show_header=True)
        self.adj_handlers = {}
        for row, par in zip(count(), MODEL.pars):
            # Thawed (i.e. fit the parameter)
            frozen = params_table[row, 0] = QtWidgets.QCheckBox()
            # frozen.set_active(not par.frozen)
            # frozen.connect('toggled', self.frozen_toggled, par)

            # par full name
            params_table[row, 1] = QtWidgets.QLabel(par.full_name)
            # params_table[row, 1].set_alignment(0, 0.5)

            # Slider
            # incr = (par.max - par.min) / 100.0
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)  # (par.val, par.min, par.max, incr, incr, 0.0)
            slider.setMinimum(par.min)
            slider.setMaximum(par.max)
            slider.setValue(par.val)
            # slider.set_update_policy(gtk.UPDATE_CONTINUOUS)
            # slider.set_draw_value(False)
            # slider.set_size_request(70, -1)
            params_table[row, 4] = slider
            # handler = adj.connect('value_changed', self.slider_changed, row)
            # self.adj_handlers[row] = handler

            # Value
            entry = params_table[row, 2] = QtWidgets.QLineEdit()
            # entry.set_width_chars(10)
            entry.setText(par.fmt.format(par.val))
            # entry.connect('activate', self.par_attr_changed, adj, par, 'val')

            # Min of slider
            entry = params_table[row, 3] = QtWidgets.QLineEdit()
            entry.setText(par.fmt.format(par.min))
            # entry.set_width_chars(4)
            # entry.connect('activate', self.par_attr_changed, adj, par, 'min')

            # Max of slider
            entry = params_table[row, 5] = QtWidgets.QLineEdit()
            entry.setText(par.fmt.format(par.max))
            # entry.set_width_chars(6)
            # entry.connect('activate', self.par_attr_changed, adj, par, 'max')

        self.pack_start(params_table.table, True, True, padding=10)
        self.params_table = params_table

    def frozen_toggled(self, widget, par):
        par.frozen = not widget.get_active()

    def par_attr_changed(self, widget, adj, par, par_attr):
        """Min, val, or max Entry box value changed.  Update the slider (adj)
        limits and the par min/val/max accordingly.
        """
        try:
            val = float(widget.get_text())
        except ValueError:
            pass
        else:
            if par_attr == 'min':
                adj.set_lower(val)
            elif par_attr == 'max':
                adj.set_upper(val)
            elif par_attr == 'val':
                adj.set_value(val)
            incr = (adj.get_upper() - adj.get_lower()) / 100.0
            adj.set_step_increment(incr)
            adj.set_page_increment(incr)
            setattr(par, par_attr, val)

    def slider_changed(self, widget, row):
        parval = widget.value  # widget is an adjustment
        par = MODEL.pars[row]
        par.val = parval
        self.params_table[row, 2].setText(par.fmt.format(parval))
        self.plots_panel.update()

    def update(self):
        for row, par in enumerate(MODEL.pars):
            val_label = self.params_table[row, 2]
            par_val_text = par.fmt.format(par.val)
            if str(val_label.text) != par_val_text:
                val_label.setText(par_val_text)
                # Change the slider value but block the signal to update the plot
                # adj = self.params_table[row, 4].get_adjustment()
                # adj.handler_block(self.adj_handlers[row])
                # adj.set_value(par.val)
                # adj.handler_unblock(self.adj_handlers[row])


class ConsolePanel(Panel):
    def __init__(self):
        Panel.__init__(self, orient='v')
        self.pack_start(QtWidgets.QLabel('console_panel'), False, False, 0)


class ControlButtonsPanel(Panel):
    def __init__(self):
        Panel.__init__(self, orient='h')

        self.fit_button = QtWidgets.QPushButton("Fit")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.save_button = QtWidgets.QPushButton("Save")
        self.add_plot_button = self.make_add_plot_button()
        self.update_status = QtWidgets.QLabel()
        # self.update_status.set_width_chars(10)
        self.quit_button = QtWidgets.QPushButton('Quit')
        self.command_entry = QtWidgets.QLineEdit()
        # self.command_entry.set_width_chars(10)
        # self.command_entry.setText('')
        self.command_panel = Panel()
        self.command_panel.pack_start(QtWidgets.QLabel('Command:'), False, False, 0)
        self.command_panel.pack_start(self.command_entry, False, False, 0)

        self.pack_start(self.fit_button, False, False, 0)
        self.pack_start(self.stop_button, False, False, 0)
        self.pack_start(self.save_button, False, False, 0)
        self.pack_start(self.add_plot_button, False, False, 0)
        self.pack_start(self.update_status, False, False, 0)
        self.pack_start(self.command_panel, False, False, 0)
        self.add_stretch(1)
        self.pack_start(self.quit_button, False, False, 0)

    def make_add_plot_button(self):
        apb = QtWidgets.QComboBox()
        apb.addItem('Add plot...')

        plot_names = ['{} {}'.format(comp.name, attr[5:])
                      for comp in MODEL.comps
                      for attr in dir(comp)
                      if attr.startswith('plot_')]

        self.plot_names = plot_names
        for plot_name in plot_names:
            apb.addItem(plot_name)

        return apb


class MainLeftPanel(Panel):
    def __init__(self, main_window):
        Panel.__init__(self, orient='v')
        self.control_buttons_panel = ControlButtonsPanel()
        self.plots_box = PlotsBox(main_window)
        self.pack_start(self.control_buttons_panel, False, False, 0)
        self.pack_start(self.plots_box)
        self.add_stretch(1)
        # self.plots_panel.add_plot_panel('pftank2t fit_resid')
        # self.box.addStretch(1)


class MainRightPanel(Panel):
    def __init__(self, plots_panel):
        Panel.__init__(self, orient='v')
        self.params_panel = ParamsPanel(plots_panel)
        self.console_panel = ConsolePanel()

        okButton = QtWidgets.QPushButton("OK right")
        self.pack_start(self.params_panel)
        self.pack_start(self.console_panel, False)
        # self.pack_start(okButton, False)
        self.box.addWidget(okButton)
        # self.box.addStretch(1)


class MainWindow(object):
    # This is a callback function. The data arguments are ignored
    # in this example. More on callbacks below.
    def __init__(self):
        # create a new window
        self.window = QtWidgets.QWidget()
        # self.window.connect("destroy", self.destroy)
        self.window.setGeometry(0, 0, *gui_config.get('size', (1400, 800)))

        # hbox = QHBoxLayout()
        # hbox.addStretch(1)
        # hbox.addWidget(okButton)
        # hbox.addWidget(cancelButton)

        # Sets the border width of the window.
        # self.window.set_border_width(10)
        self.main_box = Panel(orient='h')

        # okButton = QtGui.QPushButton("OK")
        # cancelButton = QtGui.QPushButton("Cancel")

        # This is the Layout Box that holds the top-level stuff in the main window
        main_window_hbox = QtWidgets.QHBoxLayout()
        self.window.setLayout(main_window_hbox)

        self.main_left_panel = MainLeftPanel(self)
        mlp = self.main_left_panel

        self.main_right_panel = MainRightPanel(mlp.plots_box)

        cbp = mlp.control_buttons_panel
        cbp.fit_button.clicked.connect(FIT_WORKER.start)
        cbp.fit_button.clicked.connect(self.fit_monitor)
        cbp.stop_button.clicked.connect(FIT_WORKER.terminate)
        cbp.save_button.clicked.connect(self.save_model_file)
        cbp.quit_button.clicked.connect(QtCore.QCoreApplication.instance().quit)
        cbp.add_plot_button.activated[str].connect(self.add_plot)
        # cbp.command_entry.connect('activate', self.command_activated)

        # # Add plots from previous Save
        # for plot_name in gui_config.get('plot_names', []):
        #     try:
        #         plot_index = cbp.plot_names.index(plot_name) + 1
        #         cbp.add_plot_button.set_active(plot_index)
        #         print("Adding plot {} {}".format(plot_name, plot_index))
        #         time.sleep(0.05)  # is it needed?
        #     except ValueError:
        #         print("ERROR: Unexpected plot_name {}".format(plot_name))

        # Show everything finally
        main_window_hbox.addLayout(mlp.box)
        # main_window_hbox.addStretch(1)
        main_window_hbox.addLayout(self.main_right_panel.box)

        self.window.show()

    def add_plot(self, plotname):
        sender = self.window.sender()
        # model = widget.get_model()
        # index = widget.get_active()
        pp = self.main_left_panel.plots_box
        print('Add plot {} {}'.format(plotname, sender))
        pp.add_plot_box(plotname)
        sender.setCurrentIndex(0)
        # if index:
        #     print("Add plot", model[index][0])
        #     pp.add_plot_panel(model[index][0])
        #     pp.update()
        # widget.set_active(0)

    def fit_monitor(self, *args):
        msg = None
        fit_stopped = False
        while PARENT_PIPE.poll():
            # Keep reading messages until there are no more or until getting
            # a message indicating fit is stopped.
            msg = PARENT_PIPE.recv()
            fit_stopped = msg['status'] in ('terminated', 'finished')
            if fit_stopped:
                FIT_WORKER.fit_process.wait()
                print("\n*********************************")
                print("  FIT", msg['status'].upper())
                print("*********************************\n")
                break

        if msg:
            # Update the fit_worker model parameters and then the corresponding
            # params table widget.
            MODEL.parvals = msg['parvals']
            self.main_right_panel.params_panel.update()
            self.main_left_panel.plots_box.update_plots()

        # If fit has not stopped then set another timeout 200 msec from now
        if not fit_stopped:
            QtCore.QTimer.singleShot(200, self.fit_monitor)

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
            print("ERROR: bad command: {}".format(command))
            return
        par_regexes = [fnmatch.translate(x) for x in vals[1:]]

        params_table = self.main_right_panel.params_panel.params_table
        for row, par in enumerate(MODEL.pars):
            for par_regex in par_regexes:
                if re.match(par_regex, par.full_name):
                     checkbutton = params_table[row, 0]
                     checkbutton.set_active(cmd == 'thaw')
        widget.setText('')

    def save_model_file(self, *args):
        filename = QtGui.QFileDialog.getOpenFileName(None, 'Open file', os.getcwd(),
                                                     'JSON files (*.json);; All files (*)')
        filename = str(filename)
        if filename != '':
            model_spec = MODEL.model_spec
            gui_config['plot_names'] = [x.plot_name
                                        for x in self.main_left_panel.plots_box.plot_boxes]
            gui_config['size'] = (self.window.size().width(), self.window.size().height())
            model_spec['gui_config'] = gui_config
            try:
                MODEL.write(filename, model_spec)
                gui_config['filename'] = filename
            except IOError:
                print("Error writing {}".format(filename))
                # Raise a dialog box here.


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename",
                        default='test_gui.json',
                        help="Model file")
    parser.add_argument("--days",
                        type=float,
                        default=15,  # Fix this
                        help="Number of days in fit interval (default=90")
    parser.add_argument("--stop",
                        default=DateTime() - 10,  # remove this
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
    parser.add_argument("--set-data",
                        action='append',
                        dest='set_data_exprs',
                        default=[],
                        help="Set data value as '<comp_name>=<value>'")
    parser.add_argument("--quiet",
                        default=False,
                        action='store_true',
                        help="Suppress screen output")

    return parser.parse_args()


def main():
    # Enable fully-randomized evaluation of ACIS-FP model which is desirable
    # for fitting.

    taco.set_random_salt(None)
    global MODEL, POOL, PARENT_PIPE, CHILD_PIPE, FIT_WORKER, CALC_STAT

    opt = get_options()

    src = pyc.CONTEXT['src'] if 'src' in pyc.CONTEXT else pyc.ContextDict('src')
    files = (pyc.CONTEXT['file'] if 'file' in pyc.CONTEXT else
             pyc.ContextDict('files', basedir=os.getcwd()))
    files.update(xija.files)

    sherpa_logger = logging.getLogger("sherpa")
    loggers = (fit_logger, sherpa_logger)
    if opt.quiet:
        for logger in loggers:
            for h in logger.handlers:
                logger.removeHandler(h)

    model_spec = json.load(open(opt.filename, 'r'))
    gui_config.update(model_spec.get('gui_config', {}))
    src['model'] = model_spec['name']

    # Use supplied stop time and days OR use model_spec values if stop not supplied
    if opt.stop:
        start = DateTime(DateTime(opt.stop).secs - opt.days * 86400).date[:8]
        stop = opt.stop
    else:
        start = model_spec['datestart']
        stop = model_spec['datestop']

    MODEL = xija.ThermalModel(model_spec['name'], start, stop, model_spec=model_spec)

    set_data_vals = gui_config.get('set_data_vals', {})
    for set_data_expr in opt.set_data_exprs:
        set_data_expr = re.sub('\s', '', set_data_expr)
        try:
            comp_name, val = set_data_expr.split('=')
        except ValueError:
            raise ValueError("--set_data must be in form '<comp_name>=<value>'")
        # Set data to value.  ast.literal_eval is a safe way to convert any
        # string literal into the corresponding Python object.
        set_data_vals[comp_name] = ast.literal_eval(val)

    for comp_name, val in set_data_vals.items():
        MODEL.comp[comp_name].set_data(val)

    MODEL.make()

    if opt.inherit_from:
        inherit_spec = json.load(open(opt.inherit_from, 'r'))
        inherit_pars = {par['full_name']: par for par in inherit_spec['pars']}
        for par in MODEL.pars:
            if par.full_name in inherit_pars:
                print("Inheriting par {}".format(par.full_name))
                par.val = inherit_pars[par.full_name]['val']
                par.min = inherit_pars[par.full_name]['min']
                par.max = inherit_pars[par.full_name]['max']
                par.frozen = inherit_pars[par.full_name]['frozen']
                par.fmt = inherit_pars[par.full_name]['fmt']

    dummy_data = np.zeros(1)
    dummy_times = np.arange(1)
    ui.load_arrays(1, dummy_times, dummy_data)

    ui.set_method(opt.fit_method)
    ui.get_method().config.update(sherpa_configs.get(opt.fit_method, {}))

    ui.load_user_model(sherpa_xija_model, 'xijamod')  # sets global xijamod
    ui.add_user_pars('xijamod', MODEL.parnames)
    ui.set_model(1, xijamod)
    print('model stat is {}'.format(MODEL.calc_stat()))

    # ui.load_user_stat("new_stat", custom_stat_func, custom_staterr_func)
    # ui.set_stat(new_stat)
    CALC_STAT = CalcStat()
    ui.load_user_stat('xijastat', CALC_STAT, lambda x: np.ones_like(x))
    ui.set_stat(xijastat)

    gui_config['filename'] = os.path.abspath(opt.filename)
    gui_config['set_data_vals'] = set_data_vals

    FIT_WORKER = FitWorker()

    PARENT_PIPE, CHILD_PIPE = multiprocessing.Pipe()
    POOL = multiprocessing.Pool(processes=1)

    app = QtWidgets.QApplication(sys.argv)
    MainWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
