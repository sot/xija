#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import print_function

import sys
import os
import ast
import multiprocessing
import time
import functools
import platform

from PyQt5 import QtCore, QtWidgets

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
from Ska.Matplotlib import cxctime2plotdate

logging.basicConfig(level=logging.INFO)

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


class CalcModel(object):
    def __init__(self, model):
        self.model = model

    def __call__(self, parvals, x):
        """This is the Sherpa calc_model function, but in this case calc_model does not
        actually calculate anything but instead just stores the desired paramters.  This
        allows for multiprocessing where only the fit statistic gets passed between nodes.
        """
        logging.info('Calculating params:')
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
        try:
            raise KeyError
            # fit_stat = self.cache_fit_stat[parvals_key]
            # fit_logger.info('nmass_model: Cache hit %s' % str(parvals_key))
        except KeyError:
            fit_stat = self.model.calc_stat()

        fit_logger.info('Fit statistic: %.4f' % fit_stat)
        # self.cache_fit_stat[parvals_key] = fit_stat

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
        logging.info('Fit started')

    def terminate(self, widget=None):
        """Terminate a Sherpa fit process in a controlled way by sending a
        message.  Get the final parameter values if possible.
        """
        if hasattr(self, "fit_process"):
            # Only do this if we had started a fit to begin with
            self.parent_pipe.send('terminate')
            self.fit_process.join()
            logging.info('Fit terminated')

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
                logging.info('Fit finished normally')
            except FitTerminated as err:
                calc_stat.message['status'] = 'terminated'
                logging.warning('Got FitTerminated exception {}'.format(err))

        self.child_pipe.send(calc_stat.message)


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


class Panel(object):
    def __init__(self, orient='h'):
        Box = QtWidgets.QHBoxLayout if orient == 'h' else QtWidgets.QVBoxLayout
        self.box = Box()
        self.orient = orient

    def add_stretch(self, value):
        self.box.addStretch(value)

    def pack_start(self, child):
        if isinstance(child, Panel):
            child = child.box
        if isinstance(child, QtWidgets.QBoxLayout):
            out = self.box.addLayout(child)
        else:
            out = self.box.addWidget(child)
        return out

    def pack_end(self, child):
        return self.pack_start(child)


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
    def __init__(self, parent=None):
        self.fig = Figure()

        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)
        self.updateGeometry()


class PlotBox(QtWidgets.QVBoxLayout):
    def __init__(self, plot_name, plots_box):
        super(PlotBox, self).__init__()

        comp_name, plot_method = plot_name.split()  # E.g. "tephin fit_resid"
        self.comp = plots_box.model.comp[comp_name]
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
        self.plots_box = plots_box

    def update(self, redraw=False, first=False):
        plot_func = getattr(self.comp, 'plot_' + self.plot_method)
        if redraw:
            self.fig.delaxes(self.ax)
            try:
                xaxis_type = self.plot_method.split('__')[1]
            except IndexError:
                self.ax = self.fig.add_subplot(111)
            else:
                sharex = self.plots_box.sharex.get(xaxis_type)
                self.ax = self.fig.add_subplot(111, sharex=sharex)
                if sharex is not None:
                    self.ax.autoscale(enable=False, axis='x')
                self.plots_box.sharex.setdefault(xaxis_type, self.ax)

        plot_func(fig=self.fig, ax=self.ax)
        if redraw or first:
            times = self.plots_box.model.times
            if self.plot_method.endswith("time"):
                ybot, ytop = self.ax.get_ylim()
                tplot = cxctime2plotdate(times)
                for t0, t1 in self.plots_box.model.mask_time_secs:
                    where = (times >= t0) & (times <= t1)
                    self.ax.fill_between(tplot, ybot, ytop, where=where,
                                         color='r', alpha=0.5)
        self.canvas.draw()


class PlotsBox(QtWidgets.QVBoxLayout):
    def __init__(self, model, main_window):
        super(QtWidgets.QVBoxLayout, self).__init__()
        self.main_window = main_window
        self.sharex = {}        # Shared x-axes keyed by x-axis type
        self.model = model

    def add_plot_box(self, plot_name):
        plot_name = str(plot_name)
        if plot_name == "Add plot..." or plot_name in self.plot_names:
            return
        print('Adding plot ', plot_name)
        plot_box = PlotBox(plot_name, self)
        self.addLayout(plot_box)
        plot_box.update(first=True)
        self.main_window.cbp.add_plot_button.setCurrentIndex(0)


    def delete_plot_box(self, plot_name):
        for plot_box in self.findChildren(PlotBox):
            if plot_box.plot_name == plot_name:
                self.removeItem(plot_box)
                clearLayout(plot_box)
        self.update()

    def update_plots(self, redraw=False):
        cbp = self.main_window.cbp
        cbp.update_status.setText(' BUSY... ')
        self.model.calc()
        for plot_box in self.findChildren(PlotBox):
            plot_box.update(redraw=redraw)
        cbp.update_status.setText('')

    @property
    def plot_boxes(self):
        return [plot_box for plot_box in self.findChildren(PlotBox)]

    @property
    def plot_names(self):
        return [plot_box.plot_name for plot_box in self.plot_boxes]


class PanelCheckBox(QtWidgets.QCheckBox):
    def __init__(self, par):
        super(PanelCheckBox, self).__init__() 
        self.par = par

    def frozen_toggled(self, state):
        self.par.frozen = state != QtCore.Qt.Checked


class PanelText(QtWidgets.QLineEdit):
    def __init__(self, params_panel, row, par, attr, slider):
        super(PanelText, self).__init__()
        self.par = par
        self.row = row
        self.attr = attr
        self.slider = slider
        self.params_panel = params_panel

    def _bounds_check(self):
        if self.par.val < self.par.min:
            print("Attempted to set parameter value below minimum. Setting to min value.")
            self.par.val = self.par.min
            self.params_panel.params_table[self.row, 2].setText(self.par.fmt.format(self.par.val))
        if self.par.val > self.par.max:
            print("Attempted to set parameter value below maximum. Setting to max value.")
            self.par.val = self.par.max
            self.setText(self.par.fmt.format(self.par.val))
            self.params_panel.params_table[self.row, 2].setText(self.par.fmt.format(self.par.val))

    def par_attr_changed(self):
        try:
            val = float(self.text().strip())
        except ValueError:
            pass
        else:
            setattr(self.par, self.attr, val)
            self._bounds_check()
            self.slider.update_slider_val(val, self.attr)
            self.params_panel.plots_panel.update_plots(redraw=True)


class PanelSlider(QtWidgets.QSlider):
    def __init__(self, params_panel, par, row):
        super(PanelSlider, self).__init__(QtCore.Qt.Horizontal)
        self.par = par
        self.row = row
        self.params_panel = params_panel
        self.parmin = par.min
        self.parmax = par.max
        self.parval = par.val
        self.setMinimum(0)
        self.setMaximum(100)
        self.setTickInterval(101)
        self.setSingleStep(1)
        self.setPageStep(10)
        self.set_dx()
        self.set_step_from_value(par.val)
        self.update_plots = True

    def set_dx(self):
        self.dx = 100.0/(self.parmax-self.parmin)
        self.idx = 1.0/self.dx

    def set_step_from_value(self, val):
        step = int((val-self.parmin)*self.dx)
        self.setValue(step)

    def get_value_from_step(self):
        val = self.value()*self.idx + self.parmin
        return val

    def update_slider_val(self, val, attr):
        setattr(self, "par{}".format(attr), val)
        if attr in ["min", "max"]:
            self.set_dx()
        self.set_step_from_value(self.parval)

    def block_plotting(self, block):
        self.update_plots = not block

    def slider_moved(self):
        val = self.get_value_from_step()
        setattr(self.par, "val", val)
        self.params_panel.params_table[self.row, 2].setText(self.par.fmt.format(val))
        if self.update_plots:
            self.params_panel.plots_panel.update_plots()


class ParamsPanel(Panel):
    def __init__(self, model, plots_panel):
        Panel.__init__(self, orient='v')
        self.plots_panel = plots_panel
        self.model = model

        params_table = WidgetTable(n_rows=len(self.model.pars),
                                   colnames=['fit', 'name', 'val', 'min', '', 'max'],
                                   colwidths={0: 30, 1: 250},
                                   show_header=True)

        for row, par in zip(count(), self.model.pars):
            # Thawed (i.e. fit the parameter)
            frozen = params_table[row, 0] = PanelCheckBox(par)
            frozen.setChecked(not par.frozen)
            frozen.stateChanged.connect(frozen.frozen_toggled)

            # par full name
            params_table[row, 1] = QtWidgets.QLabel(par.full_name)

            # Slider
            slider = PanelSlider(self, par, row)
            params_table[row, 4] = slider
            slider.sliderMoved.connect(slider.slider_moved)

            # Value
            entry = params_table[row, 2] = PanelText(self, row, par, 'val', slider)
            entry.setText(par.fmt.format(par.val))
            entry.returnPressed.connect(entry.par_attr_changed)

            # Min of slider
            entry = params_table[row, 3] = PanelText(self, row, par, 'min', slider)
            entry.setText(par.fmt.format(par.min))
            entry.returnPressed.connect(entry.par_attr_changed)

            # Max of slider
            entry = params_table[row, 5] = PanelText(self, row, par, 'max', slider)
            entry.setText(par.fmt.format(par.max))
            entry.returnPressed.connect(entry.par_attr_changed)

        self.pack_start(params_table.table)
        self.params_table = params_table

    def update(self):
        for row, par in enumerate(self.model.pars):
            val_label = self.params_table[row, 2]
            par_val_text = par.fmt.format(par.val)
            if str(val_label.text) != par_val_text:
                val_label.setText(par_val_text)
                # Change the slider value but block the signal to update the plot
                slider = self.params_table[row, 4]
                slider.block_plotting(True)
                slider.set_step_from_value(par.val)
                slider.block_plotting(False)


class ControlButtonsPanel(Panel):
    def __init__(self, model):
        Panel.__init__(self, orient='h')

        self.model = model

        self.fit_button = QtWidgets.QPushButton("Fit")
        self.stop_button = QtWidgets.QPushButton("Stop")
        if platform.system() == "Darwin":
            self.stop_button.setEnabled(False)
        self.save_button = QtWidgets.QPushButton("Save")
        self.add_plot_button = self.make_add_plot_button()
        self.update_status = QtWidgets.QLabel()
        self.quit_button = QtWidgets.QPushButton('Quit')
        self.command_entry = QtWidgets.QLineEdit()
        self.command_panel = Panel()
        self.command_panel.pack_start(QtWidgets.QLabel('Command:'))
        self.command_panel.pack_start(self.command_entry)

        self.pack_start(self.fit_button)
        self.pack_start(self.stop_button)
        self.pack_start(self.save_button)
        self.pack_start(self.add_plot_button)
        self.pack_start(self.update_status)
        self.pack_start(self.command_panel)
        self.add_stretch(1)
        self.pack_start(self.quit_button)

    def make_add_plot_button(self):
        apb = QtWidgets.QComboBox()
        apb.addItem('Add plot...')

        plot_names = ['{} {}'.format(comp.name, attr[5:])
                      for comp in self.model.comps
                      for attr in dir(comp)
                      if attr.startswith('plot_')]

        self.plot_names = plot_names
        for plot_name in plot_names:
            apb.addItem(plot_name)

        return apb


class MainLeftPanel(Panel):
    def __init__(self, model, main_window):
        Panel.__init__(self, orient='v')
        self.control_buttons_panel = ControlButtonsPanel(model)
        self.plots_box = PlotsBox(model, main_window)
        self.pack_start(self.control_buttons_panel)
        self.pack_start(self.plots_box)
        self.add_stretch(1)
        self.model = model


class MainRightPanel(Panel):
    def __init__(self, model, plots_panel):
        Panel.__init__(self, orient='v')
        self.params_panel = ParamsPanel(model, plots_panel)
        self.pack_start(self.params_panel)


class MainWindow(object):
    # This is a callback function. The data arguments are ignored
    # in this example. More on callbacks below.
    def __init__(self, model, fit_worker):
        self.model = model
        self.fit_worker = fit_worker
        # create a new window
        self.window = QtWidgets.QWidget()
        self.window.setGeometry(0, 0, *gui_config.get('size', (1400, 800)))

        self.main_box = Panel(orient='h')

        # This is the Layout Box that holds the top-level stuff in the main window
        main_window_hbox = QtWidgets.QHBoxLayout()
        self.window.setLayout(main_window_hbox)

        self.main_left_panel = MainLeftPanel(model, self)
        mlp = self.main_left_panel

        self.main_right_panel = MainRightPanel(model, mlp.plots_box)

        self.cbp = mlp.control_buttons_panel
        self.cbp.fit_button.clicked.connect(self.fit_worker.start)
        self.cbp.fit_button.clicked.connect(self.fit_monitor)
        self.cbp.stop_button.clicked.connect(self.fit_worker.terminate)
        self.cbp.save_button.clicked.connect(self.save_model_file)
        self.cbp.quit_button.clicked.connect(QtCore.QCoreApplication.instance().quit)
        self.cbp.add_plot_button.activated[str].connect(self.add_plot)
        self.cbp.command_entry.returnPressed.connect(self.command_activated)

        # Add plots from previous Save
        for plot_name in gui_config.get('plot_names', []):
            try:
                self.add_plot(plot_name)
                time.sleep(0.05)  # is it needed?
            except ValueError:
                print("ERROR: Unexpected plot_name {}".format(plot_name))

        # Show everything finally
        main_window_hbox.addLayout(mlp.box)
        main_window_hbox.addLayout(self.main_right_panel.box)

        self.window.show()

    def add_plot(self, plotname):
        pp = self.main_left_panel.plots_box
        pp.add_plot_box(plotname)

    def fit_monitor(self, *args):
        msg = None
        fit_stopped = False
        while self.fit_worker.parent_pipe.poll():
            # Keep reading messages until there are no more or until getting
            # a message indicating fit is stopped.
            msg = self.fit_worker.parent_pipe.recv()
            fit_stopped = msg['status'] in ('terminated', 'finished')
            if fit_stopped:
                self.fit_worker.fit_process.join()
                print("\n*********************************")
                print("  FIT", msg['status'].upper())
                print("*********************************\n")
                break

        if msg:
            # Update the fit_worker model parameters and then the corresponding
            # params table widget.
            self.fit_worker.model.parvals = msg['parvals']
            self.main_right_panel.params_panel.update()
            self.main_left_panel.plots_box.update_plots(redraw=fit_stopped)

        # If fit has not stopped then set another timeout 200 msec from now
        if not fit_stopped:
            QtCore.QTimer.singleShot(200, self.fit_monitor)

    def command_activated(self):
        """Respond to a command like "freeze solarheat*dP*" submitted via the
        command entry box.  The first word is either "freeze" or "thaw" (with
        possibility for other commands later) and the subsequent args are
        space-delimited parameter globs using the UNIX file-globbing syntax.
        This then sets the corresponding params_table checkbuttons.
        """
        widget = self.cbp.command_entry
        command = widget.text().strip()
        if command == '':
            return
        vals = command.split()
        cmd = vals[0]  # currently freeze, thaw, ignore, or notice
        if cmd not in ('freeze', 'thaw', 'ignore', 'notice') or len(vals) <= 1:
            # dialog box..
            print("ERROR: bad command: {}".format(command))
            return

        if cmd in ('freeze', 'thaw'):
            par_regexes = [fnmatch.translate(x) for x in vals[1:]]
            params_table = self.main_right_panel.params_panel.params_table
            for row, par in enumerate(self.model.pars):
                for par_regex in par_regexes:
                    if re.match(par_regex, par.full_name):
                         checkbutton = params_table[row, 0]
                         checkbutton.setChecked(cmd == 'thaw')
                         par.frozen = cmd != 'thaw'
        elif cmd in ('ignore', 'notice'):
            if cmd == "ignore":
                lim = vals[1].split("-")
                if lim[0] == "*":
                    lim[0] = self.model.datestart
                if lim[1] == "*":
                    lim[1] = self.model.datestop
                lim = DateTime(lim).date
                self.model.append_mask_times(lim)
            elif cmd == "notice":
                if vals[1] in ["**-**", "*", "all"]:
                    self.model.reset_mask_times()
                else:
                    print("ERROR: Invalid input for 'notice'!")
            self.main_left_panel.plots_box.update_plots(redraw=True)

        widget.setText('')

    def save_model_file(self, *args):
        dlg = QtWidgets.QFileDialog()
        dlg.setNameFilters(["JSON files (*.json)", "Python files (*.py)", "All files (*)"])
        dlg.selectNameFilter("JSON files (*.json)")
        dlg.selectFile(os.path.abspath(gui_config["filename"]))
        dlg.setAcceptMode(dlg.AcceptSave)
        dlg.exec_()
        filename = str(dlg.selectedFiles()[0])
        if filename != '':
            model_spec = self.model.model_spec
            gui_config['plot_names'] = [x.plot_name
                                        for x in self.main_left_panel.plots_box.plot_boxes]
            gui_config['size'] = (self.window.size().width(), self.window.size().height())
            model_spec['gui_config'] = gui_config
            try:
                self.model.write(filename, model_spec)
                gui_config['filename'] = filename
            except IOError as ioerr:
                msg = QtWidgets.QMessageBox()
                msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                msg.setText("There was a problem writing the file:")
                msg.setDetailedText("Cannot write {}. {}".format(filename, ioerr.strerror))
                msg.exec_()


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename",
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

    model = xija.ThermalModel(model_spec['name'], start, stop, model_spec=model_spec)

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
        model.comp[comp_name].set_data(val)

    model.make()

    if opt.inherit_from:
        inherit_spec = json.load(open(opt.inherit_from, 'r'))
        inherit_pars = {par['full_name']: par for par in inherit_spec['pars']}
        for par in model.pars:
            if par.full_name in inherit_pars:
                print("Inheriting par {}".format(par.full_name))
                par.val = inherit_pars[par.full_name]['val']
                par.min = inherit_pars[par.full_name]['min']
                par.max = inherit_pars[par.full_name]['max']
                par.frozen = inherit_pars[par.full_name]['frozen']
                par.fmt = inherit_pars[par.full_name]['fmt']

    gui_config['filename'] = os.path.abspath(opt.filename)
    gui_config['set_data_vals'] = set_data_vals

    fit_worker = FitWorker(model)

    model.calc()

    app = QtWidgets.QApplication(sys.argv)
    MainWindow(model, fit_worker)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
