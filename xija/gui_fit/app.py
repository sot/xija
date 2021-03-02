import sys
import os
import ast
import time

from PyQt5 import QtCore, QtWidgets, QtGui

from itertools import count
import argparse
import fnmatch

import re
import json
import logging
import numpy as np

from cxotime import CxoTime

import pyyaks.context as pyc

import acis_taco as taco
import xija

from xija.component.base import Node, TelemData

from .fitter import FitWorker, fit_logger
from .plots import PlotsBox, HistogramWindow
from .utils import in_process_console

from collections import OrderedDict

gui_config = {}


class FormattedTelemData:
    def __init__(self, telem_data):
        self.telem_data = telem_data
        self.data_names = []
        self.data_basenames = []
        self.formats = []
        for name, data in self.telem_data.items():
            if data.dvals.dtype == np.float64:
                fmt = "{0:.4f}"
            else:
                fmt = "{0}"
            if hasattr(data, 'resids'):
                self.data_names += [name, name+"_model", name+"_resid"]
                self.data_basenames += [name]*3
                self.formats += [fmt]*3
            else:
                self.data_names.append(name)
                self.data_basenames.append(name)
                self.formats.append(fmt)
        self.times = self.telem_data[self.data_basenames[0]].times

    def __iter__(self):
        return iter(self.data_names)

    _dates = None

    @property
    def dates(self):
        if self._dates is None:
            self._dates = CxoTime(self.times).date
        return self._dates

    def __getitem__(self, item):
        name = self.data_names[item]
        basenm = self.data_basenames[item]
        if name.endswith("_model"):
            val = self.telem_data[basenm].mvals
        elif name.endswith("_resid"):
            val = self.telem_data[basenm].resids
        else:
            val = self.telem_data[basenm].dvals
        return val


class WriteTableWindow(QtWidgets.QMainWindow):
    def __init__(self, model, main_window):
        super(WriteTableWindow, self).__init__()
        self.model = model
        self.mw = main_window
        self.setWindowTitle("Write Table")
        wid = QtWidgets.QWidget(self)
        self.box = QtWidgets.QVBoxLayout()
        wid.setLayout(self.box)
        self.setGeometry(0, 0, 200, 600)
        self.scroll = QtWidgets.QScrollArea()
        self.setCentralWidget(wid)

        self.last_filename = ""

        self.ftd = self.mw.fmt_telem_data
        self.write_list = self.ftd.data_names

        self.start_date = self.ftd.dates[0]
        self.stop_date = self.ftd.dates[-1]

        main_box = QtWidgets.QVBoxLayout()

        self.start_label = QtWidgets.QLabel("Start time: {}".format(self.start_date))
        self.stop_label = QtWidgets.QLabel("Stop time: {}".format(self.stop_date))
        self.list_label = QtWidgets.QLabel("Data to write:")

        self.start_text = QtWidgets.QLineEdit()
        self.start_text.returnPressed.connect(self.change_start)

        self.stop_text = QtWidgets.QLineEdit()
        self.stop_text.returnPressed.connect(self.change_stop)

        main_box.addWidget(self.start_label)
        main_box.addWidget(self.start_text)
        main_box.addWidget(self.stop_label)
        main_box.addWidget(self.stop_text)

        item_box = QtWidgets.QVBoxLayout()

        pair = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel("Write All Data")
        all_chkbox = QtWidgets.QCheckBox()
        all_chkbox.setChecked(True)
        pair.addWidget(all_chkbox)
        pair.addWidget(label)
        pair.addStretch(1)
        item_box.addLayout(pair)

        all_chkbox.stateChanged.connect(self.toggle_all_data)

        self.check_boxes = []
        for name in self.ftd.data_names:
            pair = QtWidgets.QHBoxLayout()
            label = QtWidgets.QLabel(name)
            chkbox = QtWidgets.QCheckBox()
            chkbox.setChecked(True)
            pair.addWidget(chkbox)
            pair.addWidget(label)
            pair.addStretch(1)
            item_box.addLayout(pair)
            self.check_boxes.append(chkbox)

        item_wid = QtWidgets.QWidget(self)
        item_wid.setLayout(item_box)

        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(item_wid)
        main_box.addWidget(self.scroll)
        buttons = QtWidgets.QHBoxLayout()

        write_button = QtWidgets.QPushButton('Write Table')
        write_button.clicked.connect(self.save_ascii_table)

        close_button = QtWidgets.QPushButton('Close')
        close_button.clicked.connect(self.close_window)

        buttons.addWidget(write_button)
        buttons.addWidget(close_button)

        main_box.addLayout(buttons)
        self.box.addLayout(main_box)

    def toggle_all_data(self, state):
        checked = state == QtCore.Qt.Checked
        for i, box in enumerate(self.check_boxes):
            box.setChecked(checked)

    def change_start(self):
        self.start_date = self.start_text.text()
        self.start_label.setText("Start time: {}".format(self.start_date))
        self.start_text.setText("")

    def change_stop(self):
        self.stop_date = self.stop_text.text()
        self.stop_label.setText("Stop time: {}".format(self.stop_date))
        self.stop_text.setText("")

    def close_window(self, *args):
        self.close()

    def save_ascii_table(self):
        from astropy.table import Table, Column
        dlg = QtWidgets.QFileDialog()
        dlg.setNameFilters(["DAT files (*.dat)", "TXT files (*.txt)", "All files (*)"])
        dlg.selectNameFilter("DAT files (*.dat)")
        dlg.setAcceptMode(dlg.AcceptSave)
        dlg.exec_()
        filename = str(dlg.selectedFiles()[0])
        if filename != '':
            try:
                checked = []
                for i, box in enumerate(self.check_boxes):
                    if box.isChecked():
                        checked.append(i)
                t = Table()
                ts = CxoTime([self.start_date, self.stop_date]).secs
                ts[-1] += 1.0 # a buffer to make sure we grab the last point
                istart, istop = np.searchsorted(self.ftd.times, ts)
                c = Column(self.ftd.dates[istart:istop], name="date", format="{0}")
                t.add_column(c)
                for i, key in enumerate(self.ftd):
                    if i in checked:
                        c = Column(self.ftd[i][istart:istop], name=key, format=self.ftd.formats[i])
                        t.add_column(c)
                t.write(filename, overwrite=True, format='ascii.ecsv')
                self.last_filename = filename
            except IOError as ioerr:
                msg = QtWidgets.QMessageBox()
                msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                msg.setText("There was a problem writing the file:")
                msg.setDetailedText("Cannot write {}. {}".format(filename, ioerr.strerror))
                msg.exec_()


class ModelInfoWindow(QtWidgets.QMainWindow):
    def __init__(self, model, main_window):
        super(ModelInfoWindow, self).__init__()
        self.model = model
        self.setWindowTitle("Model Info")
        wid = QtWidgets.QWidget(self)
        self.setCentralWidget(wid)
        self.box = QtWidgets.QVBoxLayout()
        wid.setLayout(self.box)
        self.setGeometry(0, 0, 300, 200)

        self.main_window = main_window
        self.checksum_label = QtWidgets.QLabel()
        self.checksum_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.update_checksum()
        self.filename_label = QtWidgets.QLabel()
        self.update_filename()

        checksum_layout = QtWidgets.QHBoxLayout()
        checksum_layout.addWidget(QtWidgets.QLabel("MD5 sum: "))
        checksum_layout.addWidget(self.checksum_label)
        checksum_layout.addStretch(1)

        main_box = QtWidgets.QVBoxLayout()
        main_box.addWidget(self.filename_label)
        main_box.addLayout(checksum_layout)
        main_box.addWidget(QtWidgets.QLabel("Start time: {}".format(model.datestart)))
        main_box.addWidget(QtWidgets.QLabel("Stop time: {}".format(model.datestop)))
        main_box.addWidget(QtWidgets.QLabel("Timestep: {:.1f} s".format(model.dt)))
        main_box.addWidget(QtWidgets.QLabel("Evolve Method: Core {}".format(model.evolve_method)))
        main_box.addWidget(QtWidgets.QLabel("Runge-Kutta Order: {}".format(4 if model.rk4 else 2)))
        main_box.addStretch(1)

        close_button = QtWidgets.QPushButton('Close')
        close_button.clicked.connect(self.close_window)

        close_box = QtWidgets.QHBoxLayout()
        close_box.addStretch(1)
        close_box.addWidget(close_button)

        main_box.addLayout(close_box)
        self.box.addLayout(main_box)

    def update_checksum(self):
        self.main_window.set_checksum()
        if self.main_window.checksum_match:
            color = 'black'
        else:
            color = 'red'
        checksum_str = self.main_window.md5sum
        self.checksum_label.setText(checksum_str)
        self.checksum_label.setStyleSheet('color: {}'.format(color))

    def update_filename(self):
        self.filename_label.setText("Filename: {}".format(gui_config["filename"]))

    def close_window(self, *args):
        self.close()
        self.main_window.model_info_window = None


class LineDataWindow(QtWidgets.QMainWindow):
    def __init__(self, model, main_window, plots_box):
        super(LineDataWindow, self).__init__()
        self.model = model
        self.setWindowTitle("Line Data")
        wid = QtWidgets.QWidget(self)
        self.setCentralWidget(wid)
        self.box = QtWidgets.QVBoxLayout()
        wid.setLayout(self.box)
        self.setGeometry(0, 0, 350, 600)

        self.plots_box = plots_box
        self.main_window = main_window
        self.ftd = self.main_window.fmt_telem_data
        self.nrows = len(self.ftd.data_names)+1

        self.table = WidgetTable(n_rows=self.nrows,
                                 colnames=['name', 'value'],
                                 colwidths={1: 200},
                                 show_header=True)

        self.table[0, 0] = QtWidgets.QLabel("date")
        self.table[0, 1] = QtWidgets.QLabel("")

        for row in range(1, self.nrows):
            name = self.ftd.data_names[row-1]
            self.table[row, 0] = QtWidgets.QLabel(name)
            self.table[row, 1] = QtWidgets.QLabel("")

        self.update_data()

        self.box.addWidget(self.table.table)

    def update_data(self):
        pos = np.searchsorted(self.plots_box.pd_times, 
                              self.plots_box.xline)
        date = self.main_window.dates[pos]
        self.table[0, 1].setText(date)
        for row in range(1, self.nrows):
            val = self.ftd[row-1]
            fmt = self.ftd.formats[row-1]
            self.table[row, 1].setText(fmt.format(val[pos]))


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


class PanelCheckBox(QtWidgets.QCheckBox):
    def __init__(self, par, main_window):
        super(PanelCheckBox, self).__init__() 
        self.par = par
        self.main_window = main_window

    def frozen_toggled(self, state):
        self.par.frozen = state != QtCore.Qt.Checked
        self.main_window.set_checksum()
        self.main_window.set_title()
        if self.main_window.model_info_window is not None:
            self.main_window.model_info_window.update_checksum()


class PanelText(QtWidgets.QLineEdit):
    def __init__(self, params_panel, row, par, attr, slider):
        super(PanelText, self).__init__()
        self.par = par
        self.row = row
        self.attr = attr
        self.slider = slider
        self.params_panel = params_panel

    def _bounds_check(self):
        msg = None
        if self.par.val < self.par.min:
            msg = "Attempted to set parameter value below minimum. Setting to min value."
            self.par.val = self.par.min
            self.params_panel.params_table[self.row, 2].setText(self.par.fmt.format(self.par.val))
        if self.par.val > self.par.max:
            msg = "Attempted to set parameter value below maximum. Setting to max value."
            self.par.val = self.par.max
            self.setText(self.par.fmt.format(self.par.val))
            self.params_panel.params_table[self.row, 2].setText(self.par.fmt.format(self.par.val))
        return msg

    def par_attr_changed(self):
        try:
            val = float(self.text().strip())
        except ValueError:
            pass
        else:
            self.set_par_attr(val)

    def set_par_attr(self, val):
        setattr(self.par, self.attr, val)
        msg = self._bounds_check()
        if msg is not None:
            print(msg)
        self.slider.update_slider_val(val, self.attr)
        self.params_panel.plots_panel.update_plots()

    def __repr__(self):
        return getattr(self.par, self.attr).__repr__()


class PanelParam(object):
    def __init__(self, val, min, max):
        self._val = val
        self._min = min
        self._max = max

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, val):
        self._val.set_par_attr(val)

    @property
    def min(self):
        return self._min

    @min.setter
    def min(self, val):
        self._min.set_par_attr(val)

    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, val):
        self._max.set_par_attr(val)

    def __repr__(self):
        return "val: {} min: {} max: {}".format(self._val, self._min, self._max)


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

        self.params_dict = OrderedDict()

        for row, par in zip(count(), self.model.pars):

            # Thawed (i.e. fit the parameter)
            frozen = params_table[row, 0] = PanelCheckBox(par, self.plots_panel.main_window)
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

            self.params_dict[par.full_name] = PanelParam(params_table[row, 2],
                                                         params_table[row, 3],
                                                         params_table[row, 5])

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
        Panel.__init__(self, orient='v')

        self.model = model

        self.fit_button = QtWidgets.QPushButton("Fit")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.save_button = QtWidgets.QPushButton("Save")
        self.hist_button = QtWidgets.QPushButton("Histogram")
        self.write_table_button = QtWidgets.QPushButton("Write Table")
        self.model_info_button = QtWidgets.QPushButton("Model Info")
        self.add_plot_button = self.make_add_plot_button()
        self.update_status = QtWidgets.QLabel()
        self.quit_button = QtWidgets.QPushButton('Quit')
        self.console_button = QtWidgets.QPushButton('Console')
        self.command_entry = QtWidgets.QLineEdit()
        self.command_panel = Panel()
        self.command_panel.pack_start(QtWidgets.QLabel('Command:'))
        self.command_panel.pack_start(self.command_entry)

        self.radzone_chkbox = QtWidgets.QCheckBox()
        self.limits_chkbox = QtWidgets.QCheckBox()
        self.line_chkbox = QtWidgets.QCheckBox()
        if len(self.model.limits) == 0:
            self.limits_chkbox.setEnabled(False)

        self.top_panel = Panel()
        self.bottom_panel = Panel()

        self.top_panel.pack_start(self.fit_button)
        self.top_panel.pack_start(self.stop_button)
        self.top_panel.pack_start(self.save_button)
        self.top_panel.pack_start(self.add_plot_button)
        self.top_panel.pack_start(self.update_status)
        self.top_panel.pack_start(self.command_panel)
        self.top_panel.pack_start(self.hist_button)
        self.top_panel.add_stretch(1)
        self.top_panel.pack_start(self.quit_button)

        self.radzone_panel = Panel()
        self.radzone_panel.pack_start(QtWidgets.QLabel('Show radzones'))
        self.radzone_panel.pack_start(self.radzone_chkbox)

        self.limits_panel = Panel()
        self.limits_panel.pack_start(QtWidgets.QLabel('Show limits'))
        self.limits_panel.pack_start(self.limits_chkbox)

        self.line_panel = Panel()
        self.line_panel.pack_start(QtWidgets.QLabel('Annotate line'))
        self.line_panel.pack_start(self.line_chkbox)

        self.bottom_panel.pack_start(self.radzone_panel)
        self.bottom_panel.pack_start(self.limits_panel)
        self.bottom_panel.pack_start(self.line_panel)
        self.bottom_panel.add_stretch(1)
        self.bottom_panel.pack_start(self.model_info_button)
        self.bottom_panel.pack_start(self.write_table_button)
        self.bottom_panel.pack_start(self.console_button)

        self.pack_start(self.top_panel)
        self.pack_start(self.bottom_panel)

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
    def __init__(self, model, fit_worker, model_file):
        import Ska.tdb
        self.model = model

        self.hist_msids = []
        for k, v in self.model.comp.items():
            if isinstance(v, Node):
                if k.startswith("fptemp") or \
                   k.startswith("tmp_fep") or \
                   k.startswith("tmp_bep"):
                    self.hist_msids.append(k)
                try:
                    Ska.tdb.msids[v.msid]
                except KeyError:
                    pass
                else:
                    self.hist_msids.append(k)

        self.checksum_match = True

        self.fit_worker = fit_worker
        # create a new window
        self.window = QtWidgets.QWidget()
        self.window.setGeometry(0, 0, *gui_config.get('size', (1400, 800)))
        self.set_title()
        self.main_box = Panel(orient='h')

        # This is the Layout Box that holds the top-level stuff in the main window
        main_window_hbox = QtWidgets.QHBoxLayout()
        self.window.setLayout(main_window_hbox)

        self.main_left_panel = MainLeftPanel(model, self)
        mlp = self.main_left_panel
        self.plots_box = self.main_left_panel.plots_box

        self.main_right_panel = MainRightPanel(model, mlp.plots_box)

        self.show_radzones = False
        self.show_limits = False
        self.show_line = False

        self.cbp = mlp.control_buttons_panel
        self.cbp.fit_button.clicked.connect(self.fit_worker.start)
        self.cbp.fit_button.clicked.connect(self.fit_monitor)
        self.cbp.stop_button.clicked.connect(self.fit_worker.terminate)
        self.cbp.save_button.clicked.connect(self.save_model_file)
        self.cbp.write_table_button.clicked.connect(self.write_table)
        self.cbp.model_info_button.clicked.connect(self.model_info)
        self.cbp.console_button.clicked.connect(self.open_console)
        self.cbp.quit_button.clicked.connect(QtCore.QCoreApplication.instance().quit)
        self.cbp.hist_button.clicked.connect(self.make_histogram)
        self.cbp.radzone_chkbox.stateChanged.connect(self.plot_radzones)
        self.cbp.limits_chkbox.stateChanged.connect(self.plot_limits)
        self.cbp.line_chkbox.stateChanged.connect(self.plot_line)
        self.cbp.add_plot_button.activated[str].connect(self.add_plot)
        self.cbp.command_entry.returnPressed.connect(self.command_activated)

        self.dates = CxoTime(self.model.times).date

        self.telem_data = {k: v for k, v in self.model.comp.items()
                           if isinstance(v, TelemData)}
        self.fmt_telem_data = FormattedTelemData(self.telem_data)

        self.set_checksum(newfile=True)

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
        self.hist_window = None
        self.model_info_window = None

    @property
    def model_spec(self):
        ms = self.model.model_spec
        ms["gui_config"] = gui_config
        return ms

    def set_checksum(self, newfile=False):
        import hashlib
        if newfile:
            model_json = open(gui_config['filename'], 'rb').read()
        else:
            model_json = json.dumps(self.model_spec, 
                                    sort_keys=True, indent=4).encode("utf-8")
        self.md5sum = hashlib.md5(model_json).hexdigest()
        if newfile:
            self.file_md5sum = self.md5sum
        self.checksum_match = self.file_md5sum == self.md5sum

    def open_console(self):

        def fit():
            """Perform a fit."""
            self.fit_worker.start()
            self.fit_monitor()

        def freeze(params):
            """Freeze the parameter or parameters which
            correspond to the given glob pattern.

            Parameters
            ----------
            params : string
                The name of the parameter to freeze.
                Multiple parameters can be specified using
                a glob/regex pattern.

            Returns
            -------

            Examples
            --------
            >>> freeze("solarheat*_P*")
            """
            self.parse_command("freeze {}".format(params))

        def thaw(params):
            """Thaw the parameter or parameters which
            correspond to the given glob pattern.

            Parameters
            ----------
            params : string
                The name of the parameter to thaw.
                Multiple parameters can be specified using
                a glob/regex pattern.

            Returns
            -------

            Examples
            --------
            >>> thaw("solarheat*_P*")
            """
            self.parse_command("thaw {}".format(params))

        def ignore(tstart, tstop):
            """Ignore specified time ranges when performing a fit.

            Parameters
            ----------
            tstart : string
                The earliest time in the range to be ignored.
                If None, this will default to the beginning of
                the imported data range.
            tstop : string
                The latest time in the range to be ignored.
                If None, this will default to the end of the
                imported data range.

            Returns
            -------

            Examples
            --------
            >>> # When only the year and DOY are included,
            >>> # assumed time is 12:00:00
            >>> ignore("2019:100:09:10:12", "2019:200")
            
            >>> ignore(None, "2019:056:17:15:10")
            """
            if tstart is None:
                tstart = "*"
            if tstop is None:
                tstop = "*"
            self.parse_command("ignore {} {}".format(tstart, tstop))

        def notice():
            """Remove all time masks which were set by the
            *ignore* command. Note: this does not remove
            "bad times".

            Parameters
            ----------

            Returns
            -------

            """
            self.parse_command("notice")

        def howto(query=None):
            if query is None:
                msg = "This is the xija_gui_fit console.\n" \
                      "Functions defined here are:\n\n" \
                      "fit()\n" \
                      "freeze(params)\n" \
                      "thaw(params)\n" \
                      "ignore(tstart, tstop)\n" \
                      "notice()\n\n" \
                      "Other data objects are:\n\n" \
                      "telem_data (dict)\n" \
                      "params (OrderedDict) \n\n" \
                      "To find out more information about any of these, " \
                      "type e.g. 'howto(freeze)'"
            elif query is params:
                msg = "The params OrderedDict gives limited access to getting " \
                      "and setting the model parameters.\n\n" \
                      "Examples\n" \
                      "--------\n" \
                      ">>> # Print value, min, max and of a model parameter\n" \
                      ">>> print(params['solarheat__1cbat__P_90'])\n" \
                      ">>> # Set value of a model parameter\n" \
                      ">>> params['dpa_power__pow_3xx0'].val = 40.0\n" \
                      ">>> # Set minimum of a model parameter\n" \
                      ">>> params['dpa_power__pow_3xx0'].min = 0.0\n" \
                      ">>> # Set maximum of a model parameter\n" \
                      ">>> params['dpa_power__pow_3xx0'].max = 100.0"
            elif query is self.telem_data:
                msg = "The telem_data dictionary gives access to the " \
                      "various telemetry data and commanded states objects " \
                      "for examination and side calculations.\n\n" \
                      "Examples\n" \
                      "--------\n" \
                      ">>> print(telem_data.keys())\n" \
                      ">>> print(telem_data['fep_count'].dvals)" 
            else:
                msg = query.__doc__
            print(msg)

        params = self.main_right_panel.params_panel.params_dict

        namespace = {"telem_data": self.telem_data, "params": params, "fit": fit,
                     "freeze": freeze, "thaw": thaw, "ignore": ignore,
                     "notice": notice, "howto": howto}
        widget = in_process_console(**namespace)
        widget.show()

    def write_table(self):
        self.write_table_window = WriteTableWindow(self.model, self)
        self.write_table_window.show()

    def model_info(self):
        self.model_info_window = ModelInfoWindow(self.model, self)
        self.model_info_window.show()

    def make_histogram(self):
        self.hist_window = HistogramWindow(self.model, self.hist_msids)
        self.hist_window.show()

    def plot_limits(self, state):
        self.show_limits = state == QtCore.Qt.Checked
        if self.show_limits:
            self.plots_box.add_annotations("limits")
        else:
            self.plots_box.remove_annotations("limits")

    def plot_line(self, state):
        self.show_line = state == QtCore.Qt.Checked
        self.main_left_panel.plots_box.update_plots()
        if self.show_line:
            self.line_data_window = LineDataWindow(self.model, self,
                                                   self.main_left_panel.plots_box)
            self.plots_box.add_annotations("line")
            self.line_data_window.show()
        else:
            self.plots_box.remove_annotations("line")
            self.line_data_window.close()

    def plot_radzones(self, state):
        self.show_radzones = state == QtCore.Qt.Checked
        if self.show_radzones:
            self.plots_box.add_annotations("radzones")
        else:
            self.plots_box.remove_annotations("radzones")

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
            self.main_left_panel.plots_box.update_plots()
            if self.show_line:
                self.line_data_window.update_data()

        # If fit has not stopped then set another timeout 200 msec from now
        if not fit_stopped:
            QtCore.QTimer.singleShot(200, self.fit_monitor)

    def command_activated(self):
        """Respond to a command like "freeze solarheat*dP*" submitted via the
        command entry box.  The first word is either "freeze" or "thaw" (with
        possibility for other commands later) and the subsequent args are
        space-delimited parameter globs using the UNIX file-globbing syntax.
        This then sets the corresponding params_table checkbuttons.

        Parameters
        ----------

        Returns
        -------

        """
        widget = self.cbp.command_entry
        command = widget.text().strip()
        if command == '':
            return
        self.parse_command(command)
        widget.setText('')

    def parse_command(self, command):
        vals = command.split()
        cmd = vals[0]  # currently freeze, thaw, ignore, or notice
        if cmd not in ('freeze', 'thaw', 'ignore', 'notice') or \
            (cmd != 'notice' and len(vals) < 2):
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
                         self.set_checksum()
                         self.set_title()
                         if self.model_info_window is not None:
                            self.model_info_window.update_checksum()
        elif cmd in ('ignore', 'notice'):
            if cmd == "ignore":
                try:
                    if vals[1] == "*":
                        vals[1] = self.model.datestart
                    if vals[2] == "*":
                        vals[2] = self.model.datestop
                    lim = CxoTime(vals[1:]).date
                except (IndexError, ValueError):
                    if len(vals) == 3:
                        print(f"Invalid input for ignore: {vals[1]} {vals[2]}")
                    else:
                        print("Ignore requires two arguments, the start time and the stop time.")
                    return
                t0, t1 = CxoTime(lim).secs
                self.plots_box.add_ignore(t0, t1)
                self.model.append_mask_times(lim)
            elif cmd == "notice":
                if len(vals) > 1:
                    print("Invalid input for notice: {}".format(vals[1:]))
                    return
                self.model.reset_mask_times()
                self.plots_box.remove_ignores()

    def set_title(self):
        title_str = gui_config['filename']
        if not self.checksum_match:
            title_str += "*"
        self.window.setWindowTitle("xija_gui_fit ({})".format(title_str))

    def save_model_file(self, *args):
        dlg = QtWidgets.QFileDialog()
        dlg.setNameFilters(["JSON files (*.json)", "All files (*)"])
        dlg.selectNameFilter("JSON files (*.json)")
        dlg.selectFile(os.path.abspath(gui_config["filename"]))
        dlg.setAcceptMode(dlg.AcceptSave)
        dlg.exec_()
        filename = str(dlg.selectedFiles()[0])
        if filename != '':
            plot_boxes = self.main_left_panel.plots_box.plot_boxes
            model_spec = self.model.model_spec
            gui_config['plot_names'] = [x.plot_name for x in plot_boxes]
            gui_config['size'] = (self.window.size().width(), 
                                  self.window.size().height())
            model_spec['gui_config'] = gui_config
            try:
                gui_config['filename'] = filename
                self.model.write(filename, model_spec)
                self.set_checksum(newfile=True)
                if self.model_info_window is not None:
                    self.model_info_window.update_checksum()
                    self.model_info_window.update_filename()
                self.set_title()
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
                        default=CxoTime() - 10,  # remove this
                        help="Stop time of fit interval (default=model values)")
    parser.add_argument("--maxiter",
                        default=1000,
                        type=int,
                        help="Maximum number of fit iterations (default=1000)")
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
        start = CxoTime(CxoTime(opt.stop).secs - opt.days * 86400).date[:8]
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

    fit_worker = FitWorker(model, opt.maxiter, method=opt.fit_method)

    model.calc()

    app = QtWidgets.QApplication(sys.argv)
    icon_path = os.path.join(os.path.dirname(__file__), "app_icon.png")
    icon = QtGui.QIcon(icon_path)
    app.setWindowIcon(icon)
    MainWindow(model, fit_worker, opt.filename)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
