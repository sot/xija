import argparse
import ast
import fnmatch
import json
import logging
import re
import sys
import time
from collections import OrderedDict
from itertools import count
from pathlib import Path

import acis_taco as taco
import astropy.units as u
import numpy as np
import pyyaks.context as pyc
from cheta.units import F_to_C
from cxotime import CxoTime
from PyQt5 import QtCore, QtGui, QtWidgets

import xija
from xija.component.base import Node, TelemData
from xija.get_model_spec import get_xija_model_spec

from .fitter import FitWorker, fit_logger
from .plots import HistogramWindow, PlotsBox

gui_config = {}


def raise_error_box(win_title, err_msg):
    msg_box = QtWidgets.QMessageBox()
    msg_box.setIcon(QtWidgets.QMessageBox.Critical)
    msg_box.setText(err_msg)
    msg_box.setWindowTitle(win_title)
    msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
    msg_box.exec_()


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
            if hasattr(data, "resids"):
                self.data_names += [name, name + "_model", name + "_resid"]
                self.data_basenames += [name] * 3
                self.formats += [fmt] * 3
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


class FiltersWindow(QtWidgets.QWidget):
    def __init__(self, model, main_window):
        super(FiltersWindow, self).__init__()
        self.mw = main_window
        self.setWindowTitle("Filters")

        header_font = QtGui.QFont()
        header_font.setBold(True)

        self.ignore_label = QtWidgets.QLabel("Add Ignore/Notice")
        self.ignore_label.setFont(header_font)
        self.start_label = QtWidgets.QLabel("Start time:")
        self.start_text = QtWidgets.QLineEdit()
        self.stop_label = QtWidgets.QLabel("Stop time:")
        self.stop_text = QtWidgets.QLineEdit()

        add_ignore_button = QtWidgets.QPushButton("Add Ignore")
        add_ignore_button.clicked.connect(self.add_ignore)

        notice_button = QtWidgets.QPushButton("Notice All")
        notice_button.clicked.connect(self.notice_pushed)

        pair = QtWidgets.QHBoxLayout()
        pair.addWidget(add_ignore_button)
        pair.addWidget(notice_button)

        self.bt_label = QtWidgets.QLabel("Add Bad Time")
        self.bt_label.setFont(header_font)
        self.bt_start_label = QtWidgets.QLabel("Start time:")
        self.bt_start_text = QtWidgets.QLineEdit()
        self.bt_stop_label = QtWidgets.QLabel("Stop time:")
        self.bt_stop_text = QtWidgets.QLineEdit()

        add_bt_button = QtWidgets.QPushButton("Add Bad Time")
        add_bt_button.clicked.connect(self.add_bad_time)

        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close_window)

        close = QtWidgets.QHBoxLayout()
        close.addWidget(close_button)
        close.addStretch(1)

        self.box = QtWidgets.QVBoxLayout()

        self.box.addWidget(self.ignore_label)
        self.box.addWidget(self.start_label)
        self.box.addWidget(self.start_text)
        self.box.addWidget(self.stop_label)
        self.box.addWidget(self.stop_text)
        self.box.addLayout(pair)
        self.box.addWidget(self.bt_label)
        self.box.addWidget(self.bt_start_label)
        self.box.addWidget(self.bt_start_text)
        self.box.addWidget(self.bt_stop_label)
        self.box.addWidget(self.bt_stop_text)
        self.box.addWidget(add_bt_button)
        self.box.addStretch(1)
        self.box.addLayout(close)

        self.setLayout(self.box)
        self.setGeometry(0, 0, 400, 400)

    def add_ignore(self):
        self.add_filter("ignore")

    def add_bad_time(self):
        self.add_filter("bad_time")

    def add_filter(self, filter_type):
        err_msg = ""
        if filter_type == "ignore":
            vals = [self.start_text.text(), self.stop_text.text()]
        elif filter_type == "bad_time":
            vals = [self.bt_start_text.text(), self.bt_stop_text.text()]
        try:
            if vals[0] == "*":
                vals[0] = self.mw.model.datestart
            if vals[1] == "*":
                vals[1] = self.mw.model.datestop
            lim = CxoTime(vals).date
            t0, t1 = CxoTime(lim).secs
            if t0 > t1:
                err_msg = "Filter stop is earlier than filter start!"
        except (IndexError, ValueError):
            if len(vals) == 2:
                err_msg = f"Invalid input for filter: {vals[0]} {vals[1]}"
            else:
                err_msg = (
                    "Filter requires two arguments, the start time and the stop time."
                )
        if len(err_msg) > 0:
            raise_error_box("Filters Error", err_msg)
        else:
            if filter_type == "ignore":
                self.mw.model.append_mask_time([lim[0], lim[1]])
                bad = False
                self.mw.plots_box.add_fill(t0, t1)
            elif filter_type == "bad_time":
                self.mw.model.append_bad_time([lim[0], lim[1]])
                bad = True
            self.mw.plots_box.add_fill(t0, t1, bad=bad)
        self.start_text.setText("")
        self.stop_text.setText("")
        self.bt_start_text.setText("")
        self.bt_stop_text.setText("")

    def notice_pushed(self):
        self.mw.plots_box.remove_ignores()
        self.mw.model.reset_mask_times()
        self.mw.plots_box.update_plots()

    def close_window(self, *args):
        self.close()


class ChangeTimesWindow(QtWidgets.QWidget):
    def __init__(self, model, main_window):
        super(ChangeTimesWindow, self).__init__()
        self.mw = main_window
        self.setWindowTitle("Change Times")

        self.start_label = QtWidgets.QLabel("Start time:")
        self.start_text = QtWidgets.QLineEdit()
        self.stop_label = QtWidgets.QLabel("Stop time:")
        self.stop_text = QtWidgets.QLineEdit()

        cancel_button = QtWidgets.QPushButton("Cancel")
        cancel_button.clicked.connect(self.close_window)

        change_button = QtWidgets.QPushButton("Change")
        change_button.clicked.connect(self.change_pushed)

        pair = QtWidgets.QHBoxLayout()
        pair.addWidget(cancel_button)
        pair.addWidget(change_button)

        self.box = QtWidgets.QVBoxLayout()

        self.box.addWidget(self.start_label)
        self.box.addWidget(self.start_text)
        self.box.addWidget(self.stop_label)
        self.box.addWidget(self.stop_text)
        self.box.addLayout(pair)

        self.setLayout(self.box)
        self.setGeometry(0, 0, 400, 200)

    def change_pushed(self):
        err_msg = ""
        vals = [self.start_text.text(), self.stop_text.text()]
        try:
            if vals[0] == "*":
                vals[0] = self.mw.model.datestart
            if vals[1] == "*":
                vals[1] = self.mw.model.datestop
            lim = CxoTime(vals).date
            t0, t1 = CxoTime(lim).secs
            if t0 > t1:
                err_msg = "Time stop is earlier than time start!"
        except (IndexError, ValueError) as e:
            if len(vals) == 2:
                err_msg = f"Invalid input for time change: {vals[0]} {vals[1]}"
            else:
                err_msg = (
                    "Changing model times requires two arguments, the start time and the stop time."
                )
        if len(err_msg) > 0:
            raise_error_box("Change Time Error", err_msg)
        else:
            self.mw.reset_model(*vals)
        self.start_text.setText("")
        self.stop_text.setText("")
        self.close()

    def close_window(self, *args):
        self.close()


class WriteTableWindow(QtWidgets.QWidget):
    def __init__(self, model, main_window):  # noqa: PLR0915
        super(WriteTableWindow, self).__init__()
        self.mw = main_window
        self.setWindowTitle("Write Table")
        self.box = QtWidgets.QVBoxLayout()
        self.setLayout(self.box)
        self.setGeometry(0, 0, 200, 600)
        self.scroll = QtWidgets.QScrollArea()

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

        write_button = QtWidgets.QPushButton("Write Table")
        write_button.clicked.connect(self.save_ascii_table)

        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close_window)

        buttons.addWidget(write_button)
        buttons.addWidget(close_button)

        main_box.addLayout(buttons)
        self.box.addLayout(main_box)

    def toggle_all_data(self, state):
        checked = state == QtCore.Qt.Checked
        for _i, box in enumerate(self.check_boxes):
            box.setChecked(checked)

    def change_start(self):
        start_date = self.start_text.text()
        try:
            _ = CxoTime(start_date).secs
            self.start_label.setText("Start time: {}".format(start_date))
            self.start_date = start_date
        except ValueError:
            raise_error_box("Write Table Error", f"Start time not valid: {start_date}")
        self.start_text.setText("")

    def change_stop(self):
        stop_date = self.stop_text.text()
        try:
            _ = CxoTime(stop_date).secs
            self.start_label.setText("Stop time: {}".format(stop_date))
            self.stop_date = stop_date
        except ValueError:
            raise_error_box("Write Table Error", f"Stop time not valid: {stop_date}")
        self.stop_text.setText("")

    def close_window(self, *args):
        self.close()

    def save_ascii_table(self):
        from astropy.table import Column, Table

        dlg = QtWidgets.QFileDialog()
        dlg.setNameFilters(["ECSV files (*.ecsv)", "All files (*)"])
        dlg.selectNameFilter("ECSV files (*.ecsv)")
        dlg.setAcceptMode(dlg.AcceptSave)
        dlg.exec_()
        filename = str(dlg.selectedFiles()[0])
        if filename != "":
            try:
                checked = []
                for i, box in enumerate(self.check_boxes):
                    if box.isChecked():
                        checked.append(i)
                t = Table()
                ts = CxoTime([self.start_date, self.stop_date]).secs
                ts[-1] += 1.0  # a buffer to make sure we grab the last point
                istart, istop = np.searchsorted(self.ftd.times, ts)
                c = Column(self.ftd.dates[istart:istop], name="date", format="{0}")
                t.add_column(c)
                for i, key in enumerate(self.ftd):
                    if i in checked:
                        c = Column(
                            self.ftd[i][istart:istop],
                            name=key,
                            format=self.ftd.formats[i],
                        )
                        t.add_column(c)
                t.write(filename, overwrite=True, format="ascii.ecsv")
                self.last_filename = filename
            except IOError as ioerr:
                msg = QtWidgets.QMessageBox()
                msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                msg.setText("There was a problem writing the file:")
                msg.setDetailedText(
                    "Cannot write {}. {}".format(filename, ioerr.strerror)
                )
                msg.exec_()


class ModelInfoWindow(QtWidgets.QWidget):
    def __init__(self, model, main_window):
        super(ModelInfoWindow, self).__init__()
        self.setWindowTitle("Model Info")
        self.box = QtWidgets.QVBoxLayout()
        self.setLayout(self.box)
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
        main_box.addWidget(
            QtWidgets.QLabel("Evolve Method: Core {}".format(model.evolve_method))
        )
        main_box.addWidget(
            QtWidgets.QLabel("Runge-Kutta Order: {}".format(4 if model.rk4 else 2))
        )
        for msid in self.main_window.model.limits:
            limits = self.main_window.model.limits[msid]
            units = limits["unit"]
            main_box.addWidget(QtWidgets.QLabel(f"{msid.upper()} limits:"))
            for limit, val in limits.items():
                if limit == "unit":
                    continue
                limit_str = f"    {limit}: {val:.1f} {units}"
                if units == "degF":
                    limit_str += f" ({F_to_C(val):.1f} degC)"
                main_box.addWidget(QtWidgets.QLabel(limit_str))
        main_box.addStretch(1)

        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close_window)

        close_box = QtWidgets.QHBoxLayout()
        close_box.addStretch(1)
        close_box.addWidget(close_button)

        main_box.addLayout(close_box)
        self.box.addLayout(main_box)

    def update_checksum(self):
        self.main_window.set_checksum()
        if self.main_window.checksum_match:
            color = "black"
        else:
            color = "red"
        checksum_str = self.main_window.md5sum
        self.checksum_label.setText(checksum_str)
        self.checksum_label.setStyleSheet("color: {}".format(color))

    def update_filename(self):
        self.filename_label.setText("Filename: {}".format(gui_config["filename"]))

    def close_window(self, *args):
        self.close()
        self.main_window.model_info_window = None


class LineDataWindow(QtWidgets.QWidget):
    def __init__(self, model, main_window, plots_box):
        super(LineDataWindow, self).__init__()
        self.setWindowTitle("Line Data")
        self.box = QtWidgets.QVBoxLayout()
        self.setLayout(self.box)
        self.setGeometry(0, 0, 350, 600)

        self.plots_box = plots_box
        self.main_window = main_window
        self.ftd = self.main_window.fmt_telem_data
        self.nrows = len(self.ftd.data_names) + 1

        self.table = WidgetTable(
            n_rows=self.nrows,
            colnames=["name", "value"],
            colwidths={1: 200},
            show_header=True,
        )

        self.table[0, 0] = QtWidgets.QLabel("date")
        self.table[0, 1] = QtWidgets.QLabel("")

        for row in range(1, self.nrows):
            name = self.ftd.data_names[row - 1]
            self.table[row, 0] = QtWidgets.QLabel(name)
            self.table[row, 1] = QtWidgets.QLabel("")

        self.update_data()

        self.box.addWidget(self.table.table)

    def update_data(self):
        pos = np.searchsorted(self.plots_box.pd_times, self.plots_box.xline)
        date = self.main_window.dates[pos]
        self.table[0, 1].setText(date)
        for row in range(1, self.nrows):
            val = self.ftd[row - 1]
            fmt = self.ftd.formats[row - 1]
            self.table[row, 1].setText(fmt.format(val[pos]))


class WidgetTable(dict):
    def __init__(
        self, n_rows, n_cols=None, colnames=None, show_header=False, colwidths=None
    ):
        if n_cols is None and colnames is None:
            raise ValueError("WidgetTable needs either n_cols or colnames")
        if colnames:
            self.colnames = colnames
            self.n_cols = len(colnames)
        else:
            self.n_cols = n_cols
            self.colnames = ["col{}".format(i + 1) for i in range(n_cols)]
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


class Panel:
    def __init__(self, orient="h"):
        Box = QtWidgets.QHBoxLayout if orient == "h" else QtWidgets.QVBoxLayout
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
            msg = (
                "Attempted to set parameter value below minimum. Setting to min value."
            )
            self.par.val = self.par.min
            self.params_panel.params_table[self.row, 2].setText(
                self.par.fmt.format(self.par.val)
            )
        if self.par.val > self.par.max:
            msg = (
                "Attempted to set parameter value below maximum. Setting to max value."
            )
            self.par.val = self.par.max
            self.setText(self.par.fmt.format(self.par.val))
            self.params_panel.params_table[self.row, 2].setText(
                self.par.fmt.format(self.par.val)
            )
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


class PanelParam:
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
        self.dx = 100.0 / (self.parmax - self.parmin)
        self.idx = 1.0 / self.dx

    def set_step_from_value(self, val):
        step = int((val - self.parmin) * self.dx)
        self.setValue(step)

    def get_value_from_step(self):
        val = self.value() * self.idx + self.parmin
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
        self.par.val = val
        self.params_panel.params_table[self.row, 2].setText(self.par.fmt.format(val))
        if self.update_plots:
            self.params_panel.plots_panel.update_plots()


class ParamsPanel(Panel):
    def __init__(self, model, plots_panel):
        Panel.__init__(self, orient="v")
        self.plots_panel = plots_panel
        self.pars = model.pars

        params_table = WidgetTable(
            n_rows=len(self.pars),
            colnames=["fit", "name", "val", "min", "", "max"],
            colwidths={0: 30, 1: 250},
            show_header=True,
        )

        self.params_dict = OrderedDict()

        for row, par in zip(count(), self.pars):
            # Thawed (i.e. fit the parameter)
            frozen = params_table[row, 0] = PanelCheckBox(
                par, self.plots_panel.main_window
            )
            frozen.setChecked(not par.frozen)
            frozen.stateChanged.connect(frozen.frozen_toggled)

            # par full name
            params_table[row, 1] = QtWidgets.QLabel(par.full_name)

            # Slider
            slider = PanelSlider(self, par, row)
            params_table[row, 4] = slider
            slider.sliderMoved.connect(slider.slider_moved)

            # Value
            entry = params_table[row, 2] = PanelText(self, row, par, "val", slider)
            entry.setText(par.fmt.format(par.val))
            entry.returnPressed.connect(entry.par_attr_changed)

            # Min of slider
            entry = params_table[row, 3] = PanelText(self, row, par, "min", slider)
            entry.setText(par.fmt.format(par.min))
            entry.returnPressed.connect(entry.par_attr_changed)

            # Max of slider
            entry = params_table[row, 5] = PanelText(self, row, par, "max", slider)
            entry.setText(par.fmt.format(par.max))
            entry.returnPressed.connect(entry.par_attr_changed)

            self.params_dict[par.full_name] = PanelParam(
                params_table[row, 2], params_table[row, 3], params_table[row, 5]
            )

        self.pack_start(params_table.table)
        self.params_table = params_table

    def update(self):
        for row, par in enumerate(self.pars):
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
    def __init__(self, model):  # noqa: PLR0915
        Panel.__init__(self, orient="v")

        self.comps = model.comps

        self.fit_button = QtWidgets.QPushButton("Fit")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.reset_plots_button = QtWidgets.QPushButton("Reset Plots")
        self.add_plot_button = self.make_add_plot_button()

        self.radzone_chkbox = QtWidgets.QCheckBox()
        self.limits_chkbox = QtWidgets.QCheckBox()
        self.line_chkbox = QtWidgets.QCheckBox()
        if len(model.limits) == 0:
            self.limits_chkbox.setEnabled(False)

        self.top_panel = Panel()
        self.bottom_panel = Panel()

        self.top_panel.pack_start(self.fit_button)
        self.top_panel.pack_start(self.stop_button)
        self.top_panel.add_stretch(1)

        self.radzone_panel = Panel()
        self.radzone_panel.pack_start(QtWidgets.QLabel("Show radzones"))
        self.radzone_panel.pack_start(self.radzone_chkbox)

        self.limits_panel = Panel()
        self.limits_panel.pack_start(QtWidgets.QLabel("Show limits"))
        self.limits_panel.pack_start(self.limits_chkbox)

        self.line_panel = Panel()
        self.line_panel.pack_start(QtWidgets.QLabel("Annotate line"))
        self.line_panel.pack_start(self.line_chkbox)

        self.bottom_panel.pack_start(self.reset_plots_button)
        self.bottom_panel.pack_start(self.add_plot_button)
        self.bottom_panel.pack_start(self.radzone_panel)
        self.bottom_panel.pack_start(self.limits_panel)
        self.bottom_panel.pack_start(self.line_panel)
        self.bottom_panel.add_stretch(1)

        self.pack_start(self.top_panel)
        self.pack_start(self.bottom_panel)

    def make_add_plot_button(self):
        apb = QtWidgets.QComboBox()
        apb.addItem("Add plot...")

        plot_names = [
            f"{comp.name} {attr[5:]}"
            for comp in self.comps
            for attr in dir(comp)
            if attr.startswith("plot_")
        ]

        self.plot_names = plot_names
        for plot_name in plot_names:
            apb.addItem(plot_name)

        return apb


class FreezeThawPanel(Panel):
    def __init__(self):
        Panel.__init__(self, orient="h")
        self.freeze_entry = QtWidgets.QLineEdit()
        self.thaw_entry = QtWidgets.QLineEdit()

        self.pack_start(QtWidgets.QLabel("Freeze:"))
        self.pack_start(self.freeze_entry)
        self.pack_start(QtWidgets.QLabel("Thaw:"))
        self.pack_start(self.thaw_entry)


class MainLeftPanel(Panel):
    def __init__(self, model, main_window):
        Panel.__init__(self, orient="v")
        self.control_buttons_panel = ControlButtonsPanel(model)
        self.plots_box = PlotsBox(model, main_window)
        self.pack_start(self.control_buttons_panel)
        self.pack_start(self.plots_box)
        self.add_stretch(1)


class MainRightPanel(Panel):
    def __init__(self, model, plots_panel):
        Panel.__init__(self, orient="v")
        self.freeze_thaw_panel = FreezeThawPanel()
        self.params_panel = ParamsPanel(model, plots_panel)
        self.pack_start(self.freeze_thaw_panel)
        self.pack_start(self.params_panel)


class MainWindow:
    # This is a callback function. The data arguments are ignored
    # in this example. More on callbacks below.
    def __init__(self, model, fit_worker, model_file):  # noqa: PLR0915
        import Ska.tdb

        self.model = model

        self.hist_msids = []
        for k, v in self.model.comp.items():
            if isinstance(v, Node):
                if k.startswith(("fptemp", "tmp_fep", "tmp_bep")):
                    self.hist_msids.append(k)
                try:
                    Ska.tdb.msids[v.msid]
                except KeyError:
                    pass
                else:
                    self.hist_msids.append(k)

        self._init_model(model)

        if gui_config["filename"] is None:
            self.checksum_match = False
        else:
            self.checksum_match = True

        self.fit_worker = fit_worker
        # create a new window
        self.mwindow = QtWidgets.QMainWindow()
        self.window = QtWidgets.QWidget()
        self.mwindow.setGeometry(0, 0, *gui_config.get("size", (1400, 800)))
        self.set_title()
        self.mwindow.setCentralWidget(self.window)
        self.main_box = Panel(orient="h")

        # This is the Layout Box that holds the top-level stuff in the main window
        main_window_hbox = QtWidgets.QHBoxLayout()
        self.window.setLayout(main_window_hbox)

        self.main_left_panel = MainLeftPanel(model, self)
        mlp = self.main_left_panel
        self.plots_box = self.main_left_panel.plots_box

        self.main_right_panel = MainRightPanel(model, mlp.plots_box)
        mrp = self.main_right_panel

        self.show_radzones = False
        self.show_limits = False
        self.show_line = False

        menu_bar = QtWidgets.QMenuBar()
        menu_bar.setNativeMenuBar(True)
        file_menu = menu_bar.addMenu("&File")
        self.save_action = QtWidgets.QAction("&Save...", self.mwindow)
        self.info_action = QtWidgets.QAction("&Info...", self.mwindow)
        self.quit_action = QtWidgets.QAction("&Quit", self.mwindow)
        self.save_action.triggered.connect(self.save_model_file)
        self.info_action.triggered.connect(self.model_info)
        self.quit_action.triggered.connect(self.quit_pushed)
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.info_action)
        file_menu.addAction(self.quit_action)
        model_menu = menu_bar.addMenu("&Model")
        self.time_action = QtWidgets.QAction("&Change Times...", self.mwindow)
        self.time_action.triggered.connect(self.change_times)
        model_menu.addAction(self.time_action)
        util_menu = menu_bar.addMenu("&Utilities")
        self.hist_action = QtWidgets.QAction("&Histogram...", self.mwindow)
        self.filt_action = QtWidgets.QAction("&Filters...", self.mwindow)
        self.table_action = QtWidgets.QAction("&Write Table...", self.mwindow)
        self.hist_action.triggered.connect(self.make_histogram)
        self.filt_action.triggered.connect(self.filters)
        self.table_action.triggered.connect(self.write_table)
        util_menu.addAction(self.hist_action)
        util_menu.addAction(self.filt_action)
        util_menu.addAction(self.table_action)
        self.mwindow.setMenuBar(menu_bar)

        self.cbp = mlp.control_buttons_panel
        self.cbp.fit_button.clicked.connect(self.fit_worker.start)
        self.cbp.fit_button.clicked.connect(self.fit_monitor)
        self.cbp.stop_button.clicked.connect(self.fit_worker.terminate)
        self.cbp.radzone_chkbox.stateChanged.connect(self.plot_radzones)
        self.cbp.limits_chkbox.stateChanged.connect(self.plot_limits)
        self.cbp.line_chkbox.stateChanged.connect(self.plot_line)
        self.cbp.add_plot_button.activated[str].connect(self.add_plot)
        self.cbp.reset_plots_button.clicked.connect(self.plots_box.reset_plots)

        self.ftp = mrp.freeze_thaw_panel
        self.ftp.freeze_entry.returnPressed.connect(self.freeze_activated)
        self.ftp.thaw_entry.returnPressed.connect(self.thaw_activated)

        self.set_checksum(newfile=True)

        self.status_bar = QtWidgets.QStatusBar()
        self.mwindow.setStatusBar(self.status_bar)

        # Add plots from previous Save
        for plot_name in gui_config.get("plot_names", []):
            try:
                self.add_plot(plot_name)
                time.sleep(0.05)  # is it needed?
            except ValueError:
                print(f"ERROR: Unexpected plot_name {plot_name}")

        # Show everything finally
        splitter = QtWidgets.QSplitter()
        left_widget = QtWidgets.QWidget()
        right_widget = QtWidgets.QWidget()
        left_widget.setLayout(mlp.box)
        right_widget.setLayout(mrp.box)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        main_window_hbox.addWidget(splitter)

        self.mwindow.show()
        self.hist_window = None
        self.model_info_window = None

    @property
    def model_spec(self):
        ms = self.model.model_spec
        ms["gui_config"] = gui_config
        return ms

    def set_checksum(self, newfile=False):
        import hashlib

        if newfile and gui_config["filename"] is not None:
            model_json = open(gui_config["filename"], "rb").read()
        else:
            model_json = json.dumps(self.model_spec, sort_keys=True, indent=4).encode(
                "utf-8"
            )
        self.md5sum = hashlib.md5(model_json).hexdigest()
        if newfile:
            self.file_md5sum = self.md5sum
        if gui_config["filename"] is None:
            self.checksum_match = False
        else:
            self.checksum_match = self.file_md5sum == self.md5sum

    def _init_model(self, model, reset=False):
        import Ska.tdb
        self.model = model
        if not reset:
            self.hist_msids = []
            for k, v in self.model.comp.items():
                if isinstance(v, Node):
                    if (
                        k.startswith("fptemp")
                        or k.startswith("tmp_fep")
                        or k.startswith("tmp_bep")
                    ):
                        self.hist_msids.append(k)
                    try:
                        Ska.tdb.msids[v.msid]
                    except KeyError:
                        pass
                    else:
                        self.hist_msids.append(k)

        self.dates = CxoTime(self.model.times).date

        self.telem_data = {
            k: v for k, v in self.model.comp.items() if isinstance(v, TelemData)
        }

        self.fmt_telem_data = FormattedTelemData(self.telem_data)

    def reset_model(self, new_start, new_stop):
        model = xija.ThermalModel(self.model.model_spec["name"], new_start,
                                  new_stop, model_spec=self.model.model_spec)

        for comp_name, val in gui_config["set_data_vals"].items():
            model.comp[comp_name].set_data(val)

        model.make()
        model.calc()

        self._init_model(model, reset=True)
        self.plots_box.reset_model(model)
        self.fit_worker.model = model

    def change_times(self):
        self.change_times_window = ChangeTimesWindow(self.model, self)
        self.change_times_window.show()

    def write_table(self):
        self.write_table_window = WriteTableWindow(self.model, self)
        self.write_table_window.show()

    def filters(self):
        self.filters_window = FiltersWindow(self.model, self)
        self.filters_window.show()

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
            self.line_data_window = LineDataWindow(
                self.model, self, self.main_left_panel.plots_box
            )
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

    def fit_monitor(self):
        self.status_bar.showMessage("BUSY... ")
        msg = None
        fit_stopped = False
        while self.fit_worker.parent_pipe.poll():
            # Keep reading messages until there are no more or until getting
            # a message indicating fit is stopped.
            msg = self.fit_worker.parent_pipe.recv()
            fit_stopped = msg["status"] in ("terminated", "finished")
            if fit_stopped:
                self.fit_worker.fit_process.join()
                print("\n*********************************")
                print("  FIT", msg["status"].upper())
                print("*********************************\n")
                self.status_bar.clearMessage()
                break

        if msg:
            # Update the fit_worker model parameters and then the corresponding
            # params table widget.
            self.fit_worker.model.parvals = msg["parvals"]
            self.main_right_panel.params_panel.update()
            self.main_left_panel.plots_box.update_plots()
            if self.show_line:
                self.line_data_window.update_data()

        # If fit has not stopped then set another timeout 200 msec from now
        if not fit_stopped:
            QtCore.QTimer.singleShot(200, self.fit_monitor)

    def freeze_activated(self):
        self.command_activated("freeze")

    def thaw_activated(self):
        self.command_activated("thaw")

    def command_activated(self, cmd_type):
        """Respond to a command like "freeze solarheat*dP*" submitted via the
        command entry box.  The first word is either "freeze" or "thaw" (with
        possibility for other commands later) and the subsequent args are
        space-delimited parameter globs using the UNIX file-globbing syntax.
        This then sets the corresponding params_table checkbuttons.

        """
        widget = getattr(self.ftp, f"{cmd_type}_entry")
        command = widget.text().strip()
        if command == "":
            return
        vals = command.split()
        if cmd_type in ("freeze", "thaw"):
            par_regexes = [fnmatch.translate(x) for x in vals]
            params_table = self.main_right_panel.params_panel.params_table
            for row, par in enumerate(self.model.pars):
                for par_regex in par_regexes:
                    if re.match(par_regex, par.full_name):
                        checkbutton = params_table[row, 0]
                        checkbutton.setChecked(cmd_type == "thaw")
                        par.frozen = cmd_type != "thaw"
                        self.set_checksum()
                        self.set_title()
                        if self.model_info_window is not None:
                            self.model_info_window.update_checksum()
        widget.setText("")

    def set_title(self):
        title_str = gui_config["filename"]
        if title_str is None:
            title_str = "no filename"
        if not self.checksum_match:
            title_str += "*"
        self.mwindow.setWindowTitle(f"xija_gui_fit v{xija.__version__} ({title_str})")

    def save_model_file(self, *args):
        dlg = QtWidgets.QFileDialog()
        dlg.setNameFilters(["JSON files (*.json)", "All files (*)"])
        dlg.selectNameFilter("JSON files (*.json)")
        if gui_config["filename"] is not None:
            dlg.selectFile(gui_config["filename"])
        dlg.setAcceptMode(dlg.AcceptSave)
        dlg.exec_()
        filename = str(dlg.selectedFiles()[0])
        if filename != "":
            plot_boxes = self.main_left_panel.plots_box.plot_boxes
            model_spec = self.model.model_spec
            gui_config["plot_names"] = [x.plot_name for x in plot_boxes]
            gui_config["size"] = (
                self.window.size().width(),
                self.window.size().height(),
            )
            model_spec["gui_config"] = gui_config
            try:
                gui_config["filename"] = filename
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
                msg.setDetailedText(
                    "Cannot write {}. {}".format(filename, ioerr.strerror)
                )
                msg.exec_()

    def quit_pushed(self):
        if not self.checksum_match:
            answer = QtWidgets.QMessageBox.question(
                self.window,
                "Save Model?",
                "Current model not saved. Would you like to save it?",
            )
            if answer == QtWidgets.QMessageBox.Yes:
                self.save_model_file()
        QtCore.QCoreApplication.instance().quit()


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        default="test_gui.json",
        help="Model or file name: '*.json' is a local spec file, otherwise this is a "
        "model name (e.g. 'acisfp_spec_matlab') that points to a like-named JSON spec "
        "file in the Ska chandra_models repo.",
    )
    parser.add_argument(
        "--model-version",
        help="Model version for model from chandra_models",
    )
    parser.add_argument(
        "--days",
        type=float,
        default=15,  # Fix this
        help="Number of days in fit interval (default=90)",
    )
    parser.add_argument(
        "--stop",
        default=CxoTime() - 10 * u.day,  # remove this
        help="Stop time of fit interval (default=model values)",
    )
    parser.add_argument(
        "--maxiter",
        default=1000,
        type=int,
        help="Maximum number of fit iterations (default=1000)",
    )
    parser.add_argument(
        "--fit-method",
        default="simplex",
        help="Sherpa fit method (simplex|moncar|levmar)",
    )
    parser.add_argument(
        "--inherit-from", help="Inherit par values from model spec file"
    )
    parser.add_argument(
        "--set-data",
        action="append",
        dest="set_data_exprs",
        default=[],
        help="Set data value as '<comp_name>=<value>'",
    )
    parser.add_argument(
        "--quiet", default=False, action="store_true", help="Suppress screen output"
    )

    return parser.parse_args()


def main():  # noqa: PLR0912, PLR0915
    # Enable fully-randomized evaluation of ACIS-FP model which is desirable
    # for fitting.
    taco.set_random_salt(None)

    opt = get_options()

    files = (
        pyc.CONTEXT["file"]
        if "file" in pyc.CONTEXT
        else pyc.ContextDict("files", basedir=str(Path.cwd()))
    )
    files.update(xija.files)

    sherpa_logger = logging.getLogger("sherpa")
    loggers = (fit_logger, sherpa_logger)
    if opt.quiet:
        for logger in loggers:
            for h in logger.handlers:
                logger.removeHandler(h)

    if opt.filename.endswith(".json"):
        model_spec = json.load(open(opt.filename, "r"))
    else:
        model_spec, model_version = get_xija_model_spec(
            opt.filename, version=opt.model_version
        )
        print(f"Using model version {model_version} from chandra_models")

    gui_config.update(model_spec.get("gui_config", {}))

    # Use supplied stop time and days OR use model_spec values if stop not supplied
    if opt.stop:
        start = CxoTime(CxoTime(opt.stop).secs - opt.days * 86400).date[:8]
        stop = opt.stop
    else:
        start = model_spec["datestart"]
        stop = model_spec["datestop"]

    model = xija.ThermalModel(model_spec["name"], start, stop, model_spec=model_spec)

    set_data_vals = gui_config.get("set_data_vals", {})
    for set_data_expr in opt.set_data_exprs:
        set_data_expr = re.sub(r"\s", "", set_data_expr)  # noqa: PLW2901
        try:
            comp_name, val = set_data_expr.split("=")
        except ValueError:
            raise ValueError(
                "--set_data must be in form '<comp_name>=<value>'"
            ) from None
        # Set data to value.  ast.literal_eval is a safe way to convert any
        # string literal into the corresponding Python object.
        set_data_vals[comp_name] = ast.literal_eval(val)

    for comp_name, val in set_data_vals.items():
        model.comp[comp_name].set_data(val)

    model.make()

    if opt.inherit_from:
        inherit_spec = json.load(open(opt.inherit_from, "r"))
        inherit_pars = {par["full_name"]: par for par in inherit_spec["pars"]}
        for par in model.pars:
            if par.full_name in inherit_pars:
                print("Inheriting par {}".format(par.full_name))
                par.val = inherit_pars[par.full_name]["val"]
                par.min = inherit_pars[par.full_name]["min"]
                par.max = inherit_pars[par.full_name]["max"]
                par.frozen = inherit_pars[par.full_name]["frozen"]
                par.fmt = inherit_pars[par.full_name]["fmt"]

    filename = Path(opt.filename)
    if filename.exists():
        gui_config["filename"] = str(filename.resolve())
    else:
        gui_config["filename"] = None
    gui_config["set_data_vals"] = set_data_vals

    fit_worker = FitWorker(model, opt.maxiter, method=opt.fit_method)

    model.calc()

    app = QtWidgets.QApplication(sys.argv)
    icon_path = str(Path(__file__).parent / "app_icon.png")
    icon = QtGui.QIcon(icon_path)
    app.setWindowIcon(icon)
    app.setApplicationName("xija_gui_fit")
    MainWindow(model, fit_worker, opt.filename)
    app.setStyleSheet("""
        QMenu {font-size: 15px}
        QMenu QWidget {font-size: 15px}
        QMenuBar {font-size: 15px}
    """)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
