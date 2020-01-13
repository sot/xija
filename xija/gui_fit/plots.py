from PyQt5 import QtWidgets, QtCore

import functools
import numpy as np

from Ska.Matplotlib import cxctime2plotdate
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.dates as mdates

def digitize_data(Ttelem, nbins=50):
    """ Digitize telemetry.

    :param Ttelem: telemetry values
    :param nbins: number of bins
    :returns: coordinates for error quantile line

    """

    # Bin boundaries
    # Note that the min/max range is expanded to keep all telemetry within the outer boundaries.
    # Also the number of boundaries is 1 more than the number of bins.
    bins = np.linspace(min(Ttelem) - 1e-6, max(Ttelem) + 1e-6, nbins + 1)
    inds = np.digitize(Ttelem, bins) - 1
    means = bins[:-1] + np.diff(bins) / 2

    return np.array([means[i] for i in inds])


def calcquantiles(errors):
    """ Calculate the error quantiles.

    :param error: model errors (telemetry - model)
    :returns: datastructure that includes errors (input) and quantile values

    """

    esort = np.sort(errors)
    q99 = esort[int(0.99 * len(esort) - 1)]
    q95 = esort[int(0.95 * len(esort) - 1)]
    q84 = esort[int(0.84 * len(esort) - 1)]
    q50 = np.median(esort)
    q16 = esort[int(0.16 * len(esort) - 1)]
    q05 = esort[int(0.05 * len(esort) - 1)]
    q01 = esort[int(0.01 * len(esort) - 1)]
    stats = {'error': errors, 'q01': q01, 'q05': q05, 'q16': q16, 'q50': q50,
             'q84': q84, 'q95': q95, 'q99': q99}
    return stats


def calcquantstats(Ttelem, error):
    """ Calculate error quantiles for individual telemetry temperatures (each count individually).

    :param Ttelem: telemetry values
    :param error: model error (telemetry - model)
    :returns: coordinates for error quantile line

    This is used for the telemetry vs. error plot (axis 3).

    """

    Tset = np.sort(list(set(Ttelem)))
    Tquant = {'key': []}
    k = -1
    for T in Tset:
        if len(Ttelem[Ttelem == T]) >= 200:
            k = k + 1
            Tquant['key'].append([k, T])
            ind = Ttelem == T
            errvals = error[ind]
            Tquant[k] = calcquantiles(errvals)

    return Tquant


def getQuantPlotPoints(quantstats, quantile):
    """ Calculate the error quantile line coordinates for the data in each telemetry count value.

    :param quantstats: output from calcquantstats()
    :param quantile: quantile (string - e.g. "q01"), used as a key into quantstats datastructure
    :returns: coordinates for error quantile line

    This is used to calculate the quantile lines plotted on the telemetry vs. error plot (axis 3)
    enclosing the data (i.e. the 1 and 99 percentile lines).

    """

    Tset = [T for (n, T) in quantstats['key']]
    diffTset = np.diff(Tset)
    Tpoints = Tset[:-1] + diffTset / 2
    Tpoints = list(np.array([Tpoints, Tpoints]).T.flatten())
    Tpoints.insert(0, Tset[0] - diffTset[0] / 2)
    Tpoints.append(Tset[-1] + diffTset[0] / 2)
    Epoints = [quantstats[num][quantile] for (num, T) in quantstats['key']]
    Epoints = np.array([Epoints, Epoints]).T.flatten()
    return Epoints, Tpoints


def get_radzones(model):
    from kadi import events
    rad_zones = events.rad_zones.filter(start=model.datestart,
                                        stop=model.datestop)
    return rad_zones


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


class HistogramWindow(QtWidgets.QMainWindow):
    def __init__(self, model, hist_msids):
        super(HistogramWindow, self).__init__()
        self.setGeometry(0, 0, 1000, 600)
        self.model = model
        self.hist_msids = hist_msids
        self.which_msid = 0
        self.comp = self.model.comp[self.hist_msids[self.which_msid]]
        self.setWindowTitle("Histogram")
        wid = QtWidgets.QWidget(self)
        self.setCentralWidget(wid)
        self.box = QtWidgets.QVBoxLayout()
        wid.setLayout(self.box)

        self.rz_masked = False
        self.fmt1_masked = False
        self.show_limits = False

        canvas = MplCanvas(parent=None)
        toolbar = NavigationToolbar(canvas, parent=None)

        msid_select = QtWidgets.QComboBox()

        for msid in self.hist_msids:
            msid_select.addItem(msid)

        msid_select.activated[str].connect(self.change_msid)

        redraw_button = QtWidgets.QPushButton('Redraw')
        redraw_button.clicked.connect(self.make_plots)

        close_button = QtWidgets.QPushButton('Close')
        close_button.clicked.connect(self.close_window)

        toolbar_box = QtWidgets.QHBoxLayout()
        toolbar_box.addWidget(toolbar)
        toolbar_box.addStretch(1)

        mask_rz_check = QtWidgets.QCheckBox()
        mask_rz_check.setChecked(False)
        mask_rz_check.stateChanged.connect(self.mask_radzones)

        mask_fmt1_check = QtWidgets.QCheckBox()
        mask_fmt1_check.setChecked(False)
        mask_fmt1_check.stateChanged.connect(self.mask_fmt1)

        limits_check = QtWidgets.QCheckBox()
        limits_check.setChecked(False)
        limits_check.stateChanged.connect(self.plot_limits)
        if len(self.model.limits) == 0:
            limits_check.setEnabled(False)

        toolbar_box.addWidget(msid_select)
        toolbar_box.addWidget(redraw_button)
        toolbar_box.addWidget(close_button)

        check_boxes = QtWidgets.QHBoxLayout()
        check_boxes.addWidget(QtWidgets.QLabel('Mask radzones'))
        check_boxes.addWidget(mask_rz_check)
        check_boxes.addWidget(QtWidgets.QLabel('Mask FMT1'))
        check_boxes.addWidget(mask_fmt1_check)
        check_boxes.addWidget(QtWidgets.QLabel('Show limits'))
        check_boxes.addWidget(limits_check)
        check_boxes.addStretch(1)

        self.box.addWidget(canvas)
        self.box.addLayout(toolbar_box)
        self.box.addLayout(check_boxes)

        self.fig = canvas.fig
        self.canvas = canvas
        self.make_plots()
        self.canvas.show()

    def close_window(self, *args):
        self.close()

    _rz_mask = None
    @property
    def rz_mask(self):
        if self._rz_mask is None:
            self._rz_mask = np.ones_like(self.comp.dvals, dtype='bool')
            rad_zones = get_radzones(self.model)
            for rz in rad_zones:
                idxs = np.logical_and(self.model.times >= rz.tstart,
                                      self.model.times <= rz.tstop)
                self._rz_mask[idxs] = False
        return self._rz_mask

    _fmt1_mask = None
    @property
    def fmt1_mask(self):
        if self._fmt1_mask is None:
            fmt = self.model.fetch("ccsdstmf", 'vals', 'nearest')
            self._fmt1_mask = fmt != "FMT1"
        return self._fmt1_mask

    def mask_fmt1(self, state):
        self.fmt1_masked = state == QtCore.Qt.Checked
        self.make_plots()

    def mask_radzones(self, state):
        self.rz_masked = state == QtCore.Qt.Checked
        self.make_plots()

    def plot_limits(self, state):
        self.show_limits = state == QtCore.Qt.Checked
        self.make_plots()

    def change_msid(self, msid):
        self.which_msid = self.hist_msids.index(msid)
        self.comp = self.model.comp[self.hist_msids[self.which_msid]]
        self.make_plots()

    def make_plots(self):
        self.fig.clf()

        msid_name = self.hist_msids[self.which_msid]

        ax1 = self.fig.add_subplot(121)

        mask = np.ones_like(self.comp.resids, dtype='bool')
        if self.comp.mask:
            mask &= self.comp.mask.mask
        if self.rz_masked:
            mask &= self.rz_mask
        if self.fmt1_masked:
            mask &= self.fmt1_mask
        resids = self.comp.resids[mask]
        dvals = self.comp.dvals[mask]
        randx = self.comp.randx[mask]

        stats = calcquantiles(resids)
        # In this case the data is not discretized to a limited number of
        # count values, or has too many possible values to work with
        # calcquantstats(), such as with tmp_fep1_mong.
        if len(np.sort(list(set(dvals)))) > 1000:
            quantized_tlm = digitize_data(dvals)
            quantstats = calcquantstats(quantized_tlm, resids)
        else:
            quantstats = calcquantstats(dvals, resids)

        ax1.plot(resids, dvals + randx, 'o', color='#386cb0',
                 alpha=1, markersize=1, markeredgecolor='#386cb0')
        ax1.grid()
        ax1.set_title('{}: data vs. residuals (data - model)'.format(msid_name))
        ax1.set_xlabel('Error')
        ax1.set_ylabel('Temperature')
        Epoints01, Tpoints01 = getQuantPlotPoints(quantstats, 'q01')
        Epoints99, Tpoints99 = getQuantPlotPoints(quantstats, 'q99')
        Epoints50, Tpoints50 = getQuantPlotPoints(quantstats, 'q50')
        ax1.plot(Epoints01, Tpoints01, color='k', linewidth=4)
        ax1.plot(Epoints99, Tpoints99, color='k', linewidth=4)
        ax1.plot(Epoints50, Tpoints50, color=[1, 1, 1], linewidth=4)
        ax1.plot(Epoints01, Tpoints01, 'k', linewidth=2)
        ax1.plot(Epoints99, Tpoints99, 'k', linewidth=2)
        ax1.plot(Epoints50, Tpoints50, 'k', linewidth=1.5)

        if self.show_limits:
            self.model.annotate_limits(ax1)

        self.ax1 = ax1

        ax2 = self.fig.add_subplot(122)

        hist, bins = np.histogram(resids, 40)
        hist = hist*100.0/self.comp.mvals.size
        hist[hist == 0.0] = np.nan
        bin_mid = 0.5*(bins[1:]+bins[:-1])
        ax2.step(bin_mid, hist, '#386cb0', where='mid')
        ax2.set_title('{}: residual histogram'.format(msid_name), y=1.0)
        ax2.set_ylim(0.0, None)
        ylim2 = ax2.get_ylim()
        ax2.axvline(stats['q01'], color='k', linestyle='--', linewidth=1.5, alpha=1)
        ax2.axvline(stats['q99'], color='k', linestyle='--', linewidth=1.5, alpha=1)
        ax2.axvline(np.nanmin(resids), color='k', linestyle='--', linewidth=1.5, alpha=1)
        ax2.axvline(np.nanmax(resids), color='k', linestyle='--', linewidth=1.5, alpha=1)
        ax2.set_xlabel('Error')
        ax2.set_ylabel('% of data')
        ax2.fill_between(bin_mid, hist, step="mid", color='#386cb0')

        # Print labels for statistical boundaries.
        ystart = (ylim2[1] + ylim2[0]) * 0.5
        xoffset = -(.2 / 25) * np.abs(np.diff(ax2.get_xlim()))
        ax2.text(stats['q01'] + xoffset * 1.1, ystart, '1% Quantile', ha="right",
                 va="center", rotation=90)

        if np.min(resids) > ax2.get_xlim()[0]:
            ax2.text(np.min(resids) + xoffset * 1.1, ystart,
                     'Minimum Error', ha="right", va="center",
                     rotation=90)
        ax2.text(stats['q99'] - xoffset * 0.9, ystart, '99% Quantile', ha="left",
                 va="center", rotation=90)

        if np.max(resids) < ax2.get_xlim()[1]:
            ax2.text(np.max(resids) - xoffset * 0.9, ystart,
                     'Maximum Error', ha="left",
                     va="center", rotation=90)

        self.ax2 = ax2

        self.canvas.draw()


class PlotBox(QtWidgets.QVBoxLayout):
    def __init__(self, plot_name, plots_box):
        super(PlotBox, self).__init__()

        comp_name, plot_method = plot_name.split()  # E.g. "tephin fit_resid"
        self.comp = plots_box.model.comp[comp_name]
        self.plot_method = plot_method
        self.comp_name = comp_name
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
        self.main_window = self.plots_box.main_window
        self.selecter = self.canvas.mpl_connect("button_press_event", self.select)
        self.releaser = self.canvas.mpl_connect("button_release_event", self.release)

    def select(self, event):
        grab = event.inaxes and self.main_window.show_line and \
               self.canvas.toolbar._active is None
        if grab:
            self.mover = self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
            self.plots_box.xline = event.xdata
            self.plots_box.update_xline()
        else:
            pass

    def on_mouse_move(self, event):
        if event.inaxes and self.main_window.show_line:
            self.plots_box.xline = event.xdata
            self.plots_box.update_xline()
        else:
            pass

    def release(self, event):
        if hasattr(self, "mover"):
            self.canvas.mpl_disconnect(self.mover)

    def update_xline(self):
        if self.plot_name.endswith("time"):
            self.ly.set_xdata(self.plots_box.xline)
            self.canvas.draw_idle()

    def update(self, redraw=False, first=False):
        pb = self.plots_box
        mw = self.main_window
        plot_func = getattr(self.comp, 'plot_' + self.plot_method)
        if redraw:
            xmin, xmax = self.ax.get_xlim()
            ymin, ymax = self.ax.get_ylim()
            self.fig.delaxes(self.ax)
            try:
                xaxis_type = self.plot_method.split('__')[1]
            except IndexError:
                self.ax = self.fig.add_subplot(111)
            else:
                sharex = pb.sharex.get(xaxis_type)
                self.ax = self.fig.add_subplot(111, sharex=sharex)
                if sharex is not None:
                    self.ax.autoscale(enable=False, axis='x')
                pb.sharex.setdefault(xaxis_type, self.ax)
            self.ax.set_xlim(xmin, xmax)
            self.ax.set_ylim(ymin, ymax)

        plot_func(fig=self.fig, ax=self.ax)
        self.ax.fmt_xdata = mdates.DateFormatter("%Y:%j:%H:%M:%S")

        if redraw or first:
            times = pb.model.times
            tplot = pb.pd_times
            if self.plot_method.endswith("time"):
                ybot, ytop = self.ax.get_ylim()
                for t0, t1 in pb.model.mask_time_secs:
                    where = (times >= t0) & (times <= t1)
                    self.ax.fill_between(tplot, ybot, ytop, where=where,
                                         color='r', alpha=0.5)
                if mw.show_radzones:
                    rad_zones = get_radzones(pb.model)
                    for rz in rad_zones:
                        t0, t1 = cxctime2plotdate([rz.tstart, rz.tstop])
                        self.ax.axvline(t0, color='g', ls='--')
                        self.ax.axvline(t1, color='g', ls='--')
                if mw.show_line:
                    self.ly = self.ax.axvline(pb.xline, color='maroon')
            if mw.show_limits and self.comp_name == mw.msid:
                if self.plot_method.endswith("resid__data"):
                    pb.model.annotate_limits(self.ax, dir='v')
                elif self.plot_method.endswith("data__time"):
                    pb.model.annotate_limits(self.ax)

        self.canvas.draw()


class PlotsBox(QtWidgets.QVBoxLayout):
    def __init__(self, model, main_window):
        super(QtWidgets.QVBoxLayout, self).__init__()
        self.main_window = main_window
        self.sharex = {}        # Shared x-axes keyed by x-axis type
        self.model = model
        self.xline = 0.5*np.sum(cxctime2plotdate([self.model.tstart,
                                                  self.model.tstop]))
        self.pd_times = cxctime2plotdate(self.model.times)
        self.plot_boxes = []
        self.plot_names = []

    def add_plot_box(self, plot_name):
        plot_name = str(plot_name)
        if plot_name == "Add plot..." or plot_name in self.plot_names:
            return
        print('Adding plot ', plot_name)
        plot_box = PlotBox(plot_name, self)
        self.addLayout(plot_box)
        plot_box.update(first=True)
        self.main_window.cbp.add_plot_button.setCurrentIndex(0)
        self.update_plot_boxes()

    def delete_plot_box(self, plot_name):
        for plot_box in self.findChildren(PlotBox):
            if plot_box.plot_name == plot_name:
                self.removeItem(plot_box)
                clearLayout(plot_box)
        self.update()
        self.update_plot_boxes()

    def update_plots(self, redraw=False):
        cbp = self.main_window.cbp
        cbp.update_status.setText(' BUSY... ')
        self.model.calc()
        for plot_box in self.plot_boxes:
            plot_box.update(redraw=redraw)
        cbp.update_status.setText('')

    def update_plot_boxes(self):
        self.plot_boxes = []
        self.plot_names = []
        for plot_box in self.findChildren(PlotBox):
            self.plot_boxes.append(plot_box)
            self.plot_names.append(plot_box.plot_name)

    def update_xline(self):
        for pb in self.plot_boxes:
            pb.update_xline()
        self.main_window.line_data_window.update_data()
