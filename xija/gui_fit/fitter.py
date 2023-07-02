import logging
import multiprocessing as mp
import time

import numpy as np
from sherpa import ui

logging.basicConfig(level=logging.INFO)

fit_logger = logging.getLogger("fit")

# Default configurations for fit methods
sherpa_configs = {
    "simplex": {
        "ftol": 1e-3,
        "finalsimplex": 0,  # converge based only on length of simplex
        "maxfev": None,
    },
}


class FitTerminated(Exception):
    pass


class CalcModel:
    def __init__(self, model):
        self.model = model

    def __call__(self, parvals, x):
        """This is the Sherpa calc_model function, but in this case calc_model does not
        actually calculate anything but instead just stores the desired paramters.  This
        allows for multiprocessing where only the fit statistic gets passed between nodes.
        """
        fit_logger.info("Calculating params:")
        for parname, parval, newparval in zip(
            self.model.parnames, self.model.parvals, parvals, strict=False
        ):
            if parval != newparval:
                fit_logger.info("  {0}: {1}".format(parname, newparval))
        self.model.parvals = parvals

        return np.ones_like(x)


class CalcStat:
    def __init__(self, model, pipe, maxiter):
        self.pipe = pipe
        self.model = model
        self.cache_fit_stat = {}
        self.min_fit_stat = None
        self.min_parvals = self.model.parvals
        self.niter = 0
        self.maxiter = maxiter

    def __call__(self, _data, _model, staterror=None, syserror=None, weight=None):
        """Calculate fit statistic for the xija model.  The args _data and _model
        are sent by Sherpa but they are fictitious -- the real data and model are
        stored in the xija model self.model.
        """
        fit_stat = self.model.calc_stat()
        fit_logger.info("Fit statistic: %.4f" % fit_stat)

        if self.min_fit_stat is None or fit_stat < self.min_fit_stat:
            self.min_fit_stat = fit_stat
            self.min_parvals = self.model.parvals

        self.message = {
            "status": "fitting",
            "time": time.time(),
            "parvals": self.model.parvals,
            "fit_stat": fit_stat,
            "min_parvals": self.min_parvals,
            "min_fit_stat": self.min_fit_stat,
        }
        self.pipe.send(self.message)

        while self.pipe.poll():
            pipe_val = self.pipe.recv()
            if pipe_val == "terminate":
                self.model.parvals = self.min_parvals
                raise FitTerminated("terminated")

        self.niter += 1
        if self.niter >= self.maxiter:
            fit_logger.warning(
                "Reached maximum number of iterations: %d" % self.maxiter
            )
            self.model.parvals = self.min_parvals
            raise FitTerminated("terminated")

        return fit_stat, np.ones(1)


class FitWorker:
    def __init__(self, model, maxiter, method="simplex"):
        self.model = model
        self.method = method
        self.parent_pipe, self.child_pipe = mp.Pipe()
        self.maxiter = maxiter

    def start(self, widget=None):
        """Start a Sherpa fit process as a spawned (non-blocking) process.

        Parameters
        ----------
        widget :
             (Default value = None)

        Returns
        -------

        """
        self.fit_process = mp.Process(target=self.fit)
        self.fit_process.start()
        fit_logger.info("Fit started")

    def terminate(self, widget=None):
        """Terminate a Sherpa fit process in a controlled way by sending a
        message.  Get the final parameter values if possible.

        Parameters
        ----------
        widget :
             (Default value = None)

        Returns
        -------

        """
        if hasattr(self, "fit_process"):
            # Only do this if we had started a fit to begin with
            self.parent_pipe.send("terminate")

    def fit(self):
        dummy_data = np.zeros(1)
        dummy_times = np.arange(1)
        ui.load_arrays(1, dummy_times, dummy_data)
        ui.set_method(self.method)
        ui.get_method().config.update(sherpa_configs.get(self.method, {}))
        ui.load_user_model(CalcModel(self.model), "xijamod")  # sets global xijamod
        ui.add_user_pars("xijamod", self.model.parnames)
        ui.set_model(1, "xijamod")
        calc_stat = CalcStat(self.model, self.child_pipe, self.maxiter)
        ui.load_user_stat("xijastat", calc_stat, lambda x: np.ones_like(x))
        ui.set_stat(xijastat)  # type: ignore  # noqa: F821, PGH003

        # Set frozen, min, and max attributes for each xijamod parameter
        for par in self.model.pars:
            xijamod_par = getattr(xijamod, par.full_name)  # type: ignore  # noqa: F821, PGH003
            xijamod_par.val = par.val
            xijamod_par.frozen = par.frozen
            xijamod_par.min = par.min
            xijamod_par.max = par.max

        if any(not par.frozen for par in self.model.pars):
            try:
                ui.fit(1)
                calc_stat.message["status"] = "finished"
                fit_logger.info("Fit finished normally")
            except FitTerminated as err:
                calc_stat.message["status"] = "terminated"
                fit_logger.warning("Got FitTerminated exception {}".format(err))

        self.child_pipe.send(calc_stat.message)
