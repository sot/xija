Design overview
==================

The key requirements that drive the Xija design are the following:

* Solve coupled first order differential equations:

.. image:: first_order_ode_dark.png
   :width: 25 %

* The A coupling matrix and B vector can both depend on time, the output data
  values Y and the model parameters p
* Handle time series up to ~1e6 elements long.
* Choice of minimization algorithms (fast versus robust)
* Handle models with up to ~100 parameters

Xija features
-----------------
* Modular and extensible
* Model definition via Python code or static JSON data structure
* Interactive and iterative model development and fitting
* Switch between predicting nodes or using truth (training) data
* Key integration steps coded in C for speed
* GUI interface for model fitting
* Fitting engine provided by `Sherpa <http://cxc.harvard.edu/contrib/sherpa>`_.


Modular and extensible
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``xija`` package provides functions and classes to assemble and
calculate a thermal model.  

At the top level there is a single class :class:`xija.XijaModel` that
encapsulates the key information about a model including the model components,
model parameters, and the times at which the model is evaluated.

Each model component is handled by a
separate Python class.  Some currently implemented examples include:

* :class:`~xija.ModelComponent` : model component base class (name, parameter methods)
* :class:`~xija.Node` : single node with a temperature, sigma, data_quantization, etc
* :class:`~xija.Coupling` : Couple two nodes together (one-way coupling)
* :class:`~xija.HeatSink` : Fixed temperature external heat bath
* :class:`~xija.SolarHeat` : Solar heating (pitch dependent)
* :class:`~xija.EarthHeat` : Earth heating of ACIS cold radiator (attitude, ephem dependent)
* :class:`~xija.PropHeater` : Proportional heater (P = k * (T - T_set) for T > T_set)
* :class:`~xija.ThermostatHeater` : Thermostat heater (with configurable deadband)
* :class:`~xija.AcisDpaStatePower` : Heating from ACIS electronics (ACIS config dependent CCDs, FEPs etc)

As needed additional model components can be added.

Single-step integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The very fast state-based analytic solutions used for purely passive
models cannot accomodate model components that depend on the node
temperatures or are continuously variable.  Instead the Xija framework
uses 2nd order Runge-Kutte integration to propagate the node
temperatures.  Model components such as heaters respond to the
most-recently calculated temperatures.  The integration code is
written in C for performance.

Model definition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The class-based framework makes it natural to define a model and do
interactive parameter fitting within the Python language.  At the same
time one needs to store the results of model fitting and potentially
iterate the fit process starting with stored parameter values.  This
is done by saving the model definition, fit parameters, and other
relevant fit meta-data to a JSON file.

Interactive and iterative fitting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The framework provides a GUI fitting tool to help with visualization
of fit results and parameter values.  This allows for interactive
fitting using CIAO/Sherpa by freezing or thawing various parameters or
groups of parameters.  The Sherpa fitting functionality is separated
from the model evaluation code.

Predictively model a node or use telemetry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One of the key methods for initially narrowing the parameter space in
a complex model is to fit parameters and predict values for a single
node only and use truth values for the other coupled nodes.  The Xija
framework easily allows nodes to be enabled or disabled from the model
fitting and prediction process.  In this way a complex model can be
gradually built up.
