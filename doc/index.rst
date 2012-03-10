.. xija documentation master file, created by
   sphinx-quickstart on Fri Mar  9 21:17:17 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to xija's documentation!
================================

Requirements
-------------

* Modular and extensible modeling framework
* Single-step integration instead of analytic state-based solutions
* Model definition via Python code or static data structure
* Interactive and iterative model development and fitting
* Predictively model a node or use telemetry during development
* GUI interface for model development
* Matlab interface

Modular and extensible
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``xija`` package provides functions and classes to assemble and calculate a thermal model.  Each model component is handled by a separate Python class.

* ``ModelComponent`` : model component base class (name, parameter methods)
* ``Node(ModelComponent)`` : single node with a temperature, sigma, data_quantization, etc
* ``Coupling(ModelComponent)`` : Couple two nodes together (one-way coupling)
* ``HeatSink(ModelComponent)`` : Fixed temperature external heat bath
* ``HeatPower(ModelComponent)`` : component that provides a direct heat power input
* ``SolarHeat(HeatPower)`` : Solar heating (pitch dependent)
* ``EarthHeat(HeatPower)`` : Earth heating of ACIS cold radiator (attitude, ephem dependent)
* ``ProportialHeater(HeatPower)`` : Proportional heater (P = k * (T - T_set) for T > T_set)
* ``ThermostatHeater(HeatPower)`` : Thermostat heater (with configurable deadband)
* ``AcisPower(HeatPower)`` : Heating from ACIS electronics (ACIS config dependent CCDs, FEPs etc)

As needed additional model components can be added.

Single-step integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The very fast state-based analytic solutions used for purely passive models cannot accomodate model components that depend on the node temperatures or are continuously variable.  Instead the Xija framework uses an integration method such as 4th order Runge-Kutte (TBD) to propagate the node temperatures.  Model components such as heaters respond to the most-recently calculated temperatures.

The initial implementation is pure Python.  As needed other options is explored for the inner integration loop.  Possibilities include Cython (a compiled Python/C hybrid), hand-coded C (general model implementation), and machine-generated C (based on specific model configuration). 

Model definition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The class-based framework will make it natural to define a model and do interactive parameter fitting within the Python language.  At the same time one needs to store the results of model fitting and potentially iterate the fit process starting with stored parameter values.  One idea is to store only the fitted parameter values in an output file and maintain the model definition as an importable code module.  This is TBR since such a strategy might be difficult if a GUI is used to create a model graphically from scratch.

Interactive and iterative fitting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The framework will provide methods to help with visualization of fit results and parameter values.  It will allow for interactive fitting using CIAO/Sherpa by freezing or thawing various parameters or groups of parameters.  The Sherpa fitting functionality is separated from the model evaluation code.

Facilities for taking advantage of multiple cores and hosts for parallel fitting using MPI + mpi4py is included.

Predictively model a node or use telemetry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One of the key methods for initially narrowing the parameter space in a complex model is to fit parameters and predict values for a single node only and use _telemetry_ (aka truth) values for the other coupled nodes.  The Xija framework will easily allow nodes to be enabled or disabled from the model fitting and prediction process.  In this way a complex model can be gradually built up.  

GUI interface (optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The underlying Xija framework will support layering of a GUI model design and fitting application on top.  There is potential benefit to presenting a simpler interface to the modeling framework.  This needs to be weighed against developer time and effort.

API docs
---------

.. automodule:: xija

Classes
^^^^^^^^^

.. autoclass:: XijaModel
   :show-inheritance:
   :members:
   :undoc-members:

.. autoclass:: ModelComponent
   :show-inheritance:
   :members:
   :undoc-members:
