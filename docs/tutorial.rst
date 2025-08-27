Tutorial
=============

Setup for Xija modeling
------------------------

When you first start working with Xija create a local copy of the Xija source code::

  % mkdir -p ~/git  # OR WHEREVER, but ~/git is easiest!
  % cd ~/git
  % git clone git://github.com/sot/xija.git  # on HEAD
  % git clone /proj/sot/ska/git/xija         # on GRETA
  % cd xija
  % setenv XIJA $PWD
  % python setup.py build_ext --inplace  # build C core module

Later on you should work in your xija repository and update to the latest development version of Xija::

  % cd ~/git/xija
  % git pull   # Update with latest dev version of xija
  % python setup.py build_ext --inplace  # build C core module

Finally set the PYTHONPATH environment variable to ensure that you import
your local version of xija from any sub-directory where you might be
working::

  % setenv PYTHONPATH $XIJA

Navigating the Xija source
---------------------------

The `Xija source <http://github.com/sot/xija>`_ is always available at `github
<http://github.com>`_.  Within the **Files** tab you will find a directory
browser.  At the top level you will see the ``xija`` directory that contains
the actual Xija package files.  There is also ``gui_fit.py`` that is the GUI
model fitting tool.  Within the ``xija`` directory the key files are::

  model.py             Top-level model class and functionality
  component/           Model components directory
            base.py    Base components (Node, HeatSink, TelemData, etc)
            heat.py    Heat components (SolarHeat, passive and active heaters)
            mask.py    Mask components

Creating and understanding models
----------------------------------

The example models show here are available in the ``examples/doc/`` directory of the Xija git repository.

Each model component is handled by a
separate Python class.  Some currently implemented examples include:

* :class:`~xija.component.base.ModelComponent` : model component base class (name, parameter methods)
* :class:`~xija.component.base.Node` : single node with a temperature, sigma, data_quantization, etc
* :class:`~xija.component.base.Coupling` : Couple two nodes together (one-way coupling)
* :class:`~xija.component.base.HeatSink` : Fixed temperature external heat bath
* :class:`~xija.component.heat.SolarHeat` : Solar heating (pitch dependent)
* :class:`~xija.component.heat.EarthHeat` : Earth heating of ACIS cold radiator (attitude, ephem dependent)
* :class:`~xija.component.heat.PropHeater` : Proportional heater (P = k * (T - T_set) for T > T_set)
* :class:`~xija.component.heat.ThermostatHeater` : Thermostat heater (with configurable deadband)
* :class:`~xija.component.heat.AcisDpaStatePower` : Heating from ACIS electronics (ACIS config dependent CCDs, FEPs etc)

Example 1: simplest model
^^^^^^^^^^^^^^^^^^^^^^^^^

Start with the simplest example with a single node with solar heating.  We use only two
bin points at 45 and 180 degrees.
::

  model = xija.XijaModel(name, start='2015:001', stop='2015:050')

  model.add(xija.Node, 'aacccdpt')

  model.add(xija.Pitch)

  model.add(xija.Eclipse)

  model.add(xija.SolarHeat,
            node='aacccdpt',
            pitch_comp='pitch',
            eclipse_comp='eclipse',
            P_pitches=[45, 180],
            Ps=[0.0, 0.0],
            ampl=0.0,
            epoch='2010:001',
           )

To make and run the model do::

  % cd $XIJA/examples/doc
  % python example1.py
  % xija_gui_fit example1.json

Points for discussion:

* What is fundamentally wrong with this model?

Example 2: add a heat sink
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Same as example 1, but add a heat sink with a temperature of -16 C and a tau of 30 ksec.
::

  model.add(xija.HeatSink,
            node='aacccdpt',
            tau=30.0,
            T=-16.0,
           )

To make and run the model do::

  % cd $XIJA/examples/doc
  % python example2.py
  % xija_gui_fit example2.json

Points for discussion:

* Twiddle each fittable parameter and observe the response.
* Use a longer interval ``xija_gui_fit example2.json --stop=2015:240 --days=400``
  for dP and solar amplitude.
* Discuss epoch: ``xija_gui_fit example2.json --stop=2015:240 --days=400 --keep-epoch``.
  It is important to verify that SolarHeat epoch is explicitly in JSON file in order
  to have auto-epoch updating.  This should be an ``"epoch"`` field in the ``"init_kwargs"``
  element of ``SolarHeat`` components. (Note: ``SolarHeatOffNomRoll`` is a bit different
  and does not have an epoch).

Example 3: add pitch bins
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Same as example 2, but now the ``SolarHeat`` component has 6 pitch bins::

  model.add(xija.SolarHeat,
            node='aacccdpt',
            pitch_comp='pitch',
            eclipse_comp='eclipse',
            [45, 70, 90, 115, 140, 180],
            [0.0] * 6,
            ampl=0.0,
            epoch='2010:001',
           )

To make and run the model do::

  % cd $XIJA/examples/doc
  % python example3.py
  % xija_gui_fit example3.json --stop=2015:240 --days=400

Points for discussion:

* Fit the model

  * Naive try.
  * Set heat sink time scale

* Managing degenerate model parameters (heatsink T, solarheat bias, solarheat P values).
* But note: eclipse data breaks degeneracy.  This can be used for short-timescale components.
* Save the best fit as ``example3_fit.json``


Working with a model
---------------------

As an example, here is the code (available in ``examples/dpa/plot_dpa_resid.py``) to plot
residuals versus temperature for the ACIS DPA model.  You can run this with
``cd examples/dpa; python plot_dpa_resid.py``.
::

  import xija
  import numpy as np
  import matplotlib.pyplot as plt
  from Ska.Matplotlib import pointpair

  start = '2010:001'
  stop = '2011:345'

  msid = '1dpamzt'
  model_spec = 'dpa.json'

  model = xija.XijaModel('dpa', start=start, stop=stop,
                            model_spec=model_spec)
  model.make()
  model.calc()

  dpa = model.get_comp(msid)
  resid = dpa.dvals - dpa.mvals

  xscatter = np.random.uniform(-0.2, 0.2, size=len(dpa.dvals))
  yscatter = np.random.uniform(-0.2, 0.2, size=len(dpa.dvals))
  plt.clf()
  plt.plot(dpa.dvals + xscatter, resid + yscatter, '.', ms=1.0, alpha=1)
  plt.xlabel('{} telemetry (degC)'.format(msid.upper()))
  plt.ylabel('Data - Model (degC)')
  plt.title('Residual vs. Data ({} - {})'.format(start, stop))

  bins = np.arange(6, 26.1, 2.0)
  r1 = []
  r99 = []
  ns = []
  xs = []
  for x0, x1 in zip(bins[:-1], bins[1:]):
      ok = (dpa.dvals >= x0) & (dpa.dvals < x1)
      val1, val99 = np.percentile(resid[ok], [1, 99])
      xs.append((x0 + x1) / 2)
      r1.append(val1)
      r99.append(val99)
      ns.append(sum(ok))

  xspp = pointpair(bins[:-1], bins[1:])
  r1pp = pointpair(r1)
  r99pp = pointpair(r99)

  plt.plot(xspp, r1pp, '-r')
  plt.plot(xspp, r99pp, '-r', label='1% and 99% limits')
  plt.grid()
  plt.ylim(-8, 14)
  plt.xlim(5, 31)

  plt.plot([5, 31], [3.5, 3.5], 'g--', alpha=1, label='+/- 3.5 degC')
  plt.plot([5, 31], [-3.5, -3.5], 'g--', alpha=1)
  for x, n, y in zip(xs, ns, r99):
      plt.text(x, max(y + 1, 5), 'N={}'.format(n),
           rotation='vertical', va='bottom', ha='center')

  plt.legend(loc='upper right')

  plt.savefig('dpa_resid_{}_{}.png'.format(start, stop))

.. Note::

   ``ThermalModel`` is a synonym for ``XijaModel`` available for back-compatibility,
   but new code should use ``XijaModel``.

Modifying an existing model
----------------------------

Much of the time the best way to create a new model is to start from an
existing model.  There are a few strategies for doing this:

* Extend an existing model at the Python API level
* Create a new model in Python and inherit existing model parameters
* Directly edit the model JSON specification
* Convert the model spec to Python and edit the Python

Extend an existing model
^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have an existing model (e.g. ``pcm03t`` from the previous examples) and
want to extend it by adding a model component, the technique is to read in the
model,  add the component, make the model, and then write out the new model.
This is illustrated in the `Xija extend model
<xija_extend_model.html>`_ notebook.

Inherit from an existing model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This option provides a way to use some of the existing (calibrated) components
from an existing model.  In particular if you want to remove a component this
is one way to do it. This is illustrated in the `Xija inherit
<xija_inherit.html>`_ IPython notebook.

Edit the model specification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Xija models are stored in a file format called `JSON
<http://en.wikipedia.org/wiki/JSON>`_.  This captures the model definition,
model parameters, and also everything about the GUI fit application (screen
size, plots, frozen / thawed parameters) when the model was saved.  

Although it requires a bit of care, sometimes the easiest way to produce a
derived model is by directly editing the JSON model specification.  

Convert model spec back to Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A very good way to modify an existing model spec is to write it back out as
Python code. This can be done in three ways:

* Within ``xija_gui_fit`` save the model with a name ending in ``.py``
* Within a Python session or script use the ``write()`` method of a Xija model::

    model = xija.XijaModel('mdl', model_spec='mdl.json')
    model.write('mdl.py')

* From the command line use the `xija.convert` module::

    % python -m xija.convert --help
    % python -m xija.convert mdl.json

Bad Times
---------

If there are one or more intervals of time where the data are effectively
bad for fitting (i.e. the thermal model is not expected to predict accurately
due to off-nominal spacecraft configuration), then one can add a ``bad_times``
tag to the JSON model file.  This would like::

  {
      "bad_times": [
          [
              "2014:001",
              "2014:003"
          ],
          [
              "2014:010",
              "2014:013"
          ]
      ],
      "comps": [
          {
              "class_name": "Mask",
              "init_args": [
                  "1dpamzt",
                  "gt",
                  20.0
              ],
              "init_kwargs": {},
              "name": "mask__1dpamzt_gt"
          },
      ...


Exercises
-----------

The exercise for both teams will be to first get familiar with the GUI fit tool
by playing with an existing calibrated model.  Do one of the following::

  % cp ~aldcroft/git/xija/examples/dpa/dpa.json ./          # ACIS
  % cp ~aldcroft/git/xija/examples/minusz/minusz.json ./    # Spacecraft

You will run ``xija_gui_fit`` specifying the stop time as ``2012:095`` and
the number of days to fit as ``90``.

Then do the following:

* Explore the different available plots.
* Try moving various sliders and see how it affects the model.
* Try fitting various parameter sets using both the check boxes and the glob
  tool to freeze and thaw.

Team ACIS
^^^^^^^^^^

**Goal**: Make a model for 1DEAMZT that is analogous to the 1DPAMZT model.

Choose the best way to derive a DEA model from the DPA model.

Team Spacecraft
^^^^^^^^^^^^^^^^

**Goal**: Make a working model for PCM03T.

The first step will be to calibrate the PCM03T model that we have created
which uses TCYLAFT6 and TCYLFMZM as known inputs.  The second step will be to
integrate the PCM03T model into the MinusZ model.

  % cp ~aldcroft/git/xija/examples/pcm/pcm.json ./    # Spacecraft

