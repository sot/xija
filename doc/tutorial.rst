Tutorial
=============

Setup for Xija modeling
------------------------

When you first start working with Xija create a local copy of the Xija source code::

  % mkdir -p ~/git
  % cd ~/git
  % git clone git://github.com/sot/xija.git  # on HEAD
  % git clone /proj/sot/ska/git/xija         # on GRETA
  % cd xija
  % python setup.py build_ext --inplace  # build C core module

Later on you should work in your xija repository and update to the latest development version of Xija::

  % cd ~/git/xija
  % git pull   # Update with latest dev version of xija
  % python setup.py build_ext --inplace  # build C core module

Finally set the PYTHONPATH environment variable to ensure that you import
your local version of xija from any sub-directory where you might be
working::

  % setenv PYTHONPATH $PWD

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

Points for discussion:

* Twiddle each fittable parameter and observe the response.
* Use a longer interval `./gui_fit.py example2.json --stop=2015:240 --days=400`
  for dP and solar amplitude.
* Discuss epoch: `./gui_fit.py example2.json --stop=2015:240 --days=400 --keep-epoch`.
  It is important to verify that SolarHeat epoch is explicitly in JSON file in order
  to have auto-epoch updating.  This should be an ``"epoch"`` field in the ``"init_kwargs"``
  element of ``SolarHeat`` components.  (Note: ``SolarHeatOffNomRoll`` is a bit different
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
Python code.  This can be done in three ways:

* Within ``gui_fit.py`` save the model with a name ending in ``.py``
* Within a Python session or script use the ``write()`` method of a Xija model::

    model = xija.XijaModel('mdl', model_spec='mdl.json')
    model.write('mdl.py')

* From the command line use the `xija.convert` module::

    % python -m xija.convert --help
    % python -m xija.convert mdl.json

Fitting a model
----------------

So far we have been manually working with a Xija model to understand a bit of
what is going on underneath and know how to make performance predictions.
However, the key task of actually calibrating the model parameters is done with
the ``gui_fit.py`` application.

GUI fit overview

The image below shows an example of fitting the ACIS DPA model with
``gui_fit.py``.

.. image:: gui_fit_guide.png
   :width: 100 %


Live demo using a Ska window::

  cd ~/git/xija/examples/pcm
  ../../gui_fit.py pcm.json --stop 2012:095 --days 30

Command line options
^^^^^^^^^^^^^^^^^^^^^

The GUI fit tool supports the following command line options::

  % ./gui_fit.py --help
  usage: gui_fit.py [-h] [--days DAYS] [--stop STOP] [--nproc NPROC]
                    [--fit-method FIT_METHOD] [--inherit-from INHERIT_FROM]
                    [--set-data SET_DATA_EXPRS] [--quiet] [--keep-epoch]
                    filename

  positional arguments:
    filename              Model file

  optional arguments:
    -h, --help            show this help message and exit
    --days DAYS           Number of days in fit interval (default=90
    --stop STOP           Stop time of fit interval (default=model values)
    --nproc NPROC         Number of processors (default=1)
    --fit-method FIT_METHOD
                          Sherpa fit method (simplex|moncar|levmar)
    --inherit-from INHERIT_FROM
                          Inherit par values from model spec file
    --set-data SET_DATA_EXPRS
                          Set data value as '<comp_name>=<value>'
    --quiet               Suppress screen output
    --keep-epoch          Maintain epoch in SolarHeat models (default=recenter
                          on fit interval)

Most of the time you should use the ``--days`` and ``--stop`` options.  Note that
if you have saved a model specification and then restart ``gui_fit.py``, the
most recently specified values will be used by default.

``--nproc``
  This option has not been tested recently though it might work.

``--fit-method``
  The default fit method is ``simplex`` which is a good compromise between speed
  and completeness.  For the fastest fitting use ``levmar``.  If already have
  somewhat decent parameters and want to try to refine for the very best fit
  then select ``moncar``.  However, do not choose this option with more than
  about 10 or 15 free parameters as it can take a long time.  Typically with
  ``moncar`` you need to start the fitting and then do something else for a
  while (many hours or more).  

``--inherit-from``
  This provides a way to construct a model which is similar to an existing
  model but has some differences.  All the model parameters which are 
  exactly the same will be taking from the inherited model specification.
 
Assuming you have created a model specification file ``my_model_spec.json``
then a typical calling sequence from the Xija source directory is::

  ./gui_fit.py --stop 2012:002 --days 180 my_model_spec.json


Manipulating plots
^^^^^^^^^^^^^^^^^^^^

Many model components have built-in plots that can be added to the fit window
via the ``Add plots...`` drop down menu.  The available plot names correspond to the
model component followed by a description of the plot.  Plots can be deleted by
pressing the corresponding ``Delete`` button.

One handy feature is that the time-based plots are always linked in the time
axis so that if you zoom in to one then all plots zoom accordingly.  When you
want to go back to the full view you can use the ``Home`` button on the plot
where you originally zoomed.

Manipulating parameters
^^^^^^^^^^^^^^^^^^^^^^^^^

One of the key features of the GUI fit tool is the ability to visualize and
manipulate the dozens of parameters in a typical Xija model.  

The parameters are on the right side panel.  Each one has a checkbox that
indicates whether it will be fit (checked) or not (unchecked).  The value is
shown, then the minimum allowed fit value, a slider bar to select the value,
and then the maximum allowed fit value.  As you change the slider the model
will be recalculated and the plots updated.  It helps to make the GUI fit
window as wide as possible to make the sliders longer.

If you want to change the min or max values just type in the box and then hit
enter.  (If you don't hit enter the new value won't apply).

You can freeze or thaw many parameters at once using the "glob" syntax in the
entry box at the top of the fit window.  Examples::

  thaw *                 # thaw all parameters
  freeze solarheat*      # freeze all the solarheat params
  freeze solarheat*_dP_* # freeze the long-term solarheat variation params

Fit strategy
^^^^^^^^^^^^^^

Fitting Xija models is a bit of an art and will it take some time to develop
skill here.  A few rules of thumb and tips:

* Start with all long-term variations frozen.  You want to begin with a single
  relatively short epoch (perhaps 2-3 months) that is centered on the model
  epoch.  The model epoch is typically defined in the solarheat component and
  defaults to 2010:001. Start by try to get the model in the
  right ballpark. Typically this means::

    freeze solarheat_*_dP_*
    freeze solarheat_*_tau
    freeze solarheat_*_ampl
    thaw solarheat_*_P_*
    thaw heatsink_*
    thaw coupling_*

* Almost always have the ``solarheat_*_bias`` terms frozen at 0.  This
  parameter is degenerate with the ``solarheat_*_P_*`` values and is used for
  certain diagnostics.

* Once you have a model that fits reasonably well over a 3-month time period
  then freeze all parameters *except* for ``solarheat_*_dP_*``.  Fit over
  a 3-month time period which is at least a couple of years separated from
  the initial fit epoch.

* Next do a fit for at least a year (but preferably more depending on the model
  complexity).  This time also thaw the ``solarheat_*_dP_*`` and
  ``solarheat_*_ampl`` parameters.  You might want to refine the
  ``solarheat_*_P_*`` parameters at this point by thawing those ones and
  freezing the long-term parameters and fitting.  Remember that if the
  time span is not long enough then ``P`` and ``dP`` are degenerate and
  the fit may not converge.

* Finally you can re-freeze all the ``solarheat_*_dP_*`` and
  ``solarheat_*_P_*`` parameters and try to nail the very long term behavior
  by fitting for just the ``solarheat_*_tau`` and ``solarheat_*_ampl`` params
  for 5 years of data.  Beyond that is probably not useful because of 
  changes on-board that probably are not captured by the model.

* It can be useful to include long normal-sun dwells in the fitting to have
  some high-temperature data in the fit dataset.

* Remember to save your model fit when you get a good fit.  It is not saved by
  default and there is currently no warning to this effect.  Often there is a
  progression of model fits and it may be useful to incrementally number the
  models, e.g. ``pcm03t_1.json``, ``pcm03t_2.json``, etc.  By convention the
  final "flight" models that get configured are called
  ``<modelname>_model_spec.json``, so avoid using this name during development.

* Saving also saves the state of plots and your parameters.

Example::

  # Initial fit for solarheat and coupling parameters.  Save as minusz_2.json
  ./gui_fit.py minusz/minusz.json --stop 2010:045 --days 90

  # Initial fit for long term variation.  Save as minusz_3.json
  ./gui_fit.py minusz/minusz_2.json --stop 2012:095 --days 90

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

You will run ``gui_fit.py`` specifying the stop time as ``2012:095`` and
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

