.. xija documentation master file, created by
   sphinx-quickstart on Fri Mar  9 21:17:17 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Xija modeling framework
================================

The Xija modeling framework is used to create, calibrate, and compute
time-series models.  This package provides a generalized framework to model
complex time series data using a network of coupled nodes with pluggable model
components that define the node interactions.  Systems that can be represented
by a set of coupled first-order differential equations are candidates for Xija
modeling.

At present the model components include thermal conduction and passive and
active heating elements, but the framework itself is fairly general and could
be used for other applications.  A key feature is a GUI fitting application
that allows for rapid evaluation of model fit results and interactive
many-parameter fits of large time-series datasets using the Sherpa fitting
package.

.. toctree::
   :maxdepth: 2

   design
   gui_fit
   tutorial
   api

