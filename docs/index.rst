.. mctpy documentation master file, created by
   sphinx-quickstart on Mon Sep  9 11:42:31 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

mctpy: Mode-Coupling Theory Solver in Python
=============================================


This module provides a python implementation of numerical routines for the
mode-coupling theory of the glass transition (MCT). In short, MCT is a
theory that predicts dynamical correlation functions for highly viscous
fluids, as solutions of certain integro-differential equations that are
solved numerically. `mctpy` supports this numerical solution alongside a
number of utility helpers that are often used in conjunction with MCT.

More information on MCT can be found in the standard references.

The core functionality of `mctpy` is the numerical treatment of evolution
equations of the form

.. math::

    \tau_0(q)\partial_t\phi(q,t) + \phi(q,t)
    + \int_0^t m(q,t-t')\partial_{t'}\phi(q,t')\,dt' = 0

solved as an initial value problem for the unknown :math:`\phi(q,t)` at
a set of given indices :math:`q` (in physical terms: the density correlation
function to a given wave number), where the memory kernel :math:`m(q,t)`
is itself a (usually nonlinear) functional of the sought-after solution,

.. math::

    m(q,t)=\mathcal F[\phi(q,t)]

The nonlinearity of this self-consistent closure causes a bifurcation
scenario, where the long-time limit :math:`f(q)=\lim_{t\to\infty}\phi(q,t)`
can show discontinuities upon a smooth change of the parameters entering
the equations. Close to such a bifurcation, the relaxation times of both
the solution and the memory kernel become arbitrarily large, and hence
adapted numerical integration schemes are needed to treat the MCT equations.

At the moment, `mctpy` is not yet a fully performant solver. It makes
heavy use for the `numba` just-in-time compiler, but this leads to relatively
long warm-up times when running even simple code. This code base is mostly
intended to provide an easy entry point for people wanting to understand,
use, modify and extend the core numerics behind MCT.


Overview
--------

The code is structured in two major groups of classes: "correlators"
and "models". Think of the correlators defining their equation of motion,
and the models defining the specific form of the memory kernel.

In particular this means that the correlator classes implement the
suitable solvers for their equations, while the model classes implement
the numerics to evaluate the mode-coupling functionals.
Often, there is a close correspondence between a correlator/solver and
a specific model, but in principle, the correlator class is more generic.

Most MCT functionals take as input the static structure factor. While
these are not strictly part of MCT, we also introduce a group of
corresponding "structurefactor" classes that implement some commonly
used approximations for these. Also, some utility methods, for example
to perform Fourier transforms of the time-domain correlators to obtain
their spectra, or the calculation of MCT's exponent parameter, are
implemented.


Contributors
------------

While this specific implementation is derived from code developed initially
during the PhD thesis of Thomas Voigtmann, it is the collected effort
of many people working on MCT over many decades. Contributions are in
particular acknowledged from

- Matthias Fuchs: initial idea for the "moment algorithm" (together with Ivo Hofacker), a lot of building blocks taken from old Fortran code, and code for sheared systems.
- Thomas Franosch: many discussions around the matrix code, non-uniform wave-number grids and the two-dimensional version of the theory.
- Liesbeth Janssen and her group: for pushing the development of modern code through their development of Julia-based MCT and GMCT code.
- Mathias Mayr: for valuable cross-checking, non-Gaussian parameter code.
- Tommaso Rizzo: for the development of multi-dimensional SBR code.



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   correlators
   models
   structurefactors
   fsolver
   util
   impl
   examples


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
