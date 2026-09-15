.. image:: ./images/uvex_logo_dark.png
   :width: 200px
   :align: center

.. _api:

API
===

This page provides the complete reference for all public classes, functions, and modules in the
UVEX Transients codebase.

Models
------
Spectral and light-curve models for individual transient classes, built on a common
:mod:`~uvex_transients.models.core` SED framework.

.. autosummary::
    :toctree: _as_gen
    :recursive:
    :template: module.rst

    uvex_transients.models

Transients
----------
Transient population classes, pairing a model with the rate/metadata needed to run a
Monte Carlo survey simulation.

.. autosummary::
    :toctree: _as_gen
    :recursive:
    :template: module.rst

    uvex_transients.transients

Simulation
----------
Monte Carlo sampling of transient populations against a survey schedule.

.. autosummary::
    :toctree: _as_gen
    :recursive:
    :template: module.rst

    uvex_transients.simulation

Surveys
-------
Representation and validation of survey schedules.

.. autosummary::
    :toctree: _as_gen
    :recursive:
    :template: module.rst

    uvex_transients.surveys

Command-Line Interface
-----------------------
The ``uvex-transients`` console script: a YAML-driven wrapper around
:mod:`uvex_transients.simulation`.

.. autosummary::
    :toctree: _as_gen
    :recursive:
    :template: module.rst

    uvex_transients.cli

Utilities
---------

.. autosummary::
    :toctree: _as_gen
    :recursive:
    :template: module.rst

    uvex_transients.dust
    uvex_transients.utils
