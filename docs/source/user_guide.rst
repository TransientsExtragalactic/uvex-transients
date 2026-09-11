.. image:: images/uvex_logo_dark.png
   :width: 200px
   :align: center

.. _user_guide:

==================
User Guide
==================

Welcome to the **UVEX Transients User Guide**. This page is the starting point for technical
documentation on how to use the library. The :ref:`api` gives exhaustive, auto-generated detail on
every class and function signature; the guides linked from this page provide a more tutorial-style overview
of the library and how to use it.

The package is organized into four layers, each importable and usable on its own:

1. (:mod:`~uvex_transients.models`) provides the **SED Models** which underlie all of the various simulated transients.
   This module can be used to build your own models, or to modify existing ones. Models need not be exclusive to a
   single transient, but may be more generic.
2. (:mod:`~uvex_transients.transients`) provides the **Transients** themselves, complete with priors on SED parameters,
   volumetric rates, and durations. This is the interface through which transient simulation is performed.
3. (:mod:`~uvex_transients.surveys`) provides tools for working with the UVEX transient survey schedules.
4. (:mod:`~uvex_transients.simulation`) provides the **Simulation** layer, which ties together the models,
   transients, and survey schedules to produce a Monte Carlo event catalog.

Several additional utility modules are also provided, including :mod:`~uvex_transients.dust` for applying Milky Way foreground
reddening to any SED model, and :mod:`~uvex_transients.utils` for miscellaneous helper functions.

----

Models
------

:mod:`uvex_transients.models` builds physical SEDs from a standard template and is the workhorse behind every transient class.
The SED framework is designed to be flexible and extensible, so that new models can be added easily.
This section covers the base SED classes, the parameter and prior system, and the concrete SED families currently implemented.

.. toctree::
   :maxdepth: 1

   user_guide/models/overview
   user_guide/models/custom_models

Transients
----------

:mod:`uvex_transients.transients` pairs a :mod:`~uvex_transients.models` SED with the metadata
needed to window a survey simulation -- a default duration and, for extragalactic populations, a
comoving volumetric rate and redshift limit -- and handles Monte Carlo sampling of
``(RA, DEC, z, t_explosion, parameter_seed)`` tuples. This section covers ``TransientBase`` and
``ExtragalacticTransient``. For the astrophysics and adopted parameters of each implemented
population (kilonovae, TDEs, LFBOTs, supernovae), see the :ref:`transients` gallery instead.

.. toctree::
   :maxdepth: 1

   user_guide/transients/overview

Surveys
-------

:mod:`uvex_transients.surveys` wraps a chronological table of spacecraft actions (observations,
slews, downlinks) in a :class:`~uvex_transients.surveys.base.SurveySchedule` and validates it
against a declarative schema, collecting every validation error rather than failing on the first
one. This section covers the schedule schema, the per-column and per-action-type checks, and how
to build or validate a schedule of your own.

.. toctree::
   :maxdepth: 1

   user_guide/surveys/overview

Simulation
----------

:mod:`uvex_transients.simulation` ties the above three layers together.
``SurveySimulator.generate_events`` samples every registered transient type against a
``SurveySchedule`` into an ``EventCatalog`` -- a pure data table that round-trips to disk and
pickles cleanly. Per-event synthetic photometry is deliberately deferred until requested. This
section covers running a simulation end to end, the two screening steps
(``filter_by_limiting_magnitude`` and ``filter_by_snr``) that narrow a population before the
expensive step, and reconstructing/simulating individual ``Event`` objects from a catalog.

.. toctree::
   :maxdepth: 1

   user_guide/simulation/overview

----

Quick Navigation
-----------------

.. grid:: 2 2 2 2
   :gutter: 3

   .. grid-item-card:: Models
      :link: user_guide_models
      :link-type: ref
      :class-card: sd-shadow-sm sd-border-1

      ``SpectralModel``, ``Lightcurve``, ``Spectrum``, parameters and priors, and the concrete
      SED families.

   .. grid-item-card:: Transients
      :link: user_guide_transients
      :link-type: ref
      :class-card: sd-shadow-sm sd-border-1

      ``TransientBase``, ``ExtragalacticTransient``, and Monte Carlo sampling of event
      parameters.

   .. grid-item-card:: Surveys
      :link: user_guide_surveys
      :link-type: ref
      :class-card: sd-shadow-sm sd-border-1

      ``SurveySchedule``, the action/column validation schema, and building your own schedule.

   .. grid-item-card:: Simulation
      :link: user_guide_simulation
      :link-type: ref
      :class-card: sd-shadow-sm sd-border-1

      ``SurveySimulator``, ``EventCatalog``, screening steps, and per-event synthetic
      photometry.
