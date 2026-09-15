.. image:: images/uvex_logo_dark.png
   :width: 200px
   :align: center

.. _user_guide:

==================
User Guide
==================

Welcome to the **UVEX Transients User Guide** -- a tutorial-style tour of the library, layer by layer.
For exhaustive per-class and per-function detail, see the :ref:`api` instead.

The package is organized into four independent layers, plus utility modules such as :mod:`~uvex_transients.dust`
and :mod:`~uvex_transients.utils`.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Models
      :class-card: sd-shadow-sm sd-border-1 guide-card

      .. container:: card-intro

         *Module:* :mod:`~uvex_transients.models`

         The :class:`~uvex_transients.models.core.base.SpectralModel` SED framework every transient class builds
         on.

      +++

      .. toctree::
         :maxdepth: 1

         user_guide/models/overview
         user_guide/models/custom_models

   .. grid-item-card:: Transients
      :class-card: sd-shadow-sm sd-border-1 guide-card

      .. container:: card-intro

         *Module:* :mod:`~uvex_transients.transients`

         Pairs a :class:`~uvex_transients.models.core.base.SpectralModel` with rate/duration metadata and samples
         event parameters via :class:`~uvex_transients.transients.base.ExtragalacticTransient`.

      +++

      .. toctree::
         :maxdepth: 1

         user_guide/transients/overview
         user_guide/transients/custom_transients

   .. grid-item-card:: Surveys
      :class-card: sd-shadow-sm sd-border-1 guide-card

      .. container:: card-intro

         *Module:* :mod:`~uvex_transients.surveys`

         Represents and validates a :class:`~uvex_transients.surveys.base.SurveySchedule` of spacecraft actions.

      +++

      .. toctree::
         :maxdepth: 1

         user_guide/surveys/overview

   .. grid-item-card:: Simulation
      :class-card: sd-shadow-sm sd-border-1 guide-card

      .. container:: card-intro

         *Module:* :mod:`~uvex_transients.simulation`

         Ties the above layers together into a :class:`~uvex_transients.simulation.event_catalog.EventCatalog` via
         :class:`~uvex_transients.simulation.core.SurveySimulator`.

      +++

      .. toctree::
         :maxdepth: 1

         user_guide/simulation/overview
