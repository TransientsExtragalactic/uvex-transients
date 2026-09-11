.. image:: images/uvex_logo_dark.png
   :width: 200px
   :align: center

.. _examples:

Examples
========

Runnable, self-contained examples showing how to drive UVEX Transients end to end: building and
validating a survey schedule, sampling and simulating transient populations against it, and
extending the code base with a new transient class of your own.

Working with Simulation Schedules
----------------------------------

Build and validate the :class:`~uvex_transients.surveys.base.SurveySchedule` that every
simulation is checked against.

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: Working with Simulation Schedules
      :link: auto_examples/schedules/index
      :link-type: doc
      :class-card: sd-shadow-sm sd-border-1

      **Build, validate, and inspect a survey schedule.**

      Construct a :class:`~uvex_transients.surveys.base.SurveySchedule`, understand how
      per-column and per-action-type validation works, and diagnose a
      :class:`~uvex_transients.surveys.base.ScheduleValidationError`.

Simulating Transients
-----------------------

Sample transient populations against a schedule and turn them into an event catalog.

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: Simulating Transients
      :link: auto_examples/simulating/index
      :link-type: doc
      :class-card: sd-shadow-sm sd-border-1

      **Run an end-to-end survey simulation.**

      Use :class:`~uvex_transients.simulation.core.SurveySimulator` to generate an
      :class:`~uvex_transients.simulation.event_catalog.EventCatalog`, screen it by limiting
      magnitude and SNR, and reconstruct individual :class:`~uvex_transients.simulation.event.Event`
      objects to simulate synthetic photometry.

Custom Transients
--------------------

Extend UVEX Transients with a new transient class of your own.

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: Custom Transients
      :link: auto_examples/custom_transients/index
      :link-type: doc
      :class-card: sd-shadow-sm sd-border-1

      **Write a new SED model and pair it with a transient class.**

      Build a new :class:`~uvex_transients.models.core.base.SpectralModel`, wrap it in a
      :class:`~uvex_transients.transients.base.TransientBase` (or
      :class:`~uvex_transients.transients.base.ExtragalacticTransient`) subclass, and register
      it with :class:`~uvex_transients.simulation.core.SurveySimulator`.

.. toctree::
   :hidden:

   auto_examples/schedules/index
   auto_examples/simulating/index
   auto_examples/custom_transients/index
