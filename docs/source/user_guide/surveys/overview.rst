.. _user_guide_surveys:

Surveys
========

.. TODO: Replace this overview with a walkthrough of the ``uvex_transients.surveys`` layer.
   Suggested content, mirroring the layer description in CLAUDE.md:

   - ``SurveySchedule`` (``surveys/base.py``): wraps a chronological table of spacecraft
     actions (observations, slews, downlinks).
   - The declarative validation schema: per-column checks (``QTableColumnSpec``) and
     per-action-type checks (``ActionSpec``), both defined in ``surveys/utils.py``.
   - ``ScheduleValidationError``: all validation errors are collected and raised together
     rather than failing on the first one -- worth a short worked example of what that error
     looks like.
   - How to build a ``SurveySchedule`` from scratch (or load UVEX's all-sky survey schedule)
     and validate it.
   - A pointer to the example gallery for a full worked schedule-building walkthrough.

This page is a placeholder -- see :ref:`user_guide` for the rest of the guide structure this
section sits in, and :mod:`uvex_transients.surveys` in the :ref:`api` reference for the
auto-generated class/function detail in the meantime.
