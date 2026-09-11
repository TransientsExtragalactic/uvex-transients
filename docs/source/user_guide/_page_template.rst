.. This is a TEMPLATE, not a built documentation page (see the `exclude_patterns`
   entry in conf.py). Copy it into the relevant layer directory (models/, transients/,
   surveys/, simulation/) as a new file, fill in every ``<...>`` placeholder, and add the new
   file to that layer's ``:toctree:`` in its ``overview.rst``. Use it for a page that goes
   deeper on one topic within a layer -- e.g. ``models/parameters.rst``,
   ``surveys/validation_schema.rst``, ``simulation/event_catalog.rst`` -- once the top-level
   ``overview.rst`` for that layer gets too long to hold it inline.

.. _user_guide_<layer>_<slug>:

<Topic Title>
===============

<One-to-two paragraph overview: what this topic covers and where it fits in the layer's
overview page (:ref:`user_guide_<layer>`). Name the concrete :mod:`uvex_transients` classes
and functions involved, e.g. :class:`~uvex_transients.<module>.<ClassName>`.>

<Section Heading>
-------------------

<Prose and/or a code example. For a runnable example use a ``.. code-block:: python`` block;
for output that should be checked against real code, prefer a ``.. doctest::`` block instead
so it's exercised by ``pytest`` / Sphinx's doctest builder rather than going stale silently.>

.. code-block:: python

    from uvex_transients.<module> import <ClassName>

    <example usage>

<Add further ``<Section Heading>`` blocks as needed. If this topic warrants literature
citations, use :footcite:t:`key` (in-sentence) or :footcite:p:`key` (parenthetical) against
``docs_bib.bib``, and close the page with:>

.. footbibliography::
