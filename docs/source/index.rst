.. raw:: html

    <p align="center">
      <img src="_static/uvex_logo_dark.png" width="400" alt="uvex_logo">
    </p>


    <h1 align="center">UVEX Transients</h1>

    <p align="center"><em>End-to-end simulations of transients from UVEX all-sky surveys.</em></p>

    <p align="center">
      <a href="https://pypi.org/project/uvex-transients/" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/pypi/v/uvex-transients" alt="PyPI version"></a>
      <a href="https://pypi.org/project/uvex-transients/" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/pypi/pyversions/uvex-transients" alt="Supported Python versions"></a>
      <a href="https://numpydoc.readthedocs.io/en/latest/" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/docstyle-numpydoc-459db9" alt="Docstring style: numpydoc"></a>
      <a href="https://github.com/astral-sh/ruff" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
      <a href="https://github.com/pre-commit/pre-commit" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit" alt="pre-commit"></a>
      <a href="https://commitizen-tools.github.io/commitizen/" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg" alt="Commit style: Conventional + Gitmoji"></a>
      <a href="https://www.conventionalcommits.org/en/v1.0.0/" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white" alt="Commit style: Conventional Commits"></a>
      <a href="https://transientsextragalactic.github.io/uvex-transients" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/docs-latest-brightgreen.svg" alt="Latest Docs"></a>
      <a href="https://github.com/TransientsExtragalactic/uvex-transients/graphs/contributors" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/github/contributors/TransientsExtragalactic/uvex-transients" alt="GitHub Contributors"></a>
      <a href="https://github.com/TransientsExtragalactic/uvex-transients" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/github/last-commit/TransientsExtragalactic/uvex-transients" alt="Last Commit"></a>
      <a href="http://www.astropy.org/" target="_blank" rel="noopener noreferrer"><img src="http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat" alt="Powered by Astropy"></a>
      <a href="https://numba.pydata.org/" target="_blank" rel="noopener noreferrer"><img src="http://img.shields.io/badge/powered%20by-Numba-00A3E0.svg?style=flat" alt="Powered by Numba"></a>
      <a href="https://m4opt.readthedocs.io/" target="_blank" rel="noopener noreferrer"><img src="http://img.shields.io/badge/powered%20by-m4opt-6f42c1.svg?style=flat" alt="Powered by m4opt"></a>
    </p>

Overview
=========

**UVEX Transients** is a Python toolkit for simulating transient science with `UVEX <https://www.uvex.caltech.edu/>`__.
It provides everything needed to simulate the discovery and follow-up of transient populations with UVEX, across
both target-of-opportunity (TOO) observations and UVEX's planned surveys.

.. card:: Key capabilities
   :class-card: sd-shadow-md sd-rounded-3 capabilities-box
   :margin: 3 0 0 0

   .. grid:: 1 2 2 4
      :gutter: 3

      .. grid-item::
         :class: sd-text-center

         :octicon:`calendar;2em;capabilities-icon`

         **Real survey schedules**

         Validates and simulates against real spacecraft schedules, not idealized cadences.

      .. grid-item::
         :class: sd-text-center

         :octicon:`pulse;2em;capabilities-icon`

         **Synthetic photometry**

         A flexible, pluggable photometry backend (``m4opt``) computes band-integrated fluxes and detection
         significance.

      .. grid-item::
         :class: sd-text-center

         :octicon:`telescope;2em;capabilities-icon`

         **Realistic instrument modeling**

         Spacecraft position, sky backgrounds, detector parameters, bandpasses, and foreground extinction are all
         treated in detail.

      .. grid-item::
         :class: sd-text-center

         :octicon:`graph;2em;capabilities-icon`

         **End-to-end yield simulations**

         Turns a sampled transient population into detections and light curves in a single pipeline.

Installation
=============

.. container:: install-block

    .. card::
       :class-card: sd-shadow-md sd-border-primary sd-p-3 install-card

       **Get started in seconds!**

       Install from source:

       .. code-block:: bash

          git clone https://github.com/TransientsExtragalactic/uvex-transients
          cd uvex-transients && pip install -e .

.. raw:: html

   <hr style="color:black">




Resources
=========

.. grid:: 2 2 2 2
    :padding: 3
    :gutter: 5

    .. grid-item-card::
        :img-top: images/index/stopwatch_icon.png

        User Guide
        ^^^^^^^^^^

        New to UVEX Transients? The user guide walks through the four layers of the package
        (models, transients, surveys, and simulation), how they fit together, and how to drive
        them all from the command line.

        +++

        .. button-ref:: user_guide
            :ref-type: doc
            :expand:
            :color: secondary
            :click-parent:

            User Guide

    .. grid-item-card::
        :img-top: images/index/book.png

        Transients
        ^^^^^^^^^^

        Curious what a simulated kilonova, TDE, LFBOT, or Type IIP supernova actually looks like?
        Each transient page covers the physical motivation, adopted volumetric rate, SED model and
        parameter priors, and a gallery of simulated light curves.

        +++

        .. button-ref:: transients/index
            :ref-type: doc
            :expand:
            :color: secondary
            :click-parent:

            Transients

    .. grid-item-card::
        :img-top: images/index/lightbulb.png

        Examples
        ^^^^^^^^

        Ready to run something end to end? The example gallery walks through building survey
        schedules, simulating transient populations, and writing your own custom transient class.

        +++

        .. button-ref:: auto_examples/index
            :ref-type: doc
            :expand:
            :color: secondary
            :click-parent:

            Examples

    .. grid-item-card::
        :img-top: images/index/api_icon.png

        API Reference
        ^^^^^^^^^^^^^

        Doing a deep dive into our code? Looking to contribute to development? The API reference is a comprehensive resource
        complete with source code and type hinting so that you can find every detail you might need.

        +++

        .. button-ref:: api
            :ref-type: doc
            :expand:
            :color: secondary
            :click-parent:

            API Reference

Contents
========
.. raw:: html

   <hr style="height:10px;background-color:black">

.. toctree::
   :maxdepth: 1

   user_guide
   transients/index
   auto_examples/index
   api

Indices and tables
==================

.. raw:: html

   <hr style="height:10px;background-color:black">


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. raw:: html

   <div class="affiliation-footer" align="center">
     <p class="affiliation-label">Developed at</p>
     <div class="affiliation-logos">
       <a href="https://www.berkeley.edu" target="_blank" rel="noopener noreferrer">
         <img src="_static/berkeley_logo.svg"
              alt="University of California, Berkeley"
              height="150px"
              class="affiliation-logo">
       </a>
       <img src="_static/trex_logo.png"
            alt="Transients Extragalactic (TREX)"
            class="affiliation-logo affiliation-logo-trex"
              height="150px">
     </div>
     <p class="affiliation-text">
       UVEX Transients is developed and maintained by Eliza Diggins and the
       <strong>Transients Extragalactic (TREX)</strong> in the Department
       of Astronomy at the <strong>University of California, Berkeley</strong>.
       Available under the GNU GPLv3 license.
     </p>
   </div>
