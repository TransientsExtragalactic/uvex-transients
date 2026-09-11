.. raw:: html

    <p align="center">
      <img src="_static/uvex_logo_dark.png" width="400" alt="uvex_logo">
    </p>


    <h1 align="center">UVEX Transients</h1>

    <p align="center"><em>End-to-end simulations of transients from UVEX all-sky surveys.</em></p>

    <p align="center">
      <img src="https://img.shields.io/badge/docstyle-numpydoc-459db9" alt="Docstring style: numpydoc">
      <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
      <img src="http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat" alt="Powered by Astropy">
    </p>

Overview
=========

**UVEX Transients** is a Python library for simulating populations of astrophysical transients
(kilonovae, tidal disruption events, luminous fast blue optical transients, supernovae, etc.) as they
would be observed by the UVEX all-sky survey (and other constituent surveys), from a physical SED model all the way through to a
Monte Carlo catalog of detections against a real survey schedule.

This tool is intended to enable informed modeling of UVEX's discovery and follow-up capabilities for a variety of  transient
populations and for a variety of survey strategies. It is also intended to be a flexible framework for simulating new transient populations
as need arises.

.. grid:: 2
   :gutter: 3

   .. grid-item-card::
      :class-card: sd-shadow-sm sd-border-1

      **What does it do?**

      - Provides a **Monte-Carlo based** event simulator capable of sampling realizations of abstract transient
        populations against a survey schedule.
      - Provides a **Flexible SED framework** for building new transient models, including a number of pre-built phenomenological models.
      - Provides a **Survey schedule framework** for representing and validating survey schedules, including UVEX's all-sky survey.
      - Provides tooling for end-to-end simulations, including sampling the population and determining detections using **synthetic photometry**.

   .. grid-item-card::
      :class-card: sd-shadow-sm sd-border-1

      **Core capabilities**

      - A common :mod:`~uvex_transients.models.core` SED framework that every transient model
        builds on.
      - Efficient and robust synthetic photometry using ``m4opt``, including detailed treatment of spacecraft position,
        sky backgrounds, detector parameters, bandpass, and foreground extinction.
      - A Monte Carlo survey simulator that samples transient populations against a survey schedule and determines detections.

.. container:: install-block

    .. card::
       :class-card: sd-shadow-md sd-border-primary sd-p-3 install-card

       **Get started in seconds!**

       .. code-block:: bash

          pip install uvex-transients

       Or install from source:

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

        New to UVEX Transients? The user guide walks through the four layers of the package --
        models, transients, surveys, and simulation -- and how they fit together.

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

        .. button-ref:: examples
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
   examples
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


.. |PYPI| image:: https://img.shields.io/pypi/v/uvex-transients
   :target: https://pypi.org/project/uvex-transients/
   :alt: PyPI version

.. |PYPI-PYTHON| image:: https://img.shields.io/pypi/pyversions/uvex-transients
   :target: https://pypi.org/project/uvex-transients/
   :alt: Supported Python versions

.. |RUFF| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff
    :alt: Ruff

.. |NUMPYDOC| image:: https://img.shields.io/badge/docstyle-numpydoc-459db9
   :target: https://numpydoc.readthedocs.io/en/latest/
   :alt: Docstring style: numpydoc

.. |CONTRIBUTORS| image:: https://img.shields.io/github/contributors/TransientsExtragalactic/uvex-transients
   :target: https://github.com/TransientsExtragalactic/uvex-transients/graphs/contributors
   :alt: GitHub Contributors

.. |LAST-COMMIT| image:: https://img.shields.io/github/last-commit/TransientsExtragalactic/uvex-transients
   :target: https://github.com/TransientsExtragalactic/uvex-transients
   :alt: Last Commit

.. |ASTROPY| image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
   :target: http://www.astropy.org/
   :alt: Powered by Astropy
