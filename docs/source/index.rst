.. raw:: html

   <div class="hero">
       <img src="_images/uvex_logo.png" class="hero-logo">
       <div class="hero-text">
           <h1>UVEX Transients</h1>
           <p>End-to-end simulations of transients from UVEX all-sky surveys</p>
           <div class="badges">

|PYPI| |PYPI-PYTHON| |RUFF| |NUMPYDOC| |LAST-COMMIT| |CONTRIBUTORS| |ASTROPY|

.. raw:: html

           </div>
       </div>
   </div>

Overview
=========

**UVEX Transients** is a Python library for simulating populations of astrophysical transients
(kilonovae, tidal disruption events, luminous fast blue optical transients, supernovae) as they
would be observed by the UVEX all-sky survey, from a physical SED model all the way through to a
Monte Carlo catalog of detections against a real survey schedule.

.. grid:: 2
   :gutter: 3

   .. grid-item-card::
      :class-card: sd-shadow-sm sd-border-1

      **What does it do?**

      - Provides spectral and light-curve models for several transient classes (kilonovae, TDEs,
        LFBOTs, supernovae).
      - Composes a bolometric light curve with a spectral shape into a full, cosmologically
        aware SED that can be evaluated as flux, magnitude, or band photometry.
      - Samples Monte Carlo realizations of transient populations against a validated survey
        schedule to produce an event catalog.

   .. grid-item-card::
      :class-card: sd-shadow-sm sd-border-1

      **Core capabilities**

      - A common :mod:`~uvex_transients.models.core` SED framework that every transient model
        builds on.
      - Milky Way foreground dust extinction via the PlanckGNILC E(B-V) map and the Gordon+2023
        reddening law.
      - Survey schedule validation and Monte Carlo event simulation, including magnitude- and
        SNR-based screening.

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

.. grid:: 1
    :padding: 3
    :gutter: 5

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

   api

Indices and tables
==================

.. raw:: html

   <hr style="height:10px;background-color:black">


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. raw:: html

   <div class="affiliation-footer">
     <p class="affiliation-label">Developed at</p>
     <div class="affiliation-logos">
       <a href="https://www.berkeley.edu" target="_blank" rel="noopener noreferrer">
         <img src="_static/berkeley_logo.svg"
              alt="University of California, Berkeley"
              class="affiliation-logo">
       </a>
       <img src="_static/trex_logo.png"
            alt="Transients Extragalactic (TREX)"
            class="affiliation-logo affiliation-logo-trex">
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
