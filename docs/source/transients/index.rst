.. image:: ../images/uvex_logo_dark.png
   :width: 200px
   :align: center

.. _transients:

Transients
===========

To allow users to simulate observations of transient populations with UVEX, a number of transient are already implemented
in the code base. The intention of these implementations is **intentionally minimal**, reflecting the phenomenological nature
of the models and the goals of the code base.

Each page below documents one transient class or group: what the transient physically is, how its
volumetric rate is determined, the SED model and default parameter priors used to simulate it, and
a gallery of simulated light curves drawn from those priors.

Phenomenological Models
-------------------------

Every transient class currently implemented is a **phenomenological** SED: a simple, empirically-motivated
functional form (a cooling blackbody, a broken power law, a Villar-style rise/plateau/decline, ...) fit or
anchored to real observed light curves, rather than a first-principles radiative-transfer or hydrodynamic
model. This keeps the models fast to sample and easy to reason about, at the cost of not (yet) capturing the
underlying physics in any detail.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Kilonovae
      :link: kilonovae
      :link-type: doc
      :class-card: sd-shadow-sm sd-border-1

      The UV/optical/IR counterpart of a compact-object merger: a cooling blackbody photosphere
      anchored to the AT2017gfo/GW170817 light curve.

      +++
      :math:`z \le 0.2` · 30 day window

   .. grid-item-card:: Tidal Disruption Events
      :link: tdes
      :link-type: doc
      :class-card: sd-shadow-sm sd-border-1

      A star disrupted by a supermassive black hole: a Gaussian-rise/exponential-decline
      photosphere, optionally with a late-time magnetized-disk plateau.

      +++
      :math:`z \le 2` · 200 day window

   .. grid-item-card:: Luminous Fast Blue Optical Transients
      :link: lfbots
      :link-type: doc
      :class-card: sd-shadow-sm sd-border-1

      Rare, fast-evolving, persistently blue explosions typified by AT2018cow: a rapidly
      rising and declining cooling blackbody.

      +++
      :math:`z \le 4` · 100 day window

   .. grid-item-card:: Type II Supernovae
      :link: type_ii
      :link-type: doc
      :class-card: sd-shadow-sm sd-border-1

      Hydrogen-bearing core-collapse supernovae: Type IIP events with a months-long plateau (plus
      a rarer, brighter/hotter early-excess IXF/GGI-like variant), and Type IIb events, a
      double-pulse phenomenological model spanning single- and double-peaked light curves.

      +++
      :math:`z \le 0.5`-:math:`1.2` · 100-200 day window

   .. grid-item-card:: Type I Supernovae
      :link: type_i
      :link-type: doc
      :class-card: sd-shadow-sm sd-border-1

      Hydrogen-free (stripped-envelope) Type Ib and Type Ic core-collapse supernovae: a single
      radioactively powered peak, modeled as a Bazin pulse times a cooling blackbody.

      +++
      :math:`z \le 0.5` · 100 day window

Detailed Physical Models
--------------------------

Unlike the phenomenological SEDs above, the model below is a semi-analytic solution derived from
radiation-hydrodynamics theory and calibrated against numerical simulations, rather than an
empirical shape fit directly to observed light curves.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Shock-Cooling Type IIb SNe
      :link: shock_cooling_iib
      :link-type: doc
      :class-card: sd-shadow-sm sd-border-1

      The early, days-long shock-cooling flash from a stripped-envelope core-collapse SN, per
      Morag+24's diffusion-envelope model. Covers only this early phase, not the later
      radioactive-decay peak of a full Type IIb light curve.

      +++
      :math:`z \le 1` · 20 day window

.. note::

   Other kinds of first-principles physical models (e.g. full radiative transfer or
   hydrodynamically-motivated SEDs) are not yet implemented, but the same
   :mod:`~uvex_transients.models.core` framework is designed to accommodate them as they are
   added.

.. toctree::
   :maxdepth: 1
   :hidden:

   kilonovae
   tdes
   lfbots
   type_ii
   type_i
   shock_cooling_iib
