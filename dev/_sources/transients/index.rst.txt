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

   .. grid-item-card:: Type IIP Supernovae
      :link: supernovae
      :link-type: doc
      :class-card: sd-shadow-sm sd-border-1

      Hydrogen-rich core-collapse supernovae with a months-long plateau, with an optional
      early-time UV excess from circumstellar interaction.

      +++
      :math:`z \le 2` · 200 day window

Detailed Physical Models
--------------------------

.. note::

   All models above are phenomenological. First-principles physical models (e.g. radiative
   transfer or hydrodynamically motivated SEDs) are not yet implemented, but the same
   :mod:`~uvex_transients.models.core` framework is designed to accommodate them as they are
   added.

.. toctree::
   :maxdepth: 1
   :hidden:

   kilonovae
   tdes
   lfbots
   supernovae
