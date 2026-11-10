.. _report_phase_space_gallery:

*******************************
Phase-Space Maps
*******************************

.. card::
   :class-card: sd-shadow-sm gallery-banner

   *Config:* `configs/phase_space.yaml <https://github.com/TransientsExtragalactic/uvex-transients/blob/main/configs/phase_space.yaml>`__
   · *Data:* :func:`~uvex_transients.utils.results.get_phase_space_map` · *Module:* :mod:`uvex_transients.phase_space`

   How well UVEX detects a transient of a given peak luminosity and decline time, whatever its physics. A canonical
   transient is simulated over a map of the two, in the FUV and the NUV, and the region of the map each shipped
   transient class occupies is shaded on it.

   :ref:`Back to the report <report_page>`

The idea
========

The Standard Run asks how many events of each class UVEX detects, which depends on the model of every class. A
**phase-space map** asks a more general question that holds for any transient, including ones that are not modeled
here: given an event's **peak luminosity** in a band and its **half-decline time** (the rest-frame time from the peak
until it has faded to half), what is the chance that UVEX detects it?

Every point of a map is a *canonical transient*, with a fast-rise, exponential-decay light curve and a power-law
spectrum, whose two free quantities are the axes of the map. At each point, events are drawn at random positions and
times of the survey, volume-weighted in redshift out to the distance at which the source falls below **25 AB**, and
observed against the same real UVEX schedule as the Standard Run. The expensive points are placed by a Gaussian
process on the recovery fraction, which puts each new batch where one more simulation teaches the most about the map.

What is a detection?
====================

The map of a band counts an event as **detected** if at least **two** observations reach a signal-to-noise ratio of
**5** in that band. As in the Standard Run, an event whose only qualifying observation is the first-ever visit to its
field is not counted, since there is no reference image to detect it against. Because the band both detects the event
and defines the luminosity, the **FUV** and **NUV** maps are separate.

Reading a map
=============

Each figure has four panels, over the half-decline time (horizontal) and the peak luminosity in the band, with the
equivalent absolute AB magnitude on the right:

- the **recovery fraction** :math:`f`, the fraction of the events drawn on the footprint that are detected,
- the **all-sky efficiency** :math:`\eta`, the probability that an event exploding at a random position and time during
  the survey, within the redshift limit, is detected. Unlike :math:`f` it does not depend on how the survey is divided in
  time,
- the **rate** at which one detection is expected, relative to the local core-collapse supernova rate. The lower the
  number, the rarer a transient the survey can find there,
- the **uncertainty** that remains on the fitted map, which is what the sampling spends its points to reduce.

The shaded regions are the shipped transient classes, each the smallest region that holds 68% of draws from its SED
priors, measured in the band of the map. They show **where a class sits, not how many of it UVEX detects**: two
events with the same peak and half-decline time can differ in rise, tail and colour evolution, which the map holds
fixed. The yield of a class is its row of the Standard Run.

Reading the data
================

The maps are made by ``uvex-transients phase-space configs/phase_space.yaml`` and attached to each release as one
``phase_space.zip``. :func:`~uvex_transients.utils.results.get_phase_space_map` fetches the files of one map, and
:meth:`~uvex_transients.phase_space.sampler.PhaseSpaceSampler.load` rebuilds the model from ``points.ecsv`` to evaluate
a map anywhere, at any resolution.
