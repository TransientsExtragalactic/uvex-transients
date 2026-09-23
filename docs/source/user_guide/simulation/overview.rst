.. _user_guide_simulation:

Simulation
===========

:mod:`uvex_transients.simulation` is where everything else in the package comes together:
a :mod:`~uvex_transients.transients` population, sampled against a real
:class:`~uvex_transients.surveys.base.SurveySchedule`, becomes a Monte Carlo catalog of events --
narrowed down to the ones that actually matter -- with real synthetic photometry available for
any one of them on demand. :class:`~uvex_transients.simulation.core.SurveySimulator` does the
sampling and screening; :class:`~uvex_transients.simulation.event_catalog.EventCatalog` holds the
result as a plain, portable data table; and :class:`~uvex_transients.simulation.event.Event`
reconstructs one row of that table into something you can query for a light curve or a full
synthetic detection.

This page walks through that pipeline end to end. See :ref:`user_guide_models` and
:ref:`user_guide_transients` for the layers underneath it, and :ref:`user_guide_surveys` for the
schedule this all gets sampled against.

Quick Look
----------

The fastest way to get a feel for the pipeline is to run it. As in :ref:`user_guide_surveys`, we
build a small synthetic schedule here (400 randomly-pointed 3x3 deg fields over 180 days) rather
than downloading a real one, so the example below runs offline:

.. plot::
   :include-source: true
   :context: reset

   import numpy as np
   import matplotlib.pyplot as plt
   from astropy import units as u
   from astropy.coordinates import EarthLocation, SkyCoord
   from astropy.table import QTable
   from astropy.time import Time
   from regions import RectangleSkyRegion

   from uvex_transients.surveys.base import SurveySchedule
   from uvex_transients.simulation.core import SurveySimulator
   from uvex_transients.transients.TDEs import TidalDisruptionEvent

   n = 400
   rng = np.random.default_rng(0)

   table = QTable()
   table["start_time"] = Time("2025-01-01T00:00:00") + np.sort(rng.uniform(0, 180, n)) * u.day
   table["duration"] = np.full(n, 900.0) * u.s
   table["observer_location"] = EarthLocation.from_geodetic(0 * u.deg, 0 * u.deg, 600 * u.km)
   table["action"] = np.full(n, "observe")
   table["target_coord"] = SkyCoord(
       rng.uniform(0, 360, n) * u.deg,
       np.degrees(np.arcsin(rng.uniform(-1, 1, n))) * u.deg,
   )
   table["roll"] = np.zeros(n) * u.deg
   table["field_id"] = np.arrange(n)
   table["block_id"] = np.zeros(n, dtype=int)

   fov = RectangleSkyRegion(center=SkyCoord(0 * u.deg, 0 * u.deg), width=3 * u.deg, height=3 * u.deg)
   schedule = SurveySchedule(table, fov)

   tde = TidalDisruptionEvent()
   simulator = SurveySimulator(schedule, transients={"tde": tde}, simulation_seed=0)

   catalog = simulator.generate_events(time_bins=6, nside=32)

   fig = plt.figure(figsize=(7, 4))
   ax = fig.add_subplot(111, projection="aitoff")
   ax.grid(True)
   ra = catalog.coord.ra.wrap_at(180 * u.deg).radian
   ax.scatter(ra, catalog.coord.dec.radian, s=4, color="C0")
   ax.set_title(f"{len(catalog)} sampled TDEs across the example schedule")

That's the whole shape of it: pair a transient population with a schedule, hand both to a
:meth:`~uvex_transients.simulation.core.SurveySimulator.generate_events` call, and get back
every TDE that could plausibly have exploded somewhere the schedule looked. The rest of this page
covers the simulator itself, the two screening steps that narrow that population down to
detections, and how to turn any one surviving row back into a real light curve.

.. hint::

   ``time_bins`` above is a plain integer -- the number of evenly-spaced bins to divide the
   schedule's own span into. It also accepts an explicit :class:`~astropy.time.Time` array of bin
   edges, for windows that don't line up evenly with the schedule (e.g. one bin per lunation).

----

The Simulator Object
---------------------

:class:`~uvex_transients.simulation.core.SurveySimulator` is a thin coordinator: a
:class:`~uvex_transients.surveys.base.SurveySchedule`, a ``{name: transient}`` dict of
:class:`~uvex_transients.transients.base.ExtragalacticTransient` instances to sample, and a root
seed for reproducibility, all supplied at construction:

.. code-block:: python

    from uvex_transients.simulation.core import SurveySimulator
    from uvex_transients.transients.TDEs import TidalDisruptionEvent
    from uvex_transients.transients.kilonovae import Kilonova

    simulator = SurveySimulator(
        schedule,
        transients={"tde": TidalDisruptionEvent(), "kne": Kilonova()},
        simulation_seed=42,
    )

    simulator.survey_schedule       # the SurveySchedule passed in
    simulator.transient_collection  # the {name: transient} dict
    simulator.simulation_seed       # 42

.. important::

   Every transient in ``transients`` needs its own distinct name -- this is the key that ties
   each sampled event back to the transient instance that produced it (the catalog's
   ``transient_type`` column) and, later, that reconstructs an
   :class:`~uvex_transients.simulation.event.Event` from a catalog row via
   :meth:`~uvex_transients.simulation.event_catalog.EventCatalog.get_events`. See
   :ref:`user_guide_transients` for configuring a transient's priors,
   :attr:`~uvex_transients.transients.base.TransientBase.duration_limit`, or redshift limit before
   handing it to the simulator.

Simulating several transient types together, as above, is no different from simulating one --
every step below (``generate_events``, both filtering methods, and ``EventCatalog.get_events``)
loops over every ``transient_type`` present in the catalog, keyed against this same
``transient_collection`` dict.

----

Generating the Event Catalog
------------------------------

:meth:`~uvex_transients.simulation.core.SurveySimulator.generate_events` is the main event: a
Monte Carlo realization of every registered transient type, sampled only where and when the
schedule could plausibly have caught it.

Windowed Sampling
^^^^^^^^^^^^^^^^^^

Sampling a transient's full redshift- and sky-dependent volumetric rate over the *entire* sky and
survey duration, then throwing away everything the schedule never observed, would waste almost all
of that effort -- a real survey only ever covers a small fraction of the sky at any one time. To
avoid that, :meth:`~uvex_transients.simulation.core.SurveySimulator.generate_events` divides the
schedule into ``time_bins`` windows and, for each window and each transient type, first asks
:meth:`~uvex_transients.surveys.base.SurveySchedule.get_observed_healpix_ids` which HEALPix pixels
the schedule touches at all between the start of the window and its end *plus* that transient's own
``duration_limit`` -- i.e. late enough that a transient exploding right at the end of the window
could still be caught while it's active. Events are then sampled only within those pixels, via
:meth:`~uvex_transients.transients.base.ExtragalacticTransient.sample_events_on_healpix_grid`, with
explosion times drawn uniformly within the window itself (never the padded tail), so no event is
ever double-counted across adjacent bins. See :ref:`user_guide_transients` for how that per-pixel
sampling itself works.

.. code-block:: python

    catalog = simulator.generate_events(time_bins=6, nside=32, order="nested")

``nside``/``order`` set the HEALPix resolution both the coverage lookup and the sampling grid use
-- the same tradeoff as everywhere else HEALPix shows up in this package: finer pixels track the
schedule's real footprint more closely, at the cost of more pixels to sample per bin.

.. tip::

   For a transient type with a very high intrinsic rate, ``downsample=k`` draws a random
   ``1/k`` subset of each per-bin, per-type table (without replacement, seeded off
   ``simulation_seed``) instead of sampling the full population. Multiply any downstream count
   by ``k`` to get back an estimate of the true yield; see the
   :ref:`simulating_gallery` example for this in practice. ``downsample`` can instead be a
   ``{transient key: k}`` mapping to downsample types individually -- a type left out of the
   mapping isn't downsampled at all. Either form is stashed on the returned
   :class:`~uvex_transients.simulation.event_catalog.EventCatalog` as
   :attr:`~uvex_transients.simulation.event_catalog.EventCatalog.downsample`, purely as
   provenance (nothing rescales counts back up automatically), and every cut carries it through
   to its own output catalog unchanged.

Two columns are computed once here, rather than being left for every later step to re-derive: each
event's ``luminosity_distance`` (interpolated off that transient type's own cached
:attr:`~uvex_transients.transients.base.ExtragalacticTransient.luminosity_distance_grid`, not a
fresh cosmology call per event) and its ``ebv`` (one vectorized Milky Way foreground dust-map
query, via :mod:`uvex_transients.dust`, over every sampled position at once).

----

The Event Catalog
-------------------

:meth:`~uvex_transients.simulation.core.SurveySimulator.generate_events` returns an
:class:`~uvex_transients.simulation.event_catalog.EventCatalog`: a plain data table with no live
reference back to the schedule or the transient instances it was generated from (both are supplied
again, explicitly, wherever they're needed -- see :ref:`user_guide_simulation_events` below). That
makes it trivially picklable and safe to round-trip to disk:

.. code-block:: python

    len(catalog)          # number of sampled events
    catalog.table          # the underlying astropy.table.QTable

    catalog.to_disk("tde_catalog.ecsv", overwrite=True)
    reloaded = EventCatalog.from_disk("tde_catalog.ecsv")

:meth:`~uvex_transients.simulation.event_catalog.EventCatalog.to_disk` writes the table as ECSV
with ``nside``/``order``/``time_bins``/``seed``/``downsample`` stashed in the file's header, so
:meth:`~uvex_transients.simulation.event_catalog.EventCatalog.from_disk` can reconstruct a
complete ``EventCatalog`` from the one file alone. (A file written by an older version of the
package, with no ``downsample`` in its header, still reads back fine -- it just defaults to
`None`.)

Every column of ``catalog.table`` is also available as a convenience property, returning a plain
array (or :class:`~astropy.units.Quantity`/:class:`~astropy.time.Time`/
:class:`~astropy.coordinates.SkyCoord`, as appropriate) rather than a raw table column:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Column / property
     - Type
     - Meaning
   * - ``event_id``
     - int
     - Unique id, assigned once across the whole catalog (not renumbered by filtering).
   * - ``transient_type``
     - str
     - Which entry of ``transient_collection`` this event came from.
   * - ``time_bin``
     - int
     - Index into ``time_bins`` of the window the event exploded within.
   * - ``healpix_id``, ``healpix_dx``, ``healpix_dy``
     - int, float, float
     - Sampling pixel (at this catalog's own ``nside``/``order``) and the event's
       sub-pixel offset within it.
   * - ``coord``
     - :class:`~astropy.coordinates.SkyCoord`
     - Sky position.
   * - ``redshift``
     - float
     - Cosmological redshift.
   * - ``luminosity_distance``
     - :class:`~astropy.units.Quantity`
     - Cached at generation time; see :ref:`user_guide_simulation` above.
   * - ``ebv``
     - float
     - Milky Way foreground E(B-V) at ``coord``; cached at generation time.
   * - ``t_explosion``
     - :class:`~astropy.time.Time`
     - Time of explosion.
   * - ``parameter_seed``
     - int
     - Regenerates this event's physical SED parameters on demand -- see
       :ref:`user_guide_simulation_events` below.

.. note::

   An event's physical SED parameters (amplitude, rise time, temperature, ...) are deliberately
   *not* stored as columns -- only the ``parameter_seed`` that regenerates them. Storing one
   column per parameter would mean a different schema per transient type; regenerating them
   lazily from a stored seed keeps ``EventCatalog`` -- and every filtering method below --
   agnostic to which SED any given row actually uses.

----

Filtering the Event Catalog
------------------------------

A freshly-sampled catalog is dominated by events far too faint to ever matter -- most of a
transient's redshift-limited volume is, by construction, near the limit where it's essentially
undetectable. Two progressively more expensive passes narrow it down to the events actually worth
keeping, both taking an ``EventCatalog`` and an :class:`~m4opt.missions.Mission` (for its
:class:`~m4opt.synphot.Detector`'s bandpasses) and returning a new ``EventCatalog`` over the
surviving rows -- ``nside``/``order``/``time_bins``/``seed``/``downsample`` unchanged, and
original ``event_id`` values preserved rather than renumbered:

.. tab-set::

   .. tab-item:: Limiting Magnitude

      **Method:** :meth:`~uvex_transients.simulation.core.SurveySimulator.filter_by_limiting_magnitude`

      **Description:** Schedule-independent -- it never checks whether the schedule actually
      pointed anywhere near an event. For each transient type, every event's own physical
      parameters are regenerated from its stored ``parameter_seed`` and evaluated over a shared
      ``linspace(0, duration_limit, n_phase)`` phase grid, the same grid for every event of that
      type; an event survives if its brightest requested band clears ``mag_limit`` at ``n_visits``
      or more of those phase samples.

      .. code-block:: python

          mag_filtered = simulator.filter_by_limiting_magnitude(
              catalog, mission, mag_limit=25.0,
          )

      **Uses:** A cheap first pass: "could this event, at its absolute brightest, ever be seen at
      all?" ``chunk_size`` bounds peak memory (evaluation broadcasts every event and phase sample
      into one dense array at once), and raising ``n_visits`` above its default of 1 discards
      events that only momentarily clear the limit.

   .. tab-item:: Signal-to-Noise Ratio

      **Method:** :meth:`~uvex_transients.simulation.core.SurveySimulator.filter_by_snr`

      **Description:** The real question, schedule and all: over every observation the schedule
      actually made of an event's position while it was active, is it ever detected above
      ``snr_threshold``? Every surviving event's observations are gathered with one batched
      :meth:`~uvex_transients.surveys.base.SurveySchedule.get_observation_indices_of` call per
      chunk (see :ref:`user_guide_surveys`), then evaluated together with
      :meth:`~m4opt.synphot.Detector.get_snr`.

      .. code-block:: python

          detected = simulator.filter_by_snr(
              mag_filtered, mission, snr_threshold=5.0,
          )

      **Uses:** The actual detected population. Run it against the *output* of
      ``filter_by_limiting_magnitude``, not the raw catalog -- it's the more expensive of the two
      (every remaining event costs at least one schedule query), so there's no reason to pay that
      cost on events the cheap pass would have rejected anyway.

.. plot::
   :context:
   :include-source: true

   from m4opt.missions import uvex

   mission = uvex
   mag_filtered = simulator.filter_by_limiting_magnitude(catalog, mission, mag_limit=25.0)
   detected = simulator.filter_by_snr(mag_filtered, mission, snr_threshold=5.0)

   stages = ["Sampled", "Mag < 25", "SNR > 5"]
   counts = [len(catalog), len(mag_filtered), len(detected)]

   fig, ax = plt.subplots()
   ax.bar(stages, counts, color=["#888888", "#4C72B0", "#55A868"])
   for i, count in enumerate(counts):
       ax.text(i, count, f"{count:,}", ha="center", va="bottom")
   ax.set_ylabel("Number of TDEs")
   ax.set_title("TDE detection funnel")

----

.. _user_guide_simulation_yield:

Exposure and Yield
----------------------

A raw or filtered ``EventCatalog`` count is a *realization*, not an estimate -- to turn one into a
formal expected-detection number with confidence bounds (:ref:`yield-statistics` derives every
estimator below), you also need to know how much of the sky the survey actually covered.
:meth:`~uvex_transients.simulation.core.SurveySimulator.compute_effective_exposure` answers that,
independent of any Monte Carlo draw: it reruns exactly the same per-bin, per-type footprint query
``generate_events`` restricts its own sampling to, but reduces it to a solid angle instead of a
drawn population:

.. code-block:: python

    exposure = simulator.compute_effective_exposure(time_bins=6, nside=32)

    exposure.total_effective_exposure   # {transient type: total solid-angle*time exposure}
    exposure.total_expected_events      # {transient type: mu_0, the footprint-aware expected count}
    exposure.coverage_fraction          # {transient type: fraction of the full 4*pi sky-time swept}

The returned :class:`~uvex_transients.simulation.exposure_catalog.ExposureCatalog` has one row per
``(transient type, time bin)`` -- the same ``time_bins``/``nside``/``order`` as ``generate_events``
should always be passed here too, so both describe the same footprint query. Each row's
``expected_events`` is ``effective_exposure * transient.integrated_rate`` -- :ref:`yield-statistics`'s
:math:`\mu_0` for that bin -- so summing it over every bin (``total_expected_events``) gives the
intrinsic expected count actually reachable by *this* schedule's footprint, not
:meth:`~uvex_transients.transients.base.ExtragalacticTransient.compute_all_sky_yield`'s idealized
full-sky number. :meth:`~uvex_transients.simulation.exposure_catalog.ExposureCatalog.get_exposure_between`/
:meth:`~uvex_transients.simulation.exposure_catalog.ExposureCatalog.get_expected_events_between` and
:meth:`~uvex_transients.simulation.exposure_catalog.ExposureCatalog.rebin` let you query or
re-tile that same tabulated exposure over an arbitrary sub-window or a different time binning
without re-querying the schedule.

Combine a raw (feasible) catalog, a detected (post-cut) catalog, and its exposure into one
per-transient-type summary with
:meth:`~uvex_transients.simulation.event_catalog.EventCatalog.compute_yield_summary`:

.. code-block:: python

    yields = catalog.compute_yield_summary(detected, exposure, {"tde": tde}, confidence=0.9)
    yields.table["transient_type", "uvex_intrinsic_events", "detection_probability", "expected_detections"]

The returned :class:`~uvex_transients.simulation.yield_table.YieldTable` carries, per transient
type, the rate (``integrated_rate``/``all_sky_rate``), the footprint-aware intrinsic rate/count
(``uvex_intrinsic_rate``/``uvex_intrinsic_events``, i.e. :math:`\mu_0`), the Monte Carlo detection
efficiency (``detection_probability``, :math:`\hat\epsilon=k/n`), and the final yield estimate
(``expected_detections``, :math:`\hat\lambda=\mu_0\hat\epsilon`) -- each rate-derived column with
its own ``RATE_CI``-propagated bounds, and ``detection_probability``/``expected_detections`` with
*two* separate uncertainty sources (Clopper-Pearson binomial and rate-normalization), kept apart
as ``..._binom_lower``/``_upper`` and ``..._rate_lower``/``_upper`` columns rather than combined
into one. :meth:`~uvex_transients.simulation.yield_table.YieldTable.to_ascii` writes a
human-readable summary table; :meth:`~uvex_transients.simulation.yield_table.YieldTable.to_latex`
renders both uncertainty sources as stacked LaTeX superscripts for a paper table.

.. hint::

   For a finer-grained question than "was this event detected at all" -- "how many separate
   epochs was it detected in" -- run synthetic photometry over a whole catalog with
   :meth:`~uvex_transients.simulation.event_catalog.EventCatalog.compute_photometry_catalog`
   (wrapping :meth:`~uvex_transients.simulation.event_catalog.EventCatalog.simulate_photometry`,
   below, as a :class:`~uvex_transients.simulation.photometry_catalog.PhotometryCatalog`), then
   call its own
   :meth:`~uvex_transients.simulation.photometry_catalog.PhotometryCatalog.compute_detection_count_table`.
   It generalizes ``compute_yield_summary``'s "detected at all" (:math:`N_{\rm det}\geq 1`) to
   "detected in at least :math:`k` epochs" for every :math:`k` at once, with the same two-source
   uncertainty treatment -- exactly what the ``detection-counts`` CLI command
   (:ref:`user_guide_cli`) automates.

----

.. _user_guide_simulation_events:

Reconstructing and Simulating Events
----------------------------------------

Every screening step above works with cheap, batched approximations -- a shared phase grid, a
single flux point at each band's pivot wavelength. Getting a real, per-observation synthetic
light curve for one particular event means reconstructing it as a full
:class:`~uvex_transients.simulation.event.Event`, via
:meth:`~uvex_transients.simulation.event_catalog.EventCatalog.get_events`. Because an
``EventCatalog`` holds no live reference to the schedule or the transients it came from (see
above), both are supplied again here -- typically the same ones the catalog was generated with:

.. code-block:: python

    event = detected.get_events(19, {"tde": tde}, schedule)
    print(event)
    ... <Event id=19 type='tde' z=0.7097 n_observations=2>

Building an ``Event`` runs exactly one query against the schedule
(:meth:`~uvex_transients.surveys.base.SurveySchedule.get_observations_of`) to find which scheduled
observations actually covered it while active -- available as
:attr:`~uvex_transients.simulation.event.Event.observations`
(:attr:`~uvex_transients.simulation.event.Event.n_observations`) -- alongside its ``coord``,
``redshift``, ``luminosity_distance``, ``ebv``, and ``t_explosion``. No photometry is done yet at
this point; :meth:`~uvex_transients.simulation.event.Event.sample_parameters` regenerates its
physical SED parameters (deterministically, from the same stored ``parameter_seed``) as a pure,
idempotent lookup any of the methods below can call as many times as needed.

Theoretical Curves
^^^^^^^^^^^^^^^^^^^^

:meth:`~uvex_transients.simulation.event.Event.mag`,
:meth:`~uvex_transients.simulation.event.Event.flux`, and
:meth:`~uvex_transients.simulation.event.Event.luminosity` give the noiseless truth this event's
real photometry (below) scatters around -- the first two folding in this event's own redshift,
distance, and foreground dust; the last a rest-frame, distance-independent bolometric quantity:

.. code-block:: python

    t = np.linspace(0, tde.duration_limit.to_value(u.day), 300) * u.day

    event.mag(t, mission, band="NUV")     # apparent AB magnitude
    event.flux(t, mission, band="NUV")    # observed flux density, at the band's pivot wavelength
    event.luminosity(t)                   # rest-frame bolometric L_bol(t)

Synthetic Photometry
^^^^^^^^^^^^^^^^^^^^^^

:meth:`~uvex_transients.simulation.event.Event.simulate_photometry` is the expensive step every
earlier screening pass was designed to defer: real, per-observation, per-band synthetic photometry
against every observation in ``observations``, batched into one
:meth:`~m4opt.synphot.Detector.get_snr` call per band regardless of how many observations there
are:

.. plot::
   :context:
   :include-source: true

   event = detected.get_events(19, {"tde": tde}, schedule)

   phot = event.simulate_photometry(mission)
   t_since_explosion = (phot["obs_time"] - event.t_explosion).to(u.day)
   t_theory = np.linspace(0, tde.duration_limit.to_value(u.day), 300) * u.day

   fig, ax = plt.subplots(figsize=(7, 4))
   for band, color in {"FUV": "#4C72B0", "NUV": "#DD8452"}.items():
       ax.plot(t_theory.value, event.mag(t_theory, mission, band=band).value, color=color, lw=1.5, alpha=0.6)

       in_band = np.isfinite(phot["ab_mag"]) & (phot["band"] == band)
       if np.any(in_band):
           ax.errorbar(
               t_since_explosion[in_band].value,
               phot["ab_mag"][in_band],
               yerr=phot["mag_err"][in_band],
               marker="s", mfc=color, mec="k", ecolor=color, linestyle="none", label=band,
           )

   ax.invert_yaxis()
   ax.set_xlabel("Days since explosion")
   ax.set_ylabel("AB magnitude")
   ax.set_title(f"Event {event.event_id} (z={event.redshift:.2f}, {event.n_observations} observations)")
   ax.legend()

This example schedule only ever visits a given field once, so most of its events -- like this one
-- end up with just one or two associated observations; see the :ref:`simulating_gallery` for the
same workflow against a real, densely-cadenced UVEX schedule.

Each returned row is a Gaussian realization of the true flux at that observation's implied SNR,
not the ground truth itself -- ``flux``/``ab_mag`` and their symmetric-in-flux ``n_sigma`` bounds
(``flux_upper``/``flux_lower``, transformed separately to ``mag_upper``/``mag_lower`` rather than
through a single linearized ``mag_err``) are ``nan`` wherever the noisy draw itself isn't securely
above zero flux -- the correct behavior at low SNR, not a bug. The whole event replays identically
given the same ``parameter_seed``, since every parameter draw and every band's noise realization
comes from one RNG seeded from it.

----

See the :ref:`simulating_gallery` for a full worked example of this entire pipeline against a real
UVEX schedule, and :mod:`uvex_transients.simulation` in the :ref:`api` reference for exhaustive
method-by-method detail.
