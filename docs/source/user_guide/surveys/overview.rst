.. _user_guide_surveys:

Surveys
========

At the core of **UVEX transient simulations** is the **survey schedule**: a chronological table of every spacecraft action: science
observations, slews, and communications downlinks represented by
:class:`~uvex_transients.surveys.base.SurveySchedule`. This page walks through building one,
validating it, and then using it: looking up what happened at a given time or position, and
computing cadence statistics like control time and pair counts across the sky.

Real schedules for UVEX itself are produced by the separate
`UVEX scheduler <https://github.com/m4opt/uvex-scheduler>`_ project; see
:ref:`user_guide_surveys_loading_scheduler` below for loading one of its published schedules
directly.

Quick Look
----------

The fastest way to get a feel for a schedule is to load one of the pre-configured, published
schedules and look at how it samples the sky:

.. code-block:: python

    from uvex_transients.surveys import get_schedule

    # Downloads (and locally caches) the configured default schedule from the UVEX scheduler.
    schedule = get_schedule()
    print(schedule)

:func:`~uvex_transients.surveys.utils.get_schedule` requires network access the first time it's
called for a given schedule, so the runnable example below builds a small synthetic schedule
instead and plots its **visit count distribution**, the simplest of the cadence diagnostics covered in
full under :ref:`user_guide_surveys_cadence`:

.. plot::
   :include-source: true

   import numpy as np
   import matplotlib.pyplot as plt
   from astropy import units as u
   from astropy.coordinates import EarthLocation, SkyCoord
   from astropy.table import QTable
   from astropy.time import Time
   from regions import RectangleSkyRegion

   from uvex_transients.surveys.base import SurveySchedule

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

   visit_counts, pixel_counts = schedule.compute_visit_count_histogram(nside=32)

   plt.bar(visit_counts, pixel_counts)
   plt.xlabel("Visits to a HEALPix pixel")
   plt.ylabel("Number of pixels")
   plt.title("Cadence distribution: visits per sky pixel")

----

Loading Surveys
----------------

.. _user_guide_surveys_loading_table:

Building a :class:`~uvex_transients.surveys.base.SurveySchedule` requires two components: a
chronological table of scheduled actions, and the instrument's field of view. Both are described
in full in :ref:`user_guide_surveys_anatomy` below; the rest of this section covers the four ways
to get them into a :class:`~uvex_transients.surveys.base.SurveySchedule`.

From a Table
^^^^^^^^^^^^^

Building a schedule from scratch means building the required table -- typically column by column
-- and handing it to the constructor along with the field of view:

.. code-block:: python

    import numpy as np
    from astropy import units as u
    from astropy.coordinates import EarthLocation, SkyCoord
    from astropy.table import QTable
    from astropy.time import Time
    from regions import RectangleSkyRegion

    from uvex_transients.surveys.base import SurveySchedule

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
    print(schedule)
    ... <SurveySchedule n_actions=400 start_time='2025-01-01 ...' end_time='2025-06-29 ...'>

.. hint::

   The constructor copies ``schedule_table`` and re-sorts it by ``start_time`` if it isn't
   already chronological, so the input table's own row order doesn't matter and mutating it
   afterwards never affects the schedule.

Every column is checked against ``SurveySchedule._SCHEMA``. Its Astropy/Python type, unit,
dtype, and dimensionality are validated, and, on top of that, each ``"observe"``/``"slew"``/``"downlink"`` row
is checked against a per-action :class:`~uvex_transients.surveys.utils.ActionSpec` (see
:ref:`user_guide_surveys_anatomy` for the full column and action tables). Both kinds of check
(column-level and action-level) are collected across the *entire* table before anything is
raised: a schedule with five unrelated problems reports all five in one
:class:`~uvex_transients.surveys.base.ScheduleValidationError`, rather than making you fix one,
re-run, hit the next, and repeat. For example, dropping the required ``roll`` column entirely
surfaces both the missing-column error and the resulting per-action error together:

.. code-block:: python

    >>> broken_table = table.copy()
    >>> del broken_table["roll"]
    >>> SurveySchedule(broken_table, fov)
    Traceback (most recent call last):
        ...
    uvex_transients.surveys.base.ScheduleValidationError: Survey schedule failed validation with the following errors:
      - Missing required column 'roll'.
      - Action 'observe' requires missing column 'roll'.

:class:`~uvex_transients.surveys.base.ScheduleValidationError` exposes the same messages,
unformatted, as a plain list via its ``.errors`` attribute, for programmatic handling.

.. tip::

   Both :class:`~uvex_transients.surveys.utils.QTableColumnSpec` and
   :class:`~uvex_transients.surveys.utils.ActionSpec` accept an optional ``validator`` callable
   for rules the declarative fields (type/unit/dtype/required-columns) can't express -- e.g. the
   ``"observe"`` action's "``duration`` must be positive" rule. A column validator receives the
   whole column; an action validator receives the subset of rows for that action. Either may
   return ``True``, ``False``, an error string, or (for action validators) a list of error
   strings.

.. _user_guide_surveys_loading_scheduler:

From the UVEX Scheduler
^^^^^^^^^^^^^^^^^^^^^^^^

Rather than building a table by hand, :func:`~uvex_transients.surveys.utils.get_schedule` fetches
one of a small set of pre-configured, published UVEX schedules and returns it as a ready,
validated :class:`~uvex_transients.surveys.base.SurveySchedule` (paired with
:attr:`m4opt.missions.uvex.fov` as its field of view):

.. code-block:: python

    from uvex_transients.surveys import get_schedule, list_schedules

    list_schedules()
    ... ['uvex_initial_082726', 'uvex_initial_091126']

    schedule = get_schedule("uvex_initial_091126")

Calling :func:`~uvex_transients.surveys.utils.get_schedule` with no arguments fetches whichever
schedule is registered as ``config["schedules.default_schedule"]`` (see
``uvex_transients/config.yaml``); :func:`~uvex_transients.surveys.utils.list_schedules` lists every
name known to that registry. Each fetch downloads directly via
:meth:`~astropy.table.QTable.read`, which locally caches the remote file by default (``cache=True``)
so repeated calls don't re-download it.

.. tip::

    If you want to use a different FOV, you can provide it with ``instrument_fov = ...``.

From a URL
^^^^^^^^^^

An arbitrary schedule URL works the same way, by passing
``url`` instead of ``name``:

.. code-block:: python

    from uvex_transients.surveys import get_schedule

    schedule = get_schedule(url="https://example.org/my-schedule.ecsv")

If you want a durable local copy of the file itself, rather than relying on Astropy's own
(evictable) download cache, use
:func:`~uvex_transients.surveys.utils.download_schedule_from_url` instead, and load it (together
with the field of view) via :meth:`~uvex_transients.surveys.base.SurveySchedule.from_disk`.

From Disk
^^^^^^^^^

:meth:`~uvex_transients.surveys.base.SurveySchedule.to_disk` and
:meth:`~uvex_transients.surveys.base.SurveySchedule.from_disk` round-trip a schedule to an
Astropy-table format (ECSV by default) plus a companion region file for the field of view:

.. code-block:: python

    schedule.to_disk("schedule.ecsv", fov_path="schedule_fov.reg", overwrite=True)

    reloaded = SurveySchedule.from_disk("schedule.ecsv", "schedule_fov.reg")

Pass ``fov_path`` to :meth:`~uvex_transients.surveys.base.SurveySchedule.to_disk` whenever you
intend to reload the file with :meth:`~uvex_transients.surveys.base.SurveySchedule.from_disk`,
since the field of view is otherwise not persisted at all and
:meth:`~uvex_transients.surveys.base.SurveySchedule.from_disk` always requires both paths.

----

.. _user_guide_surveys_anatomy:

Working with Surveys
--------------------

A schedule behaves like a lightweight, read-only sequence over its rows: ``len(schedule)``,
``for row in schedule``, ``schedule[i]``, and ``"observe" in schedule`` (checking against
:attr:`~uvex_transients.surveys.base.SurveySchedule.actions`) all work as expected, and
:attr:`~uvex_transients.surveys.base.SurveySchedule.table` gives direct access to the underlying
:class:`~astropy.table.QTable` (a view, not a copy) whenever you need the raw data rather than one
of the higher-level methods below.

A :class:`~uvex_transients.surveys.base.SurveySchedule` wraps two things:

1. An :class:`~astropy.table.QTable`, one row per scheduled action, whose columns are checked
   against a declarative schema (``SurveySchedule._SCHEMA``, built from
   :class:`~uvex_transients.surveys.utils.QTableColumnSpec`).
2. An **instrument field of view**, ``instrument_fov`` -- either a single
   :class:`~regions.SkyRegion` (of any shape :func:`m4opt.fov.footprint` supports, e.g.
   :class:`~regions.RectangleSkyRegion` or :class:`~regions.CircleSkyRegion`) or a
   :class:`~regions.Regions` collection of several, such as a real instrument's chip-gapped
   footprint made of multiple detector tiles (e.g. :attr:`m4opt.missions.uvex.fov`) -- defined at
   RA=0deg/Dec=0deg/PA=0deg (the same convention ``m4opt`` uses, e.g.
   :attr:`m4opt.missions.Mission.fov`), that every observation's footprint is derived from by
   translating and rolling it to that row's ``target_coord``/``roll``.

The table's required columns are:

.. list-table::
   :header-rows: 1
   :widths: 18 18 45

   * - Column
     - Type / units
     - Meaning
   * - ``start_time``
     - :class:`~astropy.time.Time`
     - Start of the scheduled action.
   * - ``duration``
     - :class:`~astropy.units.Quantity` [time]
     - Duration of the action; must be non-negative.
   * - ``observer_location``
     - :class:`~astropy.coordinates.EarthLocation`
     - Spacecraft position at ``start_time``.
   * - ``action``
     - string
     - One of the action names declared in ``SurveySchedule._ACTION_SCHEMA`` -- ``"observe"``,
       ``"slew"``, or ``"downlink"``.
   * - ``target_coord``
     - :class:`~astropy.coordinates.SkyCoord`
     - Pointing center; only required (unmasked) for ``"observe"`` rows.
   * - ``roll``
     - :class:`~astropy.units.Quantity` [angle]
     - Spacecraft roll angle; only required for ``"observe"`` rows.
   * - ``field_id``
     - integer
     - Survey field identifier; only required for ``"observe"`` rows.
   * - ``block_id``
     - integer
     - Scheduling block identifier; only required for ``"observe"`` rows.
   * - ``phase``
     - string, optional
     - Label identifying which combined survey phase a row came from -- see
       :ref:`Cutting the Survey <user_guide_surveys_cutting>`.

and, per action type:

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Action
     - Required columns
     - Custom rule
   * - ``"observe"``
     - ``target_coord``, ``roll``, ``field_id``, ``block_id``
     - ``duration`` must be strictly positive.
   * - ``"slew"``
     - ``observer_location``
     - --
   * - ``"downlink"``
     - ``observer_location``
     - --

Survey Attributes
^^^^^^^^^^^^^^^^^

A few properties summarize the schedule as a whole without you having to compute them by hand:

.. code-block:: python

    schedule.n_actions          # len(schedule)
    schedule.start_time         # earliest action's start
    schedule.end_time           # latest action's end (start_time + duration)
    schedule.duration           # end_time - start_time
    schedule.observing_time     # total time spent on "observe" actions
    schedule.time_spent("slew") # total time spent on any one action type
    schedule.fov                # the instrument field of view passed to the constructor
    schedule.bounding_radius    # angular radius of the smallest cone centered on the FOV
                                 # that fully contains it -- a cheap containment pre-filter

    schedule.summary            # one-row QTable: start/end time, duration, n_actions, observing_time
    schedule.action_summary     # one row per action type: count and total_duration

.. _user_guide_surveys_cutting:

Cutting the Survey
^^^^^^^^^^^^^^^^^^^

:attr:`~uvex_transients.surveys.base.SurveySchedule.observing_mask` is a boolean array, one entry
per row of :attr:`~uvex_transients.surveys.base.SurveySchedule.table`, flagging which rows are
``"observe"`` actions -- the quickest way to pull just the science observations out of
``schedule.table`` (equivalently, :attr:`~uvex_transients.surveys.base.SurveySchedule.observe_rows`
does this for you).

:meth:`~uvex_transients.surveys.base.SurveySchedule.get_row_at_time` and
:meth:`~uvex_transients.surveys.base.SurveySchedule.get_rows_between_times` cut the schedule down
by time instead of by action type:

.. code-block:: python

    row = schedule.get_row_at_time(schedule.start_time + 1 * u.hour)

    rows = schedule.get_rows_between_times(schedule.start_time, schedule.start_time + 10 * u.day)

:meth:`~uvex_transients.surveys.base.SurveySchedule.with_phase` labels every row of a *copy* of the
schedule with a string, purely so that once schedules are combined you can tell which input
schedule any given row came from -- it plays no role in validation or any query method. Two
chronologically non-overlapping schedules -- e.g. separate phases of a survey, or a real schedule
appended to a planned extension of it -- combine with ``+``:

.. code-block:: python

    early_phase = SurveySchedule(early_table, fov).with_phase("primary")
    late_phase = SurveySchedule(late_table, fov).with_phase("extended")

    combined = early_phase + late_phase

``other`` in ``self + other`` must share ``self``'s
:attr:`~uvex_transients.surveys.base.SurveySchedule.fov` and must not start before
:attr:`self.end_time <uvex_transients.surveys.base.SurveySchedule.end_time>`; combine more than two
by chaining, e.g. ``a + b + c``.

----

Observational Methods
---------------------

In many cases, it's important to work out which parts of the sky the instrument actually covered
during a given window -- e.g. checking whether a survey ever observed a particular transient's
position at all. Done naively, that means testing every query position against every individual
observation's exact footprint, one geometric containment check at a time. That's fine for a
one-off lookup, but it doesn't scale: checking many thousands of simulated transient positions
against a schedule of thousands of observations this way means millions of expensive containment
tests, the overwhelming majority of which are quick "no"s for observations nowhere near the query
position.

To avoid paying that cost on every query, :class:`~uvex_transients.surveys.base.SurveySchedule`
precomputes a HEALPix coverage index once and reuses it. Every ``"observe"`` row's rolled footprint
is rasterized onto a HEALPix grid in a single vectorized pass, then the resulting
``(row, pixel)`` hits are grouped by pixel into two flat arrays -- ``pixel_offsets`` and
``sorted_rows`` -- laid out like a CSR (compressed-sparse-row) matrix. Looking up which
observations covered a given pixel is then a cheap memory-offset lookup into ``sorted_rows``,
rather than a hash-table probe or a scan over every observation.
:meth:`~uvex_transients.surveys.base.SurveySchedule.get_healpix_coverage_index` builds this index
lazily and caches it on the schedule instance, keyed by ``(nside, order)``
(:attr:`~uvex_transients.surveys.base.SurveySchedule._HPX_MAP_CACHE`), so repeated queries at the
same resolution reuse the cached index rather than re-rasterizing the whole schedule from scratch.

The one thing to keep in mind is that HEALPix pixel membership here follows pixel-*center*
containment, not full-pixel overlap, so the index can occasionally *miss* a query point whose own
pixel center happens to fall just outside a footprint -- it never wrongly *includes* one. Every
method below that touches the sky in bulk, rather than as a single scalar query, is ultimately
built on this index, and treats it purely as a fast first-pass filter: each candidate it returns is
always confirmed with an exact geometric containment test afterward, so results stay exact even
though the index itself is approximate.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Method
     - Returns
   * - :meth:`~uvex_transients.surveys.base.SurveySchedule.get_observed_regions` / :meth:`~uvex_transients.surveys.base.SurveySchedule.get_observed_region`
     - The individual, or unioned, sky footprints observed during a time window.
   * - :meth:`~uvex_transients.surveys.base.SurveySchedule.get_observed_healpix_ids`
     - The (sorted, deduplicated) HEALPix pixel indices covered by any observation during a time
       window.
   * - :meth:`~uvex_transients.surveys.base.SurveySchedule.get_observations_of`
     - ``"observe"`` rows whose footprint covers one scalar sky position.
   * - :meth:`~uvex_transients.surveys.base.SurveySchedule.get_observation_indices_of`
     - The batched, index-returning equivalent, for many query positions at once.

.. code-block:: python

    footprint = schedule.get_observed_region(schedule.start_time, schedule.end_time)

    matches = schedule.get_observations_of(schedule.table["target_coord"][0])

    pixel_ids = schedule.get_observed_healpix_ids(schedule.start_time, schedule.end_time, nside=128)

:meth:`~uvex_transients.surveys.base.SurveySchedule.get_observation_indices_of` is the one to reach
for when you have many query positions -- e.g. one per simulated transient -- rather than calling
:meth:`~uvex_transients.surveys.base.SurveySchedule.get_observations_of` in a Python loop: it looks
every position up at once against the cached HEALPix coverage index, falling back to an exact
geometric containment test only for the small candidate set that index returns. It also accepts a
*per-position* time window (arrays the same length as ``coord``, rather than one shared window),
for exactly the case of checking each transient against its own explosion-time-derived window in
one batched call rather than one call per event.

.. plot::
   :include-source: true

   import numpy as np
   import matplotlib.pyplot as plt
   from astropy import units as u
   from astropy.coordinates import EarthLocation, SkyCoord
   from astropy.table import QTable
   from astropy.time import Time
   from regions import RectangleSkyRegion

   from uvex_transients.surveys.base import SurveySchedule

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

   coord = schedule.table["target_coord"]
   elapsed = (schedule.table["start_time"] - schedule.start_time).to_value(u.day)

   fig = plt.figure(figsize=(7, 4))
   ax = fig.add_subplot(111, projection="aitoff")
   sc = ax.scatter(coord.ra.wrap_at(180 * u.deg).radian, coord.dec.radian, c=elapsed, cmap="viridis", s=10)
   ax.grid(True)
   fig.colorbar(sc, label="Days since survey start", pad=0.05, shrink=0.7)
   ax.set_title("Pointing history of the example schedule")

.. _user_guide_surveys_cadence:

Survey Cadence
---------------

One of the most useful diagnostics for survey design is the **survey cadence**, which answers the
question "How well does this survey sample transients on a particular timescale?"
:class:`~uvex_transients.surveys.base.SurveySchedule` provides six such diagnostics, each built as
a full-sky HEALPix map so it can be inspected as a function of sky position. Throughout, fix one
HEALPix pixel :math:`p` and let it be visited :math:`N` times, at elapsed times (measured from the
start of the query window) sorted chronologically as :math:`t_1 < t_2 < \cdots < t_N`; every
diagnostic below is some function of that one per-pixel sequence, evaluated independently at every
pixel on the sky.

.. tab-set::

   .. tab-item:: Visit Count Distribution

      **Method:** :meth:`~uvex_transients.surveys.base.SurveySchedule.compute_visit_count`,
      :meth:`~uvex_transients.surveys.base.SurveySchedule.compute_visit_count_histogram`,
      :meth:`~uvex_transients.surveys.base.SurveySchedule.compute_visit_count_cdf`.

      **Description:** The visit count is simply :math:`N(p)`, the length of pixel :math:`p`'s
      visit-time sequence -- the number of ``"observe"`` actions whose rolled footprint covers
      :math:`p`. Collected over every pixel this gives a full-sky map :math:`\{N(p)\}`. The
      histogram bins pixels by that count,

      .. math::

         h(k) = \bigl|\{\, p : N(p) = k \,\}\bigr|, \qquad k = 0, 1, 2, \ldots,

      with :math:`h(0)` counting every never-observed pixel, and the complementary CDF restricts to
      *observed* pixels (:math:`N(p) \geq 1`) and asks what fraction were hit at least :math:`k`
      times:

      .. math::

         F(k) = \frac{\bigl|\{\, p : N(p) \geq k,\ N(p) \geq 1 \,\}\bigr|}
                      {\bigl|\{\, p : N(p) \geq 1 \,\}\bigr|}, \qquad k = 1, 2, \ldots

      **Uses:** The coarsest possible cadence summary -- it says nothing about *when* a pixel's
      visits happened, only how many there were. It's still the right first question to ask,
      though: :math:`N(p) = 0` identifies coverage gaps outright, and every other diagnostic on this
      page is only defined where :math:`N(p) \geq 2`, so this map also tells you where the rest are
      even applicable.

   .. tab-item:: Pair-wise Cadence

      **Method:** :meth:`~uvex_transients.surveys.base.SurveySchedule.compute_cadence_time_differences`,
      :meth:`~uvex_transients.surveys.base.SurveySchedule.compute_cadence_statistics`.

      **Description:** Consider every pair of pixel :math:`p`'s own visits and how far apart in
      time they are. The ``pairs`` argument selects which pairs count:

      .. math::

         \Delta_{\mathrm{all}}(p) &= \{\, t_j - t_i \;:\; 1 \leq i < j \leq N(p) \,\},
             \qquad |\Delta_{\mathrm{all}}(p)| = \binom{N(p)}{2}, \\
         \Delta_{\mathrm{consec}}(p) &= \{\, t_{i+1} - t_i \;:\; i = 1, \ldots, N(p) - 1 \,\},
             \qquad |\Delta_{\mathrm{consec}}(p)| = N(p) - 1,

      with ``pairs='all'`` (the default) giving :math:`\Delta_{\mathrm{all}}(p)` -- every unique
      pairwise separation, including ones spanning several revisits of the pixel -- and
      ``pairs='consecutive'`` giving :math:`\Delta_{\mathrm{consec}}(p)`, only the gaps between
      temporally-adjacent visits (see *Successive Gaps*).
      :meth:`~uvex_transients.surveys.base.SurveySchedule.compute_cadence_time_differences` returns
      the raw set as a full-sky, ragged array;
      :meth:`~uvex_transients.surveys.base.SurveySchedule.compute_cadence_statistics` instead
      reduces it to per-pixel ``mean``, ``median``, ``min``, ``max``, and ``std``. Pixels with
      :math:`N(p) < 2` have an empty set and are assigned ``NaN``.

      **Uses:** The general-purpose "how far apart in time were the observations of this point"
      distribution underlying every timescale-specific diagnostic below -- e.g. comparing its
      median or minimum against a transient's characteristic fade time tells you whether typical
      sampling is fine enough to resolve it at all.

   .. tab-item:: Successive Gaps

      **Method:** :meth:`~uvex_transients.surveys.base.SurveySchedule.compute_cadence_time_differences`
      / :meth:`~uvex_transients.surveys.base.SurveySchedule.compute_cadence_statistics` with
      ``pairs='consecutive'``.

      **Description:** The consecutive-pairs case of *Pair-wise Cadence* above, isolated because
      it's usually the one you actually want: the sequence of gaps between one visit and the very
      next one,

      .. math::

         g_i(p) = t_{i+1} - t_i, \qquad i = 1, \ldots, N(p) - 1,

      i.e. :math:`\Delta_{\mathrm{consec}}(p)` from the pair-wise cadence distribution, without the
      :math:`\binom{N(p)}{2} - (N(p)-1)` additional non-adjacent pairs that ``pairs='all'`` would
      mix in.

      **Uses:** Unlike the ``pairs='all'`` distribution, this can't be inflated by long baselines
      between widely separated survey passes revisiting the same pixel -- every value is a gap the
      pixel actually sat unobserved through, back to back. That makes it the more literal answer to
      "how long between one look and the next," and the natural input to *Max Gap* below.

   .. tab-item:: Max Gap

      **Method:** :meth:`~uvex_transients.surveys.base.SurveySchedule.compute_max_gap`.

      **Description:** The single worst successive gap at pixel :math:`p`,

      .. math::

         G(p) = \max_{i = 1, \ldots, N(p)-1} g_i(p) = \max_i \bigl(t_{i+1} - t_i\bigr),

      i.e. the maximum of the *Successive Gaps* set :math:`\Delta_{\mathrm{consec}}(p)` rather than
      its mean or median. As with the other pairwise diagnostics, pixels with :math:`N(p) < 2` have
      no defined gap and are assigned ``NaN``.

      **Uses:** A single worst-case number per pixel: "for how long could this point have gone
      completely unsampled?" A pixel can have a short *median* successive gap (from *Pair-wise
      Cadence*) and still have one long lapse hiding in :math:`G(p)` -- e.g. across a seasonal
      visibility gap -- during which a fast transient could rise and fade without a single
      supporting observation.

   .. tab-item:: Pair Counts

      **Method:** :meth:`~uvex_transients.surveys.base.SurveySchedule.compute_pair_counts`,
      :meth:`~uvex_transients.surveys.base.SurveySchedule.compute_pair_count_curve`.

      **Description:** Fix a characteristic transient timescale :math:`T` and a qualifying
      separation window :math:`[\,f_{\min} T,\ f_{\max} T\,]` (``minimum_factor``/``maximum_factor``,
      defaults :math:`f_{\min}=0.5`, :math:`f_{\max}=2`) -- wide enough to bracket a
      timescale-:math:`T` transient's rise or fade. The pair count is how many of pixel
      :math:`p`'s own pairs fall in that window:

      .. math::

         C(p; T) = \begin{cases}
            \bigl|\{\, (i, j) : i < j,\ f_{\min} T \leq t_j - t_i \leq f_{\max} T \,\}\bigr|,
               & \texttt{pairs='all'} \\[4pt]
            \bigl|\{\, i : f_{\min} T \leq t_{i+1} - t_i \leq f_{\max} T \,\}\bigr|,
               & \texttt{pairs='consecutive'}
         \end{cases}

      i.e. the same ``pairs`` choice as *Pair-wise Cadence*, now restricted to a timescale-specific
      window rather than reported in full. The **sensitive area** is the total solid angle of
      pixels with at least one such pair,
      :math:`A(T) = \bigl|\{\, p : C(p; T) > 0 \,\}\bigr| \times (4\pi / n_{\mathrm{pix}})`.
      :meth:`~uvex_transients.surveys.base.SurveySchedule.compute_pair_count_curve` evaluates
      :math:`A(T)` over an array of timescales, reusing every pixel's visit times across the whole
      sweep rather than recomputing them per :math:`T`.

      **Uses:** A cheap *existence* question -- "can this cadence, in principle, catch a
      timescale-:math:`T` transient rising or fading here at all?" -- the same idea behind
      LSST/Rubin's ``PairMetric``. It's the right first pass across many timescales or a whole sky
      before paying for the exact interval-bookkeeping of *Control-Time Curve* below, but it only
      answers yes/no per pixel: one lucky qualifying pair counts exactly the same as continuous
      cadence support.

   .. tab-item:: Control-Time Curve

      **Method:** :meth:`~uvex_transients.surveys.base.SurveySchedule.compute_control_time`,
      :meth:`~uvex_transients.surveys.base.SurveySchedule.compute_control_time_curve`.

      **Description:** Refines *Pair Counts* from "does a qualifying pair exist" to "for how much
      of the survey would a transient starting here actually be caught." A transient with onset
      time :math:`\tau` is assumed useful for temporal sampling only during
      :math:`[\tau,\ \tau + w]`, where :math:`w = f_{\mathrm{vis}} T` is the **visibility window**
      (``visibility_factor``, default :math:`f_{\mathrm{vis}}=3`). A pair :math:`(t_i, t_j)`,
      :math:`i<j`, with :math:`f_{\min}T \leq t_j - t_i \leq f_{\max}T` (as in *Pair Counts*) can
      catch such a transient exactly when both visits fall inside its useful window, i.e.
      :math:`\tau \leq t_i` and :math:`t_j \leq \tau + w`, equivalently

      .. math::

         \tau \in \bigl[\, t_j - w,\ \ t_i \,\bigr].

      For each later visit :math:`t_j`, only the *latest* qualifying earlier visit
      :math:`t_{i^*(j)}` is used, since it maximizes :math:`t_i` and therefore gives the widest
      possible interval for that :math:`j` -- one that contains the interval any other qualifying
      earlier visit would have produced. The control time is the total duration covered by the
      union of these per-visit intervals (merged so no candidate onset time is counted twice),
      clipped to the survey window :math:`[0, T_{\mathrm{survey}}]`:

      .. math::

         CT(p; T) = \operatorname{measure}\!\left(
            \bigcup_j \bigl[\, \max(t_j - w,\ 0),\ \ \min(t_{i^*(j)},\ T_{\mathrm{survey}}) \,\bigr]
         \right).

      :meth:`~uvex_transients.surveys.base.SurveySchedule.compute_control_time` returns
      :math:`\{CT(p; T)\}` as a full-sky map together with the survey-integrated exposure
      :math:`\sum_p CT(p; T) \times (4\pi / n_{\mathrm{pix}})` (solid angle :math:`\times` time);
      :meth:`~uvex_transients.surveys.base.SurveySchedule.compute_control_time_curve` sweeps
      :math:`CT(p; T)` (via its integrated exposure) over an array of timescales.

      **Uses:** Distinguishes a pixel that got exactly one lucky qualifying pair
      (:math:`CT(p;T) \ll T_{\mathrm{survey}}`) from one with continuous, repeated cadence support
      (:math:`CT(p;T)` comparable to :math:`T_{\mathrm{survey}}`) -- something *Pair Counts* cannot
      do, since it only asks whether :math:`C(p;T) > 0`. This is the statistic to reach for once you
      need "how much of the survey remains sensitive," not merely "was it ever sensitive at all."

The ``_curve`` variants of *Pair Counts* and *Control-Time Curve* sweep a whole array of
timescales while reusing the same (comparatively expensive) per-pixel visit-time computation across
every timescale, rather than recomputing it in a loop -- as in this control-time example, run over
a field revisited roughly nightly:

.. plot::
   :include-source: true

   import numpy as np
   import matplotlib.pyplot as plt
   from astropy import units as u
   from astropy.coordinates import EarthLocation, SkyCoord
   from astropy.table import QTable
   from astropy.time import Time
   from regions import RectangleSkyRegion

   from uvex_transients.surveys.base import SurveySchedule

   # A single field revisited roughly nightly over ~90 days -- the repeated-cadence
   # case `compute_control_time_curve` is designed to characterize.
   n = 60
   rng = np.random.default_rng(0)
   elapsed_days = np.sort(rng.choice(np.arrange(90), size=n, replace=False)) + rng.uniform(0, 0.3, n)

   table = QTable()
   table["start_time"] = Time("2025-01-01T00:00:00") + elapsed_days * u.day
   table["duration"] = np.full(n, 900.0) * u.s
   table["observer_location"] = EarthLocation.from_geodetic(0 * u.deg, 0 * u.deg, 600 * u.km)
   table["action"] = np.full(n, "observe")
   table["target_coord"] = SkyCoord(np.full(n, 150.0) * u.deg, np.full(n, 20.0) * u.deg)
   table["roll"] = np.zeros(n) * u.deg
   table["field_id"] = np.zeros(n, dtype=int)
   table["block_id"] = np.arrange(n)

   fov = RectangleSkyRegion(center=SkyCoord(0 * u.deg, 0 * u.deg), width=3 * u.deg, height=3 * u.deg)
   schedule = SurveySchedule(table, fov)

   timescales = np.geomspace(0.1, 30, 25) * u.day
   exposure = schedule.compute_control_time_curve(timescales, nside=32)

   plt.plot(timescales.to_value(u.day), exposure.to_value(u.day * u.sr))
   plt.xscale("log")
   plt.xlabel("Transient timescale [days]")
   plt.ylabel(r"Area-time exposure [day sr]")
   plt.title("Cadence sensitivity vs. transient timescale")

.. tip::

   Every method in this section accepts ``nside``/``order`` to control HEALPix resolution and an
   optional ``start_time``/``end_time`` window to restrict the calculation to part of the survey.
   Higher ``nside`` gives finer spatial resolution at the cost of more pixels to loop over in the
   pair-count/control-time calculations -- 128 (the default for most of these methods) is a
   reasonable starting point; drop to 32-64 while iterating on a large or fine timescale grid.

See the :ref:`schedules_gallery` for a full worked example of building, validating, and
inspecting a schedule end to end, and :mod:`uvex_transients.surveys` in the :ref:`api` reference
for exhaustive method-by-method detail.
