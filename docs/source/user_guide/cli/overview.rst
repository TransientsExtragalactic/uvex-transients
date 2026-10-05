.. _user_guide_cli:

Command-Line Interface
=========================

Everything covered in :ref:`user_guide_simulation` (sampling an
:class:`~uvex_transients.simulation.event_catalog.EventCatalog`, screening it down with one or
more cuts, and running synthetic photometry) is also available without writing any Python, via
the ``uvex-transients`` console script (:mod:`uvex_transients.cli`). One YAML **run-config** file
describes *what* to simulate (which transients, what schedule) and, via its ``steps:`` list, *how*
to process the baseline catalog once it's sampled -- screening cuts, set operations between named
catalogs, and post-processing actions like synthetic photometry, chained in one declared order,
GitHub-Actions style. The CLI's five subcommands (``generate``, ``cut``, ``photometry``,
``detection-counts``, and ``run``) drive that same pipeline, reading and writing plain catalog/table
files between stages.

This page is a hands-on tour of the run-config format and the commands themselves. See
:ref:`user_guide_simulation` for what each stage actually does under the hood; this page is about
driving that same pipeline from a config file and a terminal instead of a Python script.

Quick Look
----------

The repository ships a ready-to-run example, `quickstart_tde.yaml
<https://github.com/TransientsExtragalactic/uvex-transients/blob/main/configs/quickstart_tde.yaml>`__,
under ``configs/``. Once the package is installed (``pip install -e .`` from a source checkout
registers the ``uvex-transients`` command), running the whole pipeline is one line:

.. code-block:: bash

    uvex-transients run configs/quickstart_tde.yaml --out-dir quickstart_results/

That samples tidal disruption events against the default UVEX schedule, tabulates the survey's
effective exposure to them, screens them by magnitude and then by SNR, and runs synthetic
photometry on whatever survives -- leaving every checkpointed artifact in
``quickstart_results/``: ``baseline.ecsv``, ``exposure.ecsv``, one numbered file per declared step
(``01_mag_screen.ecsv``, ``02_snr_screen.ecsv``, ``03_phot.ecsv``). The rest of this page explains
the config file that made that happen and the commands it works with, so you can build your own.

----

The Run-Config File
----------------------

A run-config is a single YAML file with up to six top-level sections. Every CLI command reads the
*same* file (there's no separate config per command), but each command only needs the section(s)
it actually touches, resolved lazily by :class:`~uvex_transients.cli.config.RunConfig`: a config
with no ``steps:`` block is perfectly valid as long as you only ever run ``generate``.

.. list-table:: Top-level sections
   :header-rows: 1
   :widths: 18 15 67

   * - Section
     - Required by
     - Purpose
   * - ``schedule:``
     - all commands
     - Which :class:`~uvex_transients.surveys.base.SurveySchedule` to simulate against. Optional;
       falls back to the package's own default schedule if omitted entirely.
   * - ``mission:``
     - all commands
     - Which :class:`~m4opt.missions.Mission` (bandpasses, detector, background) to evaluate
       against. Optional; defaults to ``"uvex"``.
   * - ``transients:``
     - all commands
     - At least one :class:`~uvex_transients.transients.base.TransientBase` type to simulate.
       Required.
   * - ``generate:``
     - ``generate``, ``run``
     - :meth:`~uvex_transients.simulation.core.SurveySimulator.generate_events`'s own arguments.
       Its output is the ``"baseline"`` artifact every ``steps:`` entry can reference; a
       tabulated :class:`~uvex_transients.simulation.exposure_catalog.ExposureCatalog` over the
       same footprint is always available too, under ``"exposure"``.
   * - ``steps:``
     - ``run`` (and, ad hoc, ``cut``/``photometry``/``detection-counts``)
     - An ordered list of ``cut``/``logical_op``/``action`` steps -- see below. Optional; an
       entirely absent ``steps:`` list is fine if you only ever want the raw baseline catalog.
   * - ``keep_intermediate:``
     - ``run``
     - Whether ``run`` checkpoints each step's artifact to ``--out-dir`` by default (a step's own
       ``checkpoint:`` overrides this individually). Optional bool; defaults to ``true``.
       Overridden either way by ``run``'s own ``--keep-intermediate``/``--no-keep-intermediate``
       flag. ``exposure.ecsv`` is always written regardless.

.. important::

   ``schedule:``, ``mission:``, and ``transients:`` are the shared "setup" every command needs to
   build a :class:`~uvex_transients.simulation.core.SurveySimulator`
   (:attr:`~uvex_transients.cli.config.RunConfig.simulator`); they're listed as required by "all
   commands" above even though, individually, ``schedule:``/``mission:`` each have a fallback
   default.

The Schedule
^^^^^^^^^^^^^^

.. code-block:: yaml

    schedule:
      name: uvex_initial_main    # one of the names in `schedules.schedule_urls`

    # ...or, to fetch a URL directly instead of a name registered in the package config:
    #
    #   schedule:
    #     url: https://example.com/plan.ecsv
    #
    # ...or to read a schedule already saved to disk (as written by
    # `SurveySchedule.to_disk(path, fov_path=...)`):
    #
    #   schedule:
    #     path: my_schedule.ecsv
    #     fov_path: my_schedule.reg

``name:``, ``url:``, and ``path:``/``fov_path:`` are mutually exclusive: giving more than one
raises. See :ref:`user_guide_surveys` for what a schedule actually is and
:func:`~uvex_transients.surveys.utils.list_schedules` for the full set of registered names.

.. list-table:: ``schedule:`` parameters
   :header-rows: 1
   :widths: 18 15 67

   * - Key
     - Type
     - Description
   * - ``name``
     - str
     - A name registered in ``schedules.schedule_urls`` (the package config). Mutually exclusive
       with ``url``/``path``.
   * - ``url``
     - str
     - A direct URL to fetch a schedule table from. Mutually exclusive with ``name``/``path``.
   * - ``path``
     - str
     - A local schedule table path, as written by ``SurveySchedule.to_disk``. Requires
       ``fov_path``; mutually exclusive with ``name``/``url``.
   * - ``fov_path``
     - str
     - The companion instrument-FOV region file for ``path``. Required alongside ``path``.

The Mission
^^^^^^^^^^^^^

.. code-block:: yaml

    mission: uvex

A bare name, resolved against every :class:`~m4opt.missions.Mission` instance in
:mod:`m4opt.missions` (``rubin``, ``ultrasat``, ``uvex``, ``ztf`` as of this writing); an
unrecognized name raises, listing the real ones.

.. list-table:: ``mission:`` parameter
   :header-rows: 1
   :widths: 18 15 67

   * - Key
     - Type
     - Description
   * - (bare value)
     - str
     - A name from :mod:`m4opt.missions`. Optional; defaults to ``"uvex"``.

The Transients
^^^^^^^^^^^^^^^^

Each entry under ``transients:`` is a label of your choosing (used as the catalog's
``transient_type`` value; see :ref:`user_guide_simulation`) mapping to a ``class:`` plus whatever
overrides you want on top of its defaults:

.. code-block:: yaml

    transients:
      tde:
        class: TidalDisruptionEvent
        z_limit: 2.0                 # overrides ExtragalacticTransient.redshift_limit
        duration_limit: 200 day      # overrides TransientBase.duration_limit
        cosmology: !astropy_cosmology {name: Planck18}
        parameters:
          amplitude: !prior {type: normal, mean: 43.8, sigma: 0.3}
          sigma_rise: 1.0 day        # a bare value fixes the parameter instead

``class:`` must name a registered :class:`~uvex_transients.transients.base.TransientBase`
subclass: every built-in type (``TidalDisruptionEvent``, ``Kilonova``,
``LuminousFastBlueOpticalTransient``, ``TypeIIPSNe``, ``TypeIIPExcessSNe``, ``TypeIIbSNe``,
``ShockCoolingIIb``, ``TypeIbSNe``, ``TypeIcSNe``) is available by class name; an unrecognized one
raises, listing what *is* registered.

Everything under ``parameters:`` is applied the same way
:meth:`Event.simulate_photometry <uvex_transients.simulation.event.Event.simulate_photometry>`
expects to find it later: a plain value fixes that parameter
(:meth:`~uvex_transients.models.core.parameters.Parameter.fix`), while a ``!prior`` node replaces
its prior entirely (:meth:`~uvex_transients.models.core.parameters.Parameter.set_prior`). A value
with units (like ``sigma_rise`` above) needs a unit string (``"1.0 day"``); a bare, unitless
number only works for a genuinely dimensionless parameter. See :ref:`user_guide_models` for what a
model's parameters actually are.

.. tip::

   ``!prior`` takes a ``type:`` naming one of :class:`~uvex_transients.models.core.priors.Prior`'s
   built-in subclasses, plus that class's own fields:

   .. list-table:: ``!prior`` types
      :header-rows: 1
      :widths: 25 75

      * - ``type:``
        - Fields
      * - ``constant``
        - ``value``
      * - ``uniform``
        - ``lower``, ``upper``
      * - ``normal``
        - ``mean``, ``sigma``
      * - ``lognormal``
        - ``mean``, ``sigma`` (of the underlying normal, not the lognormal itself)
      * - ``truncated_normal``
        - ``mean``, ``sigma``, ``lower``, ``upper``
      * - ``exponential``
        - ``scale``
      * - ``power_law``
        - ``alpha``, ``lower``, ``upper``
      * - ``discrete``
        - ``values``, ``probabilities``

   An unrecognized ``type:``, a missing ``type:``, or an unknown field for the chosen type all
   raise at load time, naming the line the problem is on.

.. list-table:: ``transients:`` entry parameters
   :header-rows: 1
   :widths: 18 20 62

   * - Key
     - Type
     - Description
   * - ``class``
     - str
     - A registered :class:`~uvex_transients.transients.base.TransientBase` subclass name.
       Required.
   * - ``z_limit``
     - float
     - Overrides :attr:`~uvex_transients.transients.base.ExtragalacticTransient.redshift_limit`.
       Optional.
   * - ``duration_limit``
     - str or number
     - Overrides :attr:`~uvex_transients.transients.base.TransientBase.duration_limit` (a unit
       string like ``"200 day"``, or a bare number of days). Optional.
   * - ``photometry_pre_window``
     - str or number
     - Overrides :attr:`~uvex_transients.transients.base.TransientBase.photometry_pre_window` --
       how far before explosion :meth:`Event.simulate_photometry
       <uvex_transients.simulation.event.Event.simulate_photometry>` still generates
       background/non-detection photometry (a unit string or a bare number of days). Optional;
       defaults to no pre-explosion photometry.
   * - ``photometry_post_window``
     - str or number
     - Overrides :attr:`~uvex_transients.transients.base.TransientBase.photometry_post_window` --
       how far after explosion :meth:`Event.simulate_photometry
       <uvex_transients.simulation.event.Event.simulate_photometry>` evaluates the transient's SED
       before falling back to background/non-detection photometry (a unit string or a bare number
       of days). Optional; defaults to ``duration_limit``.
   * - ``cosmology``
     - ``!astropy_cosmology``
     - Overrides the transient's cosmology. Optional; defaults to the package's own default
       cosmology.
   * - ``parameters``
     - mapping
     - Per-SED-parameter overrides, keyed by parameter name; each value is either a fixed value
       or a ``!prior`` node. Optional.

----

Generating Events
--------------------

.. code-block:: yaml

    generate:
      time_bins: 20
      nside: 64
      downsample: 20
      seed: 42

A direct pass-through to
:meth:`~uvex_transients.simulation.core.SurveySimulator.generate_events`; see
:ref:`user_guide_simulation` for what each field means. Its output is the reserved ``"baseline"``
artifact -- the starting point every ``steps:`` entry ultimately traces back to (see below) -- and
an :class:`~uvex_transients.simulation.exposure_catalog.ExposureCatalog` over the same
``time_bins``/``nside``/``order`` is always available too, under the reserved id ``"exposure"``.

``downsample:`` can also give a different factor per transient type instead of one number for
every type -- keys are the ``transients:`` section's own keys, and a type left out isn't
downsampled at all:

.. code-block:: yaml

    generate:
      time_bins: 20
      downsample:
        tde: 20
        kilonova: 5
      seed: 42

.. code-block:: bash

    uvex-transients generate quickstart_tde.yaml --out baseline.ecsv

.. list-table:: ``generate:`` parameters
   :header-rows: 1
   :widths: 18 15 67

   * - Key
     - Type
     - Description
   * - ``time_bins``
     - int
     - Number of evenly-spaced time bins to divide the schedule's span into. Required.
   * - ``nside``
     - int
     - HEALPix resolution. Optional; falls back to the package's ``healpix.default_nside``.
   * - ``order``
     - str
     - HEALPix pixel ordering (``"nested"`` or ``"ring"``). Optional; falls back to the package's
       ``healpix.default_order``.
   * - ``downsample``
     - int or mapping of str to int
     - Draw a random ``1/downsample`` subset instead of the full population. Either a single
       factor applied to every registered transient type, or a ``{transient key: factor}``
       mapping to downsample types individually (a type left out of the mapping is not
       downsampled). Optional; no downsampling by default.
   * - ``seed``
     - int
     - Root seed for :class:`~uvex_transients.simulation.core.SurveySimulator`. Optional.

----

The Steps Pipeline
----------------------

``steps:`` is an ordered list, run strictly top-to-bottom -- like a GitHub Actions job. Every
entry needs a unique ``id:`` and a ``type:`` of ``cut``, ``logical_op``, or ``action``. Whatever
artifact a step produces (an :class:`~uvex_transients.simulation.event_catalog.EventCatalog`, a
photometry table, ...) is stored under its own ``id``, so a *later* step can reference it as an
input by name -- alongside the two reserved ids ``"baseline"``/``"exposure"`` that ``generate:``
always seeds the pipeline with. A step may only reference ``"baseline"``/``"exposure"`` or an
*earlier* step's ``id`` -- there's no forward reference, and the whole list is validated (unique
ids, resolvable inputs, known cut/op/action names, correct arity) the moment ``steps:`` is
accessed, before anything actually runs.

.. code-block:: yaml

    steps:
      - id: mag_screen
        type: cut
        cut: limiting_magnitude
        input: baseline
        params:
          mag_limit: 25.0

      - id: snr_screen
        type: cut
        cut: snr
        input: mag_screen
        params:
          snr_threshold: 5.0

      - id: phot
        type: action
        action: photometry
        inputs:
          catalog: snr_screen
        params:
          bands: [FUV, NUV]

``cut`` steps
^^^^^^^^^^^^^^^

A single-catalog predicate filter -- narrows one :class:`EventCatalog
<uvex_transients.simulation.event_catalog.EventCatalog>` down to another. ``cut:`` must name one
of :meth:`SurveySimulator.available_cuts()
<uvex_transients.simulation.core.SurveySimulator.available_cuts>`, registered via the
:func:`~uvex_transients.simulation.core.cut` decorator -- the schedule/detector-aware
``limiting_magnitude``/``snr`` pair walked through in :ref:`user_guide_simulation`, plus a further
set of cheaper or more specialized screens (redshift, type, peak brightness, detection timing,
sky position, and an arbitrary boolean expression), listed in full below. ``input:`` names the
artifact to filter; everything under ``params:`` is forwarded as keyword arguments to that cut's
underlying method.

An optional ``transient_types:`` list restricts the cut to rows whose ``transient_type`` matches
one of those names -- every other row passes through untouched, regardless of what the cut would
otherwise have done to it. This is handled generically by the step executor itself, not by any
individual cut, so it works retroactively for every registered cut, including ones a project adds
later:

.. code-block:: yaml

    steps:
      - id: tde_only_screen
        type: cut
        cut: limiting_magnitude
        input: baseline
        transient_types: [tde]
        params:
          mag_limit: 26.0

.. hint::

   Cuts are ordinary :class:`~uvex_transients.simulation.core.SurveySimulator` methods, not a
   separate plugin class: decorating a new method on a
   :class:`~uvex_transients.simulation.core.SurveySimulator` subclass with ``@cut("my_cut")``
   registers it automatically, immediately usable from a ``cut:`` in any run-config.

.. list-table:: ``cut`` step parameters, by ``cut:``
   :header-rows: 1
   :widths: 22 24 54

   * - ``cut:``
     - Required params
     - Optional params
   * - ``limiting_magnitude``
     - ``mag_limit``
     - ``bands``, ``n_phase``, ``chunk_size``, ``n_visits``
   * - ``snr``
     - ``snr_threshold``
     - ``bands``, ``chunk_size``, ``n_visits``
   * - ``redshift``
     - --
     - ``min_redshift``, ``max_redshift`` (at least one required)
   * - ``transient_type``
     - ``types``
     - --
   * - ``peak_magnitude``
     - --
     - ``min_mag``, ``max_mag`` (at least one required), ``bands``, ``n_phase``, ``chunk_size``
   * - ``peak_flux``
     - --
     - ``min_flux``, ``max_flux`` (at least one required), ``bands``, ``n_phase``, ``chunk_size``
   * - ``peak_luminosity``
     - --
     - ``min_luminosity``, ``max_luminosity`` (at least one required), ``n_phase``, ``chunk_size``
   * - ``time_to_first_detection``
     - ``snr_threshold``
     - ``min_delay``, ``max_delay`` (at least one required), ``bands``, ``chunk_size``
   * - ``baseline``
     - ``snr_threshold``
     - ``min_baseline``, ``max_baseline`` (at least one required), ``bands``, ``chunk_size``
   * - ``region``
     - ``region``
     - --
   * - ``sky_position``
     - ``frame``
     - ``phi_min``, ``phi_max``, ``theta_min``, ``theta_max``, ``mode``
   * - ``query``
     - ``expr``
     - --

``peak_magnitude``/``peak_flux``/``peak_luminosity`` are purely intrinsic-plus-distance screens
(no Milky Way dust attenuation, no detector background, no schedule) on the brightest a modeled
event ever gets -- see
:meth:`~uvex_transients.simulation.core.SurveySimulator.filter_by_peak_magnitude` for the exact
evaluation. ``time_to_first_detection``/``baseline`` are schedule-aware, like ``snr``, reusing its
own qualifying-epoch definition (see
:meth:`~uvex_transients.simulation.core.SurveySimulator.filter_by_baseline` for the two-sided
"at least one close pair *and* a long overall span" semantics). ``region`` takes a
:class:`~regions.SkyRegion`/:class:`~regions.Regions` object or a ``.reg`` file path (the same
convention the instrument FOV uses); ``sky_position`` is a simpler longitude/latitude box in any
coordinate frame, with ``mode: include`` (default) or ``mode: exclude`` (e.g. Galactic-plane
avoidance: ``frame: galactic, theta_min: -10, theta_max: 10, mode: exclude``). ``query`` evaluates
an arbitrary boolean expression string directly against the catalog's own columns (each bound to
its native :class:`~astropy.units.Quantity`/:class:`~astropy.coordinates.SkyCoord`/
:class:`~astropy.time.Time` type, so comparisons stay unit- and frame-aware) -- see
:meth:`~uvex_transients.simulation.core.SurveySimulator.filter_by_query` for the exact evaluation
scope and its "trusted author, not a sandbox" caveat.

``logical_op`` steps
^^^^^^^^^^^^^^^^^^^^^^^

Combines two or more already-produced :class:`EventCatalog
<uvex_transients.simulation.event_catalog.EventCatalog>` artifacts by ``event_id`` --
useful for comparing independent cut branches (e.g. which events *both* a magnitude screen and an
SNR screen agree on, or which ones only one branch caught). ``op:`` is one of
``union``/``intersection``/``difference`` (see
:mod:`uvex_transients.simulation.logical_ops`); ``inputs:`` is a list of earlier
``id:``\ s/``"baseline"``/``"exposure"``. ``union``/``intersection`` accept two or more inputs
(both are associative/commutative); ``difference`` requires exactly two (``a - b``, order
matters, so there's no n-ary form).

.. code-block:: yaml

    steps:
      - id: bright_branch
        type: cut
        cut: limiting_magnitude
        input: baseline
        params: {mag_limit: 25.0}

      - id: detected_branch
        type: cut
        cut: snr
        input: baseline
        params: {snr_threshold: 5.0}

      - id: agreement
        type: logical_op
        op: intersection
        inputs: [bright_branch, detected_branch]

``action`` steps
^^^^^^^^^^^^^^^^^^

Post-processing that doesn't fit ``cut``'s "one ``EventCatalog`` in, one ``EventCatalog`` out"
shape -- synthetic photometry, a yield summary, or a detection-count table. ``action:`` must name
one of :meth:`SurveySimulator.available_actions()
<uvex_transients.simulation.core.SurveySimulator.available_actions>`, registered the same way cuts
are, via :func:`~uvex_transients.simulation.core.action`. ``inputs:`` is a mapping from that
action's own parameter name(s) to artifact ids (so an action needing several inputs at once, like
``yield`` or ``detection_counts``, can name each one); ``params:`` are its remaining keyword
arguments.

.. list-table:: ``action`` step parameters, by ``action:``
   :header-rows: 1
   :widths: 18 20 20 42

   * - ``action:``
     - ``inputs:`` keys
     - Required params
     - Optional params
   * - ``photometry``
     - ``catalog``
     - --
     - ``bands``, ``n_sigma``
   * - ``yield``
     - ``raw``, ``detected``, ``exposure``
     - --
     - ``confidence`` (default ``0.9``)
   * - ``detection_counts``
     - ``catalog``, ``exposure``, ``photometry``
     - ``snr_threshold``
     - ``confidence`` (default ``0.9``)
   * - ``detection_delay``
     - ``detected``, ``raw``, ``exposure``
     - ``snr_threshold``, ``delays``
     - ``reference`` (``last_nondetection`` or ``explosion``), ``transient_types``, ``lookback``,
       ``bands``, ``confidence`` (default ``0.9``)

``detection_delay`` tabulates, per transient type, how many events are detected within each of a
shared grid of delays (``delays:``, a list of hours), without running a cut once per delay. The
delay is measured from the event's last non-detection to its first detection
(``reference: last_nondetection``, the default) or from the explosion to the first detection
(``reference: explosion``). ``transient_types:`` restricts the table to some types, all sharing the
one grid. ``detected`` is the SNR-cut catalog the delays are measured on, and ``raw`` is the full
sampled catalog, which supplies each type's denominator:

.. code-block:: yaml

    - id: young
      type: action
      action: detection_delay
      inputs: {detected: snr_screen, raw: baseline, exposure: exposure}
      params:
        snr_threshold: 5.0
        delays: [12, 24, 48]
        transient_types: [kilonova, tde]

A ``photometry`` action's output (a plain :class:`~astropy.table.QTable`, one row per (event,
observation, band)) can itself feed a later ``detection_counts`` action by ``id``, exactly like
any other artifact:

.. code-block:: yaml

    steps:
      - id: phot
        type: action
        action: photometry
        inputs: {catalog: snr_screen}

      - id: counts
        type: action
        action: detection_counts
        inputs: {catalog: snr_screen, exposure: exposure, photometry: phot}
        params: {snr_threshold: 5.0}

Checkpointing
^^^^^^^^^^^^^^^

Any step's own ``checkpoint:`` controls whether (and where) ``run`` writes its artifact to
``--out-dir``:

.. list-table:: ``checkpoint:`` values
   :header-rows: 1
   :widths: 18 82

   * - Value
     - Behavior
   * - *(unset, default)*
     - Falls back to the run's own ``keep_intermediate:`` (itself defaulting to ``true``).
   * - ``true`` / ``false``
     - Forces this step to be (or not be) checkpointed, overriding ``keep_intermediate:``.
   * - a string, e.g. ``"my_name.ecsv"``
     - Writes to that explicit filename (relative to ``--out-dir``) instead of the default
       ``{NN}_{id}.ecsv`` numbering convention, and forces the step to be checkpointed regardless
       of ``keep_intermediate:``.

Rerunning ``run`` against the same ``--out-dir`` without ``--overwrite`` **resumes**: any step
whose checkpoint file already exists is loaded from disk instead of being recomputed (and every
step downstream of it then runs against that loaded artifact). ``--overwrite`` disables this --
every step always reruns, replacing any existing checkpoint file.

----

Running Steps From the CLI
-------------------------------

.. code-block:: bash

    uvex-transients run quickstart_tde.yaml --out-dir results/

Samples ``generate:``, tabulates exposure, then runs every declared ``steps:`` entry in order, all
in one process, writing each checkpointed artifact to ``results/`` as it goes: ``baseline.ecsv``
(if checkpointed), ``exposure.ecsv`` (always), then one ``NN_<step id>.ecsv`` per checkpointed
step (or its own explicit ``checkpoint:`` filename). A config with an empty/absent ``steps:`` list
is fine too; ``run`` then does nothing beyond ``generate``/exposure.

Set the config's top-level ``keep_intermediate: false`` (or pass ``--no-keep-intermediate``,
which overrides the config either way) to checkpoint only the steps that explicitly opt in via
their own ``checkpoint:``:

.. code-block:: bash

    uvex-transients run quickstart_tde.yaml --out-dir results/ --no-keep-intermediate

For ad hoc, single-step use (useful for inspecting/debugging one stage without rerunning the
whole pipeline), three more subcommands each pick one declared step by its ``id:`` and run it
against externally supplied file(s), ignoring that step's own configured ``input:``/``inputs:``:

.. code-block:: bash

    # Every declared `cut` step, chained in order, against an explicit input catalog:
    uvex-transients cut quickstart_tde.yaml --in baseline.ecsv --out screened.ecsv

    # Or just one, by id:
    uvex-transients cut quickstart_tde.yaml mag_screen --in baseline.ecsv --out mag_only.ecsv

    # A declared `action: photometry` step, by id:
    uvex-transients photometry quickstart_tde.yaml phot --in screened.ecsv --out photometry.ecsv

    # A declared `action: detection_counts` step, by id:
    uvex-transients detection-counts quickstart_tde.yaml counts \
        --catalog baseline.ecsv --photometry photometry.ecsv --exposure exposure.ecsv \
        --out detection_counts.ecsv

``--catalog`` for ``detection-counts`` is the event catalog ``--photometry`` was computed over
(typically the same catalog ``photometry`` ran against, i.e. whatever survived every declared
cut) -- it's needed alongside the photometry table itself so that events with zero qualifying
epochs (including ones the schedule never observed at all) are still counted at
:math:`N_{\rm det}=0`.

Every command accepts ``--overwrite`` to replace an existing output file/directory contents
instead of raising.

Every command also accepts ``--dry-run``, which validates the config and prints what the real
command would do without sampling, computing photometry, or writing anything:

.. code-block:: bash

    uvex-transients run configs/full_run.yaml --out-dir results/ --dry-run

It resolves every section the command needs, so an unknown transient ``class:``, a bad
``parameters:`` override, an unknown cut/op/action name, a dangling step input, an unknown
mission, or an unreadable schedule fails here exactly as it would in the real run (as a short
``dry run failed: ...`` message). On success it reports the mission, the size of the schedule,
each transient population (class, SED, redshift limit, duration window), the ``generate:``
settings (for ``generate``/``run``), the resolved ``steps:`` list (for ``run``, or the selected
step(s) for ``cut``/``photometry``/``detection-counts``), and each output file it would write. If
a real run would refuse to overwrite one that already exists, the dry run flags it as ``WOULD
FAIL`` and exits non-zero unless ``--overwrite`` is also given. The schedule is loaded (and
downloaded on first use), but the command never reads the ``--in``/``--catalog``/
``--photometry``/``--exposure`` inputs of ``cut``/``photometry``/``detection-counts``, only checks
that they exist.

----

Using It From Python
------------------------

Every command is a thin wrapper. :mod:`uvex_transients.cli.pipeline`'s
:func:`~uvex_transients.cli.pipeline.run_generate`/:func:`~uvex_transients.cli.pipeline.run_exposure`
sample the two reserved artifacts, and :func:`uvex_transients.cli.steps.run_steps` executes the
rest of ``steps:`` against a plain ``{id: artifact}`` dict you seed yourself -- no ``click``
dependency, useful if you want the same config-driven setup inside a notebook or a larger script
instead of a fresh subprocess per stage:

.. code-block:: python

    from uvex_transients.cli.config import RunConfig
    from uvex_transients.cli.pipeline import run_exposure, run_generate
    from uvex_transients.cli.steps import run_steps

    config = RunConfig.from_yaml("quickstart_tde.yaml")

    baseline = run_generate(config)
    exposure = run_exposure(config)

    store = {"baseline": baseline, "exposure": exposure}
    for step, artifact, checkpoint_path in run_steps(config, store):
        print(step.id, type(artifact).__name__)

    screened = store["snr_screen"]     # any step's own artifact, by id
    phot = store["phot"]

``config.simulator``, ``config.schedule``, ``config.mission``, and ``config.transients`` are each
resolved once and cached, so building a ``RunConfig`` and calling the above costs one schedule
fetch and one round of transient construction, no matter how many steps run. Pass ``out_dir=`` to
`run_steps` to also checkpoint (and resume from) files on disk, exactly like the ``run`` command
does.

----

See :mod:`uvex_transients.cli` in the :ref:`api` reference for exhaustive, method-by-method
documentation of every function/class mentioned above.
