.. _user_guide_cli:

Command-Line Interface
=========================

Everything covered in :ref:`user_guide_simulation` (sampling an
:class:`~uvex_transients.simulation.event_catalog.EventCatalog`, screening it down with one or
more cuts, and running synthetic photometry) is also available without writing any Python, via
the ``uvex-transients`` console script (:mod:`uvex_transients.cli`). One YAML **run-config** file
describes *what* to simulate (which transients, what schedule, what screening); the CLI's four
subcommands (``generate``, ``cut``, ``photometry``, and ``run``) describe *when* to do each step,
reading and writing plain :class:`~uvex_transients.simulation.event_catalog.EventCatalog` files
between stages.

This page is a hands-on tour of the run-config format and the commands themselves. See
:ref:`user_guide_simulation` for what each stage actually does under the hood; this page is about
driving that same pipeline from a config file and a terminal instead of a Python script.

Quick Look
----------

The repository ships a ready-to-run example, `quickstart_tde.yaml
<https://github.com/TransientsExtragalactic/uvex-transients/blob/main/quickstart_tde.yaml>`__, at
the top of the source tree. Once the package is installed (``pip install -e .`` from a source
checkout registers the ``uvex-transients`` command), running the whole pipeline is one line:

.. code-block:: bash

    uvex-transients run quickstart_tde.yaml --out-dir quickstart_results/

That samples tidal disruption events against the default UVEX schedule, tabulates the survey's
effective exposure to them, screens them by magnitude and then by SNR, summarizes the resulting
yield, and runs synthetic photometry on whatever survives -- leaving every intermediate catalog in
``quickstart_results/``: ``00_generated.ecsv``, ``exposure.ecsv``, one numbered file per cut,
``yield_summary.ecsv``/``yield_summary.txt``, and a final ``photometry.ecsv``. The rest of this
page explains the config file that made that happen and the commands it works with, so you can
build your own.

----

The Run-Config File
----------------------

A run-config is a single YAML file with up to seven top-level sections. Every CLI command reads the
*same* file (there's no separate config per command), but each command only needs the section(s)
it actually touches, resolved lazily by :class:`~uvex_transients.cli.config.RunConfig`: a config
with no ``photometry:`` block is perfectly valid as long as you never run
``uvex-transients photometry`` against it.

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
   * - ``cuts:``
     - ``cut``, ``run``
     - Named screening passes, run in the order they're declared.
   * - ``photometry:``
     - ``photometry``, ``run``
     - :meth:`~uvex_transients.simulation.event_catalog.EventCatalog.simulate_photometry`'s
       optional ``bands``/``n_sigma`` arguments. Optional; omit entirely to use every band at the
       package's default detection significance.
   * - ``yield:``
     - ``run``
     - :meth:`~uvex_transients.simulation.event_catalog.EventCatalog.compute_yield_summary`'s
       optional ``confidence`` argument. Optional; defaults to a ``0.9`` Clopper-Pearson
       confidence level.
   * - ``detection_counts:``
     - ``detection-counts``, ``run`` (if declared)
     - :meth:`~uvex_transients.simulation.photometry_catalog.PhotometryCatalog.compute_detection_count_table`'s
       ``snr_threshold``/``confidence`` arguments. Optional; ``run`` skips this stage entirely if
       the section is omitted.
   * - ``keep_intermediate:``
     - ``run``
     - Whether ``run`` writes each *intermediate* stage's catalog to ``--out-dir``, in addition to
       the final event catalog and photometry table (always written). Optional bool; defaults to
       ``true``. Overridden either way by ``run``'s own ``--keep-intermediate``/
       ``--no-keep-intermediate`` flag.

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
:ref:`user_guide_simulation` for what each field means.

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

    uvex-transients generate quickstart_tde.yaml --out catalog.ecsv

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

Screening With Cuts
-----------------------

.. code-block:: yaml

    cuts:
      mag_screen:
        type: limiting_magnitude
        mag_limit: 25.0
      snr_screen:
        type: snr
        snr_threshold: 5.0

Every entry needs a ``type:`` naming one of
:meth:`SurveySimulator.available_cuts() <uvex_transients.simulation.core.SurveySimulator.available_cuts>`:
the two screening methods from :ref:`user_guide_simulation`, ``limiting_magnitude``
(:meth:`~uvex_transients.simulation.core.SurveySimulator.filter_by_limiting_magnitude`) and
``snr`` (:meth:`~uvex_transients.simulation.core.SurveySimulator.filter_by_snr`), registered via
the :func:`~uvex_transients.simulation.core.cut` decorator. Everything else under a cut is
forwarded straight through as keyword arguments to that method.

.. code-block:: bash

    # Run every cut declared above, in order, chained into one output catalog:
    uvex-transients cut quickstart_tde.yaml --in catalog.ecsv --out screened.ecsv

    # Or run just one, useful for debugging a single stage:
    uvex-transients cut quickstart_tde.yaml mag_screen --in catalog.ecsv --out mag_only.ecsv

.. hint::

   Cuts are ordinary :class:`~uvex_transients.simulation.core.SurveySimulator` methods, not a
   separate plugin class: decorating a new method on a
   :class:`~uvex_transients.simulation.core.SurveySimulator` subclass with ``@cut("my_cut")``
   registers it automatically, immediately usable from a ``type:`` in any run-config.

.. list-table:: ``cuts:`` entry parameters, by ``type:``
   :header-rows: 1
   :widths: 22 24 54

   * - ``type:``
     - Required params
     - Optional params
   * - ``limiting_magnitude``
     - ``mag_limit``
     - ``bands``, ``n_phase``, ``chunk_size``, ``n_visits``
   * - ``snr``
     - ``snr_threshold``
     - ``bands``, ``chunk_size``, ``n_visits``

----

Synthetic Photometry
------------------------

.. code-block:: yaml

    photometry:
      bands: [FUV, NUV]
      n_sigma: 5.0

Both fields are optional; omit the section entirely to evaluate every band the mission's detector
has, at the package's default detection significance (``simulation.detection_n_sigma`` in the
package config; see :ref:`user_guide_simulation`).

.. code-block:: bash

    uvex-transients photometry quickstart_tde.yaml --in screened.ecsv --out photometry.ecsv

The output is one row per (event, observation, band): exactly
:meth:`Event.simulate_photometry <uvex_transients.simulation.event.Event.simulate_photometry>`'s
own column schema, stacked across every event in the input catalog via
:meth:`~uvex_transients.simulation.event_catalog.EventCatalog.simulate_photometry`.

.. list-table:: ``photometry:`` parameters
   :header-rows: 1
   :widths: 18 20 62

   * - Key
     - Type
     - Description
   * - ``bands``
     - list of str
     - Which of the mission detector's bandpasses to evaluate. Optional; defaults to every band.
   * - ``n_sigma``
     - float
     - Width, in multiples of the flux error, of the reported detection interval. Optional;
       defaults to the package's ``simulation.detection_n_sigma``.

----

Yield and Exposure
----------------------

.. code-block:: yaml

    yield:
      confidence: 0.9

Optional; the one field defaults to a ``0.9`` Clopper-Pearson confidence level if the whole
section is omitted. Unlike the other sections, there's no standalone ``yield`` subcommand -- a
yield summary needs a raw (pre-cut) catalog, a detected (post-cut) catalog, *and* an exposure
tabulation all at once (see :ref:`user_guide_simulation_yield`), so it's only ever produced as
part of ``run``, which already has all three in hand.

.. list-table:: ``yield:`` parameters
   :header-rows: 1
   :widths: 18 15 67

   * - Key
     - Type
     - Description
   * - ``confidence``
     - float
     - Confidence level for the Clopper-Pearson binomial bounds. Optional; defaults to ``0.9``.

``run`` (see below) always computes an
:class:`~uvex_transients.simulation.exposure_catalog.ExposureCatalog` and a
:class:`~uvex_transients.simulation.yield_table.YieldTable` alongside the event catalog and
photometry, writing ``exposure.ecsv``, ``yield_summary.ecsv``, and a human-readable
``yield_summary.txt`` into ``--out-dir``.

----

Estimating Detection Counts
--------------------------------

.. code-block:: yaml

    detection_counts:
      snr_threshold: 5.0
      confidence: 0.9

Estimates, per transient type, how many events would show :math:`N_{\rm det}\geq k` detected
epochs, for every :math:`k` at once -- see
:meth:`~uvex_transients.simulation.photometry_catalog.PhotometryCatalog.compute_detection_count_table`
and :ref:`user_guide_simulation_yield`. Unlike ``yield:``, this section is entirely optional even
for ``run``: leaving it out of the config skips this stage everywhere, including ``run``.

.. code-block:: bash

    uvex-transients detection-counts quickstart_tde.yaml \
        --catalog catalog.ecsv --photometry photometry.ecsv --exposure exposure.ecsv \
        --out detection_counts.ecsv

.. list-table:: ``detection_counts:`` parameters
   :header-rows: 1
   :widths: 18 15 67

   * - Key
     - Type
     - Description
   * - ``snr_threshold``
     - float
     - An observation epoch counts as detected if at least one band's SNR exceeds this value.
       Required.
   * - ``confidence``
     - float
     - Confidence level for the Clopper-Pearson binomial bounds. Optional; defaults to ``0.9``.

``--catalog`` is the event catalog ``--photometry`` was computed over (typically the same catalog
``photometry`` ran against, i.e. whatever survived every declared cut) -- it's needed alongside
the photometry table itself so that events with zero qualifying epochs (including ones the
schedule never observed at all) are still counted at :math:`N_{\rm det}=0`. ``--exposure`` is an
exposure catalog as written by ``run`` (or
:meth:`~uvex_transients.simulation.core.SurveySimulator.compute_effective_exposure` directly).

If ``detection_counts:`` is declared, ``run`` runs this stage automatically too, writing
``detection_counts.ecsv`` into ``--out-dir``.

----

Running the Whole Pipeline
------------------------------

.. code-block:: bash

    uvex-transients run quickstart_tde.yaml --out-dir results/

Chains ``generate``, an :class:`~uvex_transients.simulation.exposure_catalog.ExposureCatalog`
tabulation, every declared cut (in order), a
:class:`~uvex_transients.simulation.yield_table.YieldTable` summary, ``photometry``, and (if
``detection_counts:`` is declared) a detection-count table, all in one process, writing each
stage's output to ``results/`` as it goes: ``00_generated.ecsv``, ``exposure.ecsv``, then one
``NN_<cut key>.ecsv`` per cut, then ``yield_summary.ecsv``/``yield_summary.txt``,
``photometry.ecsv``, and (if declared) ``detection_counts.ecsv``. A config with no ``cuts:``
section at all is fine here too; ``run`` goes straight from ``generate`` to the yield summary and
``photometry``.

Every stage's catalog stays in memory and feeds the next stage regardless, so writing the
*intermediate* event catalogs (everything up to, but not including, the catalog that photometry
actually runs against -- ``exposure.ecsv``/``yield_summary.*``/``photometry.ecsv``/
``detection_counts.ecsv`` are always written regardless) is purely for inspection/debugging -- set
the config's top-level ``keep_intermediate: false`` (or pass ``--no-keep-intermediate``, which
overrides the config either way) to have ``run`` skip those and write only ``final_catalog.ecsv``
(the catalog photometry ran against) instead of ``00_generated.ecsv``/``NN_<cut key>.ecsv``:

.. code-block:: bash

    uvex-transients run quickstart_tde.yaml --out-dir results/ --no-keep-intermediate

Every command accepts ``--overwrite`` to replace an existing output file/directory contents
instead of raising.

Every command also accepts ``--dry-run``, which validates the config and prints what the real
command would do without sampling, computing photometry, or writing anything:

.. code-block:: bash

    uvex-transients run configs/full_run.yaml --out-dir results/ --dry-run

It resolves every section the command needs, so an unknown transient ``class:``, a bad
``parameters:`` override, an unknown cut type or cut name, an unknown mission, or an unreadable
schedule fails here exactly as it would in the real run (as a short ``dry run failed: ...``
message). On success it reports the mission, the size of the schedule, each transient population
(class, SED, redshift limit, duration window), the ``generate:``, ``cuts:``, ``photometry:``,
``yield:`` (for ``run``), and ``detection_counts:`` (if declared) settings, and each output file it
would write. If a real run would refuse to overwrite one that already exists, the dry run flags it
as ``WOULD FAIL`` and exits non-zero unless ``--overwrite`` is also given. The schedule is loaded
(and downloaded on first use), but the command never reads the ``--in``/``--catalog``/
``--photometry``/``--exposure`` inputs of ``cut``/``photometry``/``detection-counts``, only checks
that they exist.

----

Using It From Python
------------------------

Every command is a thin wrapper: :mod:`uvex_transients.cli.pipeline`'s
:func:`~uvex_transients.cli.pipeline.run_generate`,
:func:`~uvex_transients.cli.pipeline.run_exposure`,
:func:`~uvex_transients.cli.pipeline.run_cuts`,
:func:`~uvex_transients.cli.pipeline.run_yield_summary`,
:func:`~uvex_transients.cli.pipeline.run_photometry`, and
:func:`~uvex_transients.cli.pipeline.run_detection_counts` take a parsed
:class:`~uvex_transients.cli.config.RunConfig` and do the real work, with no ``click``
dependency, useful if you want the same config-driven setup inside a notebook or a larger script
instead of a fresh subprocess per stage:

.. code-block:: python

    from uvex_transients.cli.config import RunConfig
    from uvex_transients.cli.pipeline import (
        run_cuts, run_detection_counts, run_exposure, run_generate, run_photometry, run_yield_summary,
    )

    config = RunConfig.from_yaml("quickstart_tde.yaml")

    catalog = run_generate(config)
    exposure = run_exposure(config)
    screened = run_cuts(config, catalog)                          # every declared cut, in order
    yields = run_yield_summary(config, catalog, screened, exposure)
    phot = run_photometry(config, screened)
    counts = run_detection_counts(config, screened, exposure, phot)   # only if config has 'detection_counts:'

``config.simulator``, ``config.schedule``, ``config.mission``, and ``config.transients`` are each
resolved once and cached, so building a ``RunConfig`` and calling every function above costs one
schedule fetch and one round of transient construction, no matter how many stages you run.

----

See :mod:`uvex_transients.cli` in the :ref:`api` reference for exhaustive, method-by-method
detail, and :ref:`user_guide_simulation` for the pipeline these commands are driving.
