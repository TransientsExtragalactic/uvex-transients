"""
Plain functions implementing each CLI stage, kept free of `click` so they're directly testable.

`uvex_transients.cli.main`'s subcommands are thin wrappers around these (and around
`uvex_transients.cli.steps.run_steps` for everything downstream of the raw generated
catalog) -- each one parses arguments, calls one of these, and reports a short summary.
"""

from collections.abc import Iterable
from pathlib import Path

from ..simulation.event_catalog import EventCatalog
from ..simulation.exposure_catalog import ExposureCatalog
from . import steps
from .config import RunConfig


def run_generate(config: RunConfig) -> EventCatalog:
    """
    Sample a Monte Carlo `EventCatalog` per the config's ``generate:`` section.

    Parameters
    ----------
    config : RunConfig
        The parsed run-config.

    Returns
    -------
    EventCatalog
        The generated catalog -- the ``"baseline"`` artifact every ``steps:`` entry can
        reference (see `uvex_transients.cli.config.RESERVED_ARTIFACT_IDS`).
    """
    settings = config.generate
    return config.simulator.generate_events(
        time_bins=settings.time_bins,
        nside=settings.nside,
        order=settings.order,
        downsample=settings.downsample,
    )


def run_exposure(config: RunConfig) -> ExposureCatalog:
    """
    Tabulate an `ExposureCatalog` per the config's ``generate:`` section.

    Uses the same ``time_bins``/``nside``/``order`` as `run_generate`, so both describe
    the same footprint query -- see `SurveySimulator.compute_effective_exposure`.

    Parameters
    ----------
    config : RunConfig
        The parsed run-config.

    Returns
    -------
    ExposureCatalog
        The tabulated per-(transient type, time bin) exposure -- the ``"exposure"``
        artifact every ``steps:`` entry can reference.
    """
    settings = config.generate
    return config.simulator.compute_effective_exposure(
        time_bins=settings.time_bins,
        nside=settings.nside,
        order=settings.order,
    )


def run_cut_steps(config: RunConfig, catalog: EventCatalog, step_ids: list[str] | None = None) -> EventCatalog:
    """
    Chain one or more of the config's declared ``cut``-type steps against an externally supplied catalog.

    Ad hoc, single-catalog use (the ``cut`` CLI command) -- unlike `steps.run_steps`, this
    ignores each step's own configured ``input:`` (which names a place in the full
    ``steps:`` graph) and instead threads `catalog` through every selected step in turn.

    Parameters
    ----------
    config : RunConfig
        The parsed run-config.
    catalog : EventCatalog
        The catalog to filter.
    step_ids : list of str, optional
        Which of `config.steps`' ``cut``-type step ids to run, and in what order. If
        `None` (the default), runs every declared ``cut`` step, in declared order.

    Returns
    -------
    EventCatalog
        The filtered catalog.

    Raises
    ------
    ValueError
        If a name in `step_ids` isn't a declared step, or names a non-``cut`` step.
    """
    selected = (
        [config.step_by_id(step_id) for step_id in step_ids]
        if step_ids
        else [step for step in config.steps if step.type == "cut"]
    )
    wrong_type = [step.id for step in selected if step.type != "cut"]
    if wrong_type:
        raise ValueError(f"step(s) {wrong_type} are not 'cut' steps.")

    simulator = config.simulator
    mission = config.mission
    for step in selected:
        catalog = steps.apply_cut_scoped(simulator, step.cut, catalog, mission, step.transient_types, **step.params)
    return catalog


def run_photometry_step(config: RunConfig, step_id: str, catalog: EventCatalog):
    """
    Run one ``action``-type, ``action: photometry`` step against an externally supplied catalog.

    Parameters
    ----------
    config : RunConfig
        The parsed run-config.
    step_id : str
        The step's own id in `config.steps`; must be an ``action`` step with
        ``action: photometry``.
    catalog : EventCatalog
        The catalog of events to simulate photometry for.

    Returns
    -------
    ~astropy.table.QTable
        One row per (event, observation, band) synthetic observation.

    Raises
    ------
    ValueError
        If `step_id` isn't declared, or isn't a ``photometry`` action step.
    """
    step = config.step_by_id(step_id)
    if step.type != "action" or step.action != "photometry":
        raise ValueError(f"step {step_id!r} is not a 'photometry' action step.")
    return config.simulator.run_action("photometry", config.mission, catalog=catalog, **step.params)


def dry_run_report(
    config: RunConfig,
    command: str,
    outputs: Iterable[Path] = (),
    overwrite: bool = False,
    step_ids: list[str] | None = None,
) -> tuple[list[str], bool]:
    """
    Validate `config` for `command` and describe what it would do, without sampling or writing anything.

    Resolving each section the command needs is itself the validation: an unknown transient
    class, a bad parameter override, an unknown cut/op/action name or a dangling step input, or
    an unknown mission or unreadable schedule all raise exactly as they would in a real run. The
    (potentially large) schedule *is* loaded, but no events are sampled and no output file is
    created.

    Parameters
    ----------
    config : RunConfig
        The parsed run-config.
    command : {"generate", "cut", "photometry", "run"}
        Which command is being dry-run; decides which config sections are validated. ``"run"``
        validates ``generate:`` and the whole ``steps:`` list; ``"cut"``/``"photometry"``
        validate `step_ids` against `config.steps`.
    outputs : iterable of pathlib.Path, optional
        The files the real command would write, checked for collisions.
    overwrite : bool, optional
        Whether the real command would be run with ``--overwrite``.
    step_ids : list of str, optional
        For ``"cut"``/``"photometry"``, the ad hoc step id(s) that
        would run (see `uvex_transients.cli.main`'s per-command semantics).

    Returns
    -------
    lines : list of str
        The report, one line per entry.
    ok : bool
        `False` if the real command would fail on an output file that already exists.

    Raises
    ------
    ValueError
        If a name in `step_ids` isn't a declared step, or names a step of the wrong type.
    """
    source = f" (config: {config._source})" if config._source else ""
    lines = [f"dry run: {command}{source}"]
    lines.append(f"mission:   {config.mission.name}")
    lines.append(f"schedule:  {len(config.schedule)} scheduled actions")

    transients = config.transients
    lines.append(f"transients ({len(transients)}):")
    for key, transient in transients.items():
        lines.append(
            f"  {key:<20s} {type(transient).__name__:<34s} sed={type(transient.sed).__name__:<32s} "
            f"z<={transient.redshift_limit}  duration={transient.duration_limit}"
        )

    if command in ("generate", "run"):
        settings = config.generate
        seed = (config._raw.get("generate") or {}).get("seed")
        lines.append(
            f"generate:  time_bins={settings.time_bins}, nside={settings.nside or 'default'}, "
            f"order={settings.order or 'default'}, downsample={settings.downsample or 'none'}, seed={seed}"
        )

    if command == "run":
        step_lines = steps.describe_steps(config)
        lines.append(f"steps ({len(step_lines)}, in order):")
        lines.extend(step_lines)

    if command in ("cut", "photometry"):
        expected_type = "cut" if command == "cut" else "action"
        selected = list(step_ids) if step_ids else [s.id for s in config.steps if s.type == expected_type]
        resolved = [config.step_by_id(step_id) for step_id in selected]
        wrong_type = [s.id for s in resolved if s.type != expected_type]
        if wrong_type:
            raise ValueError(f"step(s) {wrong_type} are not {expected_type!r} steps.")
        lines.append(f"steps ({len(resolved)}, in order):")
        lines.extend(steps.describe_step_specs(resolved))

    ok = True
    outputs = list(outputs)
    if outputs:
        lines.append("outputs:")
        for path in outputs:
            if path.exists() and not overwrite:
                lines.append(f"  WOULD FAIL   {path} already exists (pass --overwrite)")
                ok = False
            else:
                lines.append(f"  would write  {path}")

    lines.append("nothing was sampled or written.")
    return lines, ok
