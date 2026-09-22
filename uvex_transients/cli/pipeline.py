"""
Plain functions implementing each CLI stage, kept free of `click` so they're directly testable.

`uvex_transients.cli.main`'s subcommands are thin wrappers around these -- each one parses
arguments, calls one of these, and reports a short summary.
"""

from collections.abc import Iterable
from pathlib import Path

from astropy.table import QTable

from ..simulation.event_catalog import EventCatalog
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
        The generated catalog.
    """
    settings = config.generate
    return config.simulator.generate_events(
        time_bins=settings.time_bins,
        nside=settings.nside,
        order=settings.order,
        downsample=settings.downsample,
    )


def run_cuts(config: RunConfig, catalog: EventCatalog, names: list[str] | None = None) -> EventCatalog:
    """
    Run one or more of the config's declared ``cuts:`` against `catalog`, chained in order.

    Parameters
    ----------
    config : RunConfig
        The parsed run-config.
    catalog : EventCatalog
        The catalog to filter.
    names : list of str, optional
        Which of the config's `cuts:` keys to run, and in what order. If `None` (the
        default), runs every declared cut, in the order it was declared in the config.

    Returns
    -------
    EventCatalog
        The filtered catalog.
    """
    cuts = config.cuts

    if names is None:
        selected = list(cuts)
    else:
        unknown = [name for name in names if name not in cuts]
        if unknown:
            raise ValueError(f"Unknown cut key(s) {unknown}; available: {list(cuts)}.")
        selected = list(names)

    for key in selected:
        spec = cuts[key]
        catalog = config.simulator.run_cut(spec.type, catalog, config.mission, **spec.params)

    return catalog


def run_photometry(config: RunConfig, catalog: EventCatalog) -> QTable:
    """
    Run synthetic photometry over every event in `catalog` per the config's ``photometry:`` section.

    Parameters
    ----------
    config : RunConfig
        The parsed run-config.
    catalog : EventCatalog
        The catalog of events to simulate photometry for.

    Returns
    -------
    ~astropy.table.QTable
        One row per (event, time, band) synthetic observation.
    """
    settings = config.photometry
    return catalog.simulate_photometry(
        config.mission,
        config.transients,
        config.schedule,
        bands=settings.bands,
        n_sigma=settings.n_sigma,
    )


def dry_run_report(
    config: RunConfig,
    command: str,
    outputs: Iterable[Path] = (),
    overwrite: bool = False,
    cut_names: list[str] | None = None,
) -> tuple[list[str], bool]:
    """
    Validate `config` for `command` and describe what it would do, without sampling or writing anything.

    Resolving each section the command needs is itself the validation: an unknown transient
    class, a bad parameter override, an unknown cut type or cut name, an unknown mission, or an
    unreadable schedule all raise exactly as they would in a real run. The (potentially large)
    schedule *is* loaded, but no events are sampled and no output file is created.

    Parameters
    ----------
    config : RunConfig
        The parsed run-config.
    command : {"generate", "cut", "photometry", "run"}
        Which command is being dry-run; decides which config sections are validated. ``"run"``
        validates ``generate:``, ``cuts:`` (if declared) and ``photometry:``.
    outputs : iterable of pathlib.Path, optional
        The files the real command would write, checked for collisions.
    overwrite : bool, optional
        Whether the real command would be run with ``--overwrite``.
    cut_names : list of str, optional
        For ``"cut"``, which of the config's cuts would run (default: every declared cut).

    Returns
    -------
    lines : list of str
        The report, one line per entry.
    ok : bool
        `False` if the real command would fail on an output file that already exists.
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

    if command == "cut" or (command == "run" and config.has_section("cuts")):
        cuts = config.cuts
        selected = list(cuts) if cut_names is None or command == "run" else list(cut_names)
        unknown = [name for name in selected if name not in cuts]
        if unknown:
            raise ValueError(f"Unknown cut key(s) {unknown}; available: {list(cuts)}.")
        lines.append(f"cuts ({len(selected)}, in order):")
        lines.extend(f"  {key:<20s} {cuts[key].type:<20s} {cuts[key].params}" for key in selected)

    if command in ("photometry", "run"):
        settings = config.photometry
        lines.append(f"photometry: bands={settings.bands or 'every band'}, n_sigma={settings.n_sigma or 'default'}")

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
