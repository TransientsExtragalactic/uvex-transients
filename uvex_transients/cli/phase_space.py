"""
Parsing and running of a phase-space run-config, for ``uvex-transients phase-space``.

A phase-space run-config describes a survey and a list of maps (see `uvex_transients.phase_space`), and has nothing to
do with the ``steps:`` pipeline of `RunConfig`, so it has its own loader. Its layout is::

    schedule:                   # as for `RunConfig`
      name: uvex_v0.6.0
    mission: uvex_fast
    seed: 0
    overlay: true               # shade the shipped transient classes on every map that can (or ``{n_draws: 1000}``)

    detection:                  # shared by every map; each key is overridden by the same key of a map
      snr_threshold: 5.0
      n_visits: 2
    simulation:
      mag_limit: 25.0

    maps:
      - id: fuv_detection       # names the output directory
        detection: {band: FUV}
      - id: nuv_detection
        detection: {band: NUV}

A map is made of what a sampler needs to measure a point and what the campaign does with the measurements, and each
has a section, named for what it decides:

``detection:``
    What counts as a recovery: ``band``, ``snr_threshold``, ``n_visits``, ``exclude_first_visit_detections``, and
    ``cuts``, a list of the simulator's cuts for any other definition (below).
``simulation:``
    How a point is simulated: ``mag_limit`` (the depth that sets the redshift limit), ``time_bins``, ``nside`` and
    ``order``.
``stopping:``
    How many tries a point gets: ``chunk_size``, ``k_target`` and ``n_max``.
``settings:``
    The arguments of the sampler that belong to no section above: the shape of the canonical transient
    (``spectral_index``, ``rise_ratio``, ``fade_mag``), the ``z_max`` of a breakout. What a sampler takes is in
    ``uvex-transients phase-space kinds``.
``axes:``
    The ranges of the axes of the map, by name: ``{log_luminosity: [36.5, 45.0]}``.
``campaign:``
    How the campaign spends its points: ``n_initial``, ``batch_size``, ``max_points``, ``tolerance`` and
    ``acquisition``.
``grid_size:``
    Points per axis of the fitted map.

Every key that is left out takes its value from ``config["phase_space"]``.

``detection:``, ``simulation:``, ``stopping:``, ``settings:`` and ``axes:`` given at the top level are the defaults of
the maps of the top-level ``kind:`` (the canonical one unless it is given) and of no other kind, since what they set
(how many visits make a detection, how finely time is binned, the names of the axes) means something different for each
sampler, and a canonical default that leaked into a breakout map would silently change its definition of a detection.
A map of another kind takes its defaults from the package configuration (the section named for its kind, then the
section of the same name as the run-config's: ``phase_space.detection``, ``phase_space.simulation`` and so on) and its
own sections. ``campaign:`` and ``grid_size:``, which set how points are placed and
not what one is, are shared by every map. To share a section on purpose, use a YAML anchor (``detection: &detection
{...}`` and ``detection: {<<: *detection}``).

By default a detection is ``n_visits`` observations above ``snr_threshold``. Any other definition is a list of
``cuts:``, the cuts a run-config's ``steps:`` use (see
`uvex_transients.simulation.core.SurveySimulator.available_cuts`), applied in order to the events a point draws, with
the survivors counted as recovered. A cut that takes ``snr_threshold``, ``n_visits``,
``exclude_first_visit_detections`` or ``bands`` and is not given them gets the map's::

    maps:
      - id: fuv_bracketed
        detection:
          band: FUV
          cuts:
            - cut: snr
              params: {n_visits: 2}
            - cut: time_since_last_nondetection
              params: {max_delay: 1.0}  # days

A map is of the canonical transient unless it names another ``kind:`` of sampler (see
`uvex_transients.phase_space.sampling.base.available_samplers`)::

    maps:
      - id: breakout_fuv
        kind: breakout
        detection: {band: FUV}
        simulation: {mag_limit: 25.0}
        settings: {z_max: 8.0}
        axes: {log_L0: [43.0, 46.0], log_t0: [3.0, 4.0]}
        grid_size: 40

Each map is run as an `~uvex_transients.phase_space.experiment.Experiment`, and writes ``points.ecsv``, ``map.ecsv`` and
``map.png``, and ``overlay.ecsv`` if its sampler has an overlay (the canonical one does).

The names of the sections of an earlier layout (``recovery:``, ``sampler:``, ``transient:`` and ``map:``) are refused
with what replaces them.

The run-config says what to simulate. What is done with the files afterwards (fitting, drawing, comparing) is the job of
the other commands of ``uvex-transients phase-space``, which take the directories of maps and no run-config; see
`uvex_transients.phase_space.stages`.
"""

import functools
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astropy import units as u
from m4opt.missions import Mission

from uvex_transients.surveys.base import SurveySchedule
from uvex_transients.utils import config

from .config import _resolve_mission, _resolve_schedule
from .yaml_tags import get_run_yaml

__all__ = ["MapSpec", "PhaseSpaceConfig", "dry_run_report", "run_maps"]

#: The keys each section may hold. Anything else is an error, so that a misspelt setting never goes unnoticed. `None`
#: is a section whose keys belong to the sampler, which says what it does not take when it is built.
SECTIONS = {
    "detection": ("band", "snr_threshold", "n_visits", "exclude_first_visit_detections", "cuts"),
    "simulation": ("mag_limit", "time_bins", "nside", "order"),
    "stopping": ("chunk_size", "k_target", "n_max"),
    "campaign": ("n_initial", "batch_size", "max_points", "tolerance", "acquisition"),
    "settings": None,
    "axes": None,
}

#: The sections that are the defaults of the top-level kind only. ``campaign:`` is shared by all.
_KIND_SPECIFIC = ("detection", "simulation", "stopping", "settings", "axes")

#: The names of an earlier layout, and what replaced them.
_RENAMED = {
    "recovery": (
        "it is split into 'detection:' (band, snr_threshold, n_visits, exclude_first_visit_detections, and its "
        "'detection:' list as 'cuts:'), 'simulation:' (mag_limit, time_bins, nside, order) and 'stopping:' "
        "(chunk_size, k_target, n_max)"
    ),
    "sampler": "it is now 'campaign:'",
    "transient": "its keys are now in 'settings:'",
    "map": "the ranges of the axes are now 'axes:' (by name) and 'grid_size:' is now a key of its own",
}

_TOP_LEVEL = frozenset({"schedule", "mission", "seed", "overlay", "rate", "maps", "kind", "grid_size", *SECTIONS})
_MAP_KEYS = frozenset({"id", "kind", "grid_size", *SECTIONS})

# An id names a directory, and is later a path inside the release archive
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def _unknown_keys(keys, allowed, where: str) -> str | None:
    """Return the message about the keys of `where` not in `allowed`, or `None`, naming any that were renamed."""
    unknown = sorted(set(keys) - set(allowed))
    if not unknown:
        return None
    message = f"{where}: unknown key(s) {unknown}; allowed: {sorted(allowed)}."
    renamed = [f"'{key}:' was renamed ({_RENAMED[key]})" for key in unknown if key in _RENAMED]
    return message + (" " + "; ".join(renamed) + "." if renamed else "")


@dataclass(frozen=True)
class MapSpec:
    """One map of a run: an id, a kind and the settings of each section, with the shared ones already merged in."""

    id: str
    """str: Names the map's output directory."""

    kind: str = "canonical"
    """str: The kind of sampler of the map, one of `~uvex_transients.phase_space.sampling.base.available_samplers`."""

    detection: dict[str, Any] = field(default_factory=dict)
    """dict: What counts as a recovery (``band``, ``snr_threshold``, ``n_visits``, ``cuts``, ...)."""

    simulation: dict[str, Any] = field(default_factory=dict)
    """dict: How a point is simulated (``mag_limit``, ``time_bins``, ``nside``, ``order``)."""

    stopping: dict[str, Any] = field(default_factory=dict)
    """dict: How many tries a point gets (``chunk_size``, ``k_target``, ``n_max``)."""

    campaign: dict[str, Any] = field(default_factory=dict)
    """dict: How the campaign spends its points (``n_initial``, ``batch_size``, ``max_points``, ...)."""

    settings: dict[str, Any] = field(default_factory=dict)
    """dict: The arguments of the sampler that belong to no other section."""

    axes: dict[str, tuple[float, float]] | None = None
    """dict or None: The ranges of the axes of the map, by the name of the axis, or `None` for the sampler's own."""

    grid_size: int | None = None
    """int or None: Points per axis of the fitted map, or `None` for the configured size."""

    @property
    def band(self) -> str:
        """str: The band of the map, which is the one setting the configuration also holds a default for."""
        from uvex_transients.phase_space.sampling import SurveySettings, get_sampler

        return str(self.detection.get("band", SurveySettings.default("band", get_sampler(self.kind).kinds())))

    @property
    def sampler_options(self) -> dict[str, Any]:
        """
        dict: The arguments the sampler is built with: the detection, simulation and ``settings:`` sections, merged.

        The ``cuts:`` of the detection are the sampler's ``detection`` argument.
        """
        options = {**self.detection, **self.simulation, **self.settings}
        if "cuts" in options:
            options["detection"] = options.pop("cuts")
        return options


def _merge(shared: Mapping, own: Mapping, section: str, where: str, errors: list[str]) -> dict:
    """Return `shared` updated by `own`, recording a bad key or a section that is not a mapping in `errors`."""
    merged = {}
    for source, label in ((shared, "the top level"), (own, where)):
        if not isinstance(source, Mapping):
            errors.append(f"{label}: '{section}:' must be a mapping, got {type(source).__name__}.")
            continue
        if SECTIONS[section] is None:
            merged.update(source)
            continue
        problem = _unknown_keys(source, SECTIONS[section], f"{label}: '{section}:'")
        if problem:
            errors.append(problem)
        merged.update({key: value for key, value in source.items() if key in SECTIONS[section]})
    return merged


def _parse_maps(raw: Mapping) -> list[MapSpec]:
    """Return the maps of a parsed run-config, raising one `ValueError` that lists everything wrong with them."""
    errors: list[str] = []
    entries = raw.get("maps")
    if not isinstance(entries, list) or not entries:
        raise ValueError("The run-config needs a non-empty 'maps:' list.")

    from uvex_transients.phase_space.sampling import available_samplers, check_detection

    top_kind = str(raw.get("kind", "canonical"))
    specs, seen = [], set()
    for index, entry in enumerate(entries):
        where = f"maps[{index}]"
        if not isinstance(entry, Mapping):
            errors.append(f"{where}: must be a mapping, got {type(entry).__name__}.")
            continue
        problem = _unknown_keys(entry, _MAP_KEYS, where)
        if problem:
            errors.append(problem)

        # The id names a directory in the output, and a member path in the release archive
        map_id = entry.get("id")
        if not isinstance(map_id, str) or not _ID.fullmatch(map_id):
            errors.append(f"{where}: 'id' must be a name of letters, digits, '_', '-' and '.', got {map_id!r}.")
            continue
        where = f"map {map_id!r}"
        if map_id in seen:
            errors.append(f"{where}: the id is used twice.")
        seen.add(map_id)

        kind = str(entry.get("kind", top_kind))
        if kind not in available_samplers():
            errors.append(f"{where}: unknown kind {kind!r}; the registered kinds are {list(available_samplers())}.")
        sections = {
            name: _merge(
                raw.get(name) or {} if name not in _KIND_SPECIFIC or kind == top_kind else {},
                entry.get(name) or {},
                name,
                where,
                errors,
            )
            for name in SECTIONS
        }
        if "cuts" in sections["detection"]:
            try:
                check_detection(sections["detection"]["cuts"])
            except ValueError as error:
                errors.append(f"{where}: 'detection: cuts:' is invalid: {error}")

        axes = sections["axes"]
        if axes and not all(
            isinstance(pair, (list, tuple)) and len(pair) == 2 and all(isinstance(v, (int, float)) for v in pair)
            for pair in axes.values()
        ):
            errors.append(f"{where}: 'axes:' must map each axis name to [lo, hi].")
            axes = {}
        specs.append(
            MapSpec(
                id=map_id,
                kind=kind,
                detection=sections["detection"],
                simulation=sections["simulation"],
                stopping=sections["stopping"],
                campaign=sections["campaign"],
                settings=sections["settings"],
                axes={name: (float(pair[0]), float(pair[1])) for name, pair in axes.items()} if axes else None,
                grid_size=entry.get("grid_size", raw.get("grid_size")),
            )
        )

    if errors:
        raise ValueError("Invalid phase-space run-config:\n" + "\n".join(f"  - {error}" for error in errors))
    return specs


class PhaseSpaceConfig:
    """
    A parsed phase-space run-config, resolving the survey lazily on first access.

    Parameters
    ----------
    raw : Mapping
        The parsed run-config YAML, as a nested mapping.
    source : str or ~pathlib.Path, optional
        The config file's path, used only to make error messages more specific.

    Raises
    ------
    ValueError
        If the file has an unknown key anywhere, or a map without a valid, unique id. Every problem is reported at once.
    """

    def __init__(self, raw: Mapping, source: Path | None = None):
        self._raw = raw
        self.source = source

        # Everything that can be checked without loading a schedule is checked now
        problem = _unknown_keys(raw, _TOP_LEVEL, "The run-config")
        if problem:
            raise ValueError(problem)
        self.maps = _parse_maps(raw)
        self.seed = int(raw.get("seed", 0))

        overlay = raw.get("overlay", False)
        self.overlay = bool(overlay)
        self.overlay_draws = overlay.get("n_draws") if isinstance(overlay, Mapping) else None
        rate = raw.get("rate")
        self.rate = None if rate is None else float(rate) / u.Gpc**3 / u.yr

        self._schedule: SurveySchedule | None = None
        self._mission: Mission | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PhaseSpaceConfig":
        """
        Parse a phase-space run-config YAML file.

        Parameters
        ----------
        path : str or ~pathlib.Path
            Path to the run-config YAML file.

        Returns
        -------
        PhaseSpaceConfig
            The parsed config.
        """
        path = Path(path)
        with open(path) as f:
            raw = get_run_yaml().load(f) or {}
        return cls(raw, source=path)

    @property
    def schedule_name(self) -> str | None:
        """Str or None: The registry name of the schedule, recorded in each map; `None` if it has none."""
        section = self._raw.get("schedule") or {}
        if section.get("url") or section.get("path"):
            return None
        return section.get("name") or config["schedules.default_schedule"]

    @property
    def mission_name(self) -> str:
        """str: The name of the mission, ``config["phase_space.mission"]`` unless the file gives one."""
        return str(self._raw.get("mission", config["phase_space.mission"]))

    @property
    def schedule(self) -> SurveySchedule:
        """SurveySchedule: The resolved schedule."""
        if self._schedule is None:
            self._schedule = _resolve_schedule(self._raw.get("schedule") or {})
        return self._schedule

    @property
    def mission(self) -> Mission:
        """m4opt.missions.Mission: The resolved mission."""
        if self._mission is None:
            self._mission = _resolve_mission(self.mission_name)
        return self._mission

    def select(self, ids: tuple[str, ...] | list[str] | None) -> list[MapSpec]:
        """
        Return the maps with the given ids, or all of them if `ids` is empty.

        Parameters
        ----------
        ids : sequence of str or None
            The ids to keep.

        Returns
        -------
        list of MapSpec
            The selected maps, in the order of the file.

        Raises
        ------
        ValueError
            If an id is not in the file.
        """
        if not ids:
            return list(self.maps)
        known = [spec.id for spec in self.maps]
        missing = sorted(set(ids) - set(known))
        if missing:
            raise ValueError(f"No map(s) {missing} in the run-config; it has {known}.")
        return [spec for spec in self.maps if spec.id in ids]


def dry_run_report(
    phase_space_config: PhaseSpaceConfig, out_dir: Path, specs: list[MapSpec], overwrite: bool, resume: bool
) -> tuple[list[str], bool]:
    """
    Describe what a run would do, without loading the schedule, sampling or writing anything.

    Parameters
    ----------
    phase_space_config : PhaseSpaceConfig
        The parsed run-config.
    out_dir : pathlib.Path
        The directory the run would write to.
    specs : list of MapSpec
        The maps that would run.
    overwrite, resume : bool
        The flags the run would have.

    Returns
    -------
    lines : list of str
        The report.
    ok : bool
        Whether the run would proceed: false if a band is not one of the mission's or a map would hit existing output.
    """
    from uvex_transients.phase_space.sampling import SurveySettings, get_sampler

    source = f" (config: {phase_space_config.source})" if phase_space_config.source else ""
    bands = list(phase_space_config.mission.detector.bandpasses)
    lines = [f"dry run: phase-space{source}: {len(specs)} map(s) -> {out_dir}"]
    ok = True
    for spec in specs:
        kinds = get_sampler(spec.kind).kinds()
        settings = {key: SurveySettings.default(key, kinds) for key in ("snr_threshold", "n_visits", "mag_limit")}
        settings.update({"band": spec.band, **spec.sampler_options})
        problems = []
        if spec.band not in bands:
            problems.append(f"band {spec.band!r} is not one of the mission's {bands}")
        existing = [name for name in ("points.ecsv", "map.ecsv") if (out_dir / spec.id / name).exists()]
        if existing and not (overwrite or resume):
            problems.append(f"{existing} already exist (pass --overwrite or --resume)")
        ok &= not problems
        kind = "" if spec.kind == "canonical" else f" [{spec.kind}]"
        try:
            what = get_sampler(spec.kind).describe(settings)
        except KeyError:  # a sampler that describes itself with a setting the run-config leaves to its default
            what = spec.band
        lines.append(
            f"  {spec.id}{kind}: {what}, {settings['mag_limit']:g} AB"
            + "".join(f"\n    ERROR: {problem}" for problem in problems)
        )
    return lines, ok


def run_maps(
    phase_space_config: PhaseSpaceConfig,
    out_dir: Path,
    specs: list[MapSpec],
    *,
    overwrite: bool = False,
    resume: bool = False,
    progress: bool = False,
    workers: int = 1,
    products: bool = True,
) -> dict[str, dict[str, Path]]:
    """
    Make the maps, one after the other, each in its own subdirectory of `out_dir`.

    Parameters
    ----------
    phase_space_config : PhaseSpaceConfig
        The parsed run-config.
    out_dir : pathlib.Path
        The directory to write to; each map goes to ``out_dir / <id>``.
    specs : list of MapSpec
        The maps to make.
    overwrite, resume : bool, optional
        As for `uvex_transients.phase_space.stages.run_experiment`.
    progress : bool, optional
        Whether to show the progress bars of the simulation.
    workers : int, optional
        The number of processes that measure the points of a batch; 0 is one for each CPU. The results do not depend on
        it, since each point has its own seed. The default is 1.
    products : bool, optional
        Whether to fit and draw each map after sampling it. If `False`, only ``points.ecsv`` is written, and the rest
        is made with `uvex_transients.phase_space.stages.fit_map` and `plot_files`.

    Returns
    -------
    dict of str to dict
        For each map id, the files written (see `uvex_transients.phase_space.stages.run_experiment`).
    """
    return {
        spec.id: _run_experiment_map(
            phase_space_config,
            spec,
            out_dir,
            overwrite=overwrite,
            resume=resume,
            progress=progress,
            workers=workers,
            products=products,
        )
        for spec in specs
    }


def _run_experiment_map(
    phase_space_config: PhaseSpaceConfig,
    spec: MapSpec,
    out_dir: Path,
    *,
    overwrite: bool,
    resume: bool,
    progress: bool,
    workers: int = 1,
    products: bool = True,
) -> dict[str, Path]:
    """
    Make one map, of any kind of sampler, as an experiment.

    The map is run with `~uvex_transients.phase_space.stages.run_experiment`.

    Parameters
    ----------
    phase_space_config : PhaseSpaceConfig
        The parsed run-config.
    out_dir : pathlib.Path
        The directory to write to; the map goes to ``out_dir / <id>``.
    spec : MapSpec
        The map, of a registered kind.
    overwrite, resume : bool
        As for `uvex_transients.phase_space.stages.run_experiment`.
    progress : bool
        Whether to show the progress bars of the simulation.
    workers : int, optional
        The processes that measure a batch (see `run_maps`).
    products : bool, optional
        Whether to fit and draw the map after sampling it.

    Returns
    -------
    dict of str to pathlib.Path
        The files written (see `uvex_transients.phase_space.stages.run_experiment`).

    Raises
    ------
    ValueError
        If the ``axes:`` name an axis that the sampler does not have, or the sampler does not take a setting.
    """
    from uvex_transients.phase_space.experiment import CampaignSettings, Experiment
    from uvex_transients.phase_space.sampling import StoppingRule, get_sampler
    from uvex_transients.phase_space.stages import run_experiment

    sampler = get_sampler(spec.kind).from_run_config(phase_space_config, **spec.sampler_options)

    bounds = None
    if spec.axes is not None:
        unknown = sorted(set(spec.axes) - set(sampler.space.names))
        if unknown:
            raise ValueError(f"Map {spec.id!r}: no axis {unknown}; the axes are {list(sampler.space.names)}.")
        bounds = [spec.axes.get(axis.name, axis.bounds) for axis in sampler.space]

    campaign_keys = ("n_initial", "batch_size", "max_points", "tolerance")
    experiment = Experiment.create(
        sampler,
        bounds=bounds,
        rule=StoppingRule.from_config(kinds=sampler.kinds(), **spec.stopping),
        campaign=CampaignSettings.from_config(
            kinds=sampler.kinds(),
            seed=phase_space_config.seed,
            **{key: spec.campaign[key] for key in campaign_keys if key in spec.campaign},
        ),
        seed=phase_space_config.seed,
        **{key: spec.campaign[key] for key in ("acquisition",) if key in spec.campaign},
    )
    run = functools.partial(
        run_experiment,
        experiment,
        out_dir / spec.id,
        grid_size=spec.grid_size,
        rate=phase_space_config.rate,
        overlay=phase_space_config.overlay and sampler.HAS_OVERLAY,
        overlay_draws=phase_space_config.overlay_draws,
        resume=resume,
        overwrite=overwrite,
        progress=progress,
        products=products,
    )
    processes = (os.cpu_count() or 1) if workers == 0 else workers
    if processes <= 1:
        return run()

    # The workers are forked, since a sampler that holds a schedule cannot be sent to a process pool
    from uvex_transients.phase_space.experiment import fork_map

    with fork_map(sampler, experiment.rule, processes) as map_fn:
        return run(map_fn=map_fn)
