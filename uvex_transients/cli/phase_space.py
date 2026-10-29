"""
Parsing and running of a phase-space run-config, for ``uvex-transients phase-space``.

A phase-space run-config describes a survey and a list of recovery maps (see
`uvex_transients.phase_space`), and has nothing to do with the ``steps:`` pipeline of `RunConfig`, so it has its own
loader. Its layout is::

    schedule:                   # as for `RunConfig`
      name: uvex_v0.6.0
    mission: uvex_fast
    seed: 0
    overlay: true               # shade the shipped transient classes on every map (or ``{n_draws: 1000}``)

    recovery:                   # shared by every map; each key is overridden by the same key of a map
      snr_threshold: 5.0
      n_visits: 2
      mag_limit: 25.0

    maps:
      - id: fuv_detection       # names the output directory
        recovery: {band: FUV}
      - id: nuv_detection
        recovery: {band: NUV}

Besides ``recovery:``, the shared sections ``transient:``, ``sampler:`` and ``map:`` may be given at the top level and
in a map. Every key that is left out takes its value from ``config["phase_space"]``.
"""

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

#: The keys each section may hold. Anything else is an error, so that a misspelt setting never goes unnoticed.
SECTIONS = {
    "transient": ("spectral_index", "rise_ratio", "fade_mag"),
    "recovery": (
        "band",
        "snr_threshold",
        "n_visits",
        "exclude_first_visit_detections",
        "mag_limit",
        "time_bins",
        "chunk_size",
        "k_target",
        "n_max",
        "nside",
        "order",
    ),
    "sampler": ("n_initial", "batch_size", "max_points", "tolerance", "acquisition"),
    "map": ("log_luminosity", "log_half_time", "grid_size"),
}

_TOP_LEVEL = frozenset({"schedule", "mission", "seed", "overlay", "rate", "maps", *SECTIONS})
_MAP_KEYS = frozenset({"id", *SECTIONS})

# An id names a directory, and is later a path inside the release archive
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class MapSpec:
    """One map of a run: an id and the settings of each section, with the shared ones already merged in."""

    id: str
    """str: Names the map's output directory."""

    transient: dict[str, Any] = field(default_factory=dict)
    """dict: Arguments of `RecoveryEvaluator` that shape the transient (``spectral_index``, ``rise_ratio``, ...)."""

    recovery: dict[str, Any] = field(default_factory=dict)
    """dict: Arguments of `RecoveryEvaluator` that define a recovery (``band``, the detection and the sampling)."""

    sampler: dict[str, Any] = field(default_factory=dict)
    """dict: Arguments of `PhaseSpaceSampler` (``n_initial``, ``batch_size``, ``max_points``, ...)."""

    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None
    """tuple or None: The ranges of the log luminosity and log half-decline time, or `None` for the configured ones."""

    grid_size: int | None = None
    """int or None: Points per axis of the fitted map, or `None` for the configured size."""

    @property
    def band(self) -> str:
        """str: The band of the map, which is the one setting the configuration also holds a default for."""
        return str(self.recovery.get("band", config["phase_space.recovery.band"]))


def _merge(shared: Mapping, own: Mapping, section: str, where: str, errors: list[str]) -> dict:
    """Return `shared` updated by `own`, recording a bad key or a section that is not a mapping in `errors`."""
    merged = {}
    for source, label in ((shared, "the top level"), (own, where)):
        if not isinstance(source, Mapping):
            errors.append(f"{label}: '{section}:' must be a mapping, got {type(source).__name__}.")
            continue
        unknown = sorted(set(source) - set(SECTIONS[section]))
        if unknown:
            errors.append(f"{label}: unknown '{section}:' key(s) {unknown}; allowed: {list(SECTIONS[section])}.")
        merged.update({key: value for key, value in source.items() if key in SECTIONS[section]})
    return merged


def _parse_maps(raw: Mapping) -> list[MapSpec]:
    """Return the maps of a parsed run-config, raising one `ValueError` that lists everything wrong with them."""
    errors: list[str] = []
    entries = raw.get("maps")
    if not isinstance(entries, list) or not entries:
        raise ValueError("The run-config needs a non-empty 'maps:' list.")

    specs, seen = [], set()
    for index, entry in enumerate(entries):
        where = f"maps[{index}]"
        if not isinstance(entry, Mapping):
            errors.append(f"{where}: must be a mapping, got {type(entry).__name__}.")
            continue
        unknown = sorted(set(entry) - _MAP_KEYS)
        if unknown:
            errors.append(f"{where}: unknown key(s) {unknown}; allowed: {sorted(_MAP_KEYS)}.")

        # The id names a directory in the output, and a member path in the release archive
        map_id = entry.get("id")
        if not isinstance(map_id, str) or not _ID.fullmatch(map_id):
            errors.append(f"{where}: 'id' must be a name of letters, digits, '_', '-' and '.', got {map_id!r}.")
            continue
        where = f"map {map_id!r}"
        if map_id in seen:
            errors.append(f"{where}: the id is used twice.")
        seen.add(map_id)

        sections = {name: _merge(raw.get(name) or {}, entry.get(name) or {}, name, where, errors) for name in SECTIONS}
        bounds = None
        if "log_luminosity" in sections["map"] or "log_half_time" in sections["map"]:
            bounds = (
                tuple(sections["map"].get("log_luminosity", config["phase_space.map.log_luminosity"])),
                tuple(sections["map"].get("log_half_time", config["phase_space.map.log_half_time"])),
            )
        specs.append(
            MapSpec(
                id=map_id,
                transient=sections["transient"],
                recovery=sections["recovery"],
                sampler=sections["sampler"],
                bounds=bounds,
                grid_size=sections["map"].get("grid_size"),
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
        unknown = sorted(set(raw) - _TOP_LEVEL)
        if unknown:
            raise ValueError(f"Unknown top-level key(s) {unknown}; allowed: {sorted(_TOP_LEVEL)}.")
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
        """str: The name of the mission, ``config["phase_space.recovery.mission"]`` unless the file gives one."""
        return str(self._raw.get("mission", config["phase_space.recovery.mission"]))

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
    source = f" (config: {phase_space_config.source})" if phase_space_config.source else ""
    bands = list(phase_space_config.mission.detector.bandpasses)
    lines = [f"dry run: phase-space{source}: {len(specs)} map(s) -> {out_dir}"]
    ok = True
    for spec in specs:
        recovery = {key: config[f"phase_space.recovery.{key}"] for key in ("snr_threshold", "n_visits", "mag_limit")}
        recovery.update(spec.recovery)
        problems = []
        if spec.band not in bands:
            problems.append(f"band {spec.band!r} is not one of the mission's {bands}")
        existing = [name for name in ("points.ecsv", "map.ecsv") if (out_dir / spec.id / name).exists()]
        if existing and not (overwrite or resume):
            problems.append(f"{existing} already exist (pass --overwrite or --resume)")
        ok &= not problems
        lines.append(
            f"  {spec.id}: {spec.band}, SNR >= {recovery['snr_threshold']:g} on {recovery['n_visits']} visit(s), "
            f"{recovery['mag_limit']:g} AB" + "".join(f"\n    ERROR: {problem}" for problem in problems)
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
        As for `uvex_transients.phase_space.run.run_map`.
    progress : bool, optional
        Whether to show the progress bars of the simulation.

    Returns
    -------
    dict of str to dict
        For each map id, the files written (see `uvex_transients.phase_space.run.run_map`).
    """
    # Imported here: the optional ``phase_space`` extra (scikit-learn) is needed only by this command
    from uvex_transients.phase_space.evaluator import RecoveryEvaluator
    from uvex_transients.phase_space.run import run_map

    schedule = phase_space_config.schedule
    results = {}
    for spec in specs:
        evaluator = RecoveryEvaluator(
            schedule,
            phase_space_config.mission_name,
            seed=phase_space_config.seed,
            schedule_name=phase_space_config.schedule_name,
            **spec.transient,
            **spec.recovery,
        )
        results[spec.id] = run_map(
            evaluator,
            out_dir / spec.id,
            bounds=spec.bounds,
            sampler_settings=spec.sampler,
            grid_size=spec.grid_size,
            rate=phase_space_config.rate,
            overlay=phase_space_config.overlay,
            overlay_draws=phase_space_config.overlay_draws,
            resume=resume,
            overwrite=overwrite,
            seed=phase_space_config.seed,
            progress=progress,
        )
    return results
