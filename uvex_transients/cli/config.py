"""
Parsing/validation for a CLI run-config YAML file.

A single YAML file drives every CLI command (see `uvex_transients.cli.main`); each
command only needs the section(s) relevant to it (`generate:` for ``generate``,
`cuts:` for ``cut``, ...), so `RunConfig` resolves each section **lazily**, on first
access, rather than eagerly validating the whole file up front -- a config missing a
section a given command doesn't need is perfectly valid.
"""

import importlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import m4opt.missions
from astropy import units as u
from astropy.units import Quantity
from m4opt.missions import Mission

from uvex_transients.models.core.priors import Prior
from uvex_transients.simulation.core import SurveySimulator
from uvex_transients.surveys.base import SurveySchedule
from uvex_transients.surveys.utils import get_schedule
from uvex_transients.transients.base import TransientBase

from .yaml_tags import get_run_yaml

_DEFAULT_MISSION = "uvex"

# `run_cut`'s own positional parameters -- a cut's `params:` block may not use these
# names, since they'd otherwise collide with the call `SurveySimulator.run_cut(type, catalog,
# mission, **params)` makes.
_RESERVED_CUT_PARAMS = frozenset({"catalog", "mission"})

# Every `uvex_transients.transients` submodule that defines a concrete `TransientBase`
# subclass. Nothing in `uvex_transients`'s own import chain imports these eagerly -- a
# class only registers once its defining module has actually executed (see
# `TransientBase.registry`) -- so this list must be imported explicitly before any
# `class:` lookup. Adding a 6th transient type means adding its module name here too;
# `tests/cli/test_config.py::test_known_transient_modules_matches_the_package_directory`
# fails loudly if this list and the actual package contents ever drift apart.
_KNOWN_TRANSIENT_MODULES = ("TDEs", "LFBOTs", "kilonovae", "supernovae")


def _import_known_transients() -> None:
    """Import every module in `_KNOWN_TRANSIENT_MODULES` so `TransientBase.registry()` is fully populated."""
    for module_name in _KNOWN_TRANSIENT_MODULES:
        importlib.import_module(f"uvex_transients.transients.{module_name}")


def _parse_quantity(value: Any, default_unit: u.UnitBase) -> Quantity:
    """
    Resolve a YAML value (a `Quantity`, a unit string like ``"200 day"``, or a bare number) to a `Quantity`.

    Parameters
    ----------
    value : ~astropy.units.Quantity, str, or float
        The raw YAML value.
    default_unit : ~astropy.units.UnitBase
        Unit to apply if `value` is a bare number.

    Returns
    -------
    ~astropy.units.Quantity
        The resolved quantity.
    """
    if isinstance(value, Quantity):
        return value
    if isinstance(value, str):
        return Quantity(value)
    return Quantity(value, default_unit)


def _resolve_mission(name: str) -> Mission:
    """
    Resolve a mission name (e.g. ``"uvex"``) to its `m4opt.missions.Mission` instance.

    Parameters
    ----------
    name : str
        The mission's attribute name in `m4opt.missions`.

    Returns
    -------
    m4opt.missions.Mission
        The resolved mission.

    Raises
    ------
    ValueError
        If `name` is not a known `m4opt.missions.Mission` attribute.
    """
    mission = getattr(m4opt.missions, name, None)
    if not isinstance(mission, Mission):
        available = sorted(attr for attr, value in vars(m4opt.missions).items() if isinstance(value, Mission))
        raise ValueError(f"Unknown mission {name!r}; available: {available}.")
    return mission


def _resolve_schedule(section: Mapping) -> SurveySchedule:
    """
    Resolve a ``schedule:`` block's ``name:`` / ``url:`` / ``path:``+``fov_path:`` (mutually exclusive).

    Parameters
    ----------
    section : Mapping
        The parsed ``schedule:`` YAML block.

    Returns
    -------
    SurveySchedule
        The resolved schedule.

    Raises
    ------
    ValueError
        If more than one of ``name``/``url``/``path``+``fov_path`` is given, or
        ``path`` is given without ``fov_path`` (or vice versa).
    """
    name = section.get("name")
    url = section.get("url")
    path = section.get("path")
    fov_path = section.get("fov_path")

    given = [label for label, present in (("name", name), ("url", url), ("path/fov_path", path or fov_path)) if present]
    if len(given) > 1:
        raise ValueError(f"'schedule:' must give at most one of 'name', 'url', or 'path'+'fov_path', got {given}.")

    if path is not None or fov_path is not None:
        if path is None or fov_path is None:
            raise ValueError("'schedule:' with 'path' also requires 'fov_path' (and vice versa).")
        return SurveySchedule.from_disk(path, fov_path=fov_path)

    return get_schedule(name=name, url=url)


def _apply_parameter_overrides(sed, overrides: Mapping) -> None:
    """
    Apply a ``parameters:`` block's per-parameter overrides via `Parameter.fix`/`Parameter.set_prior`.

    Parameters
    ----------
    sed : SpectralModel, Lightcurve, or Spectrum
        The model whose parameters to override, keyed by name.
    overrides : Mapping
        The parsed ``parameters:`` YAML block.

    Raises
    ------
    KeyError
        If `overrides` names a parameter `sed` doesn't have.
    """
    for name, value in overrides.items():
        try:
            parameter = sed[name]
        except KeyError:
            raise KeyError(
                f"{sed.__class__.__name__} has no parameter named {name!r}. Valid parameters are {tuple(sed)}."
            ) from None

        if isinstance(value, Prior):
            parameter.set_prior(value)
        else:
            # A unit string (e.g. "2.0 day") fixes a unit-full parameter; a bare number
            # only works for a genuinely dimensionless one -- `Parameter.fix` itself
            # raises a clear `TypeError` if the two are incompatible.
            parameter.fix(Quantity(value) if isinstance(value, str) else value)


def _resolve_transients(section: Mapping) -> dict[str, TransientBase]:
    """
    Resolve a ``transients:`` block into ``{key: TransientBase instance}``.

    Parameters
    ----------
    section : Mapping
        The parsed ``transients:`` YAML block.

    Returns
    -------
    dict of str to TransientBase
        One constructed, configured transient instance per declared key.

    Raises
    ------
    ValueError
        If `section` is empty, an entry is missing its required ``class`` key,
        names an unknown transient class, or has unrecognized key(s) left over.
    """
    if not section:
        raise ValueError("'transients:' must declare at least one transient type.")

    _import_known_transients()
    registry = TransientBase.registry()

    transients: dict[str, TransientBase] = {}
    for key, raw_entry in section.items():
        entry = dict(raw_entry)

        class_name = entry.pop("class", None)
        if class_name is None:
            raise ValueError(f"transients.{key!r} is missing required key 'class'.")
        try:
            transient_cls = registry[class_name]
        except KeyError:
            raise ValueError(
                f"transients.{key!r}: unknown transient class {class_name!r}; available: {sorted(registry)}."
            ) from None

        cosmology = entry.pop("cosmology", None)
        transient = transient_cls(cosmology=cosmology)

        z_limit = entry.pop("z_limit", None)
        if z_limit is not None:
            transient.redshift_limit = z_limit

        duration_limit = entry.pop("duration_limit", None)
        if duration_limit is not None:
            transient.duration_limit = _parse_quantity(duration_limit, u.day)

        parameters = entry.pop("parameters", {}) or {}
        _apply_parameter_overrides(transient.sed, parameters)

        if entry:
            raise ValueError(f"transients.{key!r} has unknown key(s) {sorted(entry)}.")

        transients[key] = transient

    return transients


def _resolve_cuts(section: Mapping) -> dict[str, "CutSpec"]:
    """
    Resolve a ``cuts:`` block into ``{key: CutSpec}``, validating each ``type:`` against `SurveySimulator`.

    Parameters
    ----------
    section : Mapping
        The parsed ``cuts:`` YAML block.

    Returns
    -------
    dict of str to CutSpec
        One resolved `CutSpec` per declared key, in declared order.

    Raises
    ------
    ValueError
        If an entry is missing its required ``type`` key, names an unknown
        cut type, or uses a reserved parameter name.
    """
    available = SurveySimulator.available_cuts()

    cuts: dict[str, CutSpec] = {}
    for key, raw_entry in section.items():
        entry = dict(raw_entry)

        cut_type = entry.pop("type", None)
        if cut_type is None:
            raise ValueError(f"cuts.{key!r} is missing required key 'type'.")
        if cut_type not in available:
            raise ValueError(f"cuts.{key!r}: unknown cut type {cut_type!r}; available: {list(available)}.")

        reserved = _RESERVED_CUT_PARAMS & set(entry)
        if reserved:
            raise ValueError(f"cuts.{key!r}: params cannot use reserved name(s) {sorted(reserved)}.")

        cuts[key] = CutSpec(type=cut_type, params=entry)

    return cuts


@dataclass
class GenerateConfig:
    """Parsed ``generate:`` section -- see `SurveySimulator.generate_events`."""

    time_bins: int
    nside: int | None = None
    order: str | None = None
    downsample: int | dict[str, int] | None = None


@dataclass
class CutSpec:
    """One resolved entry of a ``cuts:`` section -- a `SurveySimulator.available_cuts()` name plus its params."""

    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class PhotometryConfig:
    """Parsed ``photometry:`` section -- see `EventCatalog.simulate_photometry`."""

    bands: list[str] | None = None
    n_sigma: float | None = None


class RunConfig:
    """
    A parsed CLI run-config, resolving each section lazily on first access.

    Every command reads the same config file; a command only touches the properties it
    actually needs (`generate` reads `.schedule`/`.transients`/`.mission`/`.generate`;
    `cut` reads `.schedule`/`.transients`/`.mission`/`.cuts`; `photometry` reads
    `.schedule`/`.transients`/`.mission`/`.photometry`), so a config missing an unrelated
    section (e.g. no `photometry:` block, if you never run that command) still works.

    Parameters
    ----------
    raw : Mapping
        The parsed run-config YAML, as a nested mapping.
    source : str or ~pathlib.Path, optional
        The config file's path, used only to make error messages more specific.
    """

    def __init__(self, raw: Mapping, source: Path | None = None):
        """
        Store the parsed config; every section is resolved lazily on first access.

        Parameters
        ----------
        raw : Mapping
            The parsed run-config YAML, as a nested mapping.
        source : str or ~pathlib.Path, optional
            The config file's path, used only to make error messages more specific.
        """
        self._raw = raw
        self._source = source

        self._schedule: SurveySchedule | None = None
        self._mission: Mission | None = None
        self._transients: dict[str, TransientBase] | None = None
        self._simulator: SurveySimulator | None = None
        self._generate: GenerateConfig | None = None
        self._cuts: dict[str, CutSpec] | None = None
        self._photometry: PhotometryConfig | None = None
        self._keep_intermediate: bool | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        """
        Parse a run-config YAML file (see `uvex_transients.cli.yaml_tags.get_run_yaml`).

        Parameters
        ----------
        path : str or ~pathlib.Path
            Path to the run-config YAML file.

        Returns
        -------
        RunConfig
            The parsed config.
        """
        path = Path(path)
        with open(path) as f:
            raw = get_run_yaml().load(f) or {}
        return cls(raw, source=path)

    def has_section(self, name: str) -> bool:
        """
        Whether the parsed config has a top-level ``name:`` section at all.

        Parameters
        ----------
        name : str
            The section name to check for.

        Returns
        -------
        bool
            Whether the section is present.
        """
        return name in self._raw

    def _require_section(self, name: str, command: str) -> Mapping:
        """
        Return a required top-level section, raising a clear error if it's missing.

        Parameters
        ----------
        name : str
            The section name to look up.
        command : str
            The CLI command that requires it, used to phrase the error message.

        Returns
        -------
        Mapping
            The section's parsed contents.

        Raises
        ------
        ValueError
            If the section is missing.
        """
        section = self._raw.get(name)
        if section is None:
            where = f" ({self._source})" if self._source else ""
            raise ValueError(f"Config{where} is missing a '{name}:' section, required by the '{command}' command.")
        return section

    @property
    def schedule(self) -> SurveySchedule:
        """
        The resolved `SurveySchedule` (``schedule:`` section; falls back to the package default).

        Returns
        -------
        SurveySchedule
            The resolved schedule.
        """
        if self._schedule is None:
            self._schedule = _resolve_schedule(self._raw.get("schedule") or {})
        return self._schedule

    @property
    def mission(self) -> Mission:
        """
        The resolved `Mission` (``mission:`` section; defaults to ``"uvex"``).

        Returns
        -------
        m4opt.missions.Mission
            The resolved mission.
        """
        if self._mission is None:
            self._mission = _resolve_mission(self._raw.get("mission", _DEFAULT_MISSION))
        return self._mission

    @property
    def transients(self) -> dict[str, TransientBase]:
        """
        The resolved ``{key: TransientBase instance}`` (``transients:`` section, required).

        Returns
        -------
        dict of str to TransientBase
            One constructed, configured transient instance per declared key.
        """
        if self._transients is None:
            self._transients = _resolve_transients(self._require_section("transients", "generate/cut/photometry"))
        return self._transients

    @property
    def simulator(self) -> SurveySimulator:
        """
        A `SurveySimulator` built from `.schedule`/`.transients` (cached across one CLI invocation).

        Returns
        -------
        SurveySimulator
            The resolved simulator.
        """
        if self._simulator is None:
            seed = (self._raw.get("generate") or {}).get("seed")
            self._simulator = SurveySimulator(self.schedule, transients=self.transients, simulation_seed=seed)
        return self._simulator

    @property
    def generate(self) -> GenerateConfig:
        """
        The parsed ``generate:`` section (required by the ``generate`` command).

        Returns
        -------
        GenerateConfig
            The parsed section.
        """
        if self._generate is None:
            section = self._require_section("generate", "generate")
            if "time_bins" not in section:
                raise ValueError("'generate:' is missing required key 'time_bins'.")
            self._generate = GenerateConfig(
                time_bins=section["time_bins"],
                nside=section.get("nside"),
                order=section.get("order"),
                downsample=section.get("downsample"),
            )
        return self._generate

    @property
    def cuts(self) -> dict[str, CutSpec]:
        """
        The parsed ``cuts:`` section, in declared order (required by the ``cut`` command).

        Returns
        -------
        dict of str to CutSpec
            One resolved `CutSpec` per declared key, in declared order.
        """
        if self._cuts is None:
            self._cuts = _resolve_cuts(self._require_section("cuts", "cut"))
        return self._cuts

    @property
    def photometry(self) -> PhotometryConfig:
        """
        The parsed ``photometry:`` section (optional; every field defaults to "every band"/the package default).

        Returns
        -------
        PhotometryConfig
            The parsed section.
        """
        if self._photometry is None:
            section = self._raw.get("photometry") or {}
            self._photometry = PhotometryConfig(bands=section.get("bands"), n_sigma=section.get("n_sigma"))
        return self._photometry

    @property
    def keep_intermediate(self) -> bool:
        """
        Whether the ``run`` command should keep each stage's catalog on disk (top-level ``keep_intermediate:``).

        Defaults to `True`; set to `False` to have ``run`` write only the final photometry
        table, discarding the generated/cut catalogs once the next stage no longer needs them.

        Returns
        -------
        bool
            Whether to keep intermediate stage files.
        """
        if self._keep_intermediate is None:
            self._keep_intermediate = bool(self._raw.get("keep_intermediate", True))
        return self._keep_intermediate
