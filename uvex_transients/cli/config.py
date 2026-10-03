"""
Parsing/validation for a CLI run-config YAML file.

A single YAML file drives every CLI command (see `uvex_transients.cli.main`); each
command only needs the section(s) relevant to it (`generate:` for ``generate``,
`steps:` for ``run``, ...), so `RunConfig` resolves each section **lazily**, on first
access, rather than eagerly validating the whole file up front -- a config missing a
section a given command doesn't need is perfectly valid.
"""

import importlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astropy import units as u
from astropy.units import Quantity
from m4opt.missions import Mission

from uvex_transients.missions import get_mission
from uvex_transients.models.core.priors import Prior
from uvex_transients.simulation.core import SurveySimulator
from uvex_transients.simulation.logical_ops import LOGICAL_OP_ARITY
from uvex_transients.surveys.base import SurveySchedule
from uvex_transients.surveys.utils import get_schedule
from uvex_transients.transients.base import TransientBase

from .yaml_tags import get_run_yaml

_DEFAULT_MISSION = "uvex"

#: Reserved ``steps:`` entry ids auto-registered before any user-declared step runs --
#: see `uvex_transients.cli.steps.run_steps`. A step may not reuse either as its own id.
RESERVED_ARTIFACT_IDS = frozenset({"baseline", "exposure"})

# A cut's `params:` block may not use these names, since they'd otherwise collide with
# the call `SurveySimulator.run_cut(type, catalog, mission, **params)` makes.
_RESERVED_CUT_PARAMS = frozenset({"catalog", "mission"})

_STEP_TYPES = frozenset({"cut", "logical_op", "action"})

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
    Resolve a mission name (e.g. ``"uvex"`` or ``"uvex_fast"``) to its `m4opt.missions.Mission`.

    Defers to `uvex_transients.missions.get_mission`, so the registered downsampled missions
    resolve and the ``missions.downsample`` configuration applies.

    Parameters
    ----------
    name : str
        A mission attribute name in `m4opt.missions`, or a registered fast mission name.

    Returns
    -------
    m4opt.missions.Mission
        The resolved mission.

    Raises
    ------
    ValueError
        If `name` is not a known mission.
    """
    return get_mission(name)


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

        photometry_pre_window = entry.pop("photometry_pre_window", None)
        if photometry_pre_window is not None:
            transient.photometry_pre_window = _parse_quantity(photometry_pre_window, u.day)

        photometry_post_window = entry.pop("photometry_post_window", None)
        if photometry_post_window is not None:
            transient.photometry_post_window = _parse_quantity(photometry_post_window, u.day)

        parameters = entry.pop("parameters", {}) or {}
        _apply_parameter_overrides(transient.sed, parameters)

        if entry:
            raise ValueError(f"transients.{key!r} has unknown key(s) {sorted(entry)}.")

        transients[key] = transient

    return transients


def _resolve_steps(section: list) -> list["StepSpec"]:
    """
    Resolve a ``steps:`` block into an ordered list of `StepSpec`.

    Every step's ``id`` must be unique and distinct from `RESERVED_ARTIFACT_IDS`; every
    input it references (``input:``/``inputs:``) must resolve to `RESERVED_ARTIFACT_IDS`
    or an *earlier* step's own ``id`` (steps run strictly in declared order -- see
    `uvex_transients.cli.steps.run_steps` -- so a forward reference is never valid);
    ``cut:``/``op:``/``action:`` names are checked against `SurveySimulator`/
    `~uvex_transients.simulation.logical_ops.LOGICAL_OPS`; a `logical_op` step's input
    count is checked against `~uvex_transients.simulation.logical_ops.LOGICAL_OP_ARITY`.

    Parameters
    ----------
    section : list
        The parsed ``steps:`` YAML block -- a list of step mappings, in declared order.

    Returns
    -------
    list of StepSpec
        One resolved `StepSpec` per declared entry, in declared order.

    Raises
    ------
    ValueError
        If `section` isn't a list, an entry is missing a required key, uses an unknown
        ``type:``/``cut:``/``op:``/``action:``, reuses an ``id``, references an
        unresolvable input, or violates a `logical_op`'s arity.
    """
    if not isinstance(section, list):
        raise ValueError(f"'steps:' must be a list of step entries, got {type(section).__name__}.")

    available_cuts = SurveySimulator.available_cuts()
    available_actions = SurveySimulator.available_actions()

    steps: list[StepSpec] = []
    seen_ids: set[str] = set(RESERVED_ARTIFACT_IDS)

    for i, raw_entry in enumerate(section):
        entry = dict(raw_entry)
        where = f"steps[{i}]"

        step_id = entry.pop("id", None)
        if step_id is None:
            raise ValueError(f"{where} is missing required key 'id'.")
        where = f"steps.{step_id!r}"
        if step_id in seen_ids:
            raise ValueError(f"{where}: duplicate step id (or reserved id {sorted(RESERVED_ARTIFACT_IDS)}).")

        step_type = entry.pop("type", None)
        if step_type is None:
            raise ValueError(f"{where} is missing required key 'type'.")
        if step_type not in _STEP_TYPES:
            raise ValueError(f"{where}: unknown step type {step_type!r}; available: {sorted(_STEP_TYPES)}.")

        checkpoint = entry.pop("checkpoint", None)
        if checkpoint is not None and not isinstance(checkpoint, (bool, str)):
            raise ValueError(f"{where}: 'checkpoint' must be a bool or a str filename, got {checkpoint!r}.")

        if step_type == "cut":
            cut_name = entry.pop("cut", None)
            if cut_name is None:
                raise ValueError(f"{where} is missing required key 'cut'.")
            if cut_name not in available_cuts:
                raise ValueError(f"{where}: unknown cut {cut_name!r}; available: {list(available_cuts)}.")

            step_input = entry.pop("input", None)
            if step_input is None:
                raise ValueError(f"{where} is missing required key 'input'.")
            if step_input not in seen_ids:
                raise ValueError(f"{where}: input {step_input!r} is not 'baseline'/'exposure' or an earlier step id.")

            transient_types = entry.pop("transient_types", None)
            if transient_types is not None and not isinstance(transient_types, list):
                raise ValueError(f"{where}: 'transient_types' must be a list, got {transient_types!r}.")

            params = entry.pop("params", {}) or {}
            reserved = _RESERVED_CUT_PARAMS & set(params)
            if reserved:
                raise ValueError(f"{where}: params cannot use reserved name(s) {sorted(reserved)}.")

            if entry:
                raise ValueError(f"{where} has unknown key(s) {sorted(entry)}.")

            steps.append(
                StepSpec(
                    id=step_id,
                    type="cut",
                    checkpoint=checkpoint,
                    cut=cut_name,
                    input=step_input,
                    transient_types=transient_types,
                    params=params,
                )
            )

        elif step_type == "logical_op":
            op = entry.pop("op", None)
            if op is None:
                raise ValueError(f"{where} is missing required key 'op'.")
            if op not in LOGICAL_OP_ARITY:
                raise ValueError(f"{where}: unknown op {op!r}; available: {sorted(LOGICAL_OP_ARITY)}.")

            step_inputs = entry.pop("inputs", None)
            if not isinstance(step_inputs, list) or not step_inputs:
                raise ValueError(f"{where}: 'inputs' must be a non-empty list of earlier step/baseline ids.")
            unresolved = [name for name in step_inputs if name not in seen_ids]
            if unresolved:
                raise ValueError(f"{where}: input(s) {unresolved} are not 'baseline'/'exposure' or earlier step ids.")

            lower, upper = LOGICAL_OP_ARITY[op]
            if len(step_inputs) < lower or (upper is not None and len(step_inputs) > upper):
                bound = f"exactly {lower}" if lower == upper else f"at least {lower}"
                raise ValueError(f"{where}: op {op!r} requires {bound} input(s), got {len(step_inputs)}.")

            if entry:
                raise ValueError(f"{where} has unknown key(s) {sorted(entry)}.")

            steps.append(StepSpec(id=step_id, type="logical_op", checkpoint=checkpoint, op=op, inputs=step_inputs))

        else:  # action
            action_name = entry.pop("action", None)
            if action_name is None:
                raise ValueError(f"{where} is missing required key 'action'.")
            if action_name not in available_actions:
                raise ValueError(f"{where}: unknown action {action_name!r}; available: {list(available_actions)}.")

            step_inputs = entry.pop("inputs", None)
            if not isinstance(step_inputs, dict) or not step_inputs:
                raise ValueError(f"{where}: 'inputs' must be a non-empty mapping of {{param name: artifact id}}.")
            unresolved = [name for name in step_inputs.values() if name not in seen_ids]
            if unresolved:
                raise ValueError(f"{where}: input(s) {unresolved} are not 'baseline'/'exposure' or earlier step ids.")

            params = entry.pop("params", {}) or {}

            if entry:
                raise ValueError(f"{where} has unknown key(s) {sorted(entry)}.")

            steps.append(
                StepSpec(
                    id=step_id,
                    type="action",
                    checkpoint=checkpoint,
                    action=action_name,
                    inputs=step_inputs,
                    params=params,
                )
            )

        seen_ids.add(step_id)

    return steps


@dataclass
class GenerateConfig:
    """Parsed ``generate:`` section -- see `SurveySimulator.generate_events`."""

    time_bins: int
    nside: int | None = None
    order: str | None = None
    downsample: int | dict[str, int] | None = None


@dataclass
class StepSpec:
    """
    One resolved entry of a ``steps:`` section.

    Shared across all three ``type:`` values; only the fields relevant to a given
    `type` are populated (see `uvex_transients.cli.steps.run_steps` for how each is
    dispatched).
    """

    id: str
    type: str
    """str: One of ``"cut"``, ``"logical_op"``, ``"action"``."""

    checkpoint: bool | str | None = None
    """bool, str, or None: Whether/where to persist this step's artifact.

    `None` (the default) defers to the run's own ``keep_intermediate`` setting; `True`/
    `False` overrides it for this step alone; a `str` gives an explicit filename
    (relative to ``--out-dir``) instead of the default ``{i:02d}_{id}.<ext>``
    convention, and also forces the step to be persisted regardless of
    ``keep_intermediate``.
    """

    # cut
    cut: str | None = None
    input: str | None = None
    transient_types: list[str] | None = None

    # logical_op
    op: str | None = None

    # action
    action: str | None = None

    # logical_op: list[str]; action: dict[str, str] ({param name: artifact id})
    inputs: list[str] | dict[str, str] | None = None

    params: dict[str, Any] = field(default_factory=dict)


class RunConfig:
    """
    A parsed CLI run-config, resolving each section lazily on first access.

    Every command reads the same config file; a command only touches the properties it
    actually needs (`generate` reads `.schedule`/`.transients`/`.mission`/`.generate`;
    `run` also reads `.steps`), so a config missing an unrelated section (e.g. no
    `steps:` block, if you only ever run ``generate``) still works.

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
        self._steps: list[StepSpec] | None = None
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
    def steps(self) -> list[StepSpec]:
        """
        The parsed ``steps:`` section, in declared order.

        Optional; an entirely absent ``steps:`` section resolves to an empty list, so a
        config that only ever needs the raw generated catalog (``"baseline"``) is valid.

        Returns
        -------
        list of StepSpec
            One resolved `StepSpec` per declared entry, in declared order.
        """
        if self._steps is None:
            self._steps = _resolve_steps(self._raw.get("steps") or [])
        return self._steps

    def step_by_id(self, step_id: str) -> StepSpec:
        """
        Look up one of `steps` by its own ``id`` (used by the ad hoc ``cut``/``photometry`` commands).

        Parameters
        ----------
        step_id : str
            The step's own ``id``.

        Returns
        -------
        StepSpec
            The matching step.

        Raises
        ------
        ValueError
            If no step in `steps` has that ``id``.
        """
        for step in self.steps:
            if step.id == step_id:
                return step
        raise ValueError(f"No step with id {step_id!r}; available: {[step.id for step in self.steps]}.")

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
