"""
Executes a `RunConfig`'s ``steps:`` list.

The config-driven generalization of the old hardcoded ``cuts -> photometry -> ...``
sequence. Each `~uvex_transients.cli.config.StepSpec` (``cut``, ``logical_op``, or ``action``) is
run strictly in declared order against a growing **artifact store**
(``{id: artifact}``), seeded by the caller with at least ``"baseline"`` (the raw
generated `~uvex_transients.simulation.event_catalog.EventCatalog`) and typically also
``"exposure"`` (see `uvex_transients.cli.config.RESERVED_ARTIFACT_IDS`). A step's own
``id`` becomes its artifact's key once it's run, so any later step can reference it as
an input regardless of its type -- a `cut` step feeds another `cut` or a `logical_op`
just as an `action` step (e.g. ``photometry``) can feed another `action` (e.g.
``detection_counts``).

Structural validation (unique ids, resolvable inputs, known registry names, arity) is
already done by `~uvex_transients.cli.config.RunConfig.steps` itself, the moment it's
accessed -- accessing it is the "validate the whole graph before running anything"
pass; `run_steps` does this up front, before executing step 1, by simply reading
`config.steps` before its loop starts.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from astropy.table import QTable

from ..simulation.event_catalog import EventCatalog
from ..simulation.logical_ops import LOGICAL_OPS
from ..simulation.yield_table import YieldTable
from .config import RunConfig, StepSpec

#: The artifact type each ``action`` name produces -- used to pick the right
#: `from_disk`/`QTable.read` when resuming a checkpointed action step. `cut`/
#: `logical_op` steps always produce an `EventCatalog`, so they don't need an entry
#: here.
_ACTION_OUTPUT_TYPES: dict[str, type] = {
    "photometry": QTable,
    "yield": YieldTable,
    "detection_counts": QTable,
    "alert": QTable,
    "detection_delay": QTable,
}


def _artifact_type(step: StepSpec) -> type:
    """Return the artifact type `step` produces, used to pick a checkpoint (de)serializer."""
    if step.type in ("cut", "logical_op"):
        return EventCatalog
    return _ACTION_OUTPUT_TYPES[step.action]


def _checkpoint_path(out_dir: Path, index: int, step: StepSpec) -> Path:
    """
    Resolve a step's checkpoint file path under `out_dir`.

    Parameters
    ----------
    out_dir : ~pathlib.Path
        The run's output directory.
    index : int
        This step's 1-based position in `RunConfig.steps`, used only for the default
        filename's numeric prefix.
    step : StepSpec
        The step being checkpointed.

    Returns
    -------
    ~pathlib.Path
        `out_dir / step.checkpoint` if `step.checkpoint` is a `str` (an explicit
        filename); otherwise the default ``{index:02d}_{step.id}.ecsv`` convention.
    """
    if isinstance(step.checkpoint, str):
        return out_dir / step.checkpoint
    return out_dir / f"{index:02d}_{step.id}.ecsv"


def _save_artifact(artifact: Any, path: Path, overwrite: bool) -> None:
    """Write `artifact` to `path`, via `to_disk` (every catalog/table type here has one) or `QTable.write`."""
    if hasattr(artifact, "to_disk"):
        artifact.to_disk(path, overwrite=overwrite)
    else:
        artifact.write(path, overwrite=overwrite)


def _load_artifact(path: Path, artifact_type: type) -> Any:
    """Read an artifact of `artifact_type` back from `path`, as written by `_save_artifact`."""
    if hasattr(artifact_type, "from_disk"):
        return artifact_type.from_disk(path)
    return QTable.read(path)


def apply_cut_scoped(
    simulator, cut_name: str, catalog: EventCatalog, mission, transient_types: list[str] | None, **params
) -> EventCatalog:
    """
    Run a `cut` step, optionally restricted to a subset of ``transient_type`` values.

    Generic at the executor level (not inside any individual ``@cut`` method, per
    `SurveySimulator.run_cut`'s own docstring): with `transient_types` given, only rows
    whose ``transient_type`` is in that list are evaluated against the cut; every other
    row passes through unchanged, regardless of what the cut would have done to it.

    Parameters
    ----------
    simulator : ~uvex_transients.simulation.core.SurveySimulator
        Supplies `~uvex_transients.simulation.core.SurveySimulator.run_cut`.
    cut_name : str
        One of `~uvex_transients.simulation.core.SurveySimulator.available_cuts`.
    catalog : EventCatalog
        The catalog to filter.
    mission : m4opt.missions.Mission
        Forwarded to `run_cut`.
    transient_types : list of str, optional
        If given, restricts the cut to rows with a matching ``transient_type``; other
        rows are kept untouched. `None` runs the cut over the whole catalog, exactly
        like calling `run_cut` directly.
    **params
        Forwarded to `run_cut`.

    Returns
    -------
    EventCatalog
        The filtered catalog.
    """
    if transient_types is None:
        return simulator.run_cut(cut_name, catalog, mission, **params)

    table = catalog.table
    mask = np.isin(np.asarray(table["transient_type"]), transient_types)

    scoped = EventCatalog(
        table=table[mask],
        nside=catalog.nside,
        order=catalog.order,
        time_bins=catalog.time_bins,
        seed=catalog.seed,
        downsample=catalog.downsample,
    )
    filtered_scoped = simulator.run_cut(cut_name, scoped, mission, **params)

    kept_ids = set(np.asarray(filtered_scoped.table["event_id"]).tolist()) | set(
        np.asarray(table["event_id"])[~mask].tolist()
    )
    keep = np.isin(np.asarray(table["event_id"]), list(kept_ids))
    return EventCatalog(
        table=table[keep],
        nside=catalog.nside,
        order=catalog.order,
        time_bins=catalog.time_bins,
        seed=catalog.seed,
        downsample=catalog.downsample,
    )


def checkpoint_targets(config: RunConfig, keep_intermediate: bool) -> list[tuple[StepSpec, Path | None]]:
    """
    Compute each step's checkpoint path, or `None`, without running anything.

    For each step in `config.steps`, the checkpoint path it would be written under
    (relative to ``--out-dir``), or `None` if it wouldn't be checkpointed at all. Used
    both by `run_steps` (each `(step, path)` pair reproduces exactly what it computes
    internally) and by a dry run / progress listing that needs to know this ahead of time.

    Parameters
    ----------
    config : RunConfig
        The parsed run-config.
    keep_intermediate : bool
        The default "should this step be checkpointed" answer for a step whose own
        `~uvex_transients.cli.config.StepSpec.checkpoint` is `None`.

    Returns
    -------
    list of (StepSpec, Path or None)
        One entry per declared step, in declared order.
    """
    targets = []
    for i, step in enumerate(config.steps, start=1):
        effective_checkpoint = step.checkpoint if step.checkpoint is not None else keep_intermediate
        targets.append((step, _checkpoint_path(Path(), i, step) if effective_checkpoint else None))
    return targets


def run_steps(
    config: RunConfig,
    store: dict[str, Any],
    out_dir: Path | None = None,
    overwrite: bool = False,
    keep_intermediate: bool = True,
) -> Iterator[tuple[StepSpec, Any, Path | None]]:
    """
    Run every step in `config.steps`, in order, against `store`.

    Parameters
    ----------
    config : RunConfig
        The parsed run-config; `config.steps` is accessed here (forcing its own
        structural validation, if not already resolved) before the first step runs.
    store : dict of str to Any
        The artifact store, seeded by the caller with at least ``"baseline"`` (and
        typically ``"exposure"``) -- see `uvex_transients.cli.config.RESERVED_ARTIFACT_IDS`.
        Mutated in place: each step's own ``id`` is added once it's run (or loaded from
        an existing checkpoint).
    out_dir : ~pathlib.Path, optional
        Directory to read/write checkpoint files from/to. If `None`, no step is ever
        checkpointed, regardless of its own `checkpoint`/`keep_intermediate`.
    overwrite : bool, optional
        Whether an existing checkpoint file may be overwritten -- and, symmetrically,
        whether an existing one is trusted and loaded instead of the step being rerun.
        With `overwrite=True`, every step always reruns and any existing checkpoint is
        replaced.
    keep_intermediate : bool, optional
        The default "should this step be checkpointed" answer for a step whose own
        `~uvex_transients.cli.config.StepSpec.checkpoint` is `None`. The default is
        `True`.

    Yields
    ------
    step : StepSpec
        The step just run (or loaded from checkpoint).
    artifact : Any
        Its resulting artifact (also written into `store[step.id]`).
    path : ~pathlib.Path or None
        The checkpoint file it was written to or loaded from, or `None` if it wasn't
        checkpointed at all.
    """
    simulator = config.simulator
    mission = config.mission

    for i, step in enumerate(config.steps, start=1):
        effective_checkpoint = step.checkpoint if step.checkpoint is not None else keep_intermediate

        path = None
        if effective_checkpoint and out_dir is not None:
            path = _checkpoint_path(out_dir, i, step)

        if path is not None and path.exists() and not overwrite:
            artifact = _load_artifact(path, _artifact_type(step))
            store[step.id] = artifact
            yield step, artifact, path
            continue

        if step.type == "cut":
            artifact = apply_cut_scoped(
                simulator, step.cut, store[step.input], mission, step.transient_types, **step.params
            )
        elif step.type == "logical_op":
            artifact = LOGICAL_OPS[step.op]([store[name] for name in step.inputs])
        else:  # action
            kwargs = {param_name: store[artifact_id] for param_name, artifact_id in step.inputs.items()}
            artifact = simulator.run_action(step.action, mission, **kwargs, **step.params)

        store[step.id] = artifact
        if path is not None:
            _save_artifact(artifact, path, overwrite)
        yield step, artifact, path


def describe_step_specs(step_list: list[StepSpec]) -> list[str]:
    """
    Describe a list of `StepSpec` as human-readable report lines, without running anything.

    Parameters
    ----------
    step_list : list of StepSpec
        The steps to describe, in the order to report them.

    Returns
    -------
    list of str
        One summary line per entry in `step_list`.
    """
    lines = []
    for step in step_list:
        if step.type == "cut":
            scope = f" [{', '.join(step.transient_types)}]" if step.transient_types else ""
            lines.append(f"  {step.id:<20s} cut          {step.cut}{scope}  input={step.input}  {step.params}")
        elif step.type == "logical_op":
            lines.append(f"  {step.id:<20s} logical_op   {step.op}  inputs={step.inputs}")
        else:
            lines.append(f"  {step.id:<20s} action       {step.action}  inputs={step.inputs}  {step.params}")
    return lines


def describe_steps(config: RunConfig) -> list[str]:
    """
    Describe `config.steps` as human-readable report lines, without running anything.

    Accessing `config.steps` here is itself the validation (see the module docstring);
    an invalid ``steps:`` block raises before any line is produced.

    Parameters
    ----------
    config : RunConfig
        The parsed run-config.

    Returns
    -------
    list of str
        One summary line per declared step, in declared order.
    """
    return describe_step_specs(config.steps)
