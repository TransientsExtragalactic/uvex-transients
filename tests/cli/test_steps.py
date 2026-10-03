"""Tests for `uvex_transients.cli.steps`."""

import io

import numpy as np
import pytest
from astropy.table import QTable

from uvex_transients.cli.config import RunConfig
from uvex_transients.cli.pipeline import run_exposure, run_generate
from uvex_transients.cli.steps import checkpoint_targets, run_steps
from uvex_transients.cli.yaml_tags import get_run_yaml
from uvex_transients.simulation.event_catalog import EventCatalog

from ..simulation.test_core import _make_catalog


def _config_from(doc: str) -> RunConfig:
    raw = get_run_yaml().load(io.StringIO(doc)) or {}
    return RunConfig(raw)


def _write_schedule(tmp_path, make_schedule):
    schedule = make_schedule()
    table_path = tmp_path / "schedule.ecsv"
    fov_path = tmp_path / "schedule.reg"
    schedule.to_disk(table_path, fov_path=fov_path)
    return f"schedule:\n  path: {table_path}\n  fov_path: {fov_path}\n"


def _config_with_steps(tmp_path, make_schedule, steps_block):
    doc = f"""
{_write_schedule(tmp_path, make_schedule)}
transients:
  tde:
    class: TidalDisruptionEvent
{steps_block}
"""
    return _config_from(doc)


_CUT_THEN_LOGICAL_OP_THEN_ACTION = """
steps:
  - id: mag
    type: cut
    cut: limiting_magnitude
    input: baseline
    params:
      mag_limit: 25.0
  - id: also_mag
    type: cut
    cut: limiting_magnitude
    input: baseline
    params:
      mag_limit: 30.0
  - id: combined
    type: logical_op
    op: intersection
    inputs: [mag, also_mag]
  - id: phot
    type: action
    action: photometry
    inputs:
      catalog: combined
"""


def test_run_steps_chains_cut_logical_op_and_action(tmp_path, make_schedule, hot_spot):
    """A `cut` -> `logical_op` -> `action` chain runs end to end, each step's artifact feeding the next."""
    config = _config_with_steps(tmp_path, make_schedule, _CUT_THEN_LOGICAL_OP_THEN_ACTION)
    catalog, *_ = _make_catalog(config.transients["tde"], hot_spot, n_events=20, seed=2)

    store = {"baseline": catalog}
    results = list(run_steps(config, store))

    ids = [step.id for step, _artifact, _path in results]
    assert ids == ["mag", "also_mag", "combined", "phot"]
    assert isinstance(store["mag"], EventCatalog)
    assert isinstance(store["combined"], EventCatalog)
    assert isinstance(store["phot"], QTable)
    # intersection of two magnitude cuts against the same baseline is exactly the tighter one.
    assert set(np.asarray(store["combined"].table["event_id"])) <= set(np.asarray(store["mag"].table["event_id"]))


def test_run_steps_transient_types_scoping_leaves_other_types_untouched(tmp_path, make_schedule, hot_spot):
    """A `transient_types`-scoped cut only filters matching rows; other types pass through untouched."""
    doc = """
steps:
  - id: scoped
    type: cut
    cut: limiting_magnitude
    input: baseline
    transient_types: [tde]
    params:
      mag_limit: -100.0
"""
    config = _config_with_steps(tmp_path, make_schedule, doc)
    catalog, *_ = _make_catalog(config.transients["tde"], hot_spot, n_events=10, seed=3)
    original = np.asarray(catalog.table["transient_type"])
    catalog.table["transient_type"] = np.where(np.arange(len(original)) % 2 == 0, "other_type", original)

    store = {"baseline": catalog}
    list(run_steps(config, store))

    result = store["scoped"]
    kept_types = set(np.asarray(result.table["transient_type"]))
    # Every 'other_type' row survives (mag_limit=-100 would fail everything of type 'tde').
    assert "other_type" in kept_types
    assert np.count_nonzero(np.asarray(result.table["transient_type"]) == "other_type") == 5
    assert np.count_nonzero(np.asarray(result.table["transient_type"]) == "tde") == 0


def test_run_steps_checkpoints_and_resumes(tmp_path, make_schedule, hot_spot):
    """A checkpointed step is written to disk, then loaded (not rerun) on a second `run_steps` call."""
    doc = """
steps:
  - id: mag
    type: cut
    cut: limiting_magnitude
    input: baseline
    checkpoint: true
    params:
      mag_limit: 25.0
"""
    config = _config_with_steps(tmp_path, make_schedule, doc)
    catalog, *_ = _make_catalog(config.transients["tde"], hot_spot, n_events=10, seed=4)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    store = {"baseline": catalog}
    (step, artifact, path) = next(run_steps(config, store, out_dir=out_dir, keep_intermediate=True))
    assert path == out_dir / "01_mag.ecsv"
    assert path.exists()

    # A fresh store/run reuses the checkpoint instead of recomputing.
    store2 = {"baseline": catalog}
    (step2, artifact2, path2) = next(run_steps(config, store2, out_dir=out_dir, keep_intermediate=True))
    assert path2 == path
    assert set(np.asarray(artifact2.table["event_id"])) == set(np.asarray(artifact.table["event_id"]))


def test_run_steps_checkpoint_custom_filename(tmp_path, make_schedule, hot_spot):
    """A `checkpoint: "<name>"` string overrides the default `{i:02d}_{id}.ecsv` convention."""
    doc = """
steps:
  - id: mag
    type: cut
    cut: limiting_magnitude
    input: baseline
    checkpoint: my_custom_name.ecsv
    params:
      mag_limit: 25.0
"""
    config = _config_with_steps(tmp_path, make_schedule, doc)
    catalog, *_ = _make_catalog(config.transients["tde"], hot_spot, n_events=5, seed=5)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    store = {"baseline": catalog}
    (_step, _artifact, path) = next(run_steps(config, store, out_dir=out_dir, keep_intermediate=False))

    assert path == out_dir / "my_custom_name.ecsv"
    assert path.exists()


def test_run_steps_checkpoint_none_defers_to_keep_intermediate(tmp_path, make_schedule, hot_spot):
    """A step with `checkpoint` unset is written iff the run's own `keep_intermediate` is True."""
    doc = """
steps:
  - id: mag
    type: cut
    cut: limiting_magnitude
    input: baseline
    params:
      mag_limit: 25.0
"""
    config = _config_with_steps(tmp_path, make_schedule, doc)
    catalog, *_ = _make_catalog(config.transients["tde"], hot_spot, n_events=5, seed=6)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    store = {"baseline": catalog}
    (_step, _artifact, path_off) = next(run_steps(config, store, out_dir=out_dir, keep_intermediate=False))
    assert path_off is None
    assert not any(out_dir.iterdir())

    store2 = {"baseline": catalog}
    (_step, _artifact, path_on) = next(run_steps(config, store2, out_dir=out_dir, keep_intermediate=True))
    assert path_on == out_dir / "01_mag.ecsv"
    assert path_on.exists()


def test_checkpoint_targets_matches_run_steps_paths(tmp_path, make_schedule, hot_spot):
    """`checkpoint_targets` predicts exactly the paths `run_steps` would actually write."""
    config = _config_with_steps(tmp_path, make_schedule, _CUT_THEN_LOGICAL_OP_THEN_ACTION)
    catalog, *_ = _make_catalog(config.transients["tde"], hot_spot, n_events=10, seed=7)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    predicted = checkpoint_targets(config, keep_intermediate=True)
    store = {"baseline": catalog}
    actual = list(run_steps(config, store, out_dir=out_dir, keep_intermediate=True))

    for (predicted_step, predicted_relpath), (actual_step, _artifact, actual_path) in zip(predicted, actual):
        assert predicted_step.id == actual_step.id
        assert (out_dir / predicted_relpath) == actual_path


def test_run_generate_and_run_exposure_seed_the_reserved_artifact_ids(tmp_path, make_schedule):
    """`run_generate`/`run_exposure` produce exactly the artifacts `run_steps` expects under 'baseline'/'exposure'."""
    doc = f"""
{_write_schedule(tmp_path, make_schedule)}
transients:
  tde:
    class: TidalDisruptionEvent
generate:
  time_bins: 1
  nside: 16
  seed: 1
"""
    config = _config_from(doc)
    baseline = run_generate(config)
    exposure = run_exposure(config)

    assert isinstance(baseline, EventCatalog)
    store = {"baseline": baseline, "exposure": exposure}
    # An empty steps: list is valid and simply does nothing further.
    assert list(run_steps(config, store)) == []
