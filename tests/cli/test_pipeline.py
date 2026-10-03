"""Tests for `uvex_transients.cli.pipeline`."""

import io

import pytest

from uvex_transients.cli.config import RunConfig
from uvex_transients.cli.pipeline import (
    run_cut_steps,
    run_detection_counts_step,
    run_generate,
    run_photometry_step,
)
from uvex_transients.cli.yaml_tags import get_run_yaml
from uvex_transients.simulation.event_catalog import EventCatalog
from uvex_transients.simulation.photometry_catalog import PhotometryCatalog

from ..simulation.test_core import _make_catalog
from ..simulation.test_photometry_catalog import _make_exposure_catalog


def _config_from(doc: str) -> RunConfig:
    """Build a `RunConfig` directly from a YAML string, without touching disk."""
    raw = get_run_yaml().load(io.StringIO(doc)) or {}
    return RunConfig(raw)


def _write_schedule(tmp_path, make_schedule):
    """Write a small synthetic schedule to disk and return its ``schedule:`` YAML block."""
    schedule = make_schedule()
    table_path = tmp_path / "schedule.ecsv"
    fov_path = tmp_path / "schedule.reg"
    schedule.to_disk(table_path, fov_path=fov_path)
    return f"schedule:\n  path: {table_path}\n  fov_path: {fov_path}\n"


def test_run_generate_returns_a_catalog_with_the_requested_metadata(tmp_path, make_schedule):
    """`run_generate` samples an `EventCatalog` shaped by the config's `generate:` section."""
    doc = f"""
{_write_schedule(tmp_path, make_schedule)}
transients:
  tde:
    class: TidalDisruptionEvent
generate:
  time_bins: 2
  nside: 16
  seed: 1
"""
    config = _config_from(doc)
    catalog = run_generate(config)

    assert isinstance(catalog, EventCatalog)
    assert catalog.nside == 16
    assert len(catalog.time_bins) - 1 == 2


_TWO_CUT_STEPS = """
steps:
  - id: cut_1
    type: cut
    cut: limiting_magnitude
    input: baseline
    params:
      mag_limit: 25.0
  - id: cut_2
    type: cut
    cut: snr
    input: cut_1
    params:
      snr_threshold: 5.0
"""


def test_run_cut_steps_default_chains_every_declared_cut_in_order(tmp_path, make_schedule, hot_spot):
    """With no explicit step ids, `run_cut_steps` applies every declared `cut` step, matching a manual sequential call."""
    doc = f"""
{_write_schedule(tmp_path, make_schedule)}
transients:
  tde:
    class: TidalDisruptionEvent
{_TWO_CUT_STEPS}
"""
    config = _config_from(doc)
    catalog, *_ = _make_catalog(config.transients["tde"], hot_spot, n_events=20, seed=2)

    chained = run_cut_steps(config, catalog)

    manual = config.simulator.run_cut("limiting_magnitude", catalog, config.mission, mag_limit=25.0)
    manual = config.simulator.run_cut("snr", manual, config.mission, snr_threshold=5.0)

    assert set(chained.event_id) == set(manual.event_id)


def test_run_cut_steps_explicit_ids_runs_only_those(tmp_path, make_schedule, hot_spot):
    """Explicit `step_ids` runs only those cut steps, in the given order, skipping the rest."""
    doc = f"""
{_write_schedule(tmp_path, make_schedule)}
transients:
  tde:
    class: TidalDisruptionEvent
{_TWO_CUT_STEPS}
"""
    config = _config_from(doc)
    catalog, *_ = _make_catalog(config.transients["tde"], hot_spot, n_events=20, seed=2)

    only_mag = run_cut_steps(config, catalog, step_ids=["cut_1"])
    expected = config.simulator.run_cut("limiting_magnitude", catalog, config.mission, mag_limit=25.0)

    assert set(only_mag.event_id) == set(expected.event_id)


def test_run_cut_steps_unknown_id_raises(tmp_path, make_schedule, hot_spot):
    """A step id not declared in the config raises, listing the real ones."""
    doc = f"""
{_write_schedule(tmp_path, make_schedule)}
transients:
  tde:
    class: TidalDisruptionEvent
steps:
  - id: cut_1
    type: cut
    cut: limiting_magnitude
    input: baseline
    params:
      mag_limit: 25.0
"""
    config = _config_from(doc)
    catalog, *_ = _make_catalog(config.transients["tde"], hot_spot, n_events=0)

    with pytest.raises(ValueError, match="No step with id 'bogus'"):
        run_cut_steps(config, catalog, step_ids=["bogus"])


def test_run_cut_steps_rejects_a_non_cut_step(tmp_path, make_schedule, hot_spot):
    """Naming a non-`cut` step id raises."""
    doc = f"""
{_write_schedule(tmp_path, make_schedule)}
transients:
  tde:
    class: TidalDisruptionEvent
steps:
  - id: phot
    type: action
    action: photometry
    inputs:
      catalog: baseline
"""
    config = _config_from(doc)
    catalog, *_ = _make_catalog(config.transients["tde"], hot_spot, n_events=0)

    with pytest.raises(ValueError, match="are not 'cut' steps"):
        run_cut_steps(config, catalog, step_ids=["phot"])


def test_run_photometry_step_matches_catalog_simulate_photometry(tmp_path, make_schedule, hot_spot):
    """`run_photometry_step` matches calling `EventCatalog.simulate_photometry` directly."""
    doc = f"""
{_write_schedule(tmp_path, make_schedule)}
transients:
  tde:
    class: TidalDisruptionEvent
steps:
  - id: phot
    type: action
    action: photometry
    inputs:
      catalog: baseline
"""
    config = _config_from(doc)
    catalog, *_ = _make_catalog(config.transients["tde"], hot_spot, n_events=5, seed=4)

    phot = run_photometry_step(config, "phot", catalog)
    expected = catalog.simulate_photometry(config.mission, config.transients, config.schedule)

    assert len(phot) == len(expected)


def test_run_photometry_step_rejects_a_non_photometry_step(tmp_path, make_schedule, hot_spot):
    """Naming a step that isn't `action: photometry` raises."""
    doc = f"""
{_write_schedule(tmp_path, make_schedule)}
transients:
  tde:
    class: TidalDisruptionEvent
{_TWO_CUT_STEPS}
"""
    config = _config_from(doc)
    catalog, *_ = _make_catalog(config.transients["tde"], hot_spot, n_events=0)

    with pytest.raises(ValueError, match="not a 'photometry' action step"):
        run_photometry_step(config, "cut_1", catalog)


def test_run_detection_counts_step_matches_direct_call(tmp_path, make_schedule, hot_spot):
    """`run_detection_counts_step` matches calling `PhotometryCatalog.compute_detection_count_table` directly."""
    doc = f"""
{_write_schedule(tmp_path, make_schedule)}
transients:
  tde:
    class: TidalDisruptionEvent
steps:
  - id: phot
    type: action
    action: photometry
    inputs:
      catalog: baseline
  - id: dc
    type: action
    action: detection_counts
    inputs:
      catalog: baseline
      exposure: exposure
      photometry: phot
    params:
      snr_threshold: 5.0
      confidence: 0.8
"""
    config = _config_from(doc)
    catalog, *_ = _make_catalog(config.transients["tde"], hot_spot, n_events=5, seed=4)
    phot = catalog.simulate_photometry(config.mission, config.transients, config.schedule)
    exposure = _make_exposure_catalog({"tde": 12.0})

    table = run_detection_counts_step(config, "dc", catalog, exposure, phot)
    expected = PhotometryCatalog(table=phot).compute_detection_count_table(
        catalog, exposure, config.transients, snr_threshold=5.0, confidence=0.8
    )

    assert set(table.colnames) == set(expected.colnames)
    assert len(table) == len(expected)
