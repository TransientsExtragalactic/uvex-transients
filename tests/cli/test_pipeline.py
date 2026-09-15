"""Tests for `uvex_transients.cli.pipeline`."""

import io

import pytest

from uvex_transients.cli.config import RunConfig
from uvex_transients.cli.pipeline import run_cuts, run_generate, run_photometry
from uvex_transients.cli.yaml_tags import get_run_yaml
from uvex_transients.simulation.event_catalog import EventCatalog

from ..simulation.test_core import _make_catalog


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


def test_run_cuts_default_chains_every_declared_cut_in_order(tmp_path, make_schedule, hot_spot):
    """With no explicit names, `run_cuts` applies every declared cut, matching a manual sequential call."""
    doc = f"""
{_write_schedule(tmp_path, make_schedule)}
transients:
  tde:
    class: TidalDisruptionEvent
cuts:
  cut_1:
    type: limiting_magnitude
    mag_limit: 25.0
  cut_2:
    type: snr
    snr_threshold: 5.0
"""
    config = _config_from(doc)
    catalog, *_ = _make_catalog(config.transients["tde"], hot_spot, n_events=20, seed=2)

    chained = run_cuts(config, catalog)

    manual = config.simulator.run_cut("limiting_magnitude", catalog, config.mission, mag_limit=25.0)
    manual = config.simulator.run_cut("snr", manual, config.mission, snr_threshold=5.0)

    assert set(chained.event_id) == set(manual.event_id)


def test_run_cuts_explicit_names_runs_only_those(tmp_path, make_schedule, hot_spot):
    """Explicit `names` runs only those cuts, in the given order, skipping the rest."""
    doc = f"""
{_write_schedule(tmp_path, make_schedule)}
transients:
  tde:
    class: TidalDisruptionEvent
cuts:
  cut_1:
    type: limiting_magnitude
    mag_limit: 25.0
  cut_2:
    type: snr
    snr_threshold: 5.0
"""
    config = _config_from(doc)
    catalog, *_ = _make_catalog(config.transients["tde"], hot_spot, n_events=20, seed=2)

    only_mag = run_cuts(config, catalog, names=["cut_1"])
    expected = config.simulator.run_cut("limiting_magnitude", catalog, config.mission, mag_limit=25.0)

    assert set(only_mag.event_id) == set(expected.event_id)


def test_run_cuts_unknown_name_raises(tmp_path, make_schedule, hot_spot):
    """An explicit cut name not declared in the config raises, listing the real ones."""
    doc = f"""
{_write_schedule(tmp_path, make_schedule)}
transients:
  tde:
    class: TidalDisruptionEvent
cuts:
  cut_1:
    type: limiting_magnitude
    mag_limit: 25.0
"""
    config = _config_from(doc)
    catalog, *_ = _make_catalog(config.transients["tde"], hot_spot, n_events=0)

    with pytest.raises(ValueError, match="Unknown cut key"):
        run_cuts(config, catalog, names=["bogus"])


def test_run_photometry_matches_catalog_simulate_photometry(tmp_path, make_schedule, hot_spot):
    """`run_photometry` is a thin wrapper over `EventCatalog.simulate_photometry`."""
    doc = f"""
{_write_schedule(tmp_path, make_schedule)}
transients:
  tde:
    class: TidalDisruptionEvent
"""
    config = _config_from(doc)
    catalog, *_ = _make_catalog(config.transients["tde"], hot_spot, n_events=5, seed=4)

    phot = run_photometry(config, catalog)
    expected = catalog.simulate_photometry(config.mission, config.transients, config.schedule)

    assert len(phot) == len(expected)
