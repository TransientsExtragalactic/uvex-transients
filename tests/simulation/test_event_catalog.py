"""Tests for `EventCatalog.simulate_photometry` and `EventCatalog.to_disk`/`from_disk`."""

import pytest
from astropy.table import QTable
from m4opt.missions._uvex import uvex

from uvex_transients.simulation.event import Event
from uvex_transients.simulation.event_catalog import EventCatalog
from uvex_transients.transients.TDEs import TidalDisruptionEvent

from .test_core import _make_catalog


def test_simulate_photometry_stacks_every_event(make_schedule, hot_spot):
    """The catalog-level result is the `vstack` of every reconstructed event's own photometry table."""
    schedule = make_schedule()
    transient = TidalDisruptionEvent()
    catalog, *_ = _make_catalog(transient, hot_spot, n_events=5, seed=3)

    phot = catalog.simulate_photometry(uvex, {"tde": transient}, schedule)

    events = catalog.get_events(catalog.event_id, {"tde": transient}, schedule)
    expected_len = sum(len(event.simulate_photometry(uvex)) for event in events)

    assert len(phot) == expected_len
    assert set(phot.colnames) == set(Event._empty_photometry_table().colnames)
    assert set(phot["event_id"]).issubset(set(catalog.event_id))


def test_simulate_photometry_empty_catalog(make_schedule, hot_spot):
    """An empty catalog produces an empty, correctly-typed table -- no `Event` reconstruction attempted."""
    schedule = make_schedule()
    transient = TidalDisruptionEvent()
    catalog, *_ = _make_catalog(transient, hot_spot, n_events=0)

    phot = catalog.simulate_photometry(uvex, {"tde": transient}, schedule)

    assert len(phot) == 0
    assert set(phot.colnames) == set(Event._empty_photometry_table().colnames)
    assert isinstance(phot, QTable)


def test_simulate_photometry_respects_bands(make_schedule, hot_spot):
    """Passing an explicit `bands` subset restricts every stacked row's `band` to that subset."""
    schedule = make_schedule()
    transient = TidalDisruptionEvent()
    catalog, *_ = _make_catalog(transient, hot_spot, n_events=3, seed=7)

    band = list(uvex.detector.bandpasses)[0]
    phot = catalog.simulate_photometry(uvex, {"tde": transient}, schedule, bands=[band])

    assert set(phot["band"]).issubset({band})


# --------------------------------------------------------------------------- #
# to_disk / from_disk: downsample round-trip                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("downsample", [None, 20, {"tde": 5, "kilonova": 2}])
def test_to_disk_from_disk_round_trips_downsample(tmp_path, hot_spot, downsample):
    """`downsample` (int, mapping, or `None`) survives an ECSV `to_disk`/`from_disk` round trip."""
    transient = TidalDisruptionEvent()
    catalog, *_ = _make_catalog(transient, hot_spot, n_events=3, seed=1)
    catalog.downsample = downsample

    path = tmp_path / "catalog.ecsv"
    catalog.to_disk(path)
    reloaded = EventCatalog.from_disk(path)

    assert reloaded.downsample == downsample


def test_from_disk_defaults_downsample_to_none_when_absent_from_older_files(tmp_path, hot_spot):
    """A catalog file written before `downsample` existed still reads back fine, defaulting to `None`."""
    transient = TidalDisruptionEvent()
    catalog, *_ = _make_catalog(transient, hot_spot, n_events=2, seed=1)

    path = tmp_path / "catalog.ecsv"
    table = catalog.table.copy()
    table.meta.update(
        {"nside": catalog.nside, "order": catalog.order, "time_bins": catalog.time_bins, "seed": catalog.seed}
    )
    table.write(path)

    reloaded = EventCatalog.from_disk(path)
    assert reloaded.downsample is None
