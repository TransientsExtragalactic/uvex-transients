"""Tests for `PhotometryCatalog` and `EventCatalog.compute_photometry_catalog`."""

import numpy as np
import pytest
from astropy.table import QTable
from astropy.time import Time
from m4opt.missions._uvex import uvex

from uvex_transients.simulation.event import Event
from uvex_transients.simulation.event_catalog import EventCatalog
from uvex_transients.simulation.exposure_catalog import ExposureCatalog
from uvex_transients.simulation.photometry_catalog import PhotometryCatalog
from uvex_transients.transients.TDEs import TidalDisruptionEvent

from .test_core import _make_catalog


def test_compute_photometry_catalog_wraps_simulate_photometry(make_schedule, hot_spot):
    """`compute_photometry_catalog` wraps `simulate_photometry`'s own table, unchanged."""
    schedule = make_schedule()
    transient = TidalDisruptionEvent()
    catalog, *_ = _make_catalog(transient, hot_spot, n_events=5, seed=3)

    expected = catalog.simulate_photometry(uvex, {"tde": transient}, schedule)
    photometry = catalog.compute_photometry_catalog(uvex, {"tde": transient}, schedule)

    assert isinstance(photometry, PhotometryCatalog)
    assert len(photometry) == len(expected)
    assert set(photometry.table.colnames) == set(Event._empty_photometry_table().colnames)


def test_column_accessors_match_table(make_schedule, hot_spot):
    """Every convenience property returns the same data as indexing `table` directly."""
    schedule = make_schedule()
    transient = TidalDisruptionEvent()
    catalog, *_ = _make_catalog(transient, hot_spot, n_events=5, seed=3)
    photometry = catalog.compute_photometry_catalog(uvex, {"tde": transient}, schedule)

    if len(photometry) == 0:
        return

    assert np.array_equal(photometry.event_id, np.asarray(photometry.table["event_id"]))
    assert np.array_equal(photometry.snr, np.asarray(photometry.table["snr"]))
    assert np.array_equal(photometry.band, np.asarray(photometry.table["band"]).astype(str))


def test_to_disk_from_disk_round_trips(tmp_path, make_schedule, hot_spot):
    """A `PhotometryCatalog` round-trips through ECSV with its table intact."""
    schedule = make_schedule()
    transient = TidalDisruptionEvent()
    catalog, *_ = _make_catalog(transient, hot_spot, n_events=5, seed=3)
    photometry = catalog.compute_photometry_catalog(uvex, {"tde": transient}, schedule)

    path = tmp_path / "photometry.ecsv"
    photometry.to_disk(path)
    reloaded = PhotometryCatalog.from_disk(path)

    assert len(reloaded) == len(photometry)
    assert set(reloaded.table.colnames) == set(photometry.table.colnames)


def test_from_disk_missing_file_raises(tmp_path):
    """`from_disk` raises `FileNotFoundError` on a nonexistent path, matching the other catalogs."""
    with pytest.raises(FileNotFoundError):
        PhotometryCatalog.from_disk(tmp_path / "does-not-exist.ecsv")


# --------------------------------------------------------------------------- #
# get_event                                                                   #
# --------------------------------------------------------------------------- #
def test_get_event_masks_to_that_id(make_schedule, hot_spot):
    """`get_event` returns exactly the rows matching that event id, and nothing else."""
    schedule = make_schedule()
    transient = TidalDisruptionEvent()
    catalog, *_ = _make_catalog(transient, hot_spot, n_events=5, seed=3)
    photometry = catalog.compute_photometry_catalog(uvex, {"tde": transient}, schedule)

    if len(photometry) == 0:
        return

    event_id = int(photometry.event_id[0])
    rows = photometry.get_event(event_id)

    assert len(rows) == int(np.sum(photometry.event_id == event_id))
    assert np.all(np.asarray(rows["event_id"]) == event_id)


def test_get_event_missing_id_raises(make_schedule, hot_spot):
    """`get_event` raises `KeyError` for an id absent from the catalog."""
    schedule = make_schedule()
    transient = TidalDisruptionEvent()
    catalog, *_ = _make_catalog(transient, hot_spot, n_events=5, seed=3)
    photometry = catalog.compute_photometry_catalog(uvex, {"tde": transient}, schedule)

    with pytest.raises(KeyError):
        photometry.get_event(-1)
