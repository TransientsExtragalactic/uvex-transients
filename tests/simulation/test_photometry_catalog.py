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


def _make_exposure_catalog(mu0_by_type: dict[str, float]) -> ExposureCatalog:
    """Build a minimal `ExposureCatalog` supplying just `total_expected_events`, for `compute_detection_count_table`."""
    table = QTable()
    table["transient_type"] = np.asarray(list(mu0_by_type))
    table["expected_events"] = np.asarray(list(mu0_by_type.values()), dtype=np.float64)
    return ExposureCatalog(table=table, nside=32, order="nested", time_bins=Time(["2025-01-01", "2025-06-01"]))


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


# --------------------------------------------------------------------------- #
# compute_detection_count_table                                              #
# --------------------------------------------------------------------------- #
def test_detection_count_table_accounts_for_every_event(make_schedule, hot_spot):
    """Every event in `event_catalog` is represented, including ones absent from the photometry table."""
    schedule = make_schedule()
    transient = TidalDisruptionEvent()
    catalog, *_ = _make_catalog(transient, hot_spot, n_events=8, seed=11)
    photometry = catalog.compute_photometry_catalog(uvex, {"tde": transient}, schedule)
    exposure = _make_exposure_catalog({"tde": 42.0})

    counts = photometry.compute_detection_count_table(catalog, exposure, {"tde": transient}, snr_threshold=5.0)

    for name in np.unique(catalog.transient_type):
        n_total_catalog = int(np.sum(catalog.transient_type == name))
        rows = counts[counts["transient_type"] == name]
        assert rows["n_total"][0] == n_total_catalog
        # n_events, summed across every n_detections bucket, recovers n_total.
        assert int(np.sum(rows["n_events"])) == n_total_catalog
        # n_at_least is a non-increasing reverse-cumulative sum of n_events.
        assert np.all(np.diff(rows["n_at_least"]) <= 0)
        assert rows["n_at_least"][0] == n_total_catalog
        # fraction is exactly n_at_least / n_total, and monotonically non-increasing in k.
        assert np.allclose(np.asarray(rows["fraction"]), np.asarray(rows["n_at_least"]) / n_total_catalog)
        assert np.all(np.diff(rows["fraction"]) <= 0)
        # The Clopper-Pearson interval always brackets the point estimate.
        assert np.all(rows["fraction_lower"] <= rows["fraction"] + 1e-12)
        assert np.all(rows["fraction"] <= rows["fraction_upper"] + 1e-12)
        # expected_events = mu0 * fraction, using exposure's own mu0 for that type.
        assert np.allclose(np.asarray(rows["expected_events"]), 42.0 * np.asarray(rows["fraction"]))


def test_detection_count_table_matches_manual_epoch_count(make_schedule, hot_spot):
    """`n_at_least` at `n_detections == 1` equals the number of events with any row above threshold."""
    schedule = make_schedule()
    transient = TidalDisruptionEvent()
    catalog, *_ = _make_catalog(transient, hot_spot, n_events=10, seed=5)
    photometry = catalog.compute_photometry_catalog(uvex, {"tde": transient}, schedule)
    exposure = _make_exposure_catalog({"tde": 10.0})

    snr_threshold = 5.0
    counts = photometry.compute_detection_count_table(
        catalog, exposure, {"tde": transient}, snr_threshold=snr_threshold
    )

    table = photometry.table
    detected_event_ids = set(np.asarray(table["event_id"])[np.asarray(table["snr"]) > snr_threshold])

    for name in np.unique(catalog.transient_type):
        type_event_ids = set(catalog.event_id[catalog.transient_type == name])
        expected_detected = len(detected_event_ids & type_event_ids)

        rows = counts[counts["transient_type"] == name]
        one_or_more = rows[rows["n_detections"] == 1]
        actual_detected = int(one_or_more["n_at_least"][0]) if len(one_or_more) else 0
        assert actual_detected == expected_detected

        # This reproduces EventCatalog.compute_detection_efficiency's own k/efficiency.
        detected_catalog = EventCatalog(
            table=catalog.table[np.isin(catalog.event_id, list(detected_event_ids & type_event_ids))],
            nside=catalog.nside,
            order=catalog.order,
            time_bins=catalog.time_bins,
        )
        efficiency = catalog.compute_detection_efficiency(detected_catalog)[name]
        assert efficiency["k"] == actual_detected
        if len(one_or_more):
            assert np.isclose(float(one_or_more["fraction"][0]), efficiency["efficiency"])


def test_detection_count_table_empty_photometry(hot_spot):
    """An empty photometry table still produces one all-zero-detections row per transient type."""
    transient = TidalDisruptionEvent()
    catalog, *_ = _make_catalog(transient, hot_spot, n_events=4, seed=2)
    photometry = PhotometryCatalog(table=Event._empty_photometry_table())
    exposure = _make_exposure_catalog({"tde": 7.0})

    counts = photometry.compute_detection_count_table(catalog, exposure, {"tde": transient}, snr_threshold=5.0)

    assert set(counts["n_detections"]) == {0}
    assert int(counts["n_events"][0]) == len(catalog)
    assert int(counts["n_total"][0]) == len(catalog)
    assert counts["fraction"][0] == 1.0
    assert counts["expected_events"][0] == 7.0


def test_detection_count_table_missing_exposure_raises(make_schedule, hot_spot):
    """A transient type present in `event_catalog` but absent from `exposure` raises `KeyError`."""
    schedule = make_schedule()
    transient = TidalDisruptionEvent()
    catalog, *_ = _make_catalog(transient, hot_spot, n_events=3, seed=1)
    photometry = catalog.compute_photometry_catalog(uvex, {"tde": transient}, schedule)
    exposure = _make_exposure_catalog({"kilonova": 1.0})

    with pytest.raises(KeyError):
        photometry.compute_detection_count_table(catalog, exposure, {"tde": transient}, snr_threshold=5.0)


def test_detection_count_table_missing_transient_raises(make_schedule, hot_spot):
    """A transient type present in `event_catalog` but absent from `transients` raises `KeyError`."""
    schedule = make_schedule()
    transient = TidalDisruptionEvent()
    catalog, *_ = _make_catalog(transient, hot_spot, n_events=3, seed=1)
    photometry = catalog.compute_photometry_catalog(uvex, {"tde": transient}, schedule)
    exposure = _make_exposure_catalog({"tde": 1.0})

    with pytest.raises(KeyError):
        photometry.compute_detection_count_table(catalog, exposure, {}, snr_threshold=5.0)
