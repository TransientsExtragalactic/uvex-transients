"""Tests for `SurveySimulator.run_detection_delay_action`."""

import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable
from astropy.time import Time
from m4opt.missions._uvex import uvex

from uvex_transients.simulation.core import SurveySimulator
from uvex_transients.simulation.event_catalog import EventCatalog
from uvex_transients.simulation.exposure_catalog import ExposureCatalog
from uvex_transients.transients.TDEs import TidalDisruptionEvent

from .test_additional_cuts import _gap_chunk
from .test_core import _make_catalog

TIME_BINS = Time(["2025-01-01", "2025-06-01"])


def _exposure(mu0_by_type: dict[str, float]) -> ExposureCatalog:
    table = QTable()
    table["transient_type"] = np.asarray(list(mu0_by_type))
    table["expected_events"] = np.asarray(list(mu0_by_type.values()), dtype=np.float64)
    return ExposureCatalog(table=table, nside=32, order="nested", time_bins=TIME_BINS)


@pytest.fixture
def sim(make_schedule):
    return SurveySimulator(make_schedule(n_sched=40), transients={"tde": TidalDisruptionEvent()}, simulation_seed=1)


@pytest.fixture
def catalog(hot_spot):
    return _make_catalog(TidalDisruptionEvent(), hot_spot, n_events=20, seed=3)[0]


@pytest.fixture
def two_type_sim(make_schedule):
    return SurveySimulator(
        make_schedule(n_sched=40),
        transients={"tde": TidalDisruptionEvent(), "other": TidalDisruptionEvent()},
        simulation_seed=1,
    )


@pytest.fixture
def hand_built(two_type_sim, monkeypatch):
    """
    Six events over two types, driven by `_gap_chunk`'s hand-built epochs.

    Events 0-2 are ``"tde"`` and 3-5 are ``"other"``. Delays from the last non-detection are
    2, 6, (none), (never detected), 3, 2 days; ages at first detection are 4, 1, 1, -, 2, 3 days.
    """
    chunk = _gap_chunk()
    monkeypatch.setattr(two_type_sim, "iter_epochs", lambda *args, **kwargs: iter([chunk]))
    table = QTable()
    table["event_id"] = np.arange(6, dtype=np.int64)
    table["transient_type"] = np.array(["tde"] * 3 + ["other"] * 3)
    events = EventCatalog(table=table, nside=64, order="nested", time_bins=TIME_BINS)
    return two_type_sim, events, _exposure({"tde": 10.0, "other": 4.0})


def _by_type(table, column):
    return {name: list(table[column][table["transient_type"] == name]) for name in np.unique(table["transient_type"])}


# --------------------------------------------------------------------------- #
# Hand-built epochs                                                           #
# --------------------------------------------------------------------------- #
def test_counts_since_last_nondetection(hand_built):
    sim, events, exposure = hand_built

    table = sim.run_detection_delay_action(
        events,
        events,
        exposure,
        uvex,
        snr_threshold=5.0,
        delays=[100, 24, 200, 48],  # unsorted, in hours
    )

    assert list(table["max_delay"][:4].to_value(u.hr)) == [24, 48, 100, 200]
    assert _by_type(table, "n_within") == {"other": [0, 1, 2, 2], "tde": [0, 1, 1, 2]}
    assert _by_type(table, "n_unbracketed") == {"other": [0] * 4, "tde": [1] * 4}
    assert np.all(table["n_total"] == 3)
    assert table.meta["reference"] == "last_nondetection"


def test_counts_since_explosion(hand_built):
    sim, events, exposure = hand_built

    table = sim.run_detection_delay_action(
        events, events, exposure, uvex, snr_threshold=5.0, delays=[24, 48, 100, 200], reference="explosion"
    )

    assert _by_type(table, "n_within") == {"other": [0, 1, 2, 2], "tde": [2, 2, 3, 3]}
    assert np.all(table["n_unbracketed"] == 0)


def test_expected_events_scale_with_the_intrinsic_count(hand_built):
    sim, events, exposure = hand_built

    table = sim.run_detection_delay_action(events, events, exposure, uvex, snr_threshold=5.0, delays=[200] * u.hr)

    tde = table[table["transient_type"] == "tde"][0]
    assert tde["fraction"] == pytest.approx(2 / 3)
    assert tde["expected_events"] == pytest.approx(10.0 * 2 / 3)
    assert tde["fraction_lower"] < tde["fraction"] < tde["fraction_upper"]


def test_denominator_comes_from_raw_not_detected(hand_built):
    sim, events, exposure = hand_built
    detected = sim._filtered(events, np.array([True, True, True, False, True, True]))

    table = sim.run_detection_delay_action(events, events, exposure, uvex, snr_threshold=5.0, delays=[200])
    cut = sim.run_detection_delay_action(detected, events, exposure, uvex, snr_threshold=5.0, delays=[200])

    assert list(cut["n_total"]) == [3, 3]
    assert list(cut["n_within"]) == list(table["n_within"])


def test_transient_types_restricts_the_table(hand_built):
    sim, events, exposure = hand_built

    full = sim.run_detection_delay_action(events, events, exposure, uvex, snr_threshold=5.0, delays=[48, 200])
    subset = sim.run_detection_delay_action(
        events, events, exposure, uvex, snr_threshold=5.0, delays=[48, 200], transient_types=["tde"]
    )

    assert set(subset["transient_type"]) == {"tde"}
    assert list(subset["n_within"]) == list(full["n_within"][full["transient_type"] == "tde"])


# --------------------------------------------------------------------------- #
# Real epochs                                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("reference", ["last_nondetection", "explosion"])
def test_matches_the_single_delay_cut_at_every_grid_point(sim, catalog, reference):
    """Each row's `n_within` equals the survivors of the corresponding cut run at that delay."""
    threshold = 3.0
    detected = sim.filter_by_snr(catalog, uvex, snr_threshold=threshold)
    grid = [3, 12, 24, 96, 24 * 30]
    cut = (
        sim.filter_by_time_since_last_nondetection
        if reference == "last_nondetection"
        else sim.filter_by_time_to_first_detection
    )

    table = sim.run_detection_delay_action(
        detected, catalog, _exposure({"tde": 5.0}), uvex, snr_threshold=threshold, delays=grid, reference=reference
    )

    expected = [len(cut(detected, uvex, snr_threshold=threshold, max_delay=hours / 24)) for hours in grid]
    assert list(table["n_within"]) == expected
    assert expected[-1] > 0  # guard against a vacuous comparison
    assert np.all(np.diff(table["n_within"]) >= 0)
    assert np.all(table["n_within"] <= len(detected))


def test_no_detections_gives_zero_counts(sim, catalog):
    empty = sim._filtered(catalog, np.zeros(len(catalog), dtype=bool))

    table = sim.run_detection_delay_action(
        empty, catalog, _exposure({"tde": 5.0}), uvex, snr_threshold=5.0, delays=[12, 24]
    )

    assert list(table["n_within"]) == [0, 0]
    assert list(table["n_total"]) == [len(catalog)] * 2
    assert np.all(table["expected_events"] == 0)


def test_dispatches_via_run_action(sim, catalog):
    detected = sim.filter_by_snr(catalog, uvex, snr_threshold=3.0)
    kwargs = {"exposure": _exposure({"tde": 5.0}), "snr_threshold": 3.0, "delays": [24, 48]}

    direct = sim.run_detection_delay_action(detected, catalog, mission=uvex, **kwargs)
    dispatched = sim.run_action("detection_delay", uvex, detected=detected, raw=catalog, **kwargs)

    assert list(direct["n_within"]) == list(dispatched["n_within"])


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("delays", [[], [0, 12], [-1], [np.nan], [[1, 2]]])
def test_bad_delays_raise(sim, catalog, delays):
    with pytest.raises(ValueError, match="delays"):
        sim.run_detection_delay_action(
            catalog, catalog, _exposure({"tde": 1.0}), uvex, snr_threshold=5.0, delays=delays
        )


def test_unknown_reference_raises(sim, catalog):
    with pytest.raises(ValueError, match="reference"):
        sim.run_detection_delay_action(
            catalog, catalog, _exposure({"tde": 1.0}), uvex, snr_threshold=5.0, delays=[12], reference="peak"
        )


def test_unknown_transient_type_raises(sim, catalog):
    with pytest.raises(ValueError, match="Unknown transient type"):
        sim.run_detection_delay_action(
            catalog,
            catalog,
            _exposure({"tde": 1.0}),
            uvex,
            snr_threshold=5.0,
            delays=[12],
            transient_types=["kilonova"],
        )


def test_missing_exposure_raises(sim, catalog):
    with pytest.raises(KeyError, match="No exposure"):
        sim.run_detection_delay_action(
            catalog, catalog, _exposure({"other": 1.0}), uvex, snr_threshold=5.0, delays=[12]
        )


def test_rejects_non_event_catalog(sim, catalog):
    with pytest.raises(TypeError, match="EventCatalog"):
        sim.run_detection_delay_action(QTable(), catalog, _exposure({"tde": 1.0}), uvex, snr_threshold=5.0, delays=[12])
