"""Tests for `SurveySimulator.run_event_summary_action`."""

import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable
from astropy.time import Time
from m4opt.missions._uvex import uvex

from uvex_transients.simulation.core import SurveySimulator, _reduce_event_summary
from uvex_transients.simulation.event_catalog import EventCatalog
from uvex_transients.simulation.exposure_catalog import ExposureCatalog
from uvex_transients.simulation.photometry_catalog import PhotometryCatalog
from uvex_transients.transients.TDEs import TidalDisruptionEvent

from ..surveys.test_base import _add_downlinks
from .test_core import _make_catalog

TIME_BINS = Time(["2025-01-01", "2025-06-01"])
T0 = Time("2030-01-01T00:00:00")


def _exposure(mu0_by_type: dict[str, float]) -> ExposureCatalog:
    table = QTable()
    table["transient_type"] = np.asarray(list(mu0_by_type))
    table["expected_events"] = np.asarray(list(mu0_by_type.values()), dtype=np.float64)
    return ExposureCatalog(table=table, nside=32, order="nested", time_bins=TIME_BINS)


def _chunk(rows, threshold=5.0):
    """
    Build one event's epoch chunk from ``(day, snr, flux, flux_err, pre_explosion)`` rows.

    ``day`` is the time since explosion; the explosion is at `T0`.
    """
    day, snr, flux, flux_err, pre = (np.array(column) for column in zip(*rows))
    chunk = QTable(
        {
            "event_id": np.zeros(len(rows), dtype=np.int64),
            "observation_index": np.arange(len(rows)) + 100,
            "t_obs": T0 + day.astype(float) * u.day,
            "t_since_explosion": day.astype(float) * u.day,
            "pre_explosion": pre.astype(bool),
            "snr": snr.astype(float),
            "flux": flux.astype(float) * u.Jy,
            "flux_err": flux_err.astype(float) * u.Jy,
        }
    )
    chunk["detected"] = (chunk["snr"] > threshold) & ~chunk["pre_explosion"]
    chunk["non_detected"] = chunk["snr"] <= threshold
    return chunk


def _reduce(rows, rise_sigma=3.0):
    chunk = _chunk(rows)
    return _reduce_event_summary(chunk, slice(0, len(chunk)), 5.0, rise_sigma)


# --------------------------------------------------------------------------- #
# The per-event reducer, on hand-built epochs                                  #
# --------------------------------------------------------------------------- #
class TestReduceEventSummary:
    def test_times_counts_and_nondetection(self):
        summary = _reduce(
            [
                (-5.0, 1.0, 0.1, 0.1, True),
                (2.0, 2.0, 0.2, 0.1, False),
                (4.0, 9.0, 0.9, 0.1, False),
                (6.0, 3.0, 0.3, 0.1, False),
                (8.0, 7.0, 0.7, 0.1, False),
            ]
        )
        assert summary["n_obs"] == 4
        assert summary["n_det"] == 2
        assert summary["t_first_obs"] == pytest.approx((T0 + 2 * u.day).jd)
        assert summary["t_last_obs"] == pytest.approx((T0 + 8 * u.day).jd)
        assert summary["t_first_det"] == pytest.approx((T0 + 4 * u.day).jd)
        assert summary["t_last_det"] == pytest.approx((T0 + 8 * u.day).jd)
        assert summary["first_det_obs_index"] == 102
        # The last non-detection before the first detection, not the one after it.
        assert summary["t_last_nondet"] == pytest.approx((T0 + 2 * u.day).jd)
        assert summary["last_nondet_snr"] == 2.0

    def test_constraining_nondetection_skips_marginal_epochs(self):
        """A 4.9 sigma non-detection right before a 5.3 sigma detection does not constrain the rise."""
        summary = _reduce(
            [
                (-5.0, 0.5, 0.05, 0.1, True),
                (10.0, 4.9, 0.49, 0.1, False),
                (10.01, 5.3, 0.53, 0.1, False),
            ]
        )
        assert summary["t_last_nondet"] == pytest.approx((T0 + 10 * u.day).jd)
        assert summary["last_nondet_snr"] == 4.9
        # Only the pre-explosion epoch lies far enough below the detection.
        assert summary["t_last_constraining_nondet"] == pytest.approx((T0 - 5 * u.day).jd)

    def test_rise_sigma_controls_which_epoch_constrains(self):
        rows = [
            (1.0, 1.0, 0.1, 0.1, False),
            (2.0, 3.0, 0.3, 0.1, False),
            (3.0, 9.0, 0.9, 0.1, False),
        ]
        strict = _reduce(rows, rise_sigma=5.0)
        loose = _reduce(rows, rise_sigma=1.0)
        assert strict["t_last_constraining_nondet"] == pytest.approx((T0 + 1 * u.day).jd)
        assert loose["t_last_constraining_nondet"] == pytest.approx((T0 + 2 * u.day).jd)

    def test_never_detected(self):
        summary = _reduce([(1.0, 1.0, 0.1, 0.1, False), (2.0, 2.0, 0.2, 0.1, False)])
        assert summary["n_obs"] == 2
        assert summary["n_det"] == 0
        for key in ("t_first_det", "t_last_det", "first_det_obs_index", "t_last_nondet", "last_nondet_snr"):
            assert summary[key] is None

    def test_unbracketed_first_observation_is_a_detection(self):
        summary = _reduce([(1.0, 9.0, 0.9, 0.1, False), (2.0, 8.0, 0.8, 0.1, False)])
        assert summary["n_det"] == 2
        assert summary["t_last_nondet"] is None
        assert summary["t_last_constraining_nondet"] is None

    def test_pre_explosion_false_positive_is_not_a_detection(self):
        summary = _reduce([(-3.0, 9.0, 0.9, 0.1, True), (2.0, 9.0, 0.9, 0.1, False)])
        assert summary["n_det"] == 1
        assert summary["t_first_det"] == pytest.approx((T0 + 2 * u.day).jd)
        assert summary["n_obs"] == 1


# --------------------------------------------------------------------------- #
# The action, on a small synthetic catalog                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture
def setup(make_schedule, hot_spot):
    transient = TidalDisruptionEvent()
    schedule = make_schedule(n_sched=40)
    catalog = _make_catalog(transient, hot_spot, n_events=20, seed=3)[0]
    catalog.pre_cut_counts = {"tde": len(catalog)}
    sim = SurveySimulator(schedule, transients={"tde": transient}, simulation_seed=1)
    detected = sim.filter_by_snr(catalog, uvex, snr_threshold=3.0)
    photometry = sim.run_photometry_action(detected, uvex)
    return sim, catalog, detected, photometry


def _summarize(sim, catalog, detected, photometry, **kwargs):
    kwargs.setdefault("snr_threshold", 3.0)
    kwargs.setdefault("footprints", [])
    return sim.run_event_summary_action(
        catalog=detected,
        exposure=_exposure({"tde": 100.0}),
        photometry=photometry,
        mission=uvex,
        **kwargs,
    )


def test_one_row_per_event_in_catalog_order(setup):
    sim, catalog, detected, photometry = setup
    summary = _summarize(sim, catalog, detected, photometry)
    assert len(detected) > 0
    assert list(summary["event_id"]) == list(detected.table["event_id"])
    assert set(summary.colnames) >= {
        "event_id",
        "transient_type",
        "parameter_seed",
        "time_bin",
        "redshift",
        "luminosity_distance",
        "ebv",
        "coord",
        "healpix_id",
        "t_explosion",
        "weight",
        "n_obs",
        "t_first_obs",
        "t_last_obs",
        "n_det",
        "t_first_det",
        "t_last_det",
        "field_id",
        "t_alert",
        "t_last_nondet",
        "last_nondet_snr",
        "t_last_constraining_nondet",
        "peak_snr",
        "peak_mag",
        "peak_color",
        "peak_time",
    }


def test_every_event_in_a_snr_cut_catalog_has_a_detection(setup):
    sim, catalog, detected, photometry = setup
    summary = _summarize(sim, catalog, detected, photometry)
    assert np.all(summary["n_det"] >= 1)
    assert not np.any(np.ma.getmaskarray(summary["t_first_det"].jd))
    first = np.ma.getdata(summary["t_first_det"].jd)
    last = np.ma.getdata(summary["t_last_det"].jd)
    assert np.all(last >= first)


def test_threshold_above_every_snr_leaves_every_detection_column_masked(setup):
    """A threshold nothing clears gives zero detections, and masks everything that depends on one."""
    sim, catalog, detected, photometry = setup
    strict = _summarize(sim, catalog, detected, photometry, snr_threshold=1e6)
    assert np.all(strict["n_det"] == 0)
    assert np.all(np.ma.getmaskarray(strict["t_first_det"].jd))
    assert np.all(np.ma.getmaskarray(strict["t_alert"].jd))
    assert np.all(np.ma.getmaskarray(strict["field_id"]))


def test_weights_sum_to_the_expected_events_over_the_raw_catalog(setup):
    sim, catalog, detected, photometry = setup
    summary = _summarize(sim, catalog, detected, photometry)
    np.testing.assert_allclose(summary["weight"], 100.0 / len(catalog))
    assert summary.meta["n_pre_cut"] == {"tde": len(catalog)}
    assert summary.meta["expected_events"] == {"tde": 100.0}
    assert summary.meta["rate_ci"] == {"tde": [float(x) for x in (TidalDisruptionEvent.RATE_CI or (1.0, 1.0))]}
    assert summary.meta["snr_threshold"] == 3.0


def test_estimate_yield_matches_the_summed_weights(setup):
    """Selecting rows and summing their weights gives the same expected events as `estimate_yield`."""
    from uvex_transients.simulation.rates import estimate_yield

    sim, catalog, detected, photometry = setup
    summary = _summarize(sim, catalog, detected, photometry)
    selection = np.asarray(summary["peak_snr"].filled(0.0) > 5.0)

    result = estimate_yield(summary, mask=selection)

    assert result["n_pre_cut"][0] == len(catalog)
    assert result["n_selected"][0] == int(selection.sum())
    assert result["expected_events"][0] == pytest.approx(float(np.sum(summary["weight"][selection])))


def test_peak_columns_match_the_photometry_table(setup):
    sim, catalog, detected, photometry = setup
    summary = _summarize(sim, catalog, detected, photometry)
    event_id = np.asarray(photometry["event_id"])
    for row in summary:
        rows = np.flatnonzero(event_id == row["event_id"])
        best = rows[np.argmax(np.asarray(photometry["snr"])[rows])]
        assert row["peak_snr"] == pytest.approx(float(photometry["snr"][best]))
        assert row["peak_time"].jd == pytest.approx(photometry["obs_time"][best].jd)


def test_peak_color_is_the_band_difference_at_the_peak_observation(setup):
    sim, catalog, detected, photometry = setup
    summary = _summarize(sim, catalog, detected, photometry)
    band = np.asarray(photometry["band"]).astype(str)
    event_id = np.asarray(photometry["event_id"])
    jd = photometry["obs_time"].jd
    checked = 0
    for row in summary:
        if np.ma.is_masked(row["peak_color"]):
            continue
        at_peak = (event_id == row["event_id"]) & (jd == row["peak_time"].jd)
        mag = {b: float(photometry["ab_mag"][at_peak & (band == b)][0]) for b in ("FUV", "NUV")}
        assert row["peak_color"] == pytest.approx(mag["FUV"] - mag["NUV"])
        checked += 1
    assert checked > 0


def test_events_missing_from_photometry_have_masked_peak_columns(setup):
    sim, catalog, detected, photometry = setup
    dropped = int(detected.table["event_id"][0])
    reduced = photometry[np.asarray(photometry["event_id"]) != dropped]
    summary = _summarize(sim, catalog, detected, reduced)
    row = summary[summary["event_id"] == dropped][0]
    assert np.ma.is_masked(row["peak_snr"])
    assert np.ma.is_masked(row["peak_color"])
    assert not np.ma.is_masked(row["n_det"])


def test_accepts_a_photometry_catalog(setup):
    sim, catalog, detected, photometry = setup
    from_table = _summarize(sim, catalog, detected, photometry)
    from_catalog = _summarize(sim, catalog, detected, PhotometryCatalog(table=photometry))
    np.testing.assert_array_equal(np.ma.getdata(from_table["peak_snr"]), np.ma.getdata(from_catalog["peak_snr"]))


# --------------------------------------------------------------------------- #
# Alert time                                                                   #
# --------------------------------------------------------------------------- #
def test_alert_is_masked_when_no_downlink_is_scheduled(setup):
    sim, catalog, detected, photometry = setup
    summary = _summarize(sim, catalog, detected, photometry)
    assert np.all(np.ma.getmaskarray(summary["t_alert"].jd))


def test_alert_is_the_downlink_completion_plus_the_processing_delay(setup):
    sim, catalog, detected, photometry = setup
    schedule = _add_downlinks(sim.survey_schedule, Time([sim.survey_schedule.end_time + 1 * u.hr]), duration=10 * u.min)
    sim_down = SurveySimulator(schedule, transients=sim.transient_collection, simulation_seed=1)
    expected = sim.survey_schedule.end_time + 1 * u.hr + 10 * u.min

    plain = _summarize(sim_down, catalog, detected, photometry)
    delayed = _summarize(sim_down, catalog, detected, photometry, processing_delay=2.0)

    assert not np.any(np.ma.getmaskarray(plain["t_alert"].jd))
    assert np.all(np.abs((plain["t_alert"] - expected).sec) < 1e-3)
    np.testing.assert_allclose((delayed["t_alert"] - plain["t_alert"]).to_value(u.hr), 2.0)
    assert delayed.meta["processing_delay_hr"] == 2.0


# --------------------------------------------------------------------------- #
# Footprints, dispatch, and validation                                         #
# --------------------------------------------------------------------------- #
def test_footprint_flags_are_named_after_the_footprint(setup, monkeypatch):
    sim, catalog, detected, photometry = setup
    monkeypatch.setattr(EventCatalog, "in_footprint", lambda self, name: np.ones(len(self), dtype=bool))
    summary = _summarize(sim, catalog, detected, photometry, footprints=["uvex:lmlz:deep", "ztf:main"])
    assert np.all(summary["in_uvex_lmlz_deep"])
    assert np.all(summary["in_ztf_main"])


def test_dispatches_via_run_action(setup):
    sim, catalog, detected, photometry = setup
    direct = _summarize(sim, catalog, detected, photometry)
    dispatched = sim.run_action(
        "event_summary",
        uvex,
        catalog=detected,
        exposure=_exposure({"tde": 100.0}),
        photometry=photometry,
        snr_threshold=3.0,
        footprints=[],
    )
    assert direct.colnames == dispatched.colnames
    assert list(direct["event_id"]) == list(dispatched["event_id"])


def test_round_trips_through_ecsv(setup, tmp_path):
    sim, catalog, detected, photometry = setup
    summary = _summarize(sim, catalog, detected, photometry)
    path = tmp_path / "summary.ecsv"
    summary.write(path)
    loaded = QTable.read(path)
    assert loaded.colnames == summary.colnames
    assert list(loaded["event_id"]) == list(summary["event_id"])


def test_rejects_non_event_catalogs(setup):
    sim, catalog, detected, photometry = setup
    with pytest.raises(TypeError, match="catalog"):
        sim.run_event_summary_action(QTable(), _exposure({"tde": 1.0}), photometry, uvex, snr_threshold=5.0)


def test_catalog_without_pre_cut_counts_raises(setup):
    sim, catalog, detected, photometry = setup
    detected.pre_cut_counts = None
    with pytest.raises(ValueError, match="pre_cut_counts"):
        _summarize(sim, catalog, detected, photometry)


def test_missing_exposure_for_a_type_raises(setup):
    sim, catalog, detected, photometry = setup
    with pytest.raises(KeyError, match="tde"):
        sim.run_event_summary_action(
            detected, _exposure({"other": 1.0}), photometry, uvex, snr_threshold=3.0, footprints=[]
        )


def test_pre_cut_type_unknown_to_the_simulator_raises(setup):
    sim, catalog, detected, photometry = setup
    detected.pre_cut_counts = {"tde": len(catalog), "mystery": 5}
    with pytest.raises(KeyError, match="mystery"):
        _summarize(sim, catalog, detected, photometry)


def test_types_with_no_rows_are_still_listed_in_meta(make_schedule, hot_spot):
    """A type that was generated but never detected still appears in meta, so its efficiency can be reported as zero."""
    transient = TidalDisruptionEvent()
    catalog = _make_catalog(transient, hot_spot, n_events=10, seed=3)[0]
    catalog.pre_cut_counts = {"tde": len(catalog), "other": 50}
    sim = SurveySimulator(
        make_schedule(n_sched=40), transients={"tde": transient, "other": TidalDisruptionEvent()}, simulation_seed=1
    )
    detected = sim.filter_by_snr(catalog, uvex, snr_threshold=3.0)
    photometry = sim.run_photometry_action(detected, uvex)

    summary = sim.run_event_summary_action(
        detected,
        _exposure({"tde": 100.0, "other": 20.0}),
        photometry,
        uvex,
        snr_threshold=3.0,
        footprints=[],
    )

    assert summary.meta["n_pre_cut"] == {"tde": len(catalog), "other": 50}
    assert set(summary["transient_type"]) == {"tde"}


def test_color_bands_must_be_a_pair(setup):
    sim, catalog, detected, photometry = setup
    with pytest.raises(ValueError, match="color_bands"):
        _summarize(sim, catalog, detected, photometry, color_bands=["FUV"])
