"""Tests for the additional `@cut`-registered `SurveySimulator` methods (redshift, peak brightness, ...)."""

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable
from astropy.time import Time
from m4opt.fov import contains as fov_contains
from m4opt.missions._uvex import uvex
from regions import CircleSkyRegion

from uvex_transients.simulation.core import SurveySimulator
from uvex_transients.simulation.event_catalog import EventCatalog
from uvex_transients.surveys.base import SurveySchedule
from uvex_transients.transients.TDEs import TidalDisruptionEvent

from .test_core import _make_catalog


@pytest.fixture
def sim(make_schedule):
    return SurveySimulator(make_schedule(n_sched=40), transients={"tde": TidalDisruptionEvent()}, simulation_seed=1)


@pytest.fixture
def catalog(hot_spot):
    return _make_catalog(TidalDisruptionEvent(), hot_spot, n_events=20, seed=3)[0]


# --------------------------------------------------------------------------- #
# redshift                                                                    #
# --------------------------------------------------------------------------- #
def test_filter_by_redshift_bounds_both_sides(sim, catalog):
    """Both `min_redshift`/`max_redshift` narrow to the expected rows."""
    result = sim.filter_by_redshift(catalog, uvex, min_redshift=0.03, max_redshift=0.06)
    redshift = np.asarray(result.table["redshift"])
    assert len(result) > 0
    assert np.all((redshift >= 0.03) & (redshift <= 0.06))

    expected = np.count_nonzero(
        (np.asarray(catalog.table["redshift"]) >= 0.03) & (np.asarray(catalog.table["redshift"]) <= 0.06)
    )
    assert len(result) == expected


def test_filter_by_redshift_requires_a_bound(sim, catalog):
    """Neither bound given raises."""
    with pytest.raises(ValueError, match="At least one of"):
        sim.filter_by_redshift(catalog, uvex)


# --------------------------------------------------------------------------- #
# transient_type                                                             #
# --------------------------------------------------------------------------- #
def test_filter_by_transient_type_keeps_only_named_types(sim, catalog):
    """Rows of an unlisted type are dropped; listed ones survive."""
    table = catalog.table.copy()
    table["transient_type"] = np.where(np.arange(len(table)) % 2 == 0, "tde", "other")
    mixed = EventCatalog(
        table=table, nside=catalog.nside, order=catalog.order, time_bins=catalog.time_bins, seed=catalog.seed
    )

    result = sim.filter_by_transient_type(mixed, uvex, types=["tde"])
    assert set(np.asarray(result.table["transient_type"])) == {"tde"}
    assert len(result) == np.count_nonzero(np.asarray(table["transient_type"]) == "tde")


def test_filter_by_transient_type_requires_nonempty_types(sim, catalog):
    with pytest.raises(ValueError, match="at least one transient type"):
        sim.filter_by_transient_type(catalog, uvex, types=[])


# --------------------------------------------------------------------------- #
# peak_magnitude / peak_flux / peak_luminosity                               #
# --------------------------------------------------------------------------- #
def test_peak_magnitude_and_peak_flux_agree(sim, catalog):
    """peak_flux_cut and peak_magnitude_cut evaluate the same underlying peak, just in different units."""
    flux = sim._peak_band_flux(catalog, uvex)
    mag = flux.to_value(u.ABmag)

    by_mag = sim.filter_by_peak_magnitude(catalog, uvex, max_mag=float(np.median(mag)))
    by_flux = sim.filter_by_peak_flux(
        catalog, uvex, min_flux=float(np.median(flux.to_value(u.erg / u.s / u.cm**2 / u.Hz)))
    )

    assert set(np.asarray(by_mag.table["event_id"])) == set(np.asarray(by_flux.table["event_id"]))


def test_peak_magnitude_requires_a_bound(sim, catalog):
    with pytest.raises(ValueError, match="At least one of"):
        sim.filter_by_peak_magnitude(catalog, uvex)


def test_peak_flux_requires_a_bound(sim, catalog):
    with pytest.raises(ValueError, match="At least one of"):
        sim.filter_by_peak_flux(catalog, uvex)


def test_peak_luminosity_bounds_correctly(sim, catalog):
    """A `min_luminosity` far above every event's peak drops everything; far below keeps everything."""
    dropped = sim.filter_by_peak_luminosity(catalog, uvex, min_luminosity=1e60)
    kept = sim.filter_by_peak_luminosity(catalog, uvex, min_luminosity=0.0)
    assert len(dropped) == 0
    assert len(kept) == len(catalog)


def test_peak_luminosity_requires_a_bound(sim, catalog):
    with pytest.raises(ValueError, match="At least one of"):
        sim.filter_by_peak_luminosity(catalog, uvex)


def test_peak_cuts_handle_empty_catalog(sim, hot_spot):
    empty = _make_catalog(TidalDisruptionEvent(), hot_spot, n_events=0)[0]
    assert len(sim.filter_by_peak_magnitude(empty, uvex, max_mag=30.0)) == 0
    assert len(sim.filter_by_peak_luminosity(empty, uvex, min_luminosity=0.0)) == 0


# --------------------------------------------------------------------------- #
# time_to_first_detection / baseline                                         #
# --------------------------------------------------------------------------- #
def test_time_to_first_detection_matches_manual_computation(sim, catalog):
    """time_to_first_detection_cut agrees with a manual reduction of `iter_epoch_snr_chunks`."""
    snr_threshold = 3.0
    epochs = list(sim.iter_epoch_snr_chunks(catalog, uvex, progress=False, detection_floor=snr_threshold))
    first_by_event = {}
    for chunk in epochs:
        above = np.asarray(chunk["snr"]) > snr_threshold
        for eid, t in zip(np.asarray(chunk["event_id"])[above], chunk["t_since_explosion"].to_value(u.day)[above]):
            first_by_event[eid] = min(first_by_event.get(eid, np.inf), t)

    max_delay = 5.0
    expected = {eid for eid, t in first_by_event.items() if t <= max_delay}
    assert 0 < len(expected) < len(first_by_event)  # guard against a vacuous all-empty/all-match comparison

    result = sim.filter_by_time_to_first_detection(catalog, uvex, snr_threshold=snr_threshold, max_delay=max_delay)
    assert set(np.asarray(result.table["event_id"])) == expected


def test_time_to_first_detection_requires_a_bound(sim, catalog):
    with pytest.raises(ValueError, match="At least one of"):
        sim.filter_by_time_to_first_detection(catalog, uvex, snr_threshold=5.0)


def test_baseline_cut_two_sided_example(sim, catalog):
    """A min_baseline/max_baseline pair keeps only events with a sub-threshold gap and a super-threshold span."""
    snr_threshold = 3.0
    stats = sim._collect_detection_epoch_stats(catalog, uvex, snr_threshold, None, None)
    assert len(stats) >= 2  # need at least two events with qualifying epochs for the split below to be meaningful

    # min_baseline above every event's own min_gap (so that side of the requirement never excludes anyone),
    # max_baseline at the median span (so that side genuinely splits the catalog in two).
    min_baseline = max(gap for _t_first, gap, _span in stats.values() if gap is not None) + 1.0
    spans = sorted(span for _t_first, _gap, span in stats.values())
    max_baseline = spans[len(spans) // 2]
    assert spans[0] <= max_baseline < spans[-1]  # guard against a vacuous all-empty/all-match comparison

    expected = {
        eid
        for eid, (_t_first, gap, span) in stats.items()
        if gap is not None and gap < min_baseline and span > max_baseline
    }
    assert 0 < len(expected) < len(stats)

    result = sim.filter_by_baseline(
        catalog, uvex, snr_threshold=snr_threshold, min_baseline=min_baseline, max_baseline=max_baseline
    )
    assert set(np.asarray(result.table["event_id"])) == expected


def test_baseline_cut_requires_a_bound(sim, catalog):
    with pytest.raises(ValueError, match="At least one of"):
        sim.filter_by_baseline(catalog, uvex, snr_threshold=5.0)


# --------------------------------------------------------------------------- #
# first_visit_detected                                                        #
# --------------------------------------------------------------------------- #
def _first_visit_catalog() -> EventCatalog:
    """A 4-event catalog with just enough columns for `filter_by_first_visit_detected`."""
    table = QTable()
    table["event_id"] = np.array([0, 1, 2, 3], dtype=np.int64)
    return EventCatalog(table=table, nside=64, order="nested", time_bins=Time(["2025-01-01", "2025-06-01"]))


def test_collect_solo_detection_observation_indices_keeps_only_single_epoch_events(sim, catalog, monkeypatch):
    """Events with exactly one qualifying epoch are kept; zero- and multi-epoch events are dropped."""
    chunk = QTable(
        {
            "event_id": np.array([0, 1, 1, 2]),
            "observation_index": np.array([5, 6, 7, 8]),
            "snr": np.array([9.0, 9.0, 9.0, 9.0]),
        }
    )
    monkeypatch.setattr(sim, "iter_epoch_snr_chunks", lambda *args, **kwargs: iter([chunk]))

    solo = sim._collect_solo_detection_observation_indices(catalog, uvex, 5.0, None, None)

    assert solo == {0: 5, 2: 8}  # event 1 has two epochs (6 and 7), so it's excluded entirely


def _revisit_schedule_sim() -> SurveySimulator:
    """A 4-observation `SurveySimulator` where row 2 revisits row 0's field; rows 1/3 are each unique."""
    schedule = SurveySchedule(
        QTable(
            {
                "start_time": Time("2025-01-01T00:00:00") + np.arange(4) * u.hour,
                "duration": np.full(4, 900.0) * u.s,
                "observer_location": uvex.observer_location(Time("2025-01-01T00:00:00") + np.arange(4) * u.hour),
                "action": np.full(4, "observe"),
                "target_coord": SkyCoord(np.zeros(4) * u.deg, np.zeros(4) * u.deg),
                "roll": np.zeros(4) * u.deg,
                "field_id": np.array([0, 1, 0, 3]),
                "block_id": np.zeros(4, dtype=int),
            }
        ),
        CircleSkyRegion(center=SkyCoord(0 * u.deg, 0 * u.deg), radius=1 * u.deg),
    )
    assert list(schedule.first_visit_mask) == [True, True, False, True]
    return SurveySimulator(schedule, transients={"tde": TidalDisruptionEvent()})


def test_first_visit_detected_drops_solo_first_visit_and_keeps_the_rest(monkeypatch):
    """
    A hand-built solo-detection map exercises all three outcomes at once: dropped
    (solo + first visit), kept (solo + revisit), and kept (no solo entry at all, standing
    in for both multi-epoch and zero-epoch events, which `filter_by_first_visit_detected`
    treats identically -- see `_collect_solo_detection_observation_indices`).
    """
    sim = _revisit_schedule_sim()
    monkeypatch.setattr(
        sim,
        "_collect_solo_detection_observation_indices",
        lambda *args, **kwargs: {0: 0, 1: 2},  # event 0 on a first visit; event 1 on a revisit
    )

    result = sim.filter_by_first_visit_detected(_first_visit_catalog(), uvex, snr_threshold=5.0)

    assert set(np.asarray(result.table["event_id"])) == {1, 2, 3}


def test_filter_by_snr_exclude_first_visit_detections_defaults_to_true(monkeypatch):
    """`exclude_first_visit_detections` is on by default, dropping a solo detection on a first-ever field visit."""
    sim = _revisit_schedule_sim()
    chunk = QTable(
        {
            "event_id": np.array([0, 1, 2, 2]),
            "observation_index": np.array([0, 2, 1, 3]),  # event 0: first visit; event 1: a revisit
            "snr": np.array([9.0, 9.0, 9.0, 9.0]),
        }
    )
    calls = []
    monkeypatch.setattr(sim, "iter_epoch_snr_chunks", lambda *args, **kwargs: (calls.append(1), iter([chunk]))[1])

    with_default = sim.filter_by_snr(_first_visit_catalog(), uvex, snr_threshold=5.0)
    assert len(calls) == 1
    without_flag = sim.filter_by_snr(
        _first_visit_catalog(), uvex, snr_threshold=5.0, exclude_first_visit_detections=False
    )
    # `exclude_first_visit_detections` piggybacks on the same pass rather than
    # re-running the (expensive) SNR evaluation a second time.
    assert len(calls) == 2

    # By default, event 0's lone epoch on a first-ever visit is excluded; event 1's lone
    # epoch on a revisit, and event 2's two epochs, both survive regardless.
    assert set(np.asarray(with_default.table["event_id"])) == {1, 2}
    # Event 3 has no qualifying epoch either way, so `n_visits` alone already excludes it.
    assert set(np.asarray(without_flag.table["event_id"])) == {0, 1, 2}


# --------------------------------------------------------------------------- #
# region / sky_position                                                      #
# --------------------------------------------------------------------------- #
def test_filter_by_region_keeps_only_events_inside(sim, catalog, hot_spot):
    """A circle centered on `hot_spot` keeps only events near it (most of `_make_catalog`'s in-footprint ones)."""
    region = CircleSkyRegion(hot_spot, 1.0 * u.deg)
    result = sim.filter_by_region(catalog, uvex, region=region)

    expected = np.asarray(fov_contains(region, catalog.table["coord"]), dtype=bool)
    assert set(np.asarray(result.table["event_id"])) == set(np.asarray(catalog.table["event_id"])[expected])


def test_filter_by_region_rejects_bad_type(sim, catalog):
    with pytest.raises(TypeError, match="str/Path/SkyRegion/Regions"):
        sim.filter_by_region(catalog, uvex, region=12345)


def test_filter_by_sky_position_include_vs_exclude(sim, catalog):
    """include/exclude modes partition the catalog exactly (every row in exactly one)."""
    included = sim.filter_by_sky_position(
        catalog, uvex, frame="galactic", theta_min=-10.0, theta_max=10.0, mode="include"
    )
    excluded = sim.filter_by_sky_position(
        catalog, uvex, frame="galactic", theta_min=-10.0, theta_max=10.0, mode="exclude"
    )
    included_ids = set(np.asarray(included.table["event_id"]))
    excluded_ids = set(np.asarray(excluded.table["event_id"]))

    assert 0 < len(included_ids) < len(catalog)  # guard against a vacuous all-included/all-excluded comparison
    assert included_ids | excluded_ids == set(np.asarray(catalog.table["event_id"]))
    assert included_ids & excluded_ids == set()


def test_filter_by_sky_position_default_phi_range_is_unconstrained(sim, catalog):
    """The default phi_min=0/phi_max=360 must behave as 'no longitude constraint', not collapse to phi==0."""
    result = sim.filter_by_sky_position(catalog, uvex, frame="galactic", mode="include")
    assert len(result) == len(catalog)


def test_filter_by_sky_position_bad_mode_raises(sim, catalog):
    with pytest.raises(ValueError, match="'mode' must be"):
        sim.filter_by_sky_position(catalog, uvex, frame="galactic", mode="bogus")


def test_filter_by_sky_position_phi_wraparound(sim, catalog):
    """phi_min > phi_max wraps through 0 rather than selecting an empty range."""
    result = sim.filter_by_sky_position(catalog, uvex, frame="icrs", phi_min=340.0, phi_max=30.0, mode="include")
    ra = catalog.table["coord"].icrs.ra.to_value(u.deg)
    expected = np.count_nonzero((ra >= 340.0) | (ra <= 30.0))
    assert 0 < expected < len(catalog)  # guard against a vacuous all-empty/all-match comparison
    assert len(result) == expected


# --------------------------------------------------------------------------- #
# query                                                                       #
# --------------------------------------------------------------------------- #
def test_filter_by_query_evaluates_a_boolean_expression(sim, catalog):
    result = sim.filter_by_query(catalog, uvex, "(redshift < 0.05) & (transient_type == 'tde')")
    redshift = np.asarray(result.table["redshift"])
    assert len(result) > 0
    assert np.all(redshift < 0.05)


def test_filter_by_query_sky_coord_expression(sim, catalog, hot_spot):
    """The query namespace includes SkyCoord/u, so a coordinate-separation expression works."""
    result = sim.filter_by_query(catalog, uvex, "coord.separation(SkyCoord(150*u.deg, 20*u.deg)) < 5*u.deg")
    sep = catalog.table["coord"].separation(SkyCoord(150 * u.deg, 20 * u.deg))
    expected = np.count_nonzero(sep < 5 * u.deg)
    assert len(result) == expected


def test_filter_by_query_bad_expression_raises_value_error(sim, catalog):
    with pytest.raises(ValueError, match="Failed to evaluate query expression"):
        sim.filter_by_query(catalog, uvex, "not_a_real_column < 5")


def test_filter_by_query_non_boolean_result_raises(sim, catalog):
    with pytest.raises(ValueError, match="must evaluate to a boolean array"):
        sim.filter_by_query(catalog, uvex, "redshift")


def test_filter_by_query_builtins_are_unreachable(sim, catalog):
    """The eval namespace has no builtins, so an attempt to reach them fails."""
    with pytest.raises(ValueError, match="Failed to evaluate query expression"):
        sim.filter_by_query(catalog, uvex, "__import__('os').system('echo hi')")
