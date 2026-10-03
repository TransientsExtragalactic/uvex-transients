"""Tests for `SurveySimulator.run_alert_action`."""

import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable, vstack
from astropy.time import Time
from m4opt.missions._uvex import uvex

from uvex_transients.simulation.core import SurveySimulator
from uvex_transients.surveys.base import SurveySchedule
from uvex_transients.transients.TDEs import TidalDisruptionEvent

from .test_core import _make_catalog


@pytest.fixture
def sim(make_schedule):
    return SurveySimulator(make_schedule(n_sched=40), transients={"tde": TidalDisruptionEvent()}, simulation_seed=1)


@pytest.fixture
def catalog(hot_spot):
    return _make_catalog(TidalDisruptionEvent(), hot_spot, n_events=20, seed=3)[0]


def _with_downlink_after_every_observation(schedule: SurveySchedule, delay: u.Quantity, duration: u.Quantity):
    """Append one ``"downlink"`` row `delay` after the schedule's last observation ends."""
    table = schedule.table.copy()
    downlink_start = Time([schedule.end_time + delay])

    # Built from scratch (rather than copying + overwriting a row of `table`) since
    # `table["action"][:] = "downlink"`-style in-place assignment silently truncates to
    # "observe"'s fixed string width instead of widening the column.
    row = QTable()
    row["start_time"] = downlink_start
    row["duration"] = u.Quantity([duration])
    row["observer_location"] = uvex.observer_location(downlink_start)
    row["action"] = np.array(["downlink"])
    row["target_coord"] = table["target_coord"][:1]
    row["roll"] = table["roll"][:1]
    row["field_id"] = np.array([-1], dtype=table["field_id"].dtype)
    row["block_id"] = table["block_id"][:1]

    return SurveySchedule(vstack([table, row], metadata_conflicts="silent"), schedule.fov)


# --------------------------------------------------------------------------- #
# No downlink ever scheduled                                                  #
# --------------------------------------------------------------------------- #
def test_run_alert_action_masked_when_no_downlink_scheduled(sim, catalog):
    """`make_schedule` has no downlink rows at all, so every detected event's alert columns are masked."""
    detected = sim.filter_by_snr(catalog, uvex, snr_threshold=3.0)
    assert len(detected) > 0  # guard against a vacuous all-empty comparison

    alerts = sim.run_alert_action(detected, uvex, snr_threshold=3.0)

    assert len(alerts) == len(detected)
    assert np.all(alerts["t_downlink"].mask)
    assert np.all(alerts["alert_time"].mask)


# --------------------------------------------------------------------------- #
# A downlink after every observation                                          #
# --------------------------------------------------------------------------- #
def test_run_alert_action_finds_the_downlink_after_detection(sim, catalog):
    """With one downlink scheduled after every observation, every detected event resolves to it."""
    original_end_time = sim.survey_schedule.end_time
    schedule = _with_downlink_after_every_observation(sim.survey_schedule, delay=1 * u.hour, duration=10 * u.min)
    sim_with_downlink = SurveySimulator(schedule, transients=sim.transient_collection, simulation_seed=1)

    detected = sim_with_downlink.filter_by_snr(catalog, uvex, snr_threshold=3.0)
    assert len(detected) > 0

    alerts = sim_with_downlink.run_alert_action(detected, uvex, snr_threshold=3.0)

    assert not np.any(alerts["t_downlink"].mask)
    expected_downlink = original_end_time + 1 * u.hour + 10 * u.min
    assert np.all(np.abs((alerts["t_downlink"] - expected_downlink).sec) < 1e-3)
    assert np.all(alerts["alert_time"] == alerts["t_downlink"])  # processing_delay defaults to 0
    assert np.all(alerts["alert_delay"] >= 0 * u.hour)


def test_run_alert_action_processing_delay_adds_on_top_of_downlink(sim, catalog):
    """A nonzero `processing_delay` shifts `alert_time` (and `alert_delay`) but not `t_downlink`."""
    schedule = _with_downlink_after_every_observation(sim.survey_schedule, delay=1 * u.hour, duration=10 * u.min)
    sim_with_downlink = SurveySimulator(schedule, transients=sim.transient_collection, simulation_seed=1)
    detected = sim_with_downlink.filter_by_snr(catalog, uvex, snr_threshold=3.0)

    baseline = sim_with_downlink.run_alert_action(detected, uvex, snr_threshold=3.0)
    delayed = sim_with_downlink.run_alert_action(detected, uvex, snr_threshold=3.0, processing_delay=2 * u.hour)

    assert np.all(delayed["t_downlink"] == baseline["t_downlink"])
    assert np.all(np.abs((delayed["alert_time"] - baseline["alert_time"]).sec - (2 * u.hour).to_value(u.s)) < 1e-3)
    assert np.all(np.abs((delayed["alert_delay"] - baseline["alert_delay"]).to_value(u.hour) - 2.0) < 1e-6)


# --------------------------------------------------------------------------- #
# Detection-epoch consistency with filter_by_snr                              #
# --------------------------------------------------------------------------- #
def test_run_alert_action_first_detection_matches_manual_computation(sim, catalog):
    """`t_first_detection` matches a manual reduction of `iter_epochs`, per event."""
    snr_threshold = 3.0
    schedule = _with_downlink_after_every_observation(sim.survey_schedule, delay=1 * u.hour, duration=10 * u.min)
    sim_with_downlink = SurveySimulator(schedule, transients=sim.transient_collection, simulation_seed=1)

    epochs = list(sim_with_downlink.iter_epochs(catalog, uvex, progress=False, detection_floor=snr_threshold))
    first_t_obs = {}
    for chunk in epochs:
        above = np.asarray(chunk["snr"]) > snr_threshold
        for eid, t_obs in zip(np.asarray(chunk["event_id"])[above], chunk["t_obs"][above]):
            if eid not in first_t_obs or t_obs < first_t_obs[eid]:
                first_t_obs[eid] = t_obs

    assert len(first_t_obs) > 0  # guard against a vacuous all-empty comparison

    detected = sim_with_downlink.filter_by_snr(catalog, uvex, snr_threshold=snr_threshold)
    alerts = sim_with_downlink.run_alert_action(detected, uvex, snr_threshold=snr_threshold)

    for event_id, t_obs in zip(alerts["event_id"], alerts["t_first_detection"]):
        assert abs((t_obs - first_t_obs[int(event_id)]).sec) < 1e-3


# --------------------------------------------------------------------------- #
# Empty results                                                               #
# --------------------------------------------------------------------------- #
def test_run_alert_action_empty_when_no_qualifying_epochs(sim, catalog):
    """A threshold no event can ever clear yields a zero-row (but well-formed) table."""
    alerts = sim.run_alert_action(catalog, uvex, snr_threshold=1e6)

    assert len(alerts) == 0
    assert alerts.colnames == [
        "event_id",
        "transient_type",
        "t_first_detection",
        "detection_band",
        "detection_snr",
        "t_downlink",
        "alert_time",
        "alert_delay",
    ]
    assert isinstance(alerts["t_first_detection"], Time)


def test_run_alert_action_dispatches_via_run_action(sim, catalog):
    """`run_action("alert", ...)` reaches `run_alert_action`."""
    detected = sim.filter_by_snr(catalog, uvex, snr_threshold=3.0)
    direct = sim.run_alert_action(detected, uvex, snr_threshold=3.0)
    dispatched = sim.run_action("alert", catalog=detected, mission=uvex, snr_threshold=3.0)

    assert list(direct["event_id"]) == list(dispatched["event_id"])


def test_run_alert_action_rejects_non_event_catalog(sim):
    with pytest.raises(TypeError, match="EventCatalog"):
        sim.run_alert_action(QTable(), uvex, snr_threshold=5.0)


def test_run_alert_action_rejects_non_quantity_processing_delay(sim, catalog):
    with pytest.raises(TypeError, match="Quantity"):
        sim.run_alert_action(catalog, uvex, snr_threshold=5.0, processing_delay=3600)
