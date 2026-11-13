"""Tests for `BracketedBreakoutSampler`: a breakout recovered only if its explosion is bracketed within its own window."""

import numpy as np
import pytest

from uvex_transients.phase_space.sampling import BracketedBreakoutSampler, BreakoutSampler, get_sampler

NSIDE = 32
MISSION = "uvex_fast"
POINT = np.array([44.0, 5.2])  # log10 L_0 / (erg/s), log10 t_0 / s: a day and a half


@pytest.fixture(scope="module")
def schedule(make_schedule):
    return make_schedule(n_sched=60)


def make_sampler(schedule, cls=BracketedBreakoutSampler, **kwargs):
    defaults = {"mission": MISSION, "nside": NSIDE, "order": "nested", "time_bins": 50, "mag_limit": 25.0}
    defaults.update(kwargs)
    return cls(schedule, **defaults)


class _Catalog:
    """The part of an event catalog that the detection reads: the redshift of each event."""

    def __init__(self, z):
        self.table = {"redshift": np.asarray(z, dtype=float)}

    def __len__(self):
        return len(self.table["redshift"])


class _Simulator:
    """A simulator that records the cut it is asked for, and keeps every event."""

    def __init__(self):
        self.calls = []

    def filter_by_time_since_last_nondetection(self, catalog, mission, **kwargs):
        self.calls.append(kwargs)
        return catalog


def test_it_is_a_registered_kind_of_sampler():
    assert get_sampler("breakout_bracketed") is BracketedBreakoutSampler
    assert issubclass(BracketedBreakoutSampler, BreakoutSampler)


def test_the_gap_is_compared_event_by_event_with_the_window_of_its_own_redshift(schedule):
    sampler = make_sampler(schedule)
    sampler._place(sampler._points(POINT))
    z = np.array([0.05, 0.3, 0.8, 1.2])
    simulator = _Simulator()

    sampler.detect(simulator, _Catalog(z))

    (call,) = simulator.calls
    # x_half t_0 (1 + z) days, one value for each event, so that a nearer event has a shorter gap
    expected = sampler.x_half * 10.0 ** POINT[1] / 86400.0 * (1.0 + z)
    np.testing.assert_allclose(call["max_delay"], expected)
    assert np.all(np.diff(call["max_delay"]) > 0)
    assert call["snr_threshold"] == sampler.snr_threshold
    assert call["bands"] == [sampler.band]


def test_the_search_reaches_back_one_window_of_the_farthest_event(schedule):
    """A non-detection before the explosion by more than an event's own gap could not bracket it, so one length serves."""
    sampler = make_sampler(schedule)
    sampler._place(sampler._points(POINT))
    simulator = _Simulator()

    sampler.detect(simulator, _Catalog([0.1, 0.5]))

    (call,) = simulator.calls
    far = float(sampler._transient.duration_limit.to_value("day"))
    assert call["lookback"] == pytest.approx(far)
    # Every event's own gap is within the farthest event's, which is the window the point follows
    assert np.all(call["max_delay"] <= far * (1.0 + 1e-9))


def test_the_gap_scales_with_gap_factor(schedule):
    simulator, doubled = _Simulator(), _Simulator()
    for factor, recorder in ((1.0, simulator), (2.0, doubled)):
        sampler = make_sampler(schedule, gap_factor=factor)
        sampler._place(sampler._points(POINT))
        sampler.detect(recorder, _Catalog([0.2, 0.9]))
    np.testing.assert_allclose(doubled.calls[0]["max_delay"], 2.0 * simulator.calls[0]["max_delay"])
    assert doubled.calls[0]["lookback"] == pytest.approx(2.0 * simulator.calls[0]["lookback"])


def test_the_window_follows_the_point_that_was_placed(schedule):
    sampler = make_sampler(schedule)
    gaps = []
    for x in (POINT, POINT + [0.0, 0.5]):
        sampler._place(sampler._points(x))
        simulator = _Simulator()
        sampler.detect(simulator, _Catalog([0.3]))
        gaps.append(simulator.calls[0]["max_delay"][0])
    assert gaps[1] == pytest.approx(gaps[0] * 10**0.5)


def test_no_events_are_left_as_they_are(schedule):
    sampler = make_sampler(schedule)
    sampler._place(sampler._points(POINT))
    simulator = _Simulator()
    empty = _Catalog([])
    assert sampler.detect(simulator, empty) is empty
    assert simulator.calls == []


@pytest.mark.parametrize("gap_factor", [0.0, -1.0, float("nan")])
def test_a_gap_factor_has_to_be_positive(schedule, gap_factor):
    with pytest.raises(ValueError, match="gap_factor"):
        make_sampler(schedule, gap_factor=gap_factor)


def test_it_defines_its_own_detection(schedule):
    with pytest.raises(ValueError, match="defines its own detection"):
        make_sampler(schedule, detection=[{"cut": "snr"}])


def test_the_gap_factor_is_recorded_and_rebuilds_the_same_sampler(schedule):
    sampler = make_sampler(schedule, gap_factor=1.5)
    assert sampler.settings["gap_factor"] == 1.5
    assert "detection" not in sampler.settings

    rebuilt = BracketedBreakoutSampler.from_settings(sampler.settings, schedule=schedule, mission=MISSION)
    assert rebuilt.gap_factor == 1.5
    assert rebuilt.settings == sampler.settings


def test_a_bracketed_breakout_is_a_detection_that_is_also_bracketed(schedule):
    """With no first-visit exclusion the plain rule is every detection, and the bracket asks for more than that.

    The plain rule's own first-visit exclusion is cruder than a bracket, which is satisfied by any earlier
    non-detection, so against the plain rule as the maps use it a few events are recovered only by the bracket.
    """
    plain = make_sampler(schedule, cls=BreakoutSampler, exclude_first_visit_detections=False)
    bracketed = make_sampler(schedule)
    n_plain = plain.draw(POINT, 200, np.random.SeedSequence(11))
    n_bracketed = bracketed.draw(POINT, 200, np.random.SeedSequence(11))
    assert 0 <= n_bracketed <= n_plain
