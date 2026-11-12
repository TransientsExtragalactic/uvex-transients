"""
Testing phase space behavior on test schedule scenarios to verify accurate behavior in
predictable scenarios.

There are several test scenarios considered here:

- **Single Field Response**: In this test, we consider a single pointing that we stare at for a 900s dwell with a
  re-visit time distributed as a Gaussian distribution with a given peak and width. We do this for two different values
  of the peak to ensure that the following occur:

  - The sensitivity drops off at durations below the revisit time.
  - The relative position of that drop varies between the two maps.

- **Map Difference**: The same two schedules, now of equal length, are fitted as experiments, and the
  difference of the two maps is checked. The denser cadence must never do worse, so the difference has to be
  negative (or consistent with zero) everywhere, strongly so where the half-decline time lies between the two revisit
  times, and the surveys must have the same volume-time, so that the difference is the cadence alone.
- **Null Difference**: One schedule is fitted twice from independent seeds. The difference of the two maps has to be
  consistent with zero, with the scatter that the maps' own uncertainties predict.
"""

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable
from astropy.time import Time
from m4opt.missions._uvex import uvex

from uvex_transients.phase_space.compare import difference_map
from uvex_transients.phase_space.experiment import CampaignSettings, Experiment
from uvex_transients.phase_space.sampling import CanonicalSampler, StoppingRule
from uvex_transients.surveys.base import SurveySchedule

MISSION = "uvex_fast"
NSIDE = 32
TIME_BINS = 6

DWELL = 900.0 * u.s
SURVEY = 1200.0  # days; the same for every revisit time, so that only the cadence differs between maps
WIDTH_FRACTION = 0.1

# The two revisit times (peak of the Gaussian, in days), one decade apart.
REVISITS = (3.0, 30.0)

# Bright enough to be seen at the first visit out to z of about 0.5, so that the recovery is set by the cadence.
LOG_LUMINOSITY = 42.5
LOG_HALF_TIME = np.arange(-1.0, 2.51, 0.25)


def revisit_schedule(fov, target, revisit, seed=0):
    """A single field, observed for `SURVEY` days for `DWELL` at a time, with Gaussian gaps of mean `revisit` days."""
    n_visits = round(SURVEY / revisit) + 1
    rng = np.random.default_rng(seed)
    gaps = rng.normal(revisit, WIDTH_FRACTION * revisit, n_visits - 1)
    gaps = np.clip(gaps, 2 * DWELL.to_value(u.day), None)
    times = Time("2025-01-01T00:00:00") + np.concatenate([[0.0], np.cumsum(gaps)]) * u.day

    table = QTable()
    table["start_time"] = times
    table["duration"] = np.full(n_visits, DWELL.to_value(u.s)) * u.s
    table["observer_location"] = uvex.observer_location(times)
    table["action"] = np.full(n_visits, "observe")
    table["target_coord"] = SkyCoord(
        np.full(n_visits, target.ra.value) * u.deg, np.full(n_visits, target.dec.value) * u.deg
    )
    table["roll"] = np.zeros(n_visits) * u.deg
    table["field_id"] = np.zeros(n_visits, dtype=int)
    table["block_id"] = np.zeros(n_visits, dtype=int)
    return SurveySchedule(table, fov)


def onset(recovery, fraction=0.25):
    """The log10 half-decline time at which the recovery fraction first reaches `fraction` of its maximum."""
    level = fraction * recovery.max()
    i = np.argmax(recovery >= level)
    return float(np.interp(level, recovery[i - 1 : i + 1], LOG_HALF_TIME[i - 1 : i + 1]))


# The rectangle of the maps: log10 L_band / (erg/s) and log10 t_half / day.
MAP_BOUNDS = [(40.5, 43.5), (-1.0, 2.5)]
MAP_GRID = 24


def fit_map(schedule, seed):
    """Fit the recovery map of `schedule` with an experiment, and return it on a `MAP_GRID` square grid."""
    pytest.importorskip("sklearn")
    sampler = CanonicalSampler(schedule, MISSION, nside=NSIDE, order="nested", time_bins=TIME_BINS, n_visits=2)
    experiment = Experiment.create(
        sampler,
        bounds=MAP_BOUNDS,
        rule=StoppingRule(chunk_size=250, k_target=50, n_max=2000),
        campaign=CampaignSettings(n_initial=24, batch_size=8, max_points=64, tolerance=0.1, seed=seed),
        seed=seed,
        n_restarts=1,
    )
    experiment.run()
    return experiment.grid_table(MAP_GRID)


@pytest.mark.slow
def test_the_schedule_revisits_one_field_at_the_requested_cadence(fov, hot_spot):
    for revisit in REVISITS:
        schedule = revisit_schedule(fov, hot_spot, revisit)
        gaps = np.diff(schedule.table["start_time"].jd)
        assert len(schedule) == round(SURVEY / revisit) + 1 and set(schedule.table["field_id"]) == {0}
        assert np.all(schedule.table["duration"] == DWELL)
        assert gaps.mean() == pytest.approx(revisit, rel=0.05)
        assert gaps.std() == pytest.approx(WIDTH_FRACTION * revisit, rel=0.3)


@pytest.fixture(scope="module")
def samplers(fov, hot_spot):
    """One `CanonicalSampler` per revisit time."""
    return {
        revisit: CanonicalSampler(
            revisit_schedule(fov, hot_spot, revisit),
            MISSION,
            nside=NSIDE,
            order="nested",
            time_bins=TIME_BINS,
            n_visits=2,
        )
        for revisit in REVISITS
    }


@pytest.fixture(scope="module")
def recovery(samplers):
    """The recoveries ``(k, n)`` along `LOG_HALF_TIME` at `LOG_LUMINOSITY`, for each revisit time."""
    rule = StoppingRule(chunk_size=250, k_target=60, n_max=2000)
    curves = {}
    for revisit, sampler in samplers.items():
        outcomes = [
            rule.run(sampler, [LOG_LUMINOSITY, log_half_time], np.random.SeedSequence([0, index]))
            for index, log_half_time in enumerate(LOG_HALF_TIME)
        ]
        curves[revisit] = np.array([outcome.k for outcome in outcomes]), np.array([outcome.n for outcome in outcomes])
    return curves


@pytest.mark.slow
class TestSingleFieldResponse:
    @pytest.mark.parametrize("revisit", REVISITS)
    def test_nothing_is_recovered_well_below_the_revisit_time(self, recovery, revisit):
        # Two visits must see the event, so one that fades within a tenth of the gap is never seen twice.
        k, _ = recovery[revisit]
        assert np.all(k[10.0**LOG_HALF_TIME <= revisit / 10.0] == 0)

    @pytest.mark.parametrize("revisit", REVISITS)
    def test_events_longer_than_the_revisit_time_are_recovered(self, recovery, revisit):
        k, n = recovery[revisit]
        assert (k / n)[10.0**LOG_HALF_TIME >= revisit].max() > 0.03

    @pytest.mark.parametrize("revisit", REVISITS)
    def test_the_sensitivity_turns_on_near_the_revisit_time(self, recovery, revisit):
        k, n = recovery[revisit]
        assert revisit / 10.0 < 10.0 ** onset(k / n) < 2.0 * revisit

    def test_the_drop_moves_with_the_revisit_time(self, recovery):
        (k_short, n_short), (k_long, n_long) = (recovery[revisit] for revisit in REVISITS)
        shift = onset(k_long / n_long) - onset(k_short / n_short)
        assert shift == pytest.approx(np.log10(REVISITS[1] / REVISITS[0]), abs=0.4)

    def test_the_surveys_have_the_same_volume_time(self, samplers):
        # The map difference is then the cadence alone, with no factor for the length of the survey.
        short, long = (samplers[revisit].volume_time([LOG_LUMINOSITY, 1.0]) for revisit in REVISITS)
        assert long == pytest.approx(short, rel=0.01)

    def test_the_denser_cadence_never_recovers_less(self, recovery):
        (k_short, n_short), (k_long, n_long) = (recovery[revisit] for revisit in REVISITS)
        f_short, f_long = k_short / n_short, k_long / n_long
        sigma = np.sqrt(f_short * (1 - f_short) / n_short + f_long * (1 - f_long) / n_long)
        assert np.all(f_short - f_long > -3.0 * sigma - 1e-3)


def on_grid(table, column):
    """A column of a map table as a ``(luminosity, half-decline time)`` array."""
    return np.asarray(table[column]).reshape(MAP_GRID, MAP_GRID)


@pytest.fixture(scope="module")
def maps(fov, hot_spot):
    """Fitted maps: the short and the long revisit schedule, and the short one again from other seeds."""
    short, long = (revisit_schedule(fov, hot_spot, revisit) for revisit in REVISITS)
    return {"short": fit_map(short, seed=0), "long": fit_map(long, seed=0), "short_again": fit_map(short, seed=1)}


@pytest.mark.slow
class TestMapDifference:
    @pytest.fixture(scope="class")
    def difference(self, maps):
        return difference_map(maps["short"], maps["long"])

    def test_the_surveys_have_the_same_volume_time_on_the_whole_map(self, maps):
        ratio = on_grid(maps["long"], "volume_time") / on_grid(maps["short"], "volume_time")
        np.testing.assert_allclose(ratio, 1.0, atol=0.01)

    def test_the_denser_cadence_is_never_significantly_worse(self, difference):
        # Delta is log10(long / short), so a positive one would mean that the long revisit recovers more.
        delta, sigma, limit = (on_grid(difference, column) for column in ("delta_dex", "sigma_dex", "limit"))
        measured = np.isfinite(delta) & (limit == 0)
        assert measured.sum() > 100
        assert np.all(delta[measured] / sigma[measured] < 3.0)

    def test_the_long_revisit_loses_most_between_the_two_revisit_times(self, difference):
        delta = on_grid(difference, "delta_dex")
        log_half_time = np.linspace(*MAP_BOUNDS[1], MAP_GRID)
        at_short_revisit = np.argmin(np.abs(log_half_time - np.log10(REVISITS[0])))
        assert np.nanmedian(delta[:, at_short_revisit]) < -1.0

    def test_the_difference_fades_for_events_much_longer_than_both_revisit_times(self, difference):
        # Not to zero: a denser cadence still gives an event at the detection threshold more chances to be seen.
        delta = on_grid(difference, "delta_dex")
        log_half_time = np.linspace(*MAP_BOUNDS[1], MAP_GRID)
        long_events = log_half_time > np.log10(REVISITS[1]) + 0.4
        assert -0.4 < np.nanmedian(delta[:, long_events]) < 0.05


@pytest.mark.slow
class TestNullDifference:
    @pytest.fixture(scope="class")
    def significance(self, maps):
        difference = difference_map(maps["short"], maps["short_again"])
        delta, sigma, limit = (on_grid(difference, column) for column in ("delta_dex", "sigma_dex", "limit"))
        measured = np.isfinite(delta) & np.isfinite(sigma) & (sigma > 0) & (limit == 0)
        assert measured.sum() > 100
        return delta[measured], delta[measured] / sigma[measured]

    def test_the_difference_is_centered_on_zero(self, significance):
        delta, _ = significance
        assert abs(np.median(delta)) < 0.05

    def test_the_difference_is_within_its_uncertainty(self, significance):
        _, z = significance
        assert np.mean(np.abs(z) < 2.0) > 0.9

    def test_the_uncertainties_are_not_too_small(self, significance):
        # The scatter of the difference in units of its uncertainty is one if the uncertainty is right.
        _, z = significance
        assert np.sqrt(np.mean(z**2)) < 1.25
