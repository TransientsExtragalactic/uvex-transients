"""Tests for the peak and decline-time reductions of sampled light curves."""

import numpy as np
import pytest
from astropy import units as u

from uvex_transients.models.core.lightcurve_metrics import (
    crossing_coordinate,
    decline_time,
    first_true_index,
    peak_luminosity,
    resolve_time_grid,
)


# ---------------------------------------------------------------------------- #
# Peak                                                                         #
# ---------------------------------------------------------------------------- #
def test_the_peak_is_at_the_brightest_grid_point_to_within_the_refinement():
    t = np.linspace(1, 10, 10) * u.day
    luminosity = np.array([[1, 2, 5, 3, 1, 1, 1, 1, 1, 1], [9, 1, 1, 1, 1, 1, 1, 1, 1, 1]]) * u.erg / u.s
    peak, t_peak = peak_luminosity(t, luminosity)
    # The second draw peaks at the first grid point, where there is no parabola to fit.
    assert peak[1].value == 9 and t_peak[1].value == 1
    assert peak[0].value == pytest.approx(5, rel=1e-2)
    assert 2.5 < t_peak[0].value < 3.5


def test_a_smooth_peak_is_found_between_the_points_of_a_coarse_grid():
    # A lognormal pulse, peaking at 7.3 days with height 1e40, on a grid only 25% apart.
    t = np.geomspace(0.5, 100, 28) * u.day
    luminosity = (1e40 * np.exp(-0.5 * (np.log(t.value / 7.3) / 0.8) ** 2))[np.newaxis, :] * u.erg / u.s
    peak, t_peak = peak_luminosity(t, luminosity)
    assert peak.value[0] == pytest.approx(1e40, rel=5e-3)
    assert t_peak.value[0] == pytest.approx(7.3, rel=5e-3)
    # The grid point alone is off by more.
    assert luminosity.value.max() < 0.999e40


# ---------------------------------------------------------------------------- #
# Decline                                                                      #
# ---------------------------------------------------------------------------- #
def test_an_exponential_decline_takes_tau_ln_two_to_halve():
    t = np.geomspace(0.01, 100, 600) * u.day
    tau = np.array([1.0, 5.0, 20.0]) * u.day
    luminosity = (np.exp(-t[np.newaxis, :] / tau[:, np.newaxis]) * 1e40) * u.erg / u.s
    result = decline_time(t, luminosity, 0.5)
    np.testing.assert_allclose(result.to_value(u.day), tau.value * np.log(2), rtol=5e-3)
    # A tenth of the peak takes ln 10 time constants.
    np.testing.assert_allclose(decline_time(t, luminosity, 0.1).to_value(u.day), tau.value * np.log(10), rtol=5e-3)


def test_the_decline_is_timed_from_the_peak_not_from_the_explosion():
    t = np.geomspace(0.01, 100, 800) * u.day
    luminosity = (np.exp(-np.abs(t.value - 10.0) / 3.0) * 1e40)[np.newaxis, :] * u.erg / u.s
    np.testing.assert_allclose(decline_time(t, luminosity, 0.5).to_value(u.day), 3.0 * np.log(2), rtol=2e-2)


def test_a_light_curve_that_never_declines_that_far_is_nan():
    t = np.geomspace(0.1, 10, 100) * u.day
    luminosity = np.array([np.exp(-t.value / 1.0), np.exp(-t.value / 1e4)]) * u.erg / u.s
    result = decline_time(t, luminosity, 0.5)
    assert np.isfinite(result[0]) and np.isnan(result[1])


def test_a_draw_without_a_valid_epoch_is_nan():
    t = np.geomspace(0.1, 10, 50) * u.day
    luminosity = np.array([np.exp(-t.value), np.full(50, np.nan)]) * u.erg / u.s
    peak, t_peak = peak_luminosity(t, luminosity)
    assert np.isfinite(peak[0]) and np.isnan(peak[1]) and np.isnan(t_peak[1])
    assert np.isnan(decline_time(t, luminosity)[1])


@pytest.mark.parametrize("fraction", [0.0, 1.0, 1.5, -0.1])
def test_a_fraction_outside_zero_to_one_raises(fraction):
    with pytest.raises(ValueError, match="between 0 and 1"):
        decline_time(np.array([1.0, 2.0]) * u.day, np.ones((1, 2)) * u.erg / u.s, fraction)


# ---------------------------------------------------------------------------- #
# Crossings                                                                    #
# ---------------------------------------------------------------------------- #
def test_the_first_true_index_is_the_first_column_and_flags_rows_without_one():
    condition = np.array([[False, True, True], [True, False, False], [False, False, False]])
    index, found = first_true_index(condition)
    np.testing.assert_array_equal(index, [1, 0, 0])
    np.testing.assert_array_equal(found, [True, True, False])


def test_a_crossing_is_interpolated_linearly_and_can_be_kept_between_the_samples():
    np.testing.assert_allclose(
        crossing_coordinate(np.array([1.0]), np.array([3.0]), np.array([0.0]), np.array([4.0]), 1.0), [1.5]
    )
    outside = crossing_coordinate(np.array([1.0]), np.array([3.0]), np.array([0.0]), np.array([4.0]), 6.0)
    clipped = crossing_coordinate(np.array([1.0]), np.array([3.0]), np.array([0.0]), np.array([4.0]), 6.0, clip=True)
    np.testing.assert_allclose(outside, [4.0])
    np.testing.assert_allclose(clipped, [3.0])


# ---------------------------------------------------------------------------- #
# Time grid                                                                    #
# ---------------------------------------------------------------------------- #
def test_log_time_grid_in_seconds():
    t = resolve_time_grid(None, 1 * u.day, 100 * u.day, 3)
    np.testing.assert_allclose(t, np.array([1.0, 10.0, 100.0]) * 86400.0)


def test_explicit_grid_ignores_the_other_arguments():
    t = resolve_time_grid(2 * u.day, 5 * u.day, 1 * u.day, 7)
    np.testing.assert_allclose(t, [2 * 86400.0])


@pytest.mark.parametrize(
    "t_min, t_max", [(None, 1 * u.day), (1 * u.day, None), (2 * u.day, 1 * u.day), (0 * u.day, 1 * u.day)]
)
def test_bad_time_grid_is_rejected(t_min, t_max):
    with pytest.raises(ValueError):
        resolve_time_grid(None, t_min, t_max, 10)
