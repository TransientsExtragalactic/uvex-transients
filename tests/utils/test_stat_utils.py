"""Tests for the reusable statistical operations."""

import numpy as np
import pytest

from uvex_transients.utils.stat_utils import (
    binomial_logit,
    clopper_pearson_interval,
    combine_fractional_errors,
    convert_CI_to_fractional,
    fractional_error_to_bounds,
    tolerance_limit,
    tolerance_min_samples,
    tolerance_rank,
)


class TestTolerance:
    """Rank selection and coverage of the distribution-free upper tolerance limit."""

    def test_the_minimum_sample_size_matches_the_textbook_value(self):
        assert tolerance_min_samples(0.95, 0.01) == 299

    def test_maximum_is_valid_exactly_at_the_minimum(self):
        assert tolerance_rank(299, 0.95, 0.01) == 1
        with pytest.raises(ValueError, match="at least 299"):
            tolerance_rank(298, 0.95, 0.01)

    @pytest.mark.parametrize("confidence, tolerance", [(0.0, 0.1), (1.0, 0.1), (0.9, 0.0), (0.9, 1.0)])
    def test_probabilities_are_validated(self, confidence, tolerance):
        with pytest.raises(ValueError):
            tolerance_rank(1000, confidence, tolerance)

    def test_rank_grows_with_samples(self):
        assert tolerance_rank(1000, 0.95, 0.01) > tolerance_rank(500, 0.95, 0.01) >= 1

    @pytest.mark.parametrize(
        "n_samples, confidence, tolerance", [(299, 0.95, 0.01), (1000, 0.95, 0.01), (200, 0.9, 0.05)]
    )
    def test_empirical_coverage_meets_confidence(self, n_samples, confidence, tolerance):
        """The r-th largest of M uniform draws leaves at most `tolerance` beyond it, `confidence` of the time."""
        rng = np.random.default_rng(1)
        rank = tolerance_rank(n_samples, confidence, tolerance)
        draws = np.sort(rng.random((4000, n_samples)), axis=1)[:, n_samples - rank]
        coverage = np.mean(1.0 - draws <= tolerance)
        # Binomial standard error of 4000 trials is under 0.01.
        assert coverage >= confidence - 0.015

    def test_tolerance_limit_picks_the_rth_largest(self):
        z_det = np.random.default_rng(2).random(1000)
        z_limit, rank, n_censored = tolerance_limit(z_det, 0.95, 0.01)
        assert z_limit == np.sort(z_det)[-rank]
        assert n_censored == 0

    def test_censored_draws_are_counted_and_propagate_when_they_set_the_limit(self):
        z_det = np.random.default_rng(2).random(400)
        z_det[:5] = np.inf
        z_limit, rank, n_censored = tolerance_limit(z_det, 0.95, 0.01)
        assert rank == 1
        assert np.isinf(z_limit)
        assert n_censored == 5


def test_binomial_logit_is_finite_at_the_extremes_and_matches_the_delta_method():
    y, variance = binomial_logit(np.array([0, 50, 1000]), np.array([1000, 100, 1000]))
    assert np.all(np.isfinite(y)) and np.all(np.isfinite(variance))
    p_hat = 50.5 / 101.0
    assert y[1] == pytest.approx(np.log(p_hat / (1 - p_hat)))
    assert variance[1] == pytest.approx(1.0 / (101.0 * p_hat * (1 - p_hat)))
    # At small p the logit noise is the Poisson relative error of the count, about 1/k.
    assert variance[0] == pytest.approx(1.0 / (1001.0 * (0.5 / 1001.0) * (1 - 0.5 / 1001.0)))


class TestClopperPearson:
    """The exact binomial interval and its boundary cases."""

    def test_the_interval_contains_the_proportion_and_is_ordered(self):
        lo, hi = clopper_pearson_interval(30, 100, 0.95)
        assert 0 < lo < 0.30 < hi < 1

    def test_no_successes_has_a_zero_lower_limit(self):
        lo, hi = clopper_pearson_interval(0, 50, 0.9)
        assert lo == 0.0 and 0 < hi < 1
        assert hi == pytest.approx(1 - 0.05 ** (1 / 50))

    def test_all_successes_has_a_unit_upper_limit(self):
        lo, hi = clopper_pearson_interval(50, 50, 0.9)
        assert hi == 1.0 and 0 < lo < 1

    def test_no_trials_is_unconstrained(self):
        assert clopper_pearson_interval(0, 0, 0.95) == (0.0, 1.0)

    def test_a_higher_confidence_is_wider(self):
        narrow, wide = clopper_pearson_interval(10, 40, 0.68), clopper_pearson_interval(10, 40, 0.99)
        assert wide[0] < narrow[0] and narrow[1] < wide[1]


class TestFractionalErrors:
    """Conversion, propagation and bounds of fractional errors."""

    def test_an_asymmetric_interval_gives_fractional_errors(self):
        assert convert_CI_to_fractional(2.0, 2.5, 1.5) == (0.25, 0.25)
        lower, upper = convert_CI_to_fractional(10.0, 13.0, 9.0)
        assert lower == pytest.approx(0.1) and upper == pytest.approx(0.3)

    def test_independent_factors_add_in_quadrature(self):
        lower, upper = combine_fractional_errors((0.3, 0.4), (0.4, 0.3))
        assert lower == pytest.approx(0.5) and upper == pytest.approx(0.5)
        assert combine_fractional_errors((0.1, 0.2)) == pytest.approx((0.1, 0.2))

    def test_bounds_are_multiplicative_and_floored_at_zero(self):
        assert fractional_error_to_bounds((0.2, 0.5)) == pytest.approx((0.8, 1.5))
        assert fractional_error_to_bounds((1.3, 0.5)) == (0.0, 1.5)
