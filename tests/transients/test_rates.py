"""Tests for `ExtragalacticTransient`'s rate/rate_ci/effective_volume/yield machinery."""

import numpy as np
import pytest
from astropy import units as u
from astropy.units import Quantity

from uvex_transients.models.tdes import VanVelzenTDESED
from uvex_transients.transients.base import ExtragalacticTransient


class _ConstantRateTransient(ExtragalacticTransient):
    """A minimal concrete `ExtragalacticTransient` with a flat, uncertain rate, for testing."""

    DEFAULT_MODEL = VanVelzenTDESED
    DEFAULT_DURATION = 1 * u.day
    DEFAULT_Z_LIM = 1

    RATE_CI = (0.5, 2.0)

    @property
    def rate(self) -> Quantity:
        return 1e-6 / (u.Mpc**3 * u.yr)

    def rate_shape(self, z):
        z = np.asarray(z)
        shape = np.ones_like(z, dtype=np.float64)
        return shape if z.ndim > 0 else shape.item()


class _UnspecifiedCITransient(_ConstantRateTransient):
    """Same as `_ConstantRateTransient`, but with `RATE_CI` left at its unset default."""

    RATE_CI = None


def test_rate_ci_scales_by_the_declared_factors():
    """`rate_ci` multiplies `rate` by the `RATE_CI` factors."""
    t = _ConstantRateTransient()
    lower, upper = t.rate_ci
    unit = t.rate.unit
    assert lower.to_value(unit) == pytest.approx(0.5 * t.rate.to_value(unit))
    assert upper.to_value(unit) == pytest.approx(2.0 * t.rate.to_value(unit))


def test_rate_ci_degenerates_when_unset():
    """With `RATE_CI` unset, `rate_ci` collapses to `(rate, rate)`."""
    t = _UnspecifiedCITransient()
    lower, upper = t.rate_ci
    unit = t.rate.unit
    assert lower.to_value(unit) == pytest.approx(t.rate.to_value(unit))
    assert upper.to_value(unit) == pytest.approx(t.rate.to_value(unit))


def test_all_sky_rate_is_integrated_rate_over_full_sky():
    """`all_sky_rate` is `integrated_rate` restored to the full `4 pi` steradians."""
    t = _ConstantRateTransient()
    expected = (t.integrated_rate * 4 * np.pi * u.sr).to(u.yr**-1)
    assert t.all_sky_rate.to_value(u.yr**-1) == pytest.approx(expected.to_value(u.yr**-1))


def test_all_sky_rate_equals_rate_times_effective_volume():
    """`all_sky_rate` is `rate * effective_volume` (the point value, i.e. `A * V`)."""
    t = _ConstantRateTransient()
    expected = (t.rate * t.effective_volume).to(u.yr**-1)
    assert t.all_sky_rate.to_value(u.yr**-1) == pytest.approx(expected.to_value(u.yr**-1))


def test_effective_volume_has_no_rate_dependence():
    """`effective_volume` is unchanged when only `rate` (not `rate_shape`) differs between instances."""

    class _DoubleRateTransient(_ConstantRateTransient):
        @property
        def rate(self):
            return 2e-6 / (u.Mpc**3 * u.yr)

    t1 = _ConstantRateTransient()
    t2 = _DoubleRateTransient()
    assert t1.effective_volume.to_value(u.Mpc**3) == pytest.approx(t2.effective_volume.to_value(u.Mpc**3))


def test_all_sky_rate_ci_uses_rate_ci_endpoints():
    """`all_sky_rate_ci` is `[R_L, R_U] * effective_volume`."""
    t = _ConstantRateTransient()
    lower, upper = t.all_sky_rate_ci
    rate_lower, rate_upper = t.rate_ci
    assert lower.to_value(u.yr**-1) == pytest.approx((rate_lower * t.effective_volume).to_value(u.yr**-1))
    assert upper.to_value(u.yr**-1) == pytest.approx((rate_upper * t.effective_volume).to_value(u.yr**-1))


def test_compute_all_sky_yield_multiplies_by_duration():
    """`compute_all_sky_yield` is `all_sky_rate * duration`, unitless."""
    t = _ConstantRateTransient()
    duration = 10 * u.day
    expected = (t.all_sky_rate * duration).to_value(u.dimensionless_unscaled)
    assert t.compute_all_sky_yield(duration) == pytest.approx(expected)


def test_compute_all_sky_yield_ci_brackets_the_point_estimate():
    """`compute_all_sky_yield_ci` brackets `compute_all_sky_yield` when `RATE_CI` widens the rate."""
    t = _ConstantRateTransient()
    duration = 10 * u.day
    lower, upper = t.compute_all_sky_yield_ci(duration)
    point = t.compute_all_sky_yield(duration)
    assert lower < point < upper


def test_compute_all_sky_yield_ci_degenerates_when_rate_ci_unset():
    """With `RATE_CI` unset, the yield CI collapses to the point yield estimate."""
    t = _UnspecifiedCITransient()
    duration = 10 * u.day
    lower, upper = t.compute_all_sky_yield_ci(duration)
    point = t.compute_all_sky_yield(duration)
    assert lower == pytest.approx(point)
    assert upper == pytest.approx(point)


def test_event_rate_is_rate_times_rate_shape():
    """`event_rate(z)` is the product of `rate` and `rate_shape(z)`, in Mpc^-3 yr^-1."""
    t = _ConstantRateTransient()
    z = np.array([0.0, 0.5, 1.0])
    expected = t.rate.to_value(u.Mpc**-3 * u.yr**-1) * np.asarray(t.rate_shape(z))
    np.testing.assert_allclose(t.event_rate(z), expected)
