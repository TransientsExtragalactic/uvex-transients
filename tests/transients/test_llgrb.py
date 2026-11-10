"""Tests for the low-luminosity GRB population."""

import numpy as np
import pytest
from astropy import units as u

from uvex_transients.models.supernovae import LLGRBSED
from uvex_transients.transients.base import TransientBase
from uvex_transients.transients.horizon import HorizonCurve, curve_path
from uvex_transients.transients.llgrb import LowLuminosityGRB


def test_llgrb_is_registered_and_paired_with_its_sed():
    """The class registers by name and builds its own `LLGRBSED`."""
    assert TransientBase.registry()["LowLuminosityGRB"] is LowLuminosityGRB
    assert isinstance(LowLuminosityGRB().sed, LLGRBSED)


def test_llgrb_rate_is_soderberg_2006():
    """The rate is 230 Gpc^-3 yr^-1 with the paper's 90% range, +490/-190, as `rate_ci`."""
    t = LowLuminosityGRB()
    assert t.rate.to_value(u.Gpc**-3 * u.yr**-1) == pytest.approx(230.0)
    lower, upper = (bound.to_value(u.Gpc**-3 * u.yr**-1) for bound in t.rate_ci)
    assert lower == pytest.approx(40.0)
    assert upper == pytest.approx(720.0)


def test_llgrb_rate_is_constant_in_redshift():
    """No evolution is assumed, so the rate shape is one at every redshift, and a scalar for a scalar."""
    t = LowLuminosityGRB()
    assert t.rate_shape(0.5) == 1.0
    np.testing.assert_array_equal(t.rate_shape(np.array([0.0, 1.0, 2.0])), np.ones(3))
    z = np.array([0.1, 1.0])
    np.testing.assert_allclose(t.event_rate(z), t.rate.to_value(u.Mpc**-3 * u.yr**-1))


def test_llgrb_redshift_limit_covers_its_stored_horizon():
    """`DEFAULT_Z_LIM` is the stored curve's limit at 24.5 AB, rounded up, never below it."""
    t = LowLuminosityGRB()
    z_limit = HorizonCurve.read(curve_path("LowLuminosityGRB")).z_limit(24.5)
    assert z_limit <= t.redshift_limit <= z_limit + 0.01
