"""Tests for the flat-spectrum top-hat transient."""

import numpy as np
import pytest
from astropy import units as u

from uvex_transients.cli.config import _KNOWN_TRANSIENT_MODULES
from uvex_transients.phase_space.transient import BANDWIDTH, _TopHatTransient
from uvex_transients.transients.base import ExtragalacticTransient, TransientBase

L_NU = 1e29 * u.erg / u.s / u.Hz
DURATION = 20 * u.day


@pytest.fixture
def transient():
    transient = _TopHatTransient()
    transient.set_point(L_NU, DURATION, 0.5)
    return transient


def _parameters(transient):
    return {name: value[0] for name, value in transient.sed.sample_parameters(size=1).items()}


def test_set_point_moves_the_point_and_the_windows(transient):
    assert transient.redshift_limit == 0.5
    assert transient.duration_limit.to_value(u.day) == pytest.approx(30.0)
    parameters = _parameters(transient)
    assert parameters["amplitude"].to_value(u.erg / u.s) == pytest.approx((L_NU * BANDWIDTH).to_value(u.erg / u.s))
    assert parameters["duration"] == DURATION

    transient.set_point(2 * L_NU, 5 * u.day, 2.0)
    assert transient.redshift_limit == 2.0
    assert transient.duration_limit.to_value(u.day) == pytest.approx(15.0)


def test_the_magnitude_is_the_flat_spectrum_formula_in_any_frequency(transient):
    redshift = np.array([0.02, 0.1, 0.3])
    d_l = transient.cosmology.luminosity_distance(redshift).to_value(u.cm)
    expected = -2.5 * np.log10((1 + redshift) * L_NU.value / (4 * np.pi * d_l**2)) - 48.6
    parameters = _parameters(transient)
    for nu in (1.5e15 * u.Hz, 1.0e15 * u.Hz, 3.0e15 * u.Hz):
        magnitude = transient.sed.mag(nu, 3 * u.day, redshift=redshift, **parameters)
        np.testing.assert_allclose(magnitude.value, expected, atol=1e-3)


@pytest.mark.parametrize("redshift", [0.1, 0.3])
def test_the_plateau_ends_at_the_time_dilated_duration(transient, redshift):
    parameters = _parameters(transient)
    nu = 1.5e15 * u.Hz
    inside = transient.sed.mag(nu, (1 + redshift) * DURATION * 0.98, redshift=redshift, **parameters)
    outside = transient.sed.mag(nu, (1 + redshift) * DURATION * 1.02, redshift=redshift, **parameters)
    assert np.isfinite(inside.value)
    assert np.isinf(outside.value)


def test_the_rate_is_one_per_cubic_megaparsec_per_year(transient):
    assert transient.rate == 1.0 * u.Mpc**-3 * u.yr**-1
    np.testing.assert_array_equal(transient.rate_shape(np.array([0.1, 1.0, 5.0])), np.ones(3))
    assert transient.rate_shape(0.5) == 1.0


def test_the_class_stays_out_of_the_shipped_classes():
    assert issubclass(_TopHatTransient, ExtragalacticTransient)
    assert "_TopHatTransient" in TransientBase.registry()
    # An underscore name is skipped by the check that every registered class is in full_run.yaml,
    # and its module is not one the CLI imports.
    assert _TopHatTransient.__name__.startswith("_")
    assert "phase_space" not in " ".join(_KNOWN_TRANSIENT_MODULES)
    assert not _TopHatTransient.HAS_STORED_HORIZON_CURVE
