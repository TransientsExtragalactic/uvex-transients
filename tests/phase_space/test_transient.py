"""Tests for the canonical (FRED light curve, power-law spectrum) transient."""

import numpy as np
import pytest
from astropy import units as u

from uvex_transients.cli.config import _KNOWN_TRANSIENT_MODULES
from uvex_transients.phase_space.transient import _CanonicalTransient, fred_time_scales
from uvex_transients.transients.base import ExtragalacticTransient, TransientBase

L_NU = 1e29 * u.erg / u.s / u.Hz
HALF_TIME = 20 * u.day
RISE_RATIO = 0.1


@pytest.fixture(params=[0.0, -1.0, -2.5, 0.5])
def transient(request):
    transient = _CanonicalTransient(spectral_index=request.param, rise_ratio=RISE_RATIO)
    transient.set_point(L_NU, HALF_TIME, 0.5)
    return transient


def _parameters(transient):
    return {name: value[0] for name, value in transient.sed.sample_parameters(size=1).items()}


def _t_peak(parameters):
    return np.sqrt(parameters["rise"] * parameters["decay"]).to(u.day)


def _l_nu(transient, t_rest, nu=None, parameters=None):
    parameters = _parameters(transient) if parameters is None else parameters
    nu = transient.reference_frequency if nu is None else nu
    return transient.sed.eval(nu, t_rest, **parameters).to_value(u.erg / u.s / u.Hz)


def test_set_point_moves_the_point_and_the_windows(transient):
    parameters = _parameters(transient)
    assert transient.redshift_limit == 0.5
    expected_window = 1.5 * transient.window_factor * HALF_TIME.to_value(u.day)
    assert transient.duration_limit.to_value(u.day) == pytest.approx(expected_window)
    assert (parameters["rise"] / parameters["decay"]).to_value(u.dimensionless_unscaled) == pytest.approx(RISE_RATIO)

    transient.set_point(2 * L_NU, 5 * u.day, 2.0)
    assert transient.redshift_limit == 2.0
    assert transient.duration_limit.to_value(u.day) == pytest.approx(3.0 * transient.window_factor * 5.0)


def test_the_peak_spectral_luminosity_at_the_reference_frequency_is_the_one_set(transient):
    parameters = _parameters(transient)
    assert _l_nu(transient, _t_peak(parameters), parameters=parameters) == pytest.approx(L_NU.value, rel=1e-6)


def test_the_luminosity_halves_one_half_time_after_the_peak(transient):
    parameters = _parameters(transient)
    t_peak = _t_peak(parameters)
    peak = _l_nu(transient, t_peak, parameters=parameters)
    half = _l_nu(transient, t_peak + HALF_TIME, parameters=parameters)
    assert half / peak == pytest.approx(0.5, rel=1e-6)
    # Nothing is brighter than the peak.
    grid = np.geomspace(0.05, 500, 400) * u.day
    assert np.max(_l_nu(transient, grid, parameters=parameters)) <= peak * (1 + 1e-9)


def test_the_window_ends_where_the_event_has_faded_by_the_fade_magnitude(transient):
    parameters = _parameters(transient)
    end = transient.window_factor * HALF_TIME
    peak = _l_nu(transient, _t_peak(parameters), parameters=parameters)
    faded = _l_nu(transient, end, parameters=parameters)
    assert -2.5 * np.log10(faded / peak) == pytest.approx(transient.fade_mag, abs=1e-6)


def test_the_magnitude_is_the_power_law_formula_with_its_k_correction(transient):
    redshift = np.array([0.02, 0.1, 0.3])
    alpha = transient.spectral_index
    d_l = transient.cosmology.luminosity_distance(redshift).to_value(u.cm)
    parameters = _parameters(transient)
    t_rest = _t_peak(parameters)
    for nu in (1.5e15 * u.Hz, 1.0e15 * u.Hz, 3.0e15 * u.Hz):
        # The observed band samples the rest-frame spectrum at (1+z) nu, and the bandwidth is compressed by 1+z.
        x = ((1 + redshift) * nu / transient.reference_frequency).to_value(u.dimensionless_unscaled)
        expected = -2.5 * np.log10((1 + redshift) * L_NU.value * x**alpha / (4 * np.pi * d_l**2)) - 48.6
        magnitude = transient.sed.mag(nu, (1 + redshift) * t_rest, redshift=redshift, **parameters)
        np.testing.assert_allclose(magnitude.value, expected, atol=1e-3)


def test_the_reference_frequency_only_fixes_what_a_luminosity_means():
    redshift = np.array([0.1])
    magnitudes = []
    for nu_0 in (5e14 * u.Hz, 2e15 * u.Hz):
        transient = _CanonicalTransient(spectral_index=-1.5, reference_frequency=nu_0, rise_ratio=RISE_RATIO)
        transient.set_point(L_NU * (nu_0 / (1e15 * u.Hz)) ** -1.5, HALF_TIME, 0.5)
        parameters = _parameters(transient)
        t_obs = 1.1 * _t_peak(parameters)
        magnitudes.append(transient.sed.mag(1.5e15 * u.Hz, t_obs, redshift=redshift, **parameters).value)
    np.testing.assert_allclose(magnitudes[0], magnitudes[1], atol=1e-6)


@pytest.mark.parametrize("rise_ratio", [0.01, 0.1, 0.3])
def test_the_time_scales_of_a_fred_curve_in_units_of_the_decay_time(rise_ratio):
    half_time, fade_time = fred_time_scales(rise_ratio, 6.0)
    peak = np.sqrt(rise_ratio)

    def curve(t):
        return np.exp(2 * peak - rise_ratio / t - t)

    assert curve(peak + half_time) == pytest.approx(0.5)
    assert curve(fade_time) == pytest.approx(10 ** (-0.4 * 6.0))
    assert fade_time > peak + half_time


def test_the_half_time_grows_with_the_rise_ratio_but_slowly():
    # A slower rise puts the peak later, which stretches the decline a little; this is why the ratio hardly matters.
    ratios = [fred_time_scales(r, 6.0)[0] for r in (0.01, 0.03, 0.1, 0.3)]
    assert np.all(np.diff(ratios) > 0)
    assert ratios[-1] / ratios[0] < 1.5


@pytest.mark.parametrize("kwargs", [{"rise_ratio": 0.0}, {"fade_mag": -1.0}])
def test_invalid_shape_settings_raise(kwargs):
    with pytest.raises(ValueError):
        _CanonicalTransient(**kwargs)


def test_the_rate_is_one_per_cubic_megaparsec_per_year(transient):
    assert transient.rate == 1.0 * u.Mpc**-3 * u.yr**-1
    np.testing.assert_array_equal(transient.rate_shape(np.array([0.1, 1.0, 5.0])), np.ones(3))
    assert transient.rate_shape(0.5) == 1.0


def test_the_class_stays_out_of_the_shipped_classes():
    assert issubclass(_CanonicalTransient, ExtragalacticTransient)
    assert "_CanonicalTransient" in TransientBase.registry()
    # An underscore name is skipped by the check that every registered class is in full_run.yaml,
    # and its module is not one the CLI imports.
    assert _CanonicalTransient.__name__.startswith("_")
    assert "phase_space" not in " ".join(_KNOWN_TRANSIENT_MODULES)
    assert not _CanonicalTransient.HAS_STORED_HORIZON_CURVE
