"""Tests for the rest-frame band luminosity of an SED, and its peak and decline time."""

import numpy as np
import pytest
import synphot
from astropy import units as u

from uvex_transients.models.core.lightcurve_metrics import decline_time, peak_luminosity
from uvex_transients.models.kilonovae import KilonovaCoolingBlackbodySED
from uvex_transients.phase_space.transient import _CanonicalTransient
from uvex_transients.utils import phot_utils
from uvex_transients.utils.phot_utils import effective_frequency, equivalent_bandwidth

L_NU = 1e29 * u.erg / u.s / u.Hz
HALF_TIME = 12 * u.day


@pytest.fixture(scope="module")
def smooth():
    return synphot.SpectralElement(synphot.Gaussian1D, mean=1500 * u.AA, stddev=100 * u.AA)


@pytest.fixture(scope="module")
def t_grid():
    return np.geomspace(0.01, 400, 500) * u.day


@pytest.mark.parametrize("spectral_index", [0.0, -1.0, -2.0, 0.5])
def test_a_measurement_of_the_canonical_sed_recovers_the_point_it_was_set_to(smooth, t_grid, spectral_index):
    nu_0 = effective_frequency(smooth)
    transient = _CanonicalTransient(spectral_index=spectral_index, reference_frequency=nu_0, rise_ratio=0.1)
    transient.set_point(L_NU, HALF_TIME, 1.0)
    draws = transient.sed.sample_parameters(size=3, rng=np.random.default_rng(0))

    luminosity = transient.sed.eval_band_luminosity(smooth, t_grid, **draws)
    assert luminosity.shape == (3, t_grid.size)
    peak, _ = peak_luminosity(t_grid, luminosity)
    expected = (L_NU * equivalent_bandwidth(smooth, spectral_index, nu_0)).to_value(u.erg / u.s)
    np.testing.assert_allclose(peak.to_value(u.erg / u.s), expected, rtol=3e-3)
    np.testing.assert_allclose(decline_time(t_grid, luminosity).to_value(u.day), HALF_TIME.value, rtol=1e-2)


def test_chunking_does_not_change_the_result(smooth, monkeypatch):
    t = np.geomspace(0.1, 100, 40) * u.day
    sed = KilonovaCoolingBlackbodySED()
    draws = sed.sample_parameters(size=7, rng=np.random.default_rng(1))
    whole = sed.eval_band_luminosity(smooth, t, **draws)
    monkeypatch.setattr(phot_utils, "_CHUNK_BYTES", 40_000)
    chunked = sed.eval_band_luminosity(smooth, t, **draws)
    np.testing.assert_allclose(chunked.value, whole.value)


def test_a_shipped_sed_gives_finite_positive_peaks_and_a_decline_time(smooth, t_grid):
    sed = KilonovaCoolingBlackbodySED()
    draws = sed.sample_parameters(size=40, rng=np.random.default_rng(2))
    luminosity = sed.eval_band_luminosity(smooth, t_grid, **draws)
    peak, t_peak = peak_luminosity(t_grid, luminosity)
    half = decline_time(t_grid, luminosity)
    assert np.all(peak > 0 * u.erg / u.s) and np.all(np.isfinite(peak))
    assert np.all(t_peak > 0 * u.day)
    assert np.isfinite(half).mean() > 0.8
    # The luminosity at the half-decline time is half the peak.
    check = np.isfinite(half)
    at_half = np.array(
        [
            np.interp(tp + h, t_grid.value, row)
            for tp, h, row in zip(t_peak.value[check], half.value[check], luminosity.value[check])
        ]
    )
    np.testing.assert_allclose(at_half, 0.5 * peak.value[check], rtol=0.03)


def test_the_sed_methods_are_the_reductions_of_the_curve(smooth, t_grid):
    sed = KilonovaCoolingBlackbodySED()
    draws = sed.sample_parameters(size=25, rng=np.random.default_rng(3))
    curve = sed.eval_band_luminosity(smooth, t_grid, **draws)

    peak, t_peak = sed.peak_luminosity(smooth, t_grid, **draws)
    expected_peak, expected_t = peak_luminosity(t_grid, curve)
    np.testing.assert_array_equal(peak.value, expected_peak.value)
    np.testing.assert_array_equal(t_peak.value, expected_t.value)

    np.testing.assert_array_equal(
        sed.decline_time(smooth, t_grid, 0.5, **draws).value, decline_time(t_grid, curve, 0.5).value
    )
    # The default fraction is a half, and a deeper decline takes longer.
    np.testing.assert_array_equal(sed.decline_time(smooth, t_grid, **draws).value, decline_time(t_grid, curve).value)
    deeper = sed.decline_time(smooth, t_grid, 0.1, **draws).value
    shallow = sed.decline_time(smooth, t_grid, 0.5, **draws).value
    ok = np.isfinite(deeper) & np.isfinite(shallow)
    assert ok.any() and np.all(deeper[ok] > shallow[ok])


def test_scalar_parameters_are_one_draw(smooth, t_grid):
    sed = KilonovaCoolingBlackbodySED()
    draws = sed.sample_parameters(size=4, rng=np.random.default_rng(4))
    one = {name: value[0] for name, value in draws.items()}
    curve = sed.eval_band_luminosity(smooth, t_grid, **one)
    assert curve.shape == (1, t_grid.size)
    np.testing.assert_allclose(curve.value[0], sed.eval_band_luminosity(smooth, t_grid, **draws).value[0])
