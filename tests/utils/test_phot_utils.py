"""Tests for the photometry utilities: AB magnitudes, bandpass arrays, and integrating a spectrum over bands."""

import numpy as np
import pytest
import synphot
from astropy import units as u
from scipy.integrate import quad
from synphot import SpectralElement
from synphot.models import Box1D, Gaussian1D

from uvex_transients.utils import phot_utils
from uvex_transients.utils.phot_utils import (
    AB_MAG_ZERO_POINT,
    band_integration_weights,
    band_luminosity_curve,
    band_weight_matrix,
    effective_frequency,
    equivalent_bandwidth,
    flux_to_ab_mag,
    hz_per_unit,
    photon_weighted_band_average,
    resolve_bandpasses,
    restframe_frequency_grid,
)

Z = np.geomspace(0.01, 1.0, 6)


@pytest.fixture(scope="module")
def box():
    return synphot.SpectralElement(synphot.Box1D, x_0=1500 * u.AA, width=300 * u.AA)


@pytest.fixture(scope="module")
def smooth():
    return synphot.SpectralElement(synphot.Gaussian1D, mean=1500 * u.AA, stddev=100 * u.AA)


@pytest.fixture(scope="module")
def grids():
    """Two smooth bands, so that the comparison below tests the quadrature and not a top-hat's edges."""
    bands = {
        "FUV": SpectralElement(Gaussian1D, amplitude=1.0, mean=1550 * u.AA, stddev=120 * u.AA),
        "NUV": SpectralElement(Gaussian1D, amplitude=1.0, mean=2300 * u.AA, stddev=250 * u.AA),
    }
    return resolve_bandpasses(bands)[1]


# ---------------------------------------------------------------------------- #
# Magnitudes and units                                                         #
# ---------------------------------------------------------------------------- #
def test_the_zero_point_flux_is_magnitude_zero_and_a_hundredfold_fainter_is_five_more():
    np.testing.assert_allclose(flux_to_ab_mag(AB_MAG_ZERO_POINT), 0.0, atol=1e-12)
    np.testing.assert_allclose(flux_to_ab_mag(np.array([1.0, 0.01]) * AB_MAG_ZERO_POINT), [0.0, 5.0], atol=1e-12)


def test_zero_flux_is_an_infinitely_faint_magnitude_without_a_warning():
    with np.errstate(all="raise"):
        assert flux_to_ab_mag(0.0) == np.inf


def test_the_zero_point_is_the_value_the_models_use():
    from uvex_transients.models._constants import AB_MAG_ZERO_POINT as models_value

    assert models_value is phot_utils.AB_MAG_ZERO_POINT


def test_a_frequency_unit_converts_by_a_factor_and_a_wavelength_unit_by_a_reciprocal():
    assert hz_per_unit(u.THz, is_wavelength=False) == pytest.approx(1e12)
    assert hz_per_unit(u.AA, is_wavelength=True) == pytest.approx(2.99792458e18)


@pytest.mark.parametrize("unit, is_wavelength", [(u.AA, False), (u.Hz, True), (u.kg, False)])
def test_a_unit_of_the_wrong_kind_is_rejected(unit, is_wavelength):
    with pytest.raises(ValueError, match="not a"):
        hz_per_unit(unit, is_wavelength=is_wavelength)


# ---------------------------------------------------------------------------- #
# Bandpasses as arrays                                                         #
# ---------------------------------------------------------------------------- #
def test_sequence_bands_are_named_by_position():
    band = SpectralElement(Box1D, amplitude=1.0, x_0=2300 * u.AA, width=500 * u.AA)
    names, grids = resolve_bandpasses([band, band])
    assert names == ("band0", "band1")
    assert len(grids) == 2
    nu, throughput = grids[0]
    assert nu.shape == throughput.shape


def test_empty_bands_are_rejected():
    with pytest.raises(ValueError, match="at least one"):
        resolve_bandpasses({})


def test_a_mapping_of_bands_keeps_its_names(box, smooth):
    names, grids = resolve_bandpasses({"FUV": box, "NUV": smooth})
    assert names == ("FUV", "NUV") and len(grids) == 2


def test_the_relative_table_is_increasing_in_frequency_with_a_unit_peak(smooth):
    nu, throughput = phot_utils.relative_bandpass_table(smooth)
    assert np.all(np.diff(nu) > 0) and throughput.max() == pytest.approx(1.0)


def test_the_equivalent_bandwidth_of_a_top_hat_is_its_width_in_frequency(box):
    expected = (1350 * u.AA).to(u.Hz, equivalencies=u.spectral()) - (1650 * u.AA).to(u.Hz, equivalencies=u.spectral())
    assert equivalent_bandwidth(box).to_value(u.Hz) == pytest.approx(expected.to_value(u.Hz), rel=0.03)


def test_the_equivalent_bandwidth_of_a_top_hat_follows_the_power_law(box):
    nu_low, nu_high = (1650 * u.AA).to_value(u.Hz, u.spectral()), (1350 * u.AA).to_value(u.Hz, u.spectral())
    nu_0 = effective_frequency(box)
    assert nu_low < nu_0.to_value(u.Hz) < nu_high
    assert nu_0.to_value(u.Hz) == pytest.approx(0.5 * (nu_low + nu_high), rel=0.01)
    for alpha in (-2.0, -1.0, 0.0, 1.0):
        exact = quad(lambda nu: (nu / nu_0.to_value(u.Hz)) ** alpha, nu_low, nu_high)[0]
        assert equivalent_bandwidth(box, alpha, nu_0).to_value(u.Hz) == pytest.approx(exact, rel=0.03)
    # The default reference frequency is the effective one, and a flat spectrum does not depend on it.
    assert equivalent_bandwidth(box, -1.0) == equivalent_bandwidth(box, -1.0, nu_0)
    assert equivalent_bandwidth(box, 0.0, 1e15 * u.Hz) == equivalent_bandwidth(box)


# ---------------------------------------------------------------------------- #
# Rest-frame grid and band weights                                             #
# ---------------------------------------------------------------------------- #
def test_frequency_grid_is_uniform_in_log_and_covers_every_band_at_every_redshift(grids):
    nu_rest, step = restframe_frequency_grid(grids, Z, 0.002)
    assert step <= 0.002
    np.testing.assert_allclose(np.diff(np.log(nu_rest)), step)
    for nu, throughput in grids:
        support = nu[throughput > 0]
        assert nu_rest[0] < support.min() * (1 + Z[0])
        assert nu_rest[-1] > support.max() * (1 + Z[-1])


def test_weights_sum_to_one_and_follow_the_band_to_the_rest_frame(grids):
    nu_rest, _ = restframe_frequency_grid(grids, Z)
    weights = band_weight_matrix(nu_rest, grids, Z)
    assert weights.shape == (nu_rest.size, 2, Z.size)
    np.testing.assert_allclose(weights.sum(axis=0), 1.0)
    # The band's centre moves redward in the observed frame, so blueward in the rest frame: the weight-averaged
    # rest-frame frequency is the observed one times (1 + z).
    centre = (weights[:, 0, :] * nu_rest[:, np.newaxis]).sum(axis=0)
    np.testing.assert_allclose(centre / centre[0], (1 + Z) / (1 + Z[0]), rtol=2e-3)


def test_the_frequency_grid_of_one_band_is_uniform_in_ln_nu_and_covers_it(box):
    nu, weights = band_integration_weights(box)
    np.testing.assert_allclose(np.diff(np.log(nu)), np.diff(np.log(nu))[0])
    low, high = (1650 * u.AA).to_value(u.Hz, u.spectral()), (1350 * u.AA).to_value(u.Hz, u.spectral())
    assert nu[0] < low and nu[-1] > high
    assert np.all(weights >= 0)


def test_the_weights_of_a_flat_spectrum_add_up_to_the_width_of_the_band(box):
    _, weights = band_integration_weights(box)
    width = (1350 * u.AA).to_value(u.Hz, u.spectral()) - (1650 * u.AA).to_value(u.Hz, u.spectral())
    assert weights.sum() == pytest.approx(width, rel=0.01)


@pytest.mark.parametrize("spectral_index", [-2.0, -1.0, 0.0, 1.0])
def test_the_weights_integrate_a_power_law_to_the_equivalent_bandwidth(smooth, spectral_index):
    nu, weights = band_integration_weights(smooth)
    nu_0 = effective_frequency(smooth).to_value(u.Hz)
    total = float(np.sum(weights * (nu / nu_0) ** spectral_index))
    expected = equivalent_bandwidth(smooth, spectral_index, nu_0 * u.Hz).to_value(u.Hz)
    assert total == pytest.approx(expected, rel=2e-3)


# ---------------------------------------------------------------------------- #
# The photon-weighted band average, two ways                                   #
# ---------------------------------------------------------------------------- #
def test_a_flat_flux_density_averages_to_itself(smooth):
    nu, throughput = phot_utils.relative_bandpass_table(smooth)
    flux = np.full((3, nu.size), 2.5e-27)
    np.testing.assert_allclose(photon_weighted_band_average(flux, nu, throughput), 2.5e-27, rtol=1e-12)


def test_the_average_weights_by_photons_so_it_follows_the_blue_edge_of_a_rising_spectrum(smooth):
    nu, throughput = phot_utils.relative_bandpass_table(smooth)
    flux = (nu / nu.mean()) ** 3
    photon = photon_weighted_band_average(flux, nu, throughput)
    energy = np.trapezoid(flux * throughput, nu) / np.trapezoid(throughput, nu)
    # Weighting by throughput over nu discounts the blue, where each photon carries more energy.
    assert photon < energy


@pytest.mark.parametrize("z", [0.3, 1.5])
@pytest.mark.parametrize("index", [-2.0, 0.0, 2.0])
def test_the_observed_frame_average_equals_the_rest_frame_weighted_sum(smooth, z, index):
    """`photon_weighted_band_average` and `band_weight_matrix` are the same average, done on different grids."""
    nu, throughput = phot_utils.relative_bandpass_table(smooth)
    direct = photon_weighted_band_average(nu**index, nu, throughput)

    grids = [(nu, throughput)]
    nu_rest, _ = restframe_frequency_grid(grids, np.array([z]))
    weights = band_weight_matrix(nu_rest, grids, np.array([z]))[:, 0, 0]
    # A source whose flux density is nu^index in the observed frame has L_nu proportional to (nu'/(1+z))^index.
    rest_frame = float(np.sum(weights * (nu_rest / (1.0 + z)) ** index))
    assert rest_frame == pytest.approx(direct, rel=3e-3)


# ---------------------------------------------------------------------------- #
# Band luminosity of an arbitrary spectrum                                     #
# ---------------------------------------------------------------------------- #
def _power_law_decay(index, nu_0):
    """ln L_nu of A (nu/nu_0)^index exp(-t / tau), in the cgs and with ``amplitude`` and ``tau`` as parameters."""

    def log_luminosity(nu, t, amplitude, tau):
        return np.log(amplitude) + index * np.log(nu / nu_0) - t / tau

    return log_luminosity


@pytest.mark.parametrize("index", [-2.0, 0.0, 1.0])
def test_the_band_luminosity_of_a_power_law_is_its_amplitude_times_the_equivalent_bandwidth(smooth, index):
    nu_0 = effective_frequency(smooth)
    t = np.array([0.0, 86400.0, 3 * 86400.0])
    parameters = {"amplitude": np.array([1e29, 5e29]), "tau": np.array([86400.0, 2 * 86400.0])}
    curve = band_luminosity_curve(_power_law_decay(index, nu_0.to_value(u.Hz)), parameters, t, smooth)
    assert curve.shape == (2, 3)
    bandwidth = equivalent_bandwidth(smooth, index, nu_0).to_value(u.Hz)
    expected = parameters["amplitude"][:, None] * np.exp(-t[None, :] / parameters["tau"][:, None]) * bandwidth
    np.testing.assert_allclose(curve, expected, rtol=3e-3)


def test_chunking_does_not_change_the_band_luminosity(smooth, monkeypatch):
    parameters = {"amplitude": np.geomspace(1e28, 1e30, 9), "tau": np.full(9, 86400.0)}
    t = np.geomspace(0.1, 10.0, 11) * 86400.0
    log_l = _power_law_decay(-1.0, 2e15)
    whole = band_luminosity_curve(log_l, parameters, t, smooth)
    monkeypatch.setattr(phot_utils, "_CHUNK_BYTES", 20_000)
    np.testing.assert_allclose(band_luminosity_curve(log_l, parameters, t, smooth), whole, rtol=1e-12)


def test_a_missing_value_in_the_band_makes_the_epoch_nan_and_others_are_kept(smooth):
    def log_l(nu, t, **_):
        return np.where(t > 1.5 * 86400.0, np.nan, np.log(1e29)) + 0 * nu

    curve = band_luminosity_curve(log_l, {"unused": np.zeros(2)}, np.array([86400.0, 2 * 86400.0]), smooth)
    assert np.isfinite(curve[:, 0]).all() and np.isnan(curve[:, 1]).all()


def test_scalar_parameters_and_times_are_one_draw_at_one_time(smooth):
    curve = band_luminosity_curve(_power_law_decay(0.0, 2e15), {"amplitude": 1e29, "tau": 86400.0}, 86400.0, smooth)
    assert curve.shape == (1, 1)
