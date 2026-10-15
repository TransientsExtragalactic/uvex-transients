"""Tests for the rest-frame band-magnitude machinery behind `get_effective_peak_magnitudes`."""

import numpy as np
import pytest
from astropy import units as u
from synphot import SpectralElement
from synphot.models import Gaussian1D

from uvex_transients.models._constants import AB_MAG_ZERO_POINT
from uvex_transients.models._restframe import band_weight_matrix, peak_band_magnitudes, restframe_frequency_grid
from uvex_transients.models.core._tolerance import resolve_bandpasses
from uvex_transients.models.kilonovae import KilonovaCoolingBlackbodySED
from uvex_transients.models._utils import to_cgs_value
from uvex_transients.utils.cosmology import resolve_cosmological_distances

Z = np.geomspace(0.01, 1.0, 6)
T = np.geomspace(0.05, 30.0, 25) * 86400.0
D_L = resolve_cosmological_distances(redshift=Z)["luminosity_distance"].cgs.value


@pytest.fixture(scope="module")
def grids():
    """Two smooth bands, so that the comparison below tests the quadrature and not a top-hat's edges."""
    bands = {
        "FUV": SpectralElement(Gaussian1D, amplitude=1.0, mean=1550 * u.AA, stddev=120 * u.AA),
        "NUV": SpectralElement(Gaussian1D, amplitude=1.0, mean=2300 * u.AA, stddev=250 * u.AA),
    }
    return resolve_bandpasses(bands)[1]


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


@pytest.mark.parametrize("d_ln_nu", [0.05, 0.002])
def test_a_flat_spectrum_is_exact_at_any_spacing(grids, d_ln_nu):
    """The weights are normalized, so a spectrum flat in L_nu loses nothing to the quadrature."""
    l_nu = 3.0e29
    parameters = {"unused": np.zeros(4)}
    mags = peak_band_magnitudes(
        lambda nu, t, **_: np.log(l_nu), parameters, Z, D_L, grids, T, d_ln_nu=d_ln_nu, progress=False
    )
    expected = -2.5 * np.log10((1 + Z) * l_nu / (4 * np.pi * D_L**2) / AB_MAG_ZERO_POINT)
    np.testing.assert_allclose(mags, np.broadcast_to(expected, mags.shape), rtol=0, atol=1e-10)


def test_agrees_with_the_observed_frame_integral(grids):
    sed = KilonovaCoolingBlackbodySED()
    parameters = {name: to_cgs_value(value) for name, value in sed.sample_parameters(size=6, rng=1).items()}
    mags = peak_band_magnitudes(sed._eval, parameters, Z, D_L, grids, T, progress=False)
    assert mags.shape == (6, 2, Z.size)
    for b, (nu, throughput) in enumerate(grids):
        for j, z in enumerate(Z):
            direct = sed.mag_band_cgs(
                nu,
                throughput,
                T[np.newaxis, :] * (1 + z),
                z,
                D_L[j],
                **{k: v[:, np.newaxis] for k, v in parameters.items()},
            )
            np.testing.assert_allclose(mags[:, b, j], np.nanmin(direct, axis=1), atol=2e-3)


def test_result_does_not_depend_on_the_chunk_size(grids):
    sed = KilonovaCoolingBlackbodySED()
    parameters = {name: to_cgs_value(value) for name, value in sed.sample_parameters(size=9, rng=1).items()}
    reference = peak_band_magnitudes(sed._eval, parameters, Z, D_L, grids, T, progress=False)
    for chunk_size in (1, 4, 100):
        other = peak_band_magnitudes(sed._eval, parameters, Z, D_L, grids, T, chunk_size=chunk_size, progress=False)
        np.testing.assert_allclose(other, reference, atol=1e-9)


class TestMissingValues:
    """Models return NaN outside their validity; those epochs must be skipped, not poison the peak."""

    parameters = {"unused": np.zeros(3)}

    def test_missing_epochs_are_skipped(self, grids):
        def model(nu, t, **_):
            return np.where(t < 1.0 * 86400.0, np.nan, np.log(1e29) - np.log(t / 86400.0))

        mags = peak_band_magnitudes(model, self.parameters, Z, D_L, grids, T, progress=False)
        first_valid = T[T >= 86400.0][0]
        expected = -2.5 * np.log10((1 + Z) * 1e29 / (first_valid / 86400.0) / (4 * np.pi * D_L**2) / AB_MAG_ZERO_POINT)
        np.testing.assert_allclose(mags, np.broadcast_to(expected, mags.shape), atol=1e-10)

    def test_a_draw_with_no_valid_epoch_stays_nan(self, grids):
        mags = peak_band_magnitudes(lambda nu, t, **_: np.nan, self.parameters, Z, D_L, grids, T, progress=False)
        assert np.isnan(mags).all()

    def test_a_missing_frequency_only_invalidates_the_bands_that_see_it(self, grids):
        # NaN below a rest-frequency that every FUV photon is above at every redshift, but that NUV photons straddle.
        fuv = grids[0][0][grids[0][1] > 0]
        nu_cut = fuv.min() * (1 + Z[0]) * 0.999

        def model(nu, t, **_):
            return np.where(nu < nu_cut, np.nan, np.log(1e29)) + 0 * t

        mags = peak_band_magnitudes(model, self.parameters, Z, D_L, grids, T, progress=False)
        assert np.isfinite(mags[:, 0, :]).all()
        # At low redshift NUV still reaches below the cut. Far enough out it has shifted past it, and is fine again.
        assert np.isnan(mags[:, 1, 0]).all()
        assert np.isfinite(mags[:, 1, -1]).all()
