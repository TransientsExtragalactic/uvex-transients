"""
Tests for the regime-of-validity handling of the Morag+24 shock-cooling SEDs.

Outside its regime of validity (Eqs. 17-18 of Morag+24) the shock-cooling luminosity diverges toward early
times, reaching ~1e45 erg/s and beyond for Type IIb parameters (GitHub issue #9). `_INVALID_FILL` sets what
is written there: zero luminosity, ``nan``, or (``None``) the raw formula.

The class is patched with ``monkeypatch`` rather than subclassed, because `test_seds` sweeps every
`SpectralModel` subclass for a missing test class.
"""

import numpy as np
import pytest
from astropy import units as u

from uvex_transients.models.supernovae import MoragShockCoolingBlackbodySED, MoragShockCoolingSED

MODELS = [MoragShockCoolingBlackbodySED, MoragShockCoolingSED]

# Spans well before and well after any window the priors produce (3R/c is ~10 minutes at the scale radius).
T_GRID = np.geomspace(1e-3, 100, 60) * u.day

# A coarse band, 1000 to 2000 Angstrom, with flat throughput.
BAND_NU = np.geomspace(1.5e15, 3.0e15, 24) * u.Hz
BAND_THROUGHPUT = np.ones(BAND_NU.size)


def _cgs(parameters):
    return {name: np.asarray(value.cgs.value) for name, value in parameters.items()}


def _single_draw(model_class, seed=3):
    return {name: value[0] for name, value in model_class().sample_parameters(size=1, rng=seed).items()}


def _valid(model_class, t, parameters):
    """Boolean mask of the epochs inside the regime of validity, from the model's own components."""
    return np.asarray(model_class._components(t.cgs.value, **_cgs(parameters)).valid)


def _set_fill(monkeypatch, model_class, fill):
    monkeypatch.setattr(model_class, "_INVALID_FILL", fill)


@pytest.mark.parametrize("model_class", MODELS)
class TestInvalidFill:
    def test_default_is_zero(self, model_class):
        assert model_class._INVALID_FILL == "zero"

    def test_zero_fill_bolometric(self, model_class, monkeypatch):
        parameters = _single_draw(model_class)
        valid = _valid(model_class, T_GRID, parameters)
        assert valid.any() and not valid.all(), "the draw should straddle its regime of validity"

        _set_fill(monkeypatch, model_class, None)
        raw = model_class.eval_bolometric(T_GRID, **parameters).to_value(u.erg / u.s)
        _set_fill(monkeypatch, model_class, "zero")
        filled = model_class.eval_bolometric(T_GRID, **parameters).to_value(u.erg / u.s)

        assert np.all(filled[~valid] == 0.0)
        assert np.all(np.isfinite(filled))
        np.testing.assert_allclose(filled[valid], raw[valid])

    def test_nan_fill_bolometric(self, model_class, monkeypatch):
        parameters = _single_draw(model_class)
        valid = _valid(model_class, T_GRID, parameters)

        _set_fill(monkeypatch, model_class, None)
        raw = model_class.eval_bolometric(T_GRID, **parameters).to_value(u.erg / u.s)
        _set_fill(monkeypatch, model_class, "nan")
        filled = model_class.eval_bolometric(T_GRID, **parameters).to_value(u.erg / u.s)

        assert np.all(np.isnan(filled[~valid]))
        np.testing.assert_allclose(filled[valid], raw[valid])

    def test_unmasked_is_finite_everywhere(self, model_class, monkeypatch):
        _set_fill(monkeypatch, model_class, None)
        parameters = _single_draw(model_class)
        assert np.all(np.isfinite(model_class.eval_bolometric(T_GRID, **parameters).value))

    @pytest.mark.parametrize("fill", ["zero", "nan"])
    def test_spectral_luminosity_matches_unmasked_inside(self, model_class, monkeypatch, fill):
        parameters = _single_draw(model_class)
        valid = _valid(model_class, T_GRID, parameters)
        nu = BAND_NU[:, np.newaxis]

        _set_fill(monkeypatch, model_class, None)
        raw = model_class.eval(nu, T_GRID[np.newaxis, :], **parameters).to_value(u.erg / u.s / u.Hz)
        _set_fill(monkeypatch, model_class, fill)
        filled = model_class.eval(nu, T_GRID[np.newaxis, :], **parameters).to_value(u.erg / u.s / u.Hz)

        np.testing.assert_allclose(filled[:, valid], raw[:, valid])
        if fill == "zero":
            assert np.all(filled[:, ~valid] == 0.0)
        else:
            assert np.all(np.isnan(filled[:, ~valid]))

    def test_zero_fill_band_photometry(self, model_class, monkeypatch):
        """A zero fill must reach a band flux of exactly 0 (and an infinite magnitude), never ``nan``."""
        parameters = _single_draw(model_class)
        valid = _valid(model_class, T_GRID, parameters)
        kwargs = dict(redshift=0.02, luminosity_distance=100 * u.Mpc, **parameters)

        flux = model_class.flux_band(BAND_NU, BAND_THROUGHPUT, T_GRID, **kwargs).value
        mag = model_class.mag_band(BAND_NU, BAND_THROUGHPUT, T_GRID, **kwargs).value

        assert np.all(flux[~valid] == 0.0)
        assert np.all(np.isposinf(mag[~valid]))
        assert np.all(flux[valid] > 0.0)
        assert np.all(np.isfinite(mag[valid]))

    def test_nan_fill_band_photometry(self, model_class, monkeypatch):
        _set_fill(monkeypatch, model_class, "nan")
        parameters = _single_draw(model_class)
        valid = _valid(model_class, T_GRID, parameters)
        kwargs = dict(redshift=0.02, luminosity_distance=100 * u.Mpc, **parameters)

        mag = model_class.mag_band(BAND_NU, BAND_THROUGHPUT, T_GRID, **kwargs).value

        assert np.all(np.isnan(mag[~valid]))
        assert np.all(np.isfinite(mag[valid]))

    @pytest.mark.parametrize("fill", ["zero", "nan", None])
    def test_temperature(self, model_class, monkeypatch, fill):
        _set_fill(monkeypatch, model_class, fill)
        parameters = _single_draw(model_class)
        valid = _valid(model_class, T_GRID, parameters)
        temperature = model_class.temperature(T_GRID, **parameters).to_value(u.K)

        assert np.all(np.isfinite(temperature[valid]))
        if fill is None:
            assert np.all(np.isfinite(temperature))
        else:
            assert np.all(np.isnan(temperature[~valid]))

    def test_unknown_fill_raises(self, model_class, monkeypatch):
        _set_fill(monkeypatch, model_class, "inf")
        with pytest.raises(ValueError, match="_INVALID_FILL"):
            model_class.eval_bolometric(T_GRID, **_single_draw(model_class))


class TestIIbLuminosityRegression:
    """GitHub issue #9: the sampled population must not reach 1e45 erg/s at epochs the model cannot describe."""

    N_DRAWS = 4000
    LUMINOSITY_CEILING = 1e46 * u.erg / u.s

    def _peak_luminosity(self, model_class):
        parameters = model_class().sample_parameters(size=self.N_DRAWS, rng=11)
        grid = {name: value[:, np.newaxis] for name, value in parameters.items()}
        luminosity = model_class.eval_bolometric(T_GRID[np.newaxis, :], **grid)
        return luminosity.max(axis=1)

    def test_masked_population_stays_below_ceiling(self):
        peak = self._peak_luminosity(MoragShockCoolingBlackbodySED)
        assert np.all(np.isfinite(peak))
        assert peak.max() < self.LUMINOSITY_CEILING

    def test_unmasked_population_exceeds_ceiling(self, monkeypatch):
        """Guards the test above: without the mask, the same draws do blow past the ceiling."""
        _set_fill(monkeypatch, MoragShockCoolingBlackbodySED, None)
        peak = self._peak_luminosity(MoragShockCoolingBlackbodySED)
        assert peak.max() > self.LUMINOSITY_CEILING

    def test_prior_scales(self):
        """The IIb-appropriate scales suggested in issue #9."""
        parameters = MoragShockCoolingSED._DEFAULT_PARAMETERS
        assert parameters["radius"].scale == 6e12 * u.cm
        assert parameters["envelope_mass"].scale == 0.05 * u.Msun
        assert parameters["core_mass"].scale == 2.8 * u.Msun
        assert parameters["v_star"].scale == 2.0 * 10**8.5 * u.cm / u.s
