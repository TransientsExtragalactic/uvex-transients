"""Tests for `HostDustMixin`, through `TypeIaSED`."""

import numpy as np
import pytest
from astropy import units as u
from scipy.integrate import quad_vec

from uvex_transients.dust import host_log_attenuation_cgs
from uvex_transients.models.arnett import LineBlanketedArnettDecaySED
from uvex_transients.models.host_dust import HostDustMixin
from uvex_transients.models.supernovae import TypeIaSED

DAY = 86400.0


def _cgs_draws(n: int = 4, seed: int = 0) -> dict[str, np.ndarray]:
    """Draws from the priors, as plain cgs arrays of shape ``(n,)``."""
    drawn = TypeIaSED().sample_parameters(size=n, rng=seed)
    return {name: np.asarray(getattr(value, "value", value)) for name, value in drawn.items()}


def _intrinsic(parameters: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: value for name, value in parameters.items() if name not in HostDustMixin._HOST_DUST_NAMES}


class TestPriors:
    def test_dust_parameters_are_appended_after_the_arnett_ones(self):
        names = list(TypeIaSED._DEFAULT_PARAMETERS)
        assert names[-2:] == ["A_V", "R_V"]
        assert names[:-2] == list(LineBlanketedArnettDecaySED._DEFAULT_PARAMETERS)

    def test_draws(self):
        draws = _cgs_draws(20000, seed=1)
        assert draws["A_V"].mean() == pytest.approx(0.4, rel=0.03)
        assert draws["A_V"].min() >= 0.0
        # Truncated at G23's lower limit, with the mean pulled up very slightly by the cut.
        assert draws["R_V"].min() >= 2.3
        assert draws["R_V"].max() <= 5.6
        assert draws["R_V"].mean() == pytest.approx(2.73, abs=0.02)

    def test_dust_law_is_a_class_attribute(self):
        assert TypeIaSED.DUST_LAW == "g23"


class TestAttenuation:
    t = np.array([5.0, 15.0, 30.0, 90.0])[np.newaxis, :, np.newaxis] * DAY
    nu = np.array([2e15, 1e15, 6e14, 3e14])[np.newaxis, np.newaxis, :]

    def _grids(self):
        return {name: value[:, np.newaxis, np.newaxis] for name, value in _cgs_draws().items()}

    def test_no_dust_is_exactly_the_wrapped_sed(self):
        parameters = self._grids()
        undusted = {**parameters, "A_V": np.zeros_like(parameters["A_V"])}
        np.testing.assert_array_equal(
            TypeIaSED._eval(self.nu, self.t, **undusted),
            LineBlanketedArnettDecaySED._eval(self.nu, self.t, **_intrinsic(parameters)),
        )

    def test_eval_is_the_intrinsic_sed_plus_the_law(self):
        parameters = self._grids()
        expected = LineBlanketedArnettDecaySED._eval(
            self.nu, self.t, **_intrinsic(parameters)
        ) + host_log_attenuation_cgs(self.nu, parameters["A_V"], parameters["R_V"])
        np.testing.assert_allclose(TypeIaSED._eval(self.nu, self.t, **parameters), expected)

    def test_dust_dims_the_uv_more_than_the_optical(self):
        parameters = self._grids()
        parameters = {**parameters, "A_V": np.ones_like(parameters["A_V"]), "R_V": np.full_like(parameters["R_V"], 3.0)}
        dimming = TypeIaSED._eval(self.nu, self.t, **parameters) - LineBlanketedArnettDecaySED._eval(
            self.nu, self.t, **_intrinsic(parameters)
        )
        assert np.all(dimming < 0.0)
        assert np.all(np.diff(dimming, axis=-1) > 0.0)  # frequencies descend, so the dimming eases

    def test_temperature_ignores_the_dust(self):
        parameters = {name: value[:, np.newaxis] for name, value in _cgs_draws().items()}
        with_dust = TypeIaSED._luminosity_and_temperature_cgs(self.t[:, :, 0], **parameters)
        without = LineBlanketedArnettDecaySED._luminosity_and_temperature_cgs(self.t[:, :, 0], **_intrinsic(parameters))
        for a, b in zip(with_dust, without):
            np.testing.assert_array_equal(a, b)


class TestBolometric:
    t = np.array([3.0, 14.0, 20.0, 60.0, 200.0])[np.newaxis, :] * DAY

    def _grids(self):
        return {name: value[:, np.newaxis] for name, value in _cgs_draws(3, seed=2).items()}

    def test_matches_adaptive_quadrature_of_the_attenuated_sed(self):
        parameters = self._grids()
        grid = np.exp(TypeIaSED._eval_bolometric(self.t, **parameters))
        reference, _ = quad_vec(
            lambda nu: np.exp(TypeIaSED._eval(nu, self.t, **parameters)), 1e12, 3e16, epsrel=1e-8, limit=500
        )
        np.testing.assert_allclose(grid, reference, rtol=5e-4)

    def test_no_dust_recovers_the_closed_form_arnett_luminosity(self):
        parameters = self._grids()
        undusted = {**parameters, "A_V": np.zeros_like(parameters["A_V"])}
        np.testing.assert_allclose(
            TypeIaSED._eval_bolometric(self.t, **undusted),
            LineBlanketedArnettDecaySED._eval_bolometric(self.t, **_intrinsic(parameters)),
            atol=5e-4,
        )

    def test_dust_lowers_the_bolometric_luminosity(self):
        parameters = self._grids()
        undusted = {**parameters, "A_V": np.zeros_like(parameters["A_V"])}
        assert np.all(TypeIaSED._eval_bolometric(self.t, **parameters) < TypeIaSED._eval_bolometric(self.t, **undusted))

    def test_the_spectrum_integrates_to_one(self):
        parameters = {name: value[0, 0] for name, value in self._grids().items()}
        t = 20.0 * DAY
        integral, _ = quad_vec(
            lambda nu: np.exp(TypeIaSED._eval_spectrum(nu, t, **parameters)), 1e12, 3e16, epsrel=1e-8
        )
        assert integral == pytest.approx(1.0, abs=5e-4)

    def test_chunking_barely_changes_the_result(self, monkeypatch):
        # Not bit-for-bit: the Arnett L(t) depends at the 1e-3 level on which times are evaluated together, and
        # chunking changes that grouping. That is already so for the dust-free SED.
        parameters = self._grids()
        whole = TypeIaSED._eval_bolometric(self.t, **parameters)
        monkeypatch.setattr(TypeIaSED, "_BOLOMETRIC_CHUNK", 2)
        np.testing.assert_allclose(TypeIaSED._eval_bolometric(self.t, **parameters), whole, atol=5e-3)

    def test_before_explosion_is_dark(self):
        parameters = self._grids()
        assert np.all(np.isneginf(TypeIaSED._eval_bolometric(np.array([[0.0]]), **parameters)))


def test_public_methods_take_physical_units():
    """The unit-aware entry points carry the dust parameters through, with ``A_V`` and ``R_V`` unitless."""
    model = TypeIaSED()
    draws = model.sample_parameters(size=2, rng=3)
    luminosity = model.eval_bolometric(20 * u.day, **{k: v[:, np.newaxis] for k, v in draws.items()})
    assert luminosity.unit.is_equivalent(u.erg / u.s)
    assert np.all(luminosity.value > 0)
