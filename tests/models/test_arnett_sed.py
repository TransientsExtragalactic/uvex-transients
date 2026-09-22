"""Model-specific tests for :class:`~uvex_transients.models.supernovae.magnetar.ArnettMagnetarSpindownSED`.

The generic ``SpectralModelContract`` checks live in ``test_seds.py``.
"""

import numpy as np
import pytest
from astropy import units as u

from uvex_transients.models._constants import SIGMA_SB_CGS
from uvex_transients.models.arnett import (
    GAUSS_CGS,
    _arnett_luminosity_cgs,
    compute_arnett_luminosity,
    get_magnetar_engine,
)
from uvex_transients.models.supernovae import ArnettMagnetarSpindownSED
from uvex_transients.models.supernovae import magnetar as magnetar_module

SED = ArnettMagnetarSpindownSED

PARAMS = {
    "spin_period": 2e-3,  # s
    "B_perp": 5e13,  # G
    "M_ej": 5.0 * 1.98841e33,  # g
    "v_ej": 1e9,  # cm/s
    "M_ns": 1.4 * 1.98841e33,  # g
    "kappa": 0.2,
    "kappa_gamma": 0.05,
    "T_floor": 6000.0,
}
T = np.geomspace(1.0, 300.0, 30) * 86400.0


class TestSingleIntegral:
    """The diffusion integral must be computed once per evaluation, never separately for L and T."""

    @pytest.fixture
    def counter(self, monkeypatch):
        calls = []
        original = magnetar_module._arnett_luminosity_cgs

        def counting(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(magnetar_module, "_arnett_luminosity_cgs", counting)
        return calls

    def test_eval_uses_one_integral(self, counter):
        SED.eval_log_cgs(np.geomspace(1e14, 1e16, 20)[:, None], T[None, :], **PARAMS)
        assert len(counter) == 1

    def test_bolometric_uses_one_integral(self, counter):
        SED.eval_bolometric_log_cgs(T, **PARAMS)
        assert len(counter) == 1

    def test_temperature_uses_one_integral(self, counter):
        SED.temperature(T * u.s, **{k: v for k, v in PARAMS.items()})
        assert len(counter) == 1


class TestLuminosity:
    """L(t) agrees with the public Arnett function and is independent of the spectral side."""

    def test_matches_compute_arnett_luminosity(self):
        engine = get_magnetar_engine(PARAMS["spin_period"] * u.s, PARAMS["B_perp"] * GAUSS_CGS, PARAMS["M_ns"] * u.g)
        expected = compute_arnett_luminosity(
            T * u.s,
            engine,
            PARAMS["M_ej"] * u.g,
            PARAMS["v_ej"] * u.cm / u.s,
            PARAMS["kappa"] * u.cm**2 / u.g,
            PARAMS["kappa_gamma"] * u.cm**2 / u.g,
            n_grid=4000,
        )
        actual = np.exp(SED.eval_bolometric_log_cgs(T, **PARAMS))
        np.testing.assert_allclose(actual, expected.value, rtol=2e-4)

    def test_bolometric_equals_integrated_spectrum(self):
        nu = np.geomspace(1e12, 1e17, 4000)
        t = T[10]
        lnu = SED.eval_cgs(nu, t, **PARAMS)
        np.testing.assert_allclose(np.trapezoid(lnu, nu), SED.eval_bolometric_cgs(t, **PARAMS), rtol=1e-2)

    def test_vanishes_at_t_zero_without_nans(self):
        result = SED.eval_log_cgs(np.array([1e14, 1e15]), 0.0, **PARAMS)
        assert np.all(result == -np.inf)
        assert SED.temperature(0.0 * u.s, **PARAMS).value == PARAMS["T_floor"]


class TestTemperature:
    """The photospheric temperature follows Stefan-Boltzmann, clipped at the floor."""

    def test_matches_stefan_boltzmann_above_floor_and_floor_below(self):
        luminosity = np.exp(SED.eval_bolometric_log_cgs(T, **PARAMS))
        photospheric = (luminosity / (4 * np.pi * SIGMA_SB_CGS * (PARAMS["v_ej"] * T) ** 2)) ** 0.25
        expected = np.maximum(photospheric, PARAMS["T_floor"])
        actual = SED.temperature(T * u.s, **PARAMS).value
        np.testing.assert_allclose(actual, expected, rtol=1e-12)
        assert np.all(actual >= PARAMS["T_floor"])
        assert np.any(photospheric > PARAMS["T_floor"]) and np.any(photospheric < PARAMS["T_floor"])

    def test_raising_the_floor_only_affects_the_spectrum_not_the_bolometric_light_curve(self):
        low = SED.eval_bolometric_log_cgs(T, **PARAMS)
        high = SED.eval_bolometric_log_cgs(T, **{**PARAMS, "T_floor": 9000.0})
        np.testing.assert_array_equal(low, high)
        assert np.all(SED.temperature(T * u.s, **{**PARAMS, "T_floor": 9000.0}).value >= 9000.0)


class TestBatching:
    """Per-event parameters and shared light curves must agree with one-at-a-time evaluation."""

    def test_batched_parameters_match_scalar_calls(self):
        spin = np.array([1e-3, 2e-3, 4e-3])
        batched = SED.eval_bolometric_log_cgs(T[None, :], **{**PARAMS, "spin_period": spin[:, None]})
        assert batched.shape == (3, T.size)
        for i, p in enumerate(spin):
            single = SED.eval_bolometric_log_cgs(T, **{**PARAMS, "spin_period": p})
            np.testing.assert_allclose(batched[i], single, rtol=1e-12)

    def test_per_event_times_and_parameters_pair_elementwise(self):
        spin = np.array([1e-3, 2e-3, 4e-3])
        t = np.array([10.0, 50.0, 120.0]) * 86400.0
        paired = SED.eval_bolometric_log_cgs(t, **{**PARAMS, "spin_period": spin})
        for i in range(3):
            single = SED.eval_bolometric_log_cgs(t[i], **{**PARAMS, "spin_period": spin[i]})
            np.testing.assert_allclose(paired[i], single, rtol=1e-3)

    def test_shared_parameters_share_one_grid(self):
        seen = []

        def spy(grid, energy, timescale):
            seen.append(grid.shape[0])
            return magnetar_module._magnetar_luminosity_cgs(grid, energy, timescale)

        _arnett_luminosity_cgs(T, 3e6, np.inf, spy, {"energy": 1e51, "timescale": 1e5})
        assert seen == [1]  # one group, however many times

    def test_time_order_does_not_matter(self):
        forward = SED.eval_bolometric_log_cgs(T, **PARAMS)
        shuffled = np.random.default_rng(0).permutation(T.size)
        np.testing.assert_allclose(SED.eval_bolometric_log_cgs(T[shuffled], **PARAMS), forward[shuffled], rtol=1e-12)

    def test_large_batch_is_chunked_consistently(self, monkeypatch):
        spin = np.linspace(1e-3, 5e-3, 40)
        reference = SED.eval_bolometric_log_cgs(60 * 86400.0, **{**PARAMS, "spin_period": spin})
        monkeypatch.setattr("uvex_transients.models.arnett._MAX_GRID_CELLS", 3000)  # forces many chunks
        chunked = SED.eval_bolometric_log_cgs(60 * 86400.0, **{**PARAMS, "spin_period": spin})
        np.testing.assert_allclose(chunked, reference, rtol=1e-12)


class TestPriors:
    """Default population draws must be physical."""

    def test_samples_are_valid_and_produce_finite_spectra(self):
        draws = SED().sample_parameters(size=200, rng=0)
        cgs = {name: np.asarray(value.cgs.value if hasattr(value, "cgs") else value) for name, value in draws.items()}
        assert np.all(cgs["v_ej"] > 0) and np.all(cgs["T_floor"] >= 3000)
        assert np.all((cgs["kappa_gamma"] >= 0.01) & (cgs["kappa_gamma"] <= 1.0))
        result = SED.eval_cgs(1e15, 60 * 86400.0, **cgs)
        assert result.shape == (200,) and np.all(np.isfinite(result)) and np.all(result >= 0)
