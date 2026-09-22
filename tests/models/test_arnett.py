"""Tests for :mod:`uvex_transients.models.arnett`."""

import numpy as np
import pytest
from astropy import units as u
from scipy.integrate import quad

from uvex_transients.models.arnett import (
    _diffusion_integral,
    _diffusion_time_cgs,
    _leakage_parameter_cgs,
    compute_arnett_luminosity,
    get_magnetar_engine,
)
from uvex_transients.models._constants import C_CGS, MSUN_G

M_EJ = 5.0 * u.Msun
V_EJ = 1e4 * u.km / u.s


def _reference_luminosity(t, source, t_diff, leakage):
    """Direct quadrature of the paper's diffusion integral (Nicholl+17 Sec. 2), at scalar ``t``."""

    def integrand(tp):
        return 2.0 * source(tp) * (tp / t_diff) * np.exp((tp / t_diff) ** 2) / t_diff

    integral, _ = quad(integrand, 0.0, t, limit=500, epsrel=1e-10)
    return np.exp(-((t / t_diff) ** 2)) * -np.expm1(-leakage / t**2) * integral


class TestPhysicsHelpers:
    """Diffusion-time and leakage-parameter formulas."""

    def test_diffusion_time_matches_formula(self):
        kappa, mass, velocity = 0.2, 5.0 * MSUN_G, 1e9
        expected = np.sqrt(2 * kappa * mass / (13.8 * C_CGS * velocity))
        assert _diffusion_time_cgs(kappa, mass, velocity) == pytest.approx(expected)

    def test_diffusion_time_is_plausible_for_slsn(self):
        # ~5 Msun at 10^4 km/s should diffuse over tens of days (SLSN-like), not hours or years.
        t_diff = _diffusion_time_cgs(0.2, 5.0 * MSUN_G, 1e9) / 86400.0
        assert 10.0 < t_diff < 100.0

    def test_leakage_parameter_matches_formula(self):
        kappa_gamma, mass, velocity = 0.01, 5.0 * MSUN_G, 1e9
        expected = 3 * kappa_gamma * mass / (4 * np.pi * velocity**2)
        assert _leakage_parameter_cgs(kappa_gamma, mass, velocity) == pytest.approx(expected)


class TestMagnetarEngine:
    """The magnetar spin-down source."""

    def test_reference_values(self):
        # 1 ms, 1.4 Msun, 1e14 G: E = 2.6e52 erg, t_mag = 1.3e5 s.
        engine = get_magnetar_engine(1 * u.ms, 1e14 * u.G)
        e_mag, t_mag = 2.6e52, 1.3e5
        np.testing.assert_allclose(engine(0.0), e_mag / t_mag)
        np.testing.assert_allclose(engine(t_mag), e_mag / t_mag / 4.0)

    def test_total_energy_is_rotational_energy(self):
        engine = get_magnetar_engine(2 * u.ms, 3e14 * u.G, 2.0 * u.Msun)
        e_mag = 2.6e52 * (2.0 / 1.4) ** 1.5 * (2.0) ** -2
        t_mag = 1.3e5 * (2.0 / 1.4) ** 1.5 * 2.0**2 * (3.0) ** -2
        total, _ = quad(lambda t: engine(t), 0.0, np.inf, limit=200)
        assert total == pytest.approx(e_mag, rel=1e-6)
        assert engine(t_mag) == pytest.approx(e_mag / t_mag / 4.0)

    def test_unit_invariance(self):
        cgs = get_magnetar_engine(2e-3, 3e14, 2.0 * MSUN_G)
        quantity = get_magnetar_engine(2 * u.ms, 3e10 * u.T, 2.0 * u.Msun)  # 1 T = 1e4 G
        t = np.geomspace(1e3, 1e8, 10)
        np.testing.assert_allclose(quantity(t * u.s), cgs(t), rtol=1e-12)
        np.testing.assert_allclose(quantity((t / 86400.0) * u.day), cgs(t), rtol=1e-12)

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_rejects_non_positive_parameters(self, bad):
        with pytest.raises(ValueError):
            get_magnetar_engine(bad * u.ms, 1e14 * u.G)
        with pytest.raises(ValueError):
            get_magnetar_engine(1 * u.ms, bad * u.G)


class TestDiffusionIntegralKernel:
    """The numba recurrence kernel."""

    def test_constant_source_closed_form(self):
        # F = const  =>  e^{-u} \int F e^{u'} du' = F (1 - e^{-u}).
        t_diff = 3.0
        t = np.linspace(0.0, 20.0, 4001)
        out = _diffusion_integral(np.diff(t**2) / t_diff**2, np.full_like(t, 7.0))
        np.testing.assert_allclose(out, 7.0 * (1 - np.exp(-((t / t_diff) ** 2))), rtol=1e-9, atol=1e-12)

    def test_does_not_overflow_at_extreme_times(self):
        t = np.linspace(0.0, 1e4, 2001)  # (t/t_diff)^2 ~ 1e8: exp(+u) would overflow
        out = _diffusion_integral(np.diff(t**2) / 1.0, np.ones_like(t))
        assert np.all(np.isfinite(out))
        assert out[-1] == pytest.approx(1.0)


class TestComputeArnettLuminosity:
    """The public Arnett luminosity function."""

    @pytest.fixture
    def setup(self):
        engine = get_magnetar_engine(2 * u.ms, 5e13 * u.G)
        t_diff = _diffusion_time_cgs(0.2, 5.0 * MSUN_G, 1e9)
        return engine, t_diff

    def test_matches_direct_quadrature_full_trapping(self, setup):
        engine, t_diff = setup
        t = np.array([0.3, 1.0, 3.0, 10.0]) * t_diff
        actual = compute_arnett_luminosity(t * u.s, engine, M_EJ, V_EJ)
        expected = [_reference_luminosity(ti, engine, t_diff, np.inf) for ti in t]
        np.testing.assert_allclose(actual.to_value(u.erg / u.s), expected, rtol=1e-4)

    def test_matches_direct_quadrature_with_leakage(self, setup):
        engine, t_diff = setup
        kappa_gamma = 0.01
        leakage = _leakage_parameter_cgs(kappa_gamma, 5.0 * MSUN_G, 1e9)
        t = np.array([0.5, 2.0, 5.0, 20.0]) * t_diff
        actual = compute_arnett_luminosity(t * u.s, engine, M_EJ, V_EJ, kappa_gamma=kappa_gamma * u.cm**2 / u.g)
        expected = [_reference_luminosity(ti, engine, t_diff, leakage) for ti in t]
        np.testing.assert_allclose(actual.to_value(u.erg / u.s), expected, rtol=1e-4)

    def test_leakage_only_suppresses_and_is_a_pure_prefactor(self, setup):
        engine, t_diff = setup
        t = np.geomspace(0.1, 30.0, 12) * t_diff * u.s
        full = compute_arnett_luminosity(t, engine, M_EJ, V_EJ)
        leaky = compute_arnett_luminosity(t, engine, M_EJ, V_EJ, kappa_gamma=0.01)
        assert np.all(leaky <= full)
        leakage = _leakage_parameter_cgs(0.01, 5.0 * MSUN_G, 1e9)
        np.testing.assert_allclose((leaky / full).value, -np.expm1(-leakage / t.value**2), rtol=1e-9)

    def test_zero_gamma_opacity_means_total_leakage(self, setup):
        engine, _ = setup
        t = np.array([1e5, 1e6, 1e7]) * u.s
        result = compute_arnett_luminosity(t, engine, M_EJ, V_EJ, kappa_gamma=0.0)
        np.testing.assert_array_equal(result.value, 0.0)

    def test_fast_diffusion_tracks_the_source(self):
        # t_diff << engine timescale: output should follow the input power.
        engine = get_magnetar_engine(1 * u.ms, 1e14 * u.G)  # t_mag = 1.3e5 s
        t = np.array([2e5, 5e5, 1e6]) * u.s
        result = compute_arnett_luminosity(t, engine, 1e-4 * u.Msun, 1e4 * u.km / u.s)
        np.testing.assert_allclose(result.value, engine(t), rtol=1e-2)

    def test_radiated_energy_never_exceeds_injected(self, setup):
        # Adiabatic expansion losses (the e^{-(t/t_d)^2} factor) mean diffusion can only lose energy.
        engine, t_diff = setup
        t_end = 60.0 * t_diff
        t = np.linspace(0.0, t_end, 20000) * u.s
        luminosity = compute_arnett_luminosity(t, engine, M_EJ, V_EJ, n_grid=4000).value
        radiated = np.trapezoid(luminosity, t.value)
        injected, _ = quad(engine, 0.0, t_end, limit=200)
        assert 0.0 < radiated < injected

    def test_radiated_energy_matches_injected_when_diffusion_is_fast(self):
        engine = get_magnetar_engine(1 * u.ms, 1e14 * u.G)  # t_mag = 1.3e5 s
        t = np.geomspace(1.0, 1e9, 20000) * u.s
        # t_d ~ 4e2 s is far below the default 1e-4 * t_max grid start, so resolve it with t_min.
        luminosity = compute_arnett_luminosity(
            t, engine, 1e-7 * u.Msun, 1e4 * u.km / u.s, t_min=1.0 * u.s, n_grid=4000
        ).value
        radiated = np.trapezoid(luminosity, t.value)
        assert radiated == pytest.approx(2.6e52, rel=1e-2)

    def test_unit_invariance(self, setup):
        engine, _ = setup
        t_days = np.array([5.0, 20.0, 60.0])
        with_units = compute_arnett_luminosity(
            t_days * u.day, engine, 5.0 * u.Msun, 1e4 * u.km / u.s, 0.2 * u.cm**2 / u.g, 0.01 * u.cm**2 / u.g
        )
        bare_cgs = compute_arnett_luminosity(t_days * 86400.0, engine, 5.0 * MSUN_G, 1e9, 0.2, 0.01)
        np.testing.assert_allclose(with_units.value, bare_cgs.value, rtol=1e-12)

    def test_output_units_and_shape(self, setup):
        engine, _ = setup
        assert compute_arnett_luminosity(10 * u.day, engine, M_EJ, V_EJ).unit == u.erg / u.s
        assert compute_arnett_luminosity(10 * u.day, engine, M_EJ, V_EJ).shape == ()
        t = np.linspace(1, 50, 12).reshape(3, 4) * u.day
        assert compute_arnett_luminosity(t, engine, M_EJ, V_EJ).shape == (3, 4)

    def test_zero_time_gives_zero(self, setup):
        engine, _ = setup
        np.testing.assert_array_equal(compute_arnett_luminosity([0.0, 0.0] * u.s, engine, M_EJ, V_EJ).value, 0.0)
        result = compute_arnett_luminosity([0.0, 10.0] * u.day, engine, M_EJ, V_EJ, kappa_gamma=0.01)
        assert result[0].value == 0.0 and result[1].value > 0.0

    def test_result_is_independent_of_grid_choice(self, setup):
        engine, _ = setup
        t = np.array([5.0, 30.0]) * u.day
        default = compute_arnett_luminosity(t, engine, M_EJ, V_EJ)
        fine = compute_arnett_luminosity(t, engine, M_EJ, V_EJ, n_grid=8000)
        custom = compute_arnett_luminosity(t, engine, M_EJ, V_EJ, t_grid=np.linspace(0, 40, 3000) * u.day)
        np.testing.assert_allclose(default.value, fine.value, rtol=1e-4)
        np.testing.assert_allclose(default.value, custom.value, rtol=1e-4)

    def test_t_eval_beyond_custom_grid_is_still_evaluated(self, setup):
        engine, _ = setup
        result = compute_arnett_luminosity([10.0, 100.0] * u.day, engine, M_EJ, V_EJ, t_grid=[0.0, 5.0] * u.day)
        assert np.all(np.isfinite(result.value)) and np.all(result.value > 0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"t_eval": [-1.0] * u.s},
            {"t_eval": [np.nan] * u.s},
            {"ejecta_mass": 0.0 * u.Msun},
            {"ejecta_velocity": -1.0 * u.km / u.s},
            {"kappa": 0.0},
            {"kappa_gamma": -0.1},
            {"t_grid": [-1.0, 5.0] * u.s},
            {"t_min": 1e9 * u.s},
        ],
    )
    def test_validation_errors(self, setup, kwargs):
        engine, _ = setup
        args = {"t_eval": [10.0] * u.day, "energy_function": engine, "ejecta_mass": M_EJ, "ejecta_velocity": V_EJ}
        with pytest.raises(ValueError):
            compute_arnett_luminosity(**{**args, **kwargs})

    def test_rejects_bad_energy_function(self):
        with pytest.raises(ValueError):
            compute_arnett_luminosity([10.0] * u.day, lambda t: np.ones(3), M_EJ, V_EJ)
        with pytest.raises(ValueError):
            compute_arnett_luminosity([10.0] * u.day, lambda t: np.full_like(t, np.nan), M_EJ, V_EJ)

    def test_incompatible_units_raise(self, setup):
        engine, _ = setup
        with pytest.raises(u.UnitConversionError):
            compute_arnett_luminosity([10.0] * u.day, engine, 5.0 * u.km, V_EJ)
