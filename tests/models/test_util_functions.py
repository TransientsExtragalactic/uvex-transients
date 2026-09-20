"""Tests for :mod:`uvex_transients.models._util_functions`."""

import numpy as np
import pytest
from scipy.integrate import quad

from uvex_transients.models._constants import C_CGS, H_CGS, K_B_CGS, LOG_SIGMA_SB_CGS
from uvex_transients.models._util_functions import (
    cooling_temperature_cgs,
    log_expm1,
    planck_Bnu_log_cgs,
    planck_shape_log_cgs,
)


def _planck_Bnu_reference(nu, temperature):
    """Direct (non-log-stable) Planck function, for cross-checking `planck_Bnu_log_cgs`."""
    x = H_CGS * nu / (K_B_CGS * temperature)
    return (2.0 * H_CGS * nu**3 / C_CGS**2) / np.expm1(x)


class TestPlanckBnuLogCgs:
    def test_matches_direct_evaluation(self):
        nu = np.geomspace(1e13, 1e16, 25)
        temperature = 1e4
        expected = np.log(_planck_Bnu_reference(nu, temperature))
        actual = planck_Bnu_log_cgs(nu, temperature)
        np.testing.assert_allclose(actual, expected, rtol=1e-8)

    def test_broadcasts_over_temperature(self):
        nu = 1e14
        temperature = np.array([1e3, 1e4, 1e5])
        result = planck_Bnu_log_cgs(nu, temperature)
        assert result.shape == temperature.shape


class TestPlanckShapeLogCgs:
    def test_relates_to_Bnu_by_normalization(self):
        nu = np.geomspace(1e13, 1e16, 25)
        temperature = 8e3
        shape = planck_shape_log_cgs(nu, temperature)
        Bnu = planck_Bnu_log_cgs(nu, temperature)
        expected = Bnu + np.log(np.pi) - LOG_SIGMA_SB_CGS - 4.0 * np.log(temperature)
        np.testing.assert_allclose(shape, expected, rtol=1e-12)

    def test_integrates_to_one(self):
        temperature = 1.2e4

        def integrand(log_nu):
            nu = np.exp(log_nu)
            return np.exp(planck_shape_log_cgs(nu, temperature)) * nu

        integral, _ = quad(integrand, np.log(1e5), np.log(1e20), limit=200)
        assert integral == pytest.approx(1.0, rel=1e-4)


class TestCoolingTemperatureCgs:
    def test_limits(self):
        T0, T_floor, timescale, alpha = 2e4, 5e3, 10.0, 1.5
        assert cooling_temperature_cgs(0.0, T0=T0, T_floor=T_floor, timescale=timescale, alpha=alpha) == pytest.approx(
            T0
        )
        late = cooling_temperature_cgs(1e6, T0=T0, T_floor=T_floor, timescale=timescale, alpha=alpha)
        assert late == pytest.approx(T_floor, abs=1.0)

    def test_matches_closed_form(self):
        t = np.geomspace(0.1, 1e3, 10)
        T0, T_floor, timescale, alpha = 1.5e4, 4e3, 8.0, 0.8
        expected = T_floor + (T0 - T_floor) * (1.0 + t / timescale) ** (-alpha)
        actual = cooling_temperature_cgs(t, T0=T0, T_floor=T_floor, timescale=timescale, alpha=alpha)
        np.testing.assert_allclose(actual, expected)


class TestLogExpm1:
    def test_matches_direct_computation_for_moderate_x(self):
        x = np.linspace(0.1, 20.0, 50)
        np.testing.assert_allclose(log_expm1(x), np.log(np.expm1(x)), rtol=1e-10)

    def test_asymptotic_form_for_large_x(self):
        x = np.array([50.0, 100.0, 700.0])
        np.testing.assert_allclose(log_expm1(x), x)
