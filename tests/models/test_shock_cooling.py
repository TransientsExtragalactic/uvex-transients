"""Tests for :mod:`uvex_transients.models.shock_cooling`."""

import numpy as np
import pytest
from astropy import units as u
from scipy.integrate import solve_ivp

from uvex_transients.models import shock_cooling as sc
from uvex_transients.models._constants import C_CGS, MSUN_G

# A representative extended-envelope event (SN IIb-like).
E = 1e50 * u.erg
M = 0.05 * u.Msun
R = 3e13 * u.cm
KAPPA = 0.34 * u.cm**2 / u.g
N, DELTA = 10.0, 1.1

PARAMS = {"E_envelope": E, "M_envelope": M, "kappa": KAPPA, "n": N, "delta": DELTA}


class TestScales:
    """Characteristic velocity and timescales."""

    def test_normalization_matches_paper_value(self):
        # The paper quotes K = 0.119 for n = 10, delta = 1.1.
        assert sc.compute_ejecta_normalization(10.0, 1.1) == pytest.approx(0.119, abs=5e-4)

    def test_transition_velocity_formula(self):
        coef = ((N - 5) * (5 - DELTA)) / ((N - 3) * (3 - DELTA))
        expected = np.sqrt(coef * 2 * E.value / (M.value * MSUN_G))
        v_t = sc.compute_transition_velocity(E, M, N, DELTA)
        assert v_t.unit == u.cm / u.s
        assert v_t.value == pytest.approx(expected)

    def test_photospheric_time_is_tied_to_diffusion_time(self):
        # Paper, Sec. 2: t_ph = (c / 2 v_t)^(1/2) t_d.
        v_t = sc.compute_transition_velocity(E, M, N, DELTA).value
        t_d = sc.compute_diffusion_time(**PARAMS).value
        t_ph = sc.compute_photospheric_time(**PARAMS).value
        assert t_ph == pytest.approx(np.sqrt(C_CGS / (2 * v_t)) * t_d)

    def test_timescales_are_plausible(self):
        # ~0.05 Msun of extended material should diffuse out in hours to days, not seconds or months.
        t_d = sc.compute_diffusion_time(**PARAMS).to(u.day).value
        assert 0.01 < t_d < 10


class TestLuminosity:
    """Bolometric luminosity, Eqs. 17 and 20."""

    def test_continuous_at_diffusion_time(self):
        t_d = sc.compute_diffusion_time(**PARAMS)
        eps = 1e-9
        lo = sc.compute_luminosity(t_d * (1 - eps), R_envelope=R, **PARAMS)
        hi = sc.compute_luminosity(t_d * (1 + eps), R_envelope=R, **PARAMS)
        assert lo.value == pytest.approx(hi.value, rel=1e-6)

    def test_early_power_law_index(self):
        t_d = sc.compute_diffusion_time(**PARAMS)
        t = t_d * np.array([0.01, 0.1])
        L = sc.compute_luminosity(t, R_envelope=R, **PARAMS).value
        slope = np.log(L[1] / L[0]) / np.log(10)
        assert slope == pytest.approx(-4 / (N - 2))

    def test_late_time_matches_thermal_energy_ode(self):
        # Eqs. 18 and 19, integrated numerically from t_d, must reproduce the closed form of Eq. 20.
        t_d = sc.compute_diffusion_time(**PARAMS).value
        L_d = sc.compute_luminosity(t_d, R_envelope=R, **PARAMS).value
        E_d = L_d * t_d  # from L = t E_th / t_d^2, so E_th(t_d) = L(t_d) t_d.

        def rhs(t, y):
            return [-t * y[0] / t_d**2 - y[0] / t]

        t_eval = t_d * np.array([1.0, 1.5, 2.0, 3.0])
        sol = solve_ivp(rhs, (t_d, t_eval[-1]), [E_d], t_eval=t_eval, rtol=1e-11, atol=0)
        L_ode = t_eval * sol.y[0] / t_d**2
        L_model = sc.compute_luminosity(t_eval, R_envelope=R, **PARAMS).value
        np.testing.assert_allclose(L_model, L_ode, rtol=1e-6)

    def test_scales_linearly_with_initial_radius(self):
        t = 0.1 * u.day
        L1 = sc.compute_luminosity(t, R_envelope=R, **PARAMS)
        L2 = sc.compute_luminosity(t, R_envelope=2 * R, **PARAMS)
        assert (L2 / L1).value == pytest.approx(2.0)


class TestPhotosphericRadius:
    """Photospheric radius, Eqs. 7 and 9."""

    def test_continuous_at_photospheric_time(self):
        t_ph = sc.compute_photospheric_time(**PARAMS)
        eps = 1e-9
        lo = sc.compute_photospheric_radius(t_ph * (1 - eps), **PARAMS)
        hi = sc.compute_photospheric_radius(t_ph * (1 + eps), **PARAMS)
        assert lo.value == pytest.approx(hi.value, rel=1e-6)

    def test_equals_transition_radius_at_photospheric_time(self):
        t_ph = sc.compute_photospheric_time(**PARAMS)
        v_t = sc.compute_transition_velocity(E, M, N, DELTA)
        r = sc.compute_photospheric_radius(t_ph, **PARAMS)
        assert r.to_value(u.cm) == pytest.approx((v_t * t_ph).to_value(u.cm))

    def test_early_power_law_index(self):
        t_ph = sc.compute_photospheric_time(**PARAMS)
        r = sc.compute_photospheric_radius(t_ph * np.array([0.01, 0.1]), **PARAMS).value
        assert np.log(r[1] / r[0]) / np.log(10) == pytest.approx(1 - 2 / (N - 1))

    def test_late_time_asymptote(self):
        # Eq. 10 for t >> t_ph.
        t_ph = sc.compute_photospheric_time(**PARAMS)
        v_t = sc.compute_transition_velocity(E, M, N, DELTA).value
        t = 1e3 * t_ph
        expected = ((N - 1) / (DELTA - 1)) ** (1 / (DELTA - 1)) * (t_ph / t) ** (2 / (DELTA - 1)) * v_t * t
        got = sc.compute_photospheric_radius(t, **PARAMS)
        assert got.value == pytest.approx(expected.value, rel=1e-3)


class TestTemperature:
    """Blackbody temperature, Eq. 22."""

    def test_early_power_law_index(self):
        # Fig. 1 of the paper: T ~ t^(-1/2 - 1/(n-2) + 1/(n-1)) before t_d.
        t_d = sc.compute_diffusion_time(**PARAMS)
        T = sc.compute_temperature(t_d * np.array([0.01, 0.1]), R_envelope=R, **PARAMS).value
        assert np.log(T[1] / T[0]) / np.log(10) == pytest.approx(-0.5 - 1 / (N - 2) + 1 / (N - 1))

    def test_stefan_boltzmann_consistency(self):
        from astropy.constants import sigma_sb

        t = 0.3 * u.day
        L = sc.compute_luminosity(t, R_envelope=R, **PARAMS)
        r = sc.compute_photospheric_radius(t, **PARAMS)
        T = sc.compute_temperature(t, R_envelope=R, **PARAMS)
        assert (4 * np.pi * r**2 * sigma_sb * T**4).to_value(u.erg / u.s) == pytest.approx(L.value)

    def test_is_plausible(self):
        T = sc.compute_temperature(0.5 * u.day, R_envelope=R, **PARAMS)
        assert T.unit == u.K
        assert 5e3 < T.value < 1e5


class TestInputs:
    """Unit handling, broadcasting, and validation of the public functions."""

    def test_unitless_inputs_are_cgs(self):
        with_units = sc.compute_luminosity(1 * u.day, R_envelope=R, **PARAMS)
        bare = sc.compute_luminosity(
            u.day.to(u.s), E_envelope=E.value, R_envelope=R.value, M_envelope=M.to_value(u.g),
            kappa=KAPPA.value, n=N, delta=DELTA,
        )  # fmt: skip
        assert bare.value == pytest.approx(with_units.value)

    def test_other_units_are_converted(self):
        a = sc.compute_luminosity(1 * u.day, R_envelope=R, **PARAMS)
        b = sc.compute_luminosity(24 * u.hr, R_envelope=R.to(u.Rsun), **{**PARAMS, "E_envelope": E.to(u.J)})
        assert b.value == pytest.approx(a.value)

    def test_scalar_in_scalar_out(self):
        assert sc.compute_luminosity(1 * u.day, R_envelope=R, **PARAMS).shape == ()

    def test_broadcasts_parameters_against_time(self):
        t = np.geomspace(0.01, 5, 7) * u.day
        masses = np.array([0.01, 0.05, 0.1])[:, None] * u.Msun
        params = {**PARAMS, "M_envelope": masses}
        assert sc.compute_luminosity(t, R_envelope=R, **params).shape == (3, 7)
        assert sc.compute_photospheric_radius(t, **params).shape == (3, 7)
        assert sc.compute_temperature(t, R_envelope=R, **params).shape == (3, 7)

    def test_array_matches_scalar_evaluation(self):
        # Times straddle both t_d and t_ph so that every branch is exercised in a single array call.
        t = np.geomspace(1e-3, 30, 9) * u.day
        L = sc.compute_luminosity(t, R_envelope=R, **PARAMS)
        for ti, Li in zip(t, L):
            assert sc.compute_luminosity(ti, R_envelope=R, **PARAMS).value == pytest.approx(Li.value)

    @pytest.mark.parametrize("bad_t", [0 * u.s, -1 * u.s, np.nan * u.s])
    def test_rejects_non_positive_times(self, bad_t):
        with pytest.raises(ValueError, match="t must be"):
            sc.compute_luminosity(bad_t, R_envelope=R, **PARAMS)

    @pytest.mark.parametrize("n", [5.0, 3.0])
    def test_rejects_n_not_above_five(self, n):
        with pytest.raises(ValueError, match="n must be"):
            sc.compute_luminosity(1 * u.day, R_envelope=R, **{**PARAMS, "n": n})

    @pytest.mark.parametrize("delta", [1.0, 3.0, 0.5])
    def test_rejects_delta_outside_open_interval(self, delta):
        with pytest.raises(ValueError, match="delta must be"):
            sc.compute_photospheric_radius(1 * u.day, **{**PARAMS, "delta": delta})

    def test_rejects_non_positive_parameters(self):
        with pytest.raises(ValueError, match="M_envelope must be"):
            sc.compute_diffusion_time(**{**PARAMS, "M_envelope": 0 * u.Msun})
        with pytest.raises(ValueError, match="R_envelope must be"):
            sc.compute_temperature(1 * u.day, R_envelope=-R, **PARAMS)
