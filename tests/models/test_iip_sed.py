"""Tests for `TypeIIPSED`: the specification's regression values, its UV blanketing and its hierarchical population."""

import warnings

import numpy as np
import pytest
from astropy import units as u
from scipy import stats

from uvex_transients.models.core.parameters import Parameter
from uvex_transients.models.core.priors import UniformPrior
from uvex_transients.models.supernovae import TypeIIPSED
from uvex_transients.models.supernovae.IIp import _log_normalization, _normalization_table
from uvex_transients.simulation.core import _sample_parameters_from_seeds

ERG_S = u.erg / u.s
C_SPEC = 2.99792e10  # the specification's value, in cm/s
PC10_SPEC = 10 * 3.0857e18  # its 10 pc, in cm

# =========================================================================== #
# Specification parameter sets (Sections 3.4 and 6)                           #
# =========================================================================== #
DEFAULTS = {
    "csm": 0.0,
    "L50": 1e42 * ERG_S,
    "t_p": 105 * u.day,
    "s": 0.5,
    "a_e": 1.5,
    "tau_e": 8 * u.day,
    "t_r": 0.3 * u.day,
    "M_Ni": 0.03 * u.Msun,
    "v_ph": 8e8 * u.cm / u.s,
    "R0": 3.5e13 * u.cm,
    "T_rec": 6000 * u.K,
    "L_c": 0 * ERG_S,
    "t_c": 3 * u.day,
    "k": 2.0,
    "tau_d": 3 * u.day,
    "w": 5 * u.day,
    "T0": 400 * u.day,
    "f_c": 1.25,
    "n": 6.0,
    "lam_c": 3975 * u.AA,
    "beta_max": 3.6,
    "T_b": 8000 * u.K,
    "dT": 660 * u.K,
}
WITH_CSM = {
    **DEFAULTS,
    "csm": 1.0,
    "L_c": 1e43 * ERG_S,
    "t_c": 3.0 * u.day,
    "k": 4.0,
    "tau_d": 1.5 * u.day,
    "R0": 1.4e14 * u.cm,
    "t_r": 1.2 * u.day,
}
SN2023IXF = {
    **DEFAULTS,
    "csm": 1.0,
    "L50": 2.246e42 * ERG_S,
    "t_p": 82.0 * u.day,
    "s": 1.99,
    "a_e": 6.0,
    "tau_e": 6.73 * u.day,
    "M_Ni": 0.0628 * u.Msun,
    "t_r": 5.11 * u.day,
    "L_c": 2.137e43 * ERG_S,
    "t_c": 5.27 * u.day,
    "k": 12.0,
    "tau_d": 0.863 * u.day,
    "R0": 2.26e14 * u.cm,
    "v_ph": 1.269e9 * u.cm / u.s,
    "T_rec": 5079 * u.K,
}
SN2024GGI = {
    **DEFAULTS,
    "csm": 1.0,
    "L50": 1.353e42 * ERG_S,
    "a_e": 1.51,
    "tau_e": 20.35 * u.day,
    "t_r": 8.0 * u.day,
    "L_c": 9.012e42 * ERG_S,
    "t_c": 2.37 * u.day,
    "k": 12.0,
    "tau_d": 2.59 * u.day,
    "R0": 8.61e13 * u.cm,
    "v_ph": 1.081e9 * u.cm / u.s,
}

EPOCHS = np.array([1, 5, 10, 50, 120.0]) * u.day

# Section 7: L_bol (erg/s), T_col (K), M_V and M_UV at the epochs above, from the reference implementation.
SPEC_VALUES = {
    "defaults": dict(
        params=DEFAULTS,
        L_bol=[2.9119e42, 2.2180e42, 1.7189e42, 1.0029e42, 1.6543e41],
        T_col=[30973.3, 15144.1, 10346.6, 6097.4, 6000.5],
        M_V=[-14.679, -16.122, -16.516, -16.265, -14.297],
        M_UV=[-15.671, -16.251, -15.726, -12.065, -10.010],
        rtol=1e-4,
        atol_mag=2e-3,
    ),
    "with_csm": dict(
        params=WITH_CSM,
        L_bol=[3.2347e42, 7.3522e42, 1.9489e42, 1.0029e42, 1.6543e41],
        T_col=[22438.3, 18082.8, 9994.6, 6089.4, 6000.5],
        M_V=[-15.635, -17.041, -16.708, -16.264, -14.297],
        M_UV=[-16.331, -17.459, -15.786, -12.057, -10.010],
        rtol=1e-4,
        atol_mag=2e-3,
    ),
    # The input parameters of Section 6 are rounded to three or four figures, which sets the looser tolerance.
    "sn2023ixf": dict(
        params=SN2023IXF,
        L_bol=[1.2800e42, 3.3209e43, 1.1017e43, 2.2516e42, 2.8803e41],
        T_col=[14052.6, 20875.9, 12133.1, 5244.5, 5079.5],
        M_V=[-15.673, -18.341, -18.276, -16.988, -14.709],
        M_UV=[-15.660, -18.952, -17.941, -11.969, -9.509],
        rtol=3e-3,
        atol_mag=5e-3,
    ),
}


def _top_hat_abs_mag(params, t, lo_angstrom, hi_angstrom):
    """AB absolute magnitude through a top-hat band, averaging ``L_nu`` over 40 points linear in frequency (Sec. 7)."""
    nu = np.linspace(C_SPEC / (hi_angstrom * 1e-8), C_SPEC / (lo_angstrom * 1e-8), 40)
    l_nu = TypeIIPSED.eval(nu[None, :] * u.Hz, t[:, None], **params).to_value(u.erg / u.s / u.Hz).mean(axis=1)
    return -2.5 * np.log10(l_nu / (4 * np.pi * PC10_SPEC**2)) - 48.6


def _blackbody_shape(nu, temperature):
    """``pi B_nu(T) / sigma T^4``, the unit-integral blackbody shape, in 1/Hz, from the closed form."""
    h, k_b, c, sigma = 6.62607015e-27, 1.380649e-16, 2.99792458e10, 5.670374419e-5
    return np.pi * 2 * h * nu**3 / c**2 / np.expm1(h * nu / (k_b * temperature)) / (sigma * temperature**4)


def _cgs_value(x):
    return x.cgs.value if isinstance(x, u.Quantity) else x


# =========================================================================== #
# Regression values of the specification                                      #
# =========================================================================== #
class TestSpecificationRegression:
    """The values the reference implementation gives at 1, 5, 10, 50 and 120 d (Section 7)."""

    @pytest.mark.parametrize("case", SPEC_VALUES)
    def test_bolometric_luminosity(self, case):
        spec = SPEC_VALUES[case]
        got = TypeIIPSED.eval_bolometric(EPOCHS, **spec["params"]).to_value(ERG_S)
        np.testing.assert_allclose(got, spec["L_bol"], rtol=spec["rtol"])

    @pytest.mark.parametrize("case", SPEC_VALUES)
    def test_colour_temperature(self, case):
        spec = SPEC_VALUES[case]
        got = TypeIIPSED.temperature(EPOCHS, **spec["params"]).to_value(u.K)
        np.testing.assert_allclose(got, spec["T_col"], rtol=spec["rtol"])

    @pytest.mark.parametrize("case", SPEC_VALUES)
    def test_optical_and_uv_top_hat_magnitudes(self, case):
        spec = SPEC_VALUES[case]
        np.testing.assert_allclose(
            _top_hat_abs_mag(spec["params"], EPOCHS, 5000, 5900), spec["M_V"], atol=spec["atol_mag"]
        )
        np.testing.assert_allclose(
            _top_hat_abs_mag(spec["params"], EPOCHS, 2250, 2950), spec["M_UV"], atol=spec["atol_mag"]
        )

    def test_spectral_luminosity_at_ten_days(self):
        nu = C_SPEC / (np.array([1600.0, 2600.0, 5500.0]) * 1e-8) * u.Hz
        got = TypeIIPSED.eval(nu, 10 * u.day, **DEFAULTS).to_value(u.erg / u.s / u.Hz)
        np.testing.assert_allclose(got, [1.2611e26, 8.7707e26, 1.7543e27], rtol=1e-3)

    @pytest.mark.parametrize(
        ("temperature", "expected"), [(6000.0, 0.93922), (8000.0, 0.88459), (12000.0, 0.99767), (30000.0, 1.00002)]
    )
    def test_normalization_table(self, temperature, expected):
        n = np.exp(_log_normalization(np.array(temperature), 3975e-8, 3.6, 8000.0, 660.0))
        assert n == pytest.approx(expected, abs=2e-5)

    def test_parameters_are_those_of_the_specification(self):
        per_object = {
            "L50", "t_p", "s", "a_e", "tau_e", "t_r", "M_Ni", "v_ph", "R0", "T_rec", "L_c", "t_c", "k", "tau_d",
        }  # fmt: skip
        fixed = {"w", "T0", "f_c", "n", "lam_c", "beta_max", "T_b", "dT"}
        assert set(TypeIIPSED._DEFAULT_PARAMETERS) == per_object | fixed | {"csm"}


# =========================================================================== #
# Light curve                                                                 #
# =========================================================================== #
class TestLightCurve:
    """The envelope, the radioactive tail and the CSM pulse."""

    @staticmethod
    def _components(params, t_days):
        t = np.atleast_1d(np.asarray(t_days, dtype=float)) * 86400.0
        cgs = {name: _cgs_value(value) for name, value in params.items()}
        return tuple(np.exp(c) for c in TypeIIPSED._log_components(t, **cgs))

    def test_the_plateau_luminosity_at_day_50_is_L50(self):
        """The tail is gated off, so the observed luminosity at day 50 is L50 times the small early excess."""
        (env, ni, csm), p = self._components(DEFAULTS, 50.0), DEFAULTS
        plateau_not_yet_over = 1.0 / (1.0 + np.exp((50.0 - 105.0) / 5.0))  # F(50 d)
        assert env[0] == pytest.approx(1e42 * (1 + p["a_e"] * np.exp(-50.0 / 8.0)) * plateau_not_yet_over, rel=1e-9)
        assert ni[0] < 1e-5 * env[0]  # gated by 1 - F(50 d) = 1.7e-5
        assert csm[0] == 0.0

    def test_the_tail_is_the_nickel_cobalt_chain_gated_by_trapping(self):
        t = 200.0
        env, ni, _ = self._components(DEFAULTS, t)
        msun = 1.98841e33
        chain = 0.03 * msun * ((3.9e10 - 6.78e9) * np.exp(-t / 8.8) + 6.78e9 * np.exp(-t / 111.3))
        assert ni[0] == pytest.approx(chain * (1 - np.exp(-((400.0 / t) ** 2))), rel=1e-4)
        assert env[0] < 1e-6 * ni[0]  # the plateau is over

    def test_the_plateau_ends_at_t_p(self):
        env_at_end, _, _ = self._components(DEFAULTS, 105.0)
        env_before, _, _ = self._components(DEFAULTS, 105.0 - 40.0)
        # F is 1/2 at t_p; compared with the same epoch without the drop
        assert env_at_end[0] == pytest.approx(
            0.5 * 1e42 * 10 ** (-0.4 * 0.5 * 55 / 100) * (1 + 1.5 * np.exp(-105 / 8)), rel=1e-3
        )
        assert env_before[0] > env_at_end[0]

    def test_the_csm_pulse_peaks_at_L_c_at_t_c_with_zero_slope(self):
        p = WITH_CSM
        t_c = 3.0
        eps = 1e-4
        _, _, csm = self._components(p, [t_c - eps, t_c, t_c + eps])
        assert csm[1] == pytest.approx(1e43, rel=1e-12)
        assert csm[0] == pytest.approx(csm[1], rel=1e-6)  # zero slope on the rising side
        assert csm[2] == pytest.approx(csm[1], rel=1e-6)  # and on the decaying side

    def test_the_csm_pulse_rises_as_a_power_law_and_decays_exponentially(self):
        # Deep in the rise (x = t/t_c << 1) the pulse is x^k e^k, so the slope in log-log is k = 4.
        _, _, csm = self._components(WITH_CSM, [3e-4, 3e-3])
        assert np.log10(csm[1] / csm[0]) == pytest.approx(4.0, abs=0.01)
        # After the peak it is exp[1 - sqrt(1 + ((t - t_c)/tau_d)^2)], which decays as exp(-t/tau_d) far from it.
        _, _, late = self._components(WITH_CSM, [30.0, 31.0])
        y = (np.array([30.0, 31.0]) - 3.0) / 1.5
        assert late[1] / late[0] == pytest.approx(np.exp(-(np.sqrt(1 + y[1] ** 2) - np.sqrt(1 + y[0] ** 2))), rel=1e-9)
        assert np.log(late[0] / late[1]) == pytest.approx(1.0 / 1.5, rel=3e-3)

    def test_a_plain_event_ignores_the_unused_csm_parameters(self):
        t = np.array([0.5, 3.0, 20.0, 100.0]) * u.day
        other = {**DEFAULTS, "t_c": 9 * u.day, "k": 7.0, "tau_d": 0.5 * u.day}
        np.testing.assert_array_equal(
            TypeIIPSED.eval_bolometric(t, **DEFAULTS).value, TypeIIPSED.eval_bolometric(t, **other).value
        )

    def test_the_csm_pulse_adds_to_the_bolometric_luminosity(self):
        t = np.array([1.0, 3.0, 5.0, 10.0]) * u.day
        plain = TypeIIPSED.eval_bolometric(t, **DEFAULTS).value
        with_csm = TypeIIPSED.eval_bolometric(t, **{**DEFAULTS, **{k: WITH_CSM[k] for k in ("L_c", "t_c", "k")}}).value
        assert np.all(with_csm > plain)

    def test_finite_and_warning_free_from_explosion_to_late_times(self):
        t = np.concatenate([[0.0], np.geomspace(1e-4, 2000.0, 60)]) * u.day
        nu = np.geomspace(1e14, 3e16, 12) * u.Hz
        for params in (DEFAULTS, WITH_CSM, SN2023IXF, SN2024GGI):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                lbol = TypeIIPSED.eval_bolometric(t, **params)
                l_nu = TypeIIPSED.eval(nu[None, :], t[:, None], **params)
                temperature = TypeIIPSED.temperature(t, **params)
            assert np.all(np.isfinite(lbol.value)) and np.all(lbol.value > 0)
            assert np.all(np.isfinite(l_nu.value)) and np.all(np.isfinite(temperature.value))

    def test_sn2024ggi_set_has_a_few_day_csm_peak(self):
        t = np.geomspace(0.5, 200.0, 400) * u.day
        lbol = TypeIIPSED.eval_bolometric(t, **SN2024GGI).to_value(ERG_S)
        assert 2.0 < t[np.argmax(lbol)].value < 6.0
        assert 3e42 < lbol.max() < 3e43


# =========================================================================== #
# Temperature and spectrum                                                    #
# =========================================================================== #
class TestPhotosphereAndSpectrum:
    """The cooling photosphere and the line-blanketed, energy-conserving blackbody."""

    def test_the_photosphere_cools_onto_the_recombination_floor(self):
        t = np.geomspace(1.0, 300.0, 60) * u.day
        temperature = TypeIIPSED.temperature(t, **DEFAULTS).to_value(u.K)
        assert temperature[0] > 20000.0
        assert temperature[-1] == pytest.approx(6000.0, rel=1e-3)
        assert np.all(np.diff(temperature[t.value < 100]) < 0)

    def test_the_shape_integrates_to_one_so_l_bol_is_the_emitted_luminosity(self):
        nu = np.geomspace(1e12, 3e17, 6000)
        for t_days in (0.5, 3.0, 10.0, 40.0, 120.0):
            for params in (DEFAULTS, WITH_CSM):
                shape = TypeIIPSED.eval_spectrum(nu * u.Hz, t_days * u.day, **params).to_value(1 / u.Hz)
                assert np.trapezoid(shape, nu) == pytest.approx(1.0, abs=5e-3), (t_days, params["csm"])

    def test_a_hot_photosphere_is_an_unsuppressed_blackbody(self):
        nu = np.array([6e14, 1.5e15, 3e15])
        hot = {**DEFAULTS, "T_rec": 6000 * u.K}
        shape = TypeIIPSED.eval_spectrum(nu * u.Hz, 0.5 * u.day, **hot).to_value(1 / u.Hz)
        temperature = TypeIIPSED.temperature(0.5 * u.day, **hot).to_value(u.K)
        assert temperature > 2e4
        plain = _blackbody_shape(nu, temperature)
        np.testing.assert_allclose(shape, plain, rtol=2e-3)

    def test_a_cool_photosphere_loses_the_uv_and_the_optical_gains_it(self):
        nu = np.array([C_SPEC / 2500e-8, C_SPEC / 5500e-8])
        t = 100 * u.day
        shape = TypeIIPSED.eval_spectrum(nu * u.Hz, t, **DEFAULTS).to_value(1 / u.Hz)
        temperature = TypeIIPSED.temperature(t, **DEFAULTS).to_value(u.K)
        plain = _blackbody_shape(nu, temperature)
        n = float(np.exp(_log_normalization(np.array(temperature), 3975e-8, 3.6, 8000.0, 660.0)))
        beta = 3.6 / (1.0 + np.exp((temperature - 8000.0) / 660.0))
        suppression = (2500.0 / 3975.0) ** beta  # blueward of the 3975 A cutoff; 1 redward of it
        assert beta > 3.0 and suppression < 0.25
        assert shape[0] / plain[0] == pytest.approx(suppression / n, rel=1e-3)
        assert shape[1] / plain[1] == pytest.approx(1.0 / n, rel=1e-3)  # redward: renormalized by 1/N, not lost

    def test_uv_parameters_that_differ_between_elements_each_get_their_own_table(self):
        temperature = np.array([5000.0, 7000.0, 9000.0, 12000.0])
        lam_c = np.array([3975e-8, 3000e-8, 3975e-8, 5000e-8])
        grouped = _log_normalization(temperature, lam_c, 3.6, 8000.0, 660.0)
        one_by_one = [_log_normalization(np.array(t), lc, 3.6, 8000.0, 660.0) for t, lc in zip(temperature, lam_c)]
        np.testing.assert_allclose(grouped, one_by_one, rtol=1e-12)

    def test_the_normalization_table_is_cached_per_set_of_uv_parameters(self):
        _normalization_table.cache_clear()
        _log_normalization(np.array([6000.0, 7000.0]), 3975e-8, 3.6, 8000.0, 660.0)
        _log_normalization(np.array([8000.0]), 3975e-8, 3.6, 8000.0, 660.0)
        assert _normalization_table.cache_info().misses == 1


# =========================================================================== #
# Population                                                                  #
# =========================================================================== #
N_DRAWS = 20000


@pytest.fixture(scope="module")
def population():
    """20000 draws from the default population, as plain floats in the model's cgs units."""
    draws = TypeIIPSED().sample_parameters(size=N_DRAWS, rng=11)
    return {name: _cgs_value(value) for name, value in draws.items()}


def _log10(x):
    return np.log10(np.asarray(x, dtype=float))


class TestPopulationPriors:
    """The marginal and conditional priors of Section 5."""

    def test_csm_fraction_is_0_36(self, population):
        fraction = np.mean(population["csm"] == 1.0)
        assert set(np.unique(population["csm"])) == {0.0, 1.0}
        assert fraction == pytest.approx(0.36, abs=4 * np.sqrt(0.36 * 0.64 / N_DRAWS))

    def test_L50_and_L_c_follow_their_truncated_normals(self, population):
        csm = population["csm"] == 1.0
        l50 = stats.truncnorm(a=(40.8 - 41.26) / 0.51, b=(43.3 - 41.26) / 0.51, loc=41.26, scale=0.51)
        l_c = stats.truncnorm(a=(41.0 - 43.0) / 0.4, b=(44.8 - 43.0) / 0.4, loc=43.0, scale=0.4)
        assert stats.kstest(_log10(population["L50"]), l50.cdf).pvalue > 1e-3
        assert stats.kstest(_log10(population["L_c"][csm]), l_c.cdf).pvalue > 1e-3

    def test_a_plain_event_has_no_csm_pulse(self, population):
        plain = population["csm"] == 0.0
        assert np.all(population["L_c"][plain] == 0.0)
        np.testing.assert_allclose(population["t_c"][plain] / 86400.0, 3.0)
        np.testing.assert_allclose(population["k"][plain], 2.0)
        np.testing.assert_allclose(population["tau_d"][plain] / 86400.0, 3.0)
        assert np.all(population["L_c"][~plain] > 0.0)

    def test_every_draw_is_inside_the_bounds_of_its_prior(self, population):
        days = 86400.0
        checks = {
            "t_p": (60 * days, 150 * days),
            "s": (0.0, 3.0),
            "T_rec": (4500.0, 7000.0),
        }
        for name, (lo, hi) in checks.items():
            assert population[name].min() >= lo and population[name].max() <= hi, name
        assert _log10(population["a_e"]).min() >= -1.0 and _log10(population["a_e"]).max() <= 1.0
        assert 0.5 <= _log10(population["tau_e"] / days).min() and _log10(population["tau_e"] / days).max() <= 1.5
        msun = 1.98841e33
        assert -3.3 <= _log10(population["M_Ni"] / msun).min() and _log10(population["M_Ni"] / msun).max() <= -0.5
        csm = population["csm"] == 1.0
        assert 0.0 <= _log10(population["t_c"][csm] / days).min()
        assert _log10(population["t_c"][csm] / days).max() <= 1.0
        assert 0.18 <= _log10(population["k"][csm]).min() and _log10(population["k"][csm]).max() <= 1.18
        assert -0.5 <= _log10(population["tau_d"][csm] / days).min()
        assert _log10(population["tau_d"][csm] / days).max() <= 1.0

    def test_conditionals_on_a_fixed_L50(self):
        """Given L50 the parameters are the shifted, truncated normals of Section 5.2."""
        for log_l50 in (40.9, 42.0, 43.1):
            model = TypeIIPSED(L50=10**log_l50 * ERG_S)
            d = {k: _cgs_value(v) for k, v in model.sample_parameters(size=6000, rng=3).items()}
            shift = log_l50 - 42.0
            s = stats.truncnorm(a=(0 - (0.8 + 0.5 * shift)) / 0.5, b=(3 - (0.8 + 0.5 * shift)) / 0.5,
                                loc=0.8 + 0.5 * shift, scale=0.5)  # fmt: skip
            m_ni_mean = -1.45 + 1.0 * shift
            m_ni = stats.truncnorm(a=(-3.3 - m_ni_mean) / 0.17, b=(-0.5 - m_ni_mean) / 0.17, loc=m_ni_mean, scale=0.17)
            assert stats.kstest(d["s"], s.cdf).pvalue > 1e-3, log_l50
            assert stats.kstest(_log10(d["M_Ni"] / 1.98841e33), m_ni.cdf).pvalue > 1e-3, log_l50
            v = stats.norm(loc=np.log10(8e8) + 0.33 * shift, scale=0.08)
            assert stats.kstest(_log10(d["v_ph"]), v.cdf).pvalue > 1e-3, log_l50

    def test_t_c_follows_the_brightness_of_the_csm_pulse(self):
        for log_l_c in (42.0, 43.15, 44.4):
            model = TypeIIPSED(L_c=10**log_l_c * ERG_S)
            d = {k: _cgs_value(v) for k, v in model.sample_parameters(size=6000, rng=4).items()}
            mean = np.log10(3.5) + 0.5 * (log_l_c - 43.15)
            t_c = stats.truncnorm(a=(0.0 - mean) / 0.15, b=(1.0 - mean) / 0.15, loc=mean, scale=0.15)
            assert np.all(d["csm"] == 1.0)
            assert stats.kstest(_log10(d["t_c"] / 86400.0), t_c.cdf).pvalue > 1e-3, log_l_c

    def test_the_two_classes_of_event_have_their_own_rise_time_and_radius(self, population):
        csm = population["csm"] == 1.0
        days = 86400.0
        assert np.median(population["t_r"][~csm] / days) == pytest.approx(1.0, rel=0.05)
        assert np.median(population["t_r"][csm] / days) == pytest.approx(1.2, rel=0.08)
        assert np.median(population["R0"][~csm]) == pytest.approx(3.5e13, rel=0.04)
        assert np.median(population["R0"][csm]) == pytest.approx(1.4e14, rel=0.08)
        assert population["R0"][csm].max() == 3e14  # capped, not truncated
        assert np.mean(population["R0"][csm] == 3e14) == pytest.approx(
            stats.norm.sf(np.log10(3e14 / 1.4e14) / 0.2), abs=0.01
        )

    def test_magnitude_limited_sample_is_about_60_percent_csm(self, population):
        """Median peak M_V of Section 7 and the CSM share of a magnitude-limited sample (weights 10^(-0.6 M_V))."""
        idx = np.arange(1500)
        t = np.geomspace(0.3, 120.0, 70) * u.day
        nu = np.linspace(C_SPEC / 5900e-8, C_SPEC / 5000e-8, 24)
        sub = {k: v[idx][:, None, None] for k, v in population.items()}
        l_nu = np.exp(TypeIIPSED.eval_log_cgs(nu[None, None, :], t.to_value(u.s)[None, :, None], **sub)).mean(axis=2)
        peak = (-2.5 * np.log10(l_nu / (4 * np.pi * PC10_SPEC**2)) - 48.6).min(axis=1)
        csm = population["csm"][idx] == 1.0
        assert np.median(peak[~csm]) == pytest.approx(-15.7, abs=0.2)
        assert np.median(peak[csm]) == pytest.approx(-16.9, abs=0.25)
        weight = 10 ** (-0.6 * peak)
        assert weight[csm].sum() / weight.sum() == pytest.approx(0.60, abs=0.06)

    @pytest.mark.slow
    def test_v_band_peak_luminosity_function_follows_asas_sn(self):
        """Section 8: 0.06 to 0.10 dex rms over the eight bins from M_V = -15.4 to -20.0, after one normalization."""
        from uvex_transients.utils.luminosity_function_archive import LuminosityFunctionArchive

        model, n_draws, chunk = TypeIIPSED(), 40000, 2000
        rng = np.random.default_rng(7)
        t = np.geomspace(0.3, 150.0, 80) * u.day
        nu = np.linspace(C_SPEC / 5900e-8, C_SPEC / 5000e-8, 24)
        peaks = []
        for _ in range(n_draws // chunk):
            d = {k: _cgs_value(v)[:, None, None] for k, v in model.sample_parameters(size=chunk, rng=rng).items()}
            l_nu = np.exp(TypeIIPSED.eval_log_cgs(nu[None, None, :], t.to_value(u.s)[None, :, None], **d)).mean(axis=2)
            peaks.append((-2.5 * np.log10(l_nu / (4 * np.pi * PC10_SPEC**2)) - 48.6).min(axis=1))
        peak = np.concatenate(peaks)

        archive = LuminosityFunctionArchive().table("supernovae/II", "ASAS-SN")
        centers, density = np.asarray(archive["M"].value), np.asarray(archive["dNdM"].value)
        midpoints = 0.5 * (centers[1:] + centers[:-1])
        edges = np.concatenate([[2 * centers[0] - midpoints[0]], midpoints, [2 * centers[-1] - midpoints[-1]]])
        counts = np.array([np.sum((peak >= edges[i]) & (peak < edges[i + 1])) for i in range(len(centers))])
        model_density = counts / n_draws / np.diff(edges)
        use = centers > -20.5  # the brightest bin, M_V = -21.2, is not reproduced and is excluded
        assert use.sum() == 8
        residual = np.log10(density[use]) - np.log10(model_density[use])
        assert np.sqrt(np.mean((residual - residual.mean()) ** 2)) < 0.15


# =========================================================================== #
# Fixing and replacing parameters                                             #
# =========================================================================== #
class TestFixingAndReplacing:
    """The population's rules for a parameter that the user pins or replaces."""

    def test_csm_fixed_to_zero_gives_only_plain_events(self):
        d = {k: _cgs_value(v) for k, v in TypeIIPSED(csm=0.0).sample_parameters(size=500, rng=1).items()}
        assert np.all(d["csm"] == 0.0) and np.all(d["L_c"] == 0.0)

    def test_csm_fixed_to_one_gives_only_csm_events(self):
        d = {k: _cgs_value(v) for k, v in TypeIIPSED(csm=1.0).sample_parameters(size=500, rng=1).items()}
        assert np.all(d["csm"] == 1.0) and np.all(d["L_c"] > 0.0)

    def test_fixing_L_c_makes_every_event_a_csm_event_unless_csm_is_fixed(self):
        d = {k: _cgs_value(v) for k, v in TypeIIPSED(L_c=1e43 * ERG_S).sample_parameters(size=500, rng=1).items()}
        assert np.all(d["csm"] == 1.0) and np.all(d["L_c"] == 1e43)
        d = {
            k: _cgs_value(v)
            for k, v in TypeIIPSED(L_c=1e43 * ERG_S, csm=0.0).sample_parameters(size=500, rng=1).items()
        }
        assert np.all(d["csm"] == 0.0) and np.all(d["L_c"] == 0.0)

    def test_a_fixed_parameter_has_its_value_in_both_classes(self):
        for name, value, cgs in (("t_r", 2.0 * u.day, 2.0 * 86400.0), ("R0", 5e13 * u.cm, 5e13)):
            d = {k: _cgs_value(v) for k, v in TypeIIPSED(**{name: value}).sample_parameters(size=500, rng=1).items()}
            np.testing.assert_allclose(d[name], cgs)
            assert 0.0 < d["csm"].mean() < 1.0

    def test_a_replaced_prior_that_is_not_a_normal_is_drawn_without_the_trend(self):
        replacement = Parameter(prior=UniformPrior(lower=-2.0, upper=-1.0), scale=1 * u.Msun, transform="log10")
        d = {k: _cgs_value(v) for k, v in TypeIIPSED(M_Ni=replacement).sample_parameters(size=4000, rng=2).items()}
        log_m = _log10(d["M_Ni"] / 1.98841e33)
        assert log_m.min() >= -2.0 and log_m.max() <= -1.0
        assert abs(np.corrcoef(log_m, _log10(d["L50"]))[0, 1]) < 0.06

    def test_a_replaced_truncated_normal_keeps_the_trend(self):
        from uvex_transients.models.core.priors import TruncatedNormalPrior

        replacement = Parameter(
            prior=TruncatedNormalPrior(mean=-1.0, sigma=0.1, lower=-3.0, upper=0.0), scale=1 * u.Msun, transform="log10"
        )
        d = {k: _cgs_value(v) for k, v in TypeIIPSED(M_Ni=replacement).sample_parameters(size=4000, rng=2).items()}
        slope = np.polyfit(_log10(d["L50"]) - 42.0, _log10(d["M_Ni"] / 1.98841e33), 1)[0]
        assert slope == pytest.approx(1.0, abs=0.1)

    def test_a_subset_of_parameters_can_be_requested(self):
        draws = TypeIIPSED().sample_parameters(size=5, rng=0, parameters=["L50", "M_Ni"])
        assert set(draws) == {"L50", "M_Ni"}
        assert all(len(v) == 5 for v in draws.values())


# =========================================================================== #
# Reproducibility                                                             #
# =========================================================================== #
class TestSeeding:
    """An event's parameters follow from its seed alone, however they are asked for."""

    def test_the_per_event_and_catalog_paths_draw_the_same_parameters(self):
        model = TypeIIPSED()
        seeds = np.array([5, 17, 123456, 2**31 + 7])
        from_seeds = _sample_parameters_from_seeds(model, seeds)
        for i, seed in enumerate(seeds):
            event = model.sample_parameters(size=1, rng=np.random.default_rng(int(seed)))
            for name in event:
                np.testing.assert_array_equal(_cgs_value(event[name])[0], _cgs_value(from_seeds[name])[i])

    def test_the_seed_fixes_the_event_class_and_every_parameter(self):
        first = TypeIIPSED().sample_parameters(size=200, rng=9)
        again = TypeIIPSED().sample_parameters(size=200, rng=9)
        other = TypeIIPSED().sample_parameters(size=200, rng=10)
        for name in first:
            np.testing.assert_array_equal(_cgs_value(first[name]), _cgs_value(again[name]))
        assert not np.array_equal(_cgs_value(first["L50"]), _cgs_value(other["L50"]))
