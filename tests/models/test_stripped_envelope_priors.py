"""Tests for the priors of the Type Ib and Ic SEDs: the log-normal ``M_Ni`` and the host dust."""

import numpy as np
import pytest
from astropy import units as u

from uvex_transients.models.host_dust import HostDustMixin
from uvex_transients.models.supernovae import TypeIbSED, TypeIcBLSED, TypeIcSED, TypeIIbSED
from uvex_transients.models.supernovae.Ibc import _TypeIbIntrinsicSED


def _draws(model_class, n: int = 100000, seed: int = 0) -> dict:
    """Draws from the priors, as `sample_parameters` returns them (``Quantity`` where a parameter has units)."""
    return model_class().sample_parameters(size=n, rng=seed)


@pytest.mark.parametrize("model_class", [TypeIbSED, TypeIcSED])
class TestTypeIbAndIc:
    def test_nickel_mass_is_a_truncated_log_normal(self, model_class):
        ni = _draws(model_class)["M_Ni"].to_value(u.Msun)
        assert ni.min() >= 0.005 * (1 - 1e-6) and ni.max() <= 1.5 * (1 + 1e-6)
        # Median 0.06 and 16 to 84% of 0.03 to 0.12, from sigma_ln = 0.7 (the truncation barely moves them)
        np.testing.assert_allclose(np.percentile(ni, [16, 50, 84]), [0.03, 0.06, 0.12], rtol=0.04)
        assert np.percentile(ni, 97.5) == pytest.approx(0.24, rel=0.05)

    def test_host_dust_follows_the_type_ia_construction(self, model_class):
        names = list(model_class._DEFAULT_PARAMETERS)
        assert names[-2:] == ["A_V", "R_V"]
        draws = _draws(model_class)
        assert draws["A_V"].mean() == pytest.approx(0.6, rel=0.03)
        assert draws["A_V"].min() >= 0.0
        # R_V is the mixin's default, shared with Type Ia
        assert model_class._DEFAULT_PARAMETERS["R_V"] is HostDustMixin.HOST_DUST_PARAMETERS["R_V"]

    def test_the_dust_dims_the_bolometric_luminosity(self, model_class):
        draws = _draws(model_class, n=5, seed=3)
        clean = model_class.eval_bolometric(20 * u.day, **{**draws, "A_V": np.zeros(5)})
        dusty = model_class.eval_bolometric(20 * u.day, **{**draws, "A_V": np.ones(5)})
        assert np.all(dusty < clean)


class TestTypeIIbHasNoHostDust:
    """`TypeIIbSED` takes the nickel priors of Type Ib, but not its dust, which would only dim half the SED."""

    def test_no_dust_parameters(self):
        assert not set(HostDustMixin._HOST_DUST_NAMES) & set(TypeIIbSED._DEFAULT_PARAMETERS)

    def test_nickel_component_is_the_dust_free_type_ib(self):
        assert TypeIIbSED._ARNETT_SED is _TypeIbIntrinsicSED
        assert not set(HostDustMixin._HOST_DUST_NAMES) & set(TypeIIbSED._ARNETT_PARAMETERS)


class TestTypeIcBL:
    """`TypeIcBLSED` is the Type Ic SED with the Ic-BL subsample's ``M_Ni``, ``M_ej`` and ``v_ej``."""

    OWN = ("M_Ni", "M_ej", "v_ej")

    def test_only_the_three_sample_priors_differ_from_type_ic(self):
        icbl, ic = TypeIcBLSED(), TypeIcSED()
        assert list(icbl) == list(ic)
        differ = {name for name in icbl if repr(icbl[name].prior) != repr(ic[name].prior)}
        assert differ == set(self.OWN)

    def test_quantiles_of_the_sample_priors(self):
        draws = _draws(TypeIcBLSED, n=200000)
        np.testing.assert_allclose(
            np.percentile(draws["M_Ni"].to_value(u.Msun), [16, 50, 84]), [0.093, 0.19, 0.29], rtol=0.02
        )
        np.testing.assert_allclose(
            np.percentile(draws["M_ej"].to_value(u.Msun), [16, 50, 84]), [1.17, 2.9, 4.86], rtol=0.03
        )
        np.testing.assert_allclose(
            np.percentile(draws["v_ej"].to_value(u.km / u.s) / 1e3, [16, 50, 84]), [13.3, 16.8, 20.3], rtol=0.01
        )
