"""Tests for `uvex_transients.dust`'s reddening-law resolution and callable binding."""

import numpy as np
import pytest
from astropy import units as u
from dust_extinction.parameter_averages import G23

from uvex_transients.dust import attenuation_callable, get_dust_law, host_log_attenuation_cgs, log_attenuation
from uvex_transients.utils import config


def test_get_dust_law_default_matches_config():
    """With no argument, `get_dust_law` resolves `config["physics.default_dust_law"]`."""
    law = get_dust_law()
    assert isinstance(law, G23)
    assert isinstance(get_dust_law(config["physics.default_dust_law"]), G23)


def test_get_dust_law_is_case_insensitive_and_cached():
    """Repeated lookups of the same name return the identical cached instance."""
    assert get_dust_law("g23") is get_dust_law("G23") is get_dust_law()


def test_get_dust_law_unknown_name_raises():
    """An unregistered name fails loudly, naming the laws that *are* registered."""
    with pytest.raises(KeyError, match="bogus"):
        get_dust_law("bogus")


def test_attenuation_callable_matches_direct_log_attenuation():
    """`attenuation_callable(ebv)(nu)` is exactly `log_attenuation(nu, ebv)`."""
    nu = np.linspace(1000, 3000, 5) * u.AA
    ebv = 0.2

    bound = attenuation_callable(ebv)
    np.testing.assert_array_equal(bound(nu), log_attenuation(nu, ebv))


def test_attenuation_callable_is_a_plain_bound_callable():
    """The whole point: callers get a ready-to-use callable, no `functools.partial` of their own."""
    bound = attenuation_callable(0.1)
    assert callable(bound)
    assert bound(1500 * u.AA).shape == ()


# ----------------------------------------------------------------------- #
# Host-galaxy (A_V, R_V) attenuation                                       #
# ----------------------------------------------------------------------- #
_C_CGS = 2.99792458e10


def _nu_hz(wavelength_um):
    return _C_CGS / (np.asarray(wavelength_um) * 1e-4)


@pytest.mark.parametrize("rv", [2.3, 2.7, 3.1, 4.4, 5.6])
def test_host_attenuation_matches_the_law_at_any_rv(rv):
    """The tabulated affine-in-1/R_V coefficients reproduce G23 at an R_V the table was not built at."""
    wavelength = np.geomspace(0.0913, 31.0, 400)
    expected = -0.4 * np.log(10.0) * 1.7 * np.asarray(G23(Rv=rv)(wavelength * u.micron))
    np.testing.assert_allclose(host_log_attenuation_cgs(_nu_hz(wavelength), 1.7, rv), expected, atol=1e-4)


def test_host_attenuation_agrees_with_the_line_of_sight_helper_at_the_default_rv():
    """A_V = R_V * E(B-V) at R_V = 3.1 is the Milky Way `log_attenuation`."""
    wavelength = np.geomspace(0.1, 5.0, 50)
    nu = _nu_hz(wavelength)
    np.testing.assert_allclose(host_log_attenuation_cgs(nu, 3.1 * 0.2, 3.1), log_attenuation(nu * u.Hz, 0.2), atol=1e-4)


def test_host_attenuation_broadcasts_and_vanishes_without_dust():
    """Per-draw A_V, R_V broadcast against nu; A_V = 0 is exactly no attenuation."""
    nu = _nu_hz([0.15, 0.55, 1.2])[np.newaxis, :]
    a_v = np.array([0.0, 0.5, 1.0])[:, np.newaxis]
    r_v = np.array([2.5, 3.0, 3.5])[:, np.newaxis]
    result = host_log_attenuation_cgs(nu, a_v, r_v)
    assert result.shape == (3, 3)
    np.testing.assert_array_equal(result[0], 0.0)
    assert np.all(result[1:] < 0.0)


def test_host_attenuation_holds_the_edge_value_outside_the_law():
    """Beyond 912 Angstrom and 32 microns the curve is flat at the edge, never NaN (so bolometric sums stay finite)."""
    inside = host_log_attenuation_cgs(_nu_hz([0.09121, 31.99]), 1.0, 3.1)
    outside = host_log_attenuation_cgs(_nu_hz([0.03, 100.0]), 1.0, 3.1)
    assert np.all(np.isfinite(outside))
    np.testing.assert_allclose(outside, inside, atol=2e-3)


def test_host_attenuation_unknown_law_raises():
    with pytest.raises(KeyError, match="bogus"):
        host_log_attenuation_cgs(_nu_hz([0.5]), 1.0, 3.1, dust_law="bogus")
