"""Tests for `uvex_transients.dust`'s reddening-law resolution and callable binding."""

import numpy as np
import pytest
from astropy import units as u
from dust_extinction.parameter_averages import G23

from uvex_transients.dust import attenuation_callable, get_dust_law, log_attenuation
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
