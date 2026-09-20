"""Tests for :mod:`uvex_transients.models._constants`."""

import numpy as np
from astropy import constants as const
from astropy import units as u

from uvex_transients.models import _constants as c


def test_fundamental_constants_match_astropy():
    assert c.H_CGS == const.h.cgs.value
    assert c.C_CGS == const.c.cgs.value
    assert c.K_B_CGS == const.k_B.cgs.value
    assert c.SIGMA_SB_CGS == const.sigma_sb.cgs.value


def test_log_variants_are_logs_of_linear_values():
    for linear_name, log_name in [
        ("H_CGS", "LOG_H_CGS"),
        ("C_CGS", "LOG_C_CGS"),
        ("K_B_CGS", "LOG_K_B_CGS"),
        ("SIGMA_SB_CGS", "LOG_SIGMA_SB_CGS"),
        ("EV_TO_ERG", "LOG_EV_TO_ERG"),
        ("KELVIN_PER_EV", "LOG_KELVIN_PER_EV"),
    ]:
        assert np.log(getattr(c, linear_name)) == getattr(c, log_name)


def test_unit_conversion_factors():
    assert c.MSUN_G == (1.0 * u.Msun).to_value(u.g)
    assert c.SECONDS_PER_DAY == (1.0 * u.day).to_value(u.s)
    assert c.SECONDS_PER_HOUR == (1.0 * u.hr).to_value(u.s)
    assert c.EV_TO_ERG == (1.0 * u.eV).to_value(u.erg)
    assert c.KELVIN_PER_EV == (1.0 * u.eV / const.k_B).to_value(u.K)


def test_ab_mag_zero_point():
    assert c.AB_MAG_ZERO_POINT == 3631e-23
