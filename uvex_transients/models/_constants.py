r"""
Shared CGS physical constants and unit-conversion factors for :mod:`uvex_transients.models`.

Every value here is derived once, at import time, from :mod:`astropy.constants` /
:mod:`astropy.units` rather than hardcoded -- the single source of truth for a
constant reused across multiple model files (e.g. the Planck function's ``h``,
``c``, ``k_B`` appearing in both :mod:`uvex_transients.models.spectra.thermal` and
:mod:`uvex_transients.models.supernovae.IIb`). ``LOG_``-prefixed siblings are
precomputed natural logs, matching the rest of the codebase's unit-free
natural-log CGS convention, since several of these sit on hot per-event SED
evaluation paths.
"""

import numpy as np
from astropy import constants as const
from astropy import units as u

__all__ = [
    "H_CGS",
    "LOG_H_CGS",
    "C_CGS",
    "LOG_C_CGS",
    "K_B_CGS",
    "LOG_K_B_CGS",
    "SIGMA_SB_CGS",
    "LOG_SIGMA_SB_CGS",
    "AB_MAG_ZERO_POINT",
    "MSUN_G",
    "SECONDS_PER_DAY",
    "SECONDS_PER_HOUR",
    "EV_TO_ERG",
    "LOG_EV_TO_ERG",
    "KELVIN_PER_EV",
    "LOG_KELVIN_PER_EV",
]

# ------------------------------------------ #
# Fundamental Physical Constants (cgs)       #
# ------------------------------------------ #
H_CGS: float = const.h.cgs.value
"""float: The Planck constant, in erg s."""
LOG_H_CGS: float = np.log(H_CGS)

C_CGS: float = const.c.cgs.value
"""float: The speed of light, in cm/s."""
LOG_C_CGS: float = np.log(C_CGS)

K_B_CGS: float = const.k_B.cgs.value
"""float: The Boltzmann constant, in erg/K."""
LOG_K_B_CGS: float = np.log(K_B_CGS)

SIGMA_SB_CGS: float = const.sigma_sb.cgs.value
"""float: The Stefan-Boltzmann constant, in erg/s/cm^2/K^4."""
LOG_SIGMA_SB_CGS: float = np.log(SIGMA_SB_CGS)

# ------------------------------------------ #
# Unit-Conversion Factors                    #
# ------------------------------------------ #
AB_MAG_ZERO_POINT: float = 3631e-23
"""float: The CGS zero-point flux density in the AB mag system, in erg/s/cm^2/Hz."""

MSUN_G: float = (1.0 * u.Msun).to_value(u.g)
"""float: One solar mass, in grams."""

SECONDS_PER_DAY: float = (1.0 * u.day).to_value(u.s)
SECONDS_PER_HOUR: float = (1.0 * u.hr).to_value(u.s)

EV_TO_ERG: float = (1.0 * u.eV).to_value(u.erg)
"""float: Conversion factor from electronvolts to ergs."""
LOG_EV_TO_ERG: float = np.log(EV_TO_ERG)

KELVIN_PER_EV: float = (1.0 * u.eV / const.k_B).to_value(u.K)
"""float: Kelvin per electronvolt, i.e. :math:`1/k_B` in eV/K -- multiply a temperature in eV by
this to get Kelvin."""
LOG_KELVIN_PER_EV: float = np.log(KELVIN_PER_EV)

# ------------------------------------------ #
# Decay Efficiencies                         #
# ------------------------------------------ #
nickel_decay_yield: u.Quantity = 3.9e10 * u.Unit("erg g^-1 s^-1")
""" ~astropy.units.Quantity: The yield from decay of Nickel-56.

Taken from :footcite:t:`
"""
cobalt_decay_yield: u.Quantity = 6.78e9 * u.Unit("erg g^-1 s^-1")
nickel_decay_time: u.Quantity = 8.8 * u.day
cobalt_decay_time: u.Quantity = 113.6 * u.day
