"""Kilonova (binary neutron star merger) population."""

from typing import Union

import numpy as np
from astropy import units as u
from astropy.units import Quantity
from numpy.typing import NDArray

from uvex_transients.models.kilonovae import KilonovaCoolingBlackbodySED

from .base import ExtragalacticTransient

# GW170817-like BNS merger rate, Fishbach et al. 2026: 53 (+176/-49) Gpc^-3 yr^-1.
# Taken as constant with redshift and as a conservative estimate of the rate of
# BNS mergers with plausible EM counterparts, relative to the ~28-300 Gpc^-3 yr^-1
# total BNS merger rate reported by the same work (not every merger is expected
# to produce a detectable kilonova).
_KNE_RATE: Quantity = 53 / (u.Gpc**3 * u.yr)

# The 90% CI endpoints implied by the quoted +176/-49 Gpc^-3 yr^-1, expressed as
# multiplicative factors on `_KNE_RATE` (see `ExtragalacticTransient.RATE_CI`).
_KNE_RATE_CI: tuple[float, float] = ((53 - 49) / 53, (53 + 176) / 53)


class Kilonova(ExtragalacticTransient):
    """
    GW170817-like kilonova population.

    Modeled with `KilonovaCoolingBlackbodySED` (a Gaussian-rise/broken-power-law-decline
    light curve with a cooling blackbody photosphere, calibrated against GW170817), a
    constant volumetric rate of 53 Gpc^-3 yr^-1 out to z=0.2, and a 30-day duration window.

    The z=0.2 redshift limit is conservative: a source with bolometric luminosity
    below 1e43 erg/s (well above anything reported in the literature) would still
    remain detectable at UVEX's m < 27 band limit out to z=2, assuming a flat
    spectrum and no K-correction. The 30-day duration is a conservative upper
    bound on the total light curve -- the blue/early kilonova component this SED
    targets fades below detectability closer to ~10 days.
    """

    DEFAULT_MODEL = KilonovaCoolingBlackbodySED
    DEFAULT_DURATION = 30 * u.day
    DEFAULT_Z_LIM = 0.2

    RATE_CI = _KNE_RATE_CI

    @property
    def rate(self) -> Quantity:
        """~astropy.units.Quantity: The volumetric kilonova rate, constant in `z` (Fishbach et al. 2026)."""
        return _KNE_RATE

    def rate_shape(self, z: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
        """
        Return the (trivial, constant) rate shape of kilonovae at a given redshift.

        Parameters
        ----------
        z : float or array-like
            Redshift(s) at which to evaluate the rate shape.

        Returns
        -------
        float or array-like
            Ones, since the kilonova rate is constant in `z` (Fishbach et al. 2026).
        """
        z = np.asarray(z)
        shape = np.ones_like(z, dtype=np.float64)
        return shape if z.ndim > 0 else shape.item()  # Return scalar if input was scalar.
