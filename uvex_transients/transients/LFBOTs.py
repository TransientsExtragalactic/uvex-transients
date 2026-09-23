"""Luminous fast blue optical transient (LFBOT) population."""

from typing import Union

import numpy as np
from astropy import units as u
from astropy.units import Quantity
from numpy.typing import NDArray

from uvex_transients.models.lfbots import LFBOTCoolingBlackbodySED

from .base import ExtragalacticTransient

# Constant (redshift-independent) volumetric rate, Perley et al. 2026 / Ho & Lu
# et al. 2026: 10 Gpc^-3 yr^-1. No published uncertainty is adopted here yet, so
# `RATE_CI` is left at its default (unset).
_LFBOT_RATE: Quantity = 10 / (u.Gpc**3 * u.yr)


class LuminousFastBlueOpticalTransient(ExtragalacticTransient):
    """
    Luminous fast blue optical transient (LFBOT) population, e.g. AT2018cow-like events.

    Modeled with `LFBOTCoolingBlackbodySED` (a Gaussian-rise/power-law-decline
    light curve with a smoothly cooling blackbody photosphere), a constant
    volumetric rate of 10 Gpc^-3 yr^-1 out to z=3 (Perley et al. 2026; Ho & Lu
    et al. 2026), and a 100-day duration window -- generous relative to the
    SED's own rise/decline timescales, to safely bound the slowly fading
    power-law tail.
    """

    DEFAULT_MODEL = LFBOTCoolingBlackbodySED
    DEFAULT_DURATION = 100 * u.day
    DEFAULT_Z_LIM = 3

    @property
    def rate(self) -> Quantity:
        """~astropy.units.Quantity: The volumetric LFBOT rate, constant in `z` (Perley/Ho & Lu et al. 2026)."""
        return _LFBOT_RATE

    def rate_shape(self, z: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
        """
        Return the (trivial, constant) rate shape of LFBOTs at a given redshift.

        Parameters
        ----------
        z : float or array-like
            Redshift(s) at which to evaluate the rate shape.

        Returns
        -------
        float or array-like
            Ones, since the LFBOT rate is constant in `z` (Perley et al. 2026; Ho & Lu et al. 2026).
        """
        z = np.asarray(z)
        shape = np.ones_like(z, dtype=np.float64)
        return shape if z.ndim > 0 else shape.item()  # Return scalar if input was scalar.
