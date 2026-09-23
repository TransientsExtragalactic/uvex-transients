"""Tidal disruption event population."""

from typing import Union

import numpy as np
from astropy import units as u
from astropy.units import Quantity
from numpy.typing import NDArray

from uvex_transients.models.tdes import VanVelzenTDESED

from .base import ExtragalacticTransient

# Yao et al. 2023 value, taken as constant with redshift: 3.1e-7 Mpc^-3 yr^-1. No published
# uncertainty is adopted here yet, so `RATE_CI` is left at its default (unset).
_TDE_RATE: Quantity = 3.1e-7 / (u.Mpc**3 * u.yr)


class TidalDisruptionEvent(ExtragalacticTransient):
    """
    Optical/UV tidal disruption event (TDE) population.

    Modeled with `VanVelzenTDESED` (a Gaussian-rise/exponential-decay light curve
    with a constant-temperature blackbody photosphere, calibrated against the ZTF
    TDE sample of Van Velzen et al. 2021), a constant volumetric rate of
    3.1e-7 Mpc^-3 yr^-1 out to z=2 (Yao et al. 2023), and a 200-day duration window.
    """

    DEFAULT_MODEL = VanVelzenTDESED
    DEFAULT_DURATION = 200 * u.day
    DEFAULT_Z_LIM = 2

    @property
    def rate(self) -> Quantity:
        """~astropy.units.Quantity: The volumetric TDE rate, constant in `z` (Yao et al. 2023)."""
        return _TDE_RATE

    def rate_shape(self, z: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
        """
        Return the (trivial, constant) rate shape of TDEs at a given redshift.

        Parameters
        ----------
        z : float or array-like
            Redshift(s) at which to evaluate the rate shape.

        Returns
        -------
        float or array-like
            Ones, since the TDE rate is constant in `z` (Yao et al. 2023).
        """
        z = np.asarray(z)
        shape = np.ones_like(z, dtype=np.float64)
        return shape if z.ndim > 0 else shape.item()  # Return scalar if input was scalar.
