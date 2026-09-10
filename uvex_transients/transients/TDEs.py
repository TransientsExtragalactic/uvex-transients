"""Tidal disruption event population."""

from typing import Union

import numpy as np
from astropy import units as u
from astropy.units import Quantity
from numpy.typing import NDArray

from uvex_transients.models.tdes import AlushStoneTDESED

from .base import ExtragalacticTransient

# Yao et al. 2023 value, taken as constant with redshift: 3.1e-7 Mpc^-3 yr^-1.
_TDE_RATE: Quantity = 3.1e-7 / (u.Mpc**3 * u.yr)


class TidalDisruptionEvent(ExtragalacticTransient):
    """
    Optical/UV tidal disruption event (TDE) population.

    Modeled with `VanVelzenTDESED` (a Gaussian-rise/exponential-decay light curve
    with a constant-temperature blackbody photosphere, calibrated against the ZTF
    TDE sample of Van Velzen et al. 2021), a constant volumetric rate of
    3.1e-7 Mpc^-3 yr^-1 out to z=1 (Yao et al. 2023), and a 200-day duration window.
    """

    DEFAULT_MODEL = AlushStoneTDESED
    DEFAULT_DURATION = 200 * u.day
    DEFAULT_Z_LIM = 1

    def event_rate(self, z: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
        """
        Return the volumetric event rate of tidal disruption events (TDEs) at a given redshift.

        Parameters
        ----------
        z : float or array-like
            Redshift(s) at which to evaluate the event rate.

        Returns
        -------
        float or array-like
            The volumetric event rate of TDEs at the specified redshift(s), in units of
            events per cubic megaparsec per year. Constant in `z` (Yao et al. 2023).
        """
        z = np.asarray(z)

        rate = np.full_like(z, _TDE_RATE.to_value(u.Mpc**-3 * u.yr**-1), dtype=np.float64)

        return rate if z.ndim > 0 else rate.item()  # Return scalar if input was scalar.
