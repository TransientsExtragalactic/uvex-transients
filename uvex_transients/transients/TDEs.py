"""Tidal disruption event population."""

from typing import Union

import numpy as np
from astropy import units as u
from astropy.units import Quantity
from numpy.typing import NDArray

from uvex_transients.models.tdes import VanVelzenTDESED

from .base import ExtragalacticTransient

# Yao et al. 2023 value, taken as constant with redshift: 3.1e-7 Mpc^-3 yr^-1.
_TDE_RATE: Quantity = 3.1e-7 / (u.Mpc**3 * u.yr)

# The 90% CI endpoints implied by the quoted +0.6/-1.0 (e-7 Mpc^-3 yr^-1), expressed as
# multiplicative factors on `_TDE_RATE` (see `ExtragalacticTransient.RATE_CI`).
_TDE_RATE_CI: tuple[float, float] = ((3.1e-7 - 1.0e-7) / 3.1e-7, (3.1e-7 + 0.6e-7) / 3.1e-7)


class TidalDisruptionEvent(ExtragalacticTransient):
    r"""
    Optical/UV tidal disruption event (TDE) population.

    Modeled with `VanVelzenTDESED` (a Gaussian-rise/exponential-decay light curve with a
    constant-temperature blackbody photosphere, calibrated against the ZTF TDE sample of
    Van Velzen et al. 2021), a constant volumetric rate of
    :math:`3.1\times10^{-7}\ \mathrm{Mpc}^{-3}\,\mathrm{yr}^{-1}` out to :math:`z=2` (Yao et al.
    2023), and a 200-day duration window.

    See Also
    --------
    uvex_transients.models.tdes.van_velzen.VanVelzenTDESED : The SED this class pairs with `DEFAULT_MODEL`.
    uvex_transients.transients.base.ExtragalacticTransient : The base class supplying rate/sampling machinery.
    """

    DEFAULT_MODEL = VanVelzenTDESED
    DEFAULT_DURATION = 200 * u.day
    DEFAULT_Z_LIM = 2
    RATE_CI = _TDE_RATE_CI

    @property
    def rate(self) -> Quantity:
        r"""
        ~astropy.units.Quantity: The volumetric TDE rate, constant in :math:`z` (Yao et al. 2023).

        See Also
        --------
        RATE_CI : The confidence bounds attached to this point estimate.
        rate_shape : The redshift dependence this rate normalizes.
        """
        return _TDE_RATE

    def rate_shape(self, z: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
        r"""
        Return the (trivial, constant) rate shape of TDEs at a given redshift.

        Parameters
        ----------
        z : float or array-like
            Redshift(s) at which to evaluate the rate shape.

        Returns
        -------
        float or array-like
            Ones, since the TDE rate is constant in :math:`z` (Yao et al. 2023).

        See Also
        --------
        rate : The (constant-in-:math:`z`) normalization this shape multiplies.
        """
        z = np.asarray(z)
        shape = np.ones_like(z, dtype=np.float64)
        return shape if z.ndim > 0 else shape.item()  # Return scalar if input was scalar.
