"""Luminous fast blue optical transient (LFBOT) population."""

from typing import Union

import numpy as np
from astropy import units as u
from astropy.units import Quantity
from numpy.typing import NDArray

from uvex_transients.models.lfbots import LFBOTCoolingBlackbodySED

from .base import ExtragalacticTransient

# Constant (redshift-independent) volumetric rate, Perley et al. 2026 / Ho & Lu
# et al. 2026: 10 Gpc^-3 yr^-1.
_LFBOT_RATE: Quantity = 10 / (u.Gpc**3 * u.yr)

# Adopted bounds are Perley et al. 2026's revised BTS-derived range, 0.9-12.5 Gpc^-3
# yr^-1 (their more heavily-sampled revision of Ho et al. 2023's own BTS estimate,
# 0.31-85.3 Gpc^-3 yr^-1, preferred here over Ho et al. 2023's CLU-derived range,
# 2.7-546 Gpc^-3 yr^-1, on the same grounds -- see the "Rate uncertainty" note on the
# LFBOTs documentation page for the full survey-selection discussion). Expressed as
# multiplicative factors on `_LFBOT_RATE` (see `ExtragalacticTransient.RATE_CI`).
_LFBOT_RATE_CI: tuple[float, float] = (0.9 / 10, 12.5 / 10)


class LuminousFastBlueOpticalTransient(ExtragalacticTransient):
    """
    Luminous fast blue optical transient (LFBOT) population, e.g. AT2018cow-like events.

    Modeled with `LFBOTCoolingBlackbodySED` (a Gaussian-rise/power-law-decline
    light curve with a smoothly cooling blackbody photosphere), a constant
    volumetric rate of 10 Gpc^-3 yr^-1 out to z=3 (Perley et al. 2026; Ho & Lu
    et al. 2026), and a 100-day duration window -- generous relative to the
    SED's own rise/decline timescales, to safely bound the slowly fading
    power-law tail.

    See Also
    --------
    uvex_transients.models.lfbots.lfbots.LFBOTCoolingBlackbodySED : The SED this class pairs with `DEFAULT_MODEL`.
    uvex_transients.transients.base.ExtragalacticTransient : The base class supplying rate/sampling machinery.
    """

    DEFAULT_MODEL = LFBOTCoolingBlackbodySED
    DEFAULT_DURATION = 100 * u.day
    DEFAULT_Z_LIM = 3
    RATE_CI = _LFBOT_RATE_CI

    @property
    def rate(self) -> Quantity:
        """
        ~astropy.units.Quantity: The volumetric LFBOT rate, constant in `z` (Perley/Ho & Lu et al. 2026).

        See Also
        --------
        RATE_CI : The confidence bounds attached to this point estimate.
        rate_shape : The redshift dependence this rate normalizes.
        """
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

        See Also
        --------
        rate : The (constant-in-`z`) normalization this shape multiplies.
        """
        z = np.asarray(z)
        shape = np.ones_like(z, dtype=np.float64)
        return shape if z.ndim > 0 else shape.item()  # Return scalar if input was scalar.
