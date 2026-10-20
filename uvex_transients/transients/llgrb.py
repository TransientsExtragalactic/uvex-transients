"""Low-luminosity gamma-ray burst (llGRB) population."""

from typing import Union

import numpy as np
from astropy import units as u
from astropy.units import Quantity
from numpy.typing import NDArray

from uvex_transients.models.supernovae import LLGRBSED

from .base import ExtragalacticTransient

__all__ = ["LowLuminosityGRB"]

# Local rate of sub-energetic GRBs, Soderberg et al. 2006 (Nature 442, 1014): 230 (+490/-190)
# Gpc^-3 yr^-1, a 90% confidence range. It comes from the two sub-energetic events seen by the
# localizing GRB missions (GRB 980425 and XRF 060218) and the volume each instrument could have
# detected them in, so the errors are dominated by Poisson statistics for two detections. It is
# the true rate, since the paper argues such events cannot be strongly beamed, so no beaming
# correction is applied here. It is a local (d < ~200 Mpc) measurement with no constraint on
# redshift evolution, and is taken as constant with redshift.
_LLGRB_RATE: Quantity = 230 / (u.Gpc**3 * u.yr)

# The 90% CI endpoints implied by the quoted +490/-190 Gpc^-3 yr^-1, expressed as
# multiplicative factors on `_LLGRB_RATE` (see `ExtragalacticTransient.RATE_CI`).
_LLGRB_RATE_CI: tuple[float, float] = ((230 - 190) / 230, (230 + 490) / 230)


class LowLuminosityGRB(ExtragalacticTransient):
    r"""
    Low-luminosity GRB population, e.g. GRB 060218 / SN 2006aj-like events.

    Modeled with `LLGRBSED` (an Arnett-powered Ic-BL supernova plus an early thermal excess with a
    power-law cooling photosphere), a constant volumetric rate of
    :math:`230\ \mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}` (Soderberg et al. 2006) out to :math:`z=2.41`, and a
    30-day duration window.

    The rate is the local rate of sub-energetic GRBs from two detected events, with a 90% confidence
    range of :math:`40`-:math:`720\ \mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}`. It is a local measurement, and
    is taken to be constant with redshift because nothing constrains its evolution.

    The :math:`z=2.41` redshift limit is the redshift beyond which, with 95% confidence, at most 1% of
    llGRBs drawn from the SED's priors are brighter than :math:`m=24.5` AB in either UVEX band (see
    `ExtragalacticTransient.get_detection_horizon`), rounded up. It is set by the bright tail of the priors on the early
    excess, which are broad and calibrated to a single event, and the typical draw is detectable only to
    :math:`z\sim1`. The 30-day duration is generous: the early excess is bright for about a day, and
    fewer than 1% of events (weighted by volume) have any epoch brighter than 24.5 AB beyond 30 days.

    See Also
    --------
    uvex_transients.models.supernovae.llgrb.LLGRBSED : The SED this class pairs with `DEFAULT_MODEL`.
    uvex_transients.transients.base.ExtragalacticTransient : The base class supplying rate/sampling machinery.
    """

    DEFAULT_MODEL = LLGRBSED
    DEFAULT_DURATION = 30 * u.day
    # Redshift beyond which, with 95% confidence, at most 1% of draws from the SED's priors are brighter than
    # 24.5 AB in either UVEX band (`ExtragalacticTransient.get_detection_horizon`), rounded up.
    DEFAULT_Z_LIM = 2.41
    RATE_CI = _LLGRB_RATE_CI

    @property
    def rate(self) -> Quantity:
        """
        ~astropy.units.Quantity: The volumetric llGRB rate, constant in `z` (Soderberg et al. 2006).

        See Also
        --------
        RATE_CI : The confidence bounds attached to this point estimate.
        rate_shape : The redshift dependence this rate normalizes.
        """
        return _LLGRB_RATE

    def rate_shape(self, z: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
        """
        Return the (trivial, constant) rate shape of llGRBs at a given redshift.

        Parameters
        ----------
        z : float or array-like
            Redshift(s) at which to evaluate the rate shape.

        Returns
        -------
        float or array-like
            Ones, since the llGRB rate is taken to be constant in `z` (Soderberg et al. 2006).

        See Also
        --------
        rate : The (constant-in-`z`) normalization this shape multiplies.
        """
        z = np.asarray(z)
        shape = np.ones_like(z, dtype=np.float64)
        return shape if z.ndim > 0 else shape.item()  # Return scalar if input was scalar.
