"""Kilonova (binary neutron star merger) population."""

from typing import Union

import numpy as np
from astropy import units as u
from astropy.units import Quantity
from numpy.typing import NDArray

from uvex_transients.models.kilonovae import KilonovaCoolingBlackbodySED

from .base import ExtragalacticTransient

# Total BNS merger rate (all mass bins, not just the GW170817-like ~1.3+1.3 Msun
# bin), Fishbach et al. 2026: 110 (+192/-82) Gpc^-3 yr^-1. Taken as constant with
# redshift. Using the total BNS rate here means every BNS merger is assumed to
# produce a feasibly GW170817-like kilonova, an assumption made for this
# simulation's convenience and not one asserted by Fishbach et al. 2026 itself
# (whose GW170817-like sub-rate is a strict subset of this total; see their Fig. 2).
_KNE_RATE: Quantity = 110 / (u.Gpc**3 * u.yr)

# The 90% CI endpoints implied by the quoted +192/-82 Gpc^-3 yr^-1, expressed as
# multiplicative factors on `_KNE_RATE` (see `ExtragalacticTransient.RATE_CI`).
_KNE_RATE_CI: tuple[float, float] = ((110 - 82) / 110, (110 + 192) / 110)


class Kilonova(ExtragalacticTransient):
    r"""
    GW170817-like kilonova population.

    Modeled with `KilonovaCoolingBlackbodySED` (a Gaussian-rise/broken-power-law-decline
    light curve with a cooling blackbody photosphere, calibrated against GW170817), a
    constant volumetric rate of :math:`110\ \mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}` out to
    :math:`z=0.2`, and a 30-day duration window. This rate is Fishbach et al. 2026's total
    BNS merger rate, not just their GW170817-like mass-bin sub-rate, so every BNS merger
    sampled here is assumed to be feasibly GW170817-like: a simplifying assumption of this
    simulation, not a claim made by that paper.

    The :math:`z=0.2` redshift limit is conservative: a source with bolometric luminosity
    below :math:`10^{43}\ \mathrm{erg\,s^{-1}}` (well above anything reported in the
    literature) would still remain detectable at UVEX's :math:`m<27` band limit out to
    :math:`z=2`, assuming a flat spectrum and no K-correction. The 30-day duration is a
    conservative upper bound on the total light curve; the blue/early kilonova component
    this SED targets fades below detectability closer to :math:`\sim10` days.

    See Also
    --------
    uvex_transients.models.kilonovae.kne.KilonovaCoolingBlackbodySED : The SED this class pairs with `DEFAULT_MODEL`.
    uvex_transients.transients.base.ExtragalacticTransient : The base class supplying rate/sampling machinery.
    """

    DEFAULT_MODEL = KilonovaCoolingBlackbodySED
    DEFAULT_DURATION = 30 * u.day
    DEFAULT_Z_LIM = 0.2

    RATE_CI = _KNE_RATE_CI

    @property
    def rate(self) -> Quantity:
        r"""
        ~astropy.units.Quantity: The volumetric kilonova rate, constant in :math:`z`.

        This is Fishbach et al. 2026's total BNS merger rate, taken as a proxy for the
        GW170817-like kilonova rate under the assumption that every BNS merger is
        feasibly GW170817-like (see module-level comment).

        See Also
        --------
        RATE_CI : The confidence bounds attached to this point estimate.
        rate_shape : The redshift dependence this rate normalizes.
        """
        return _KNE_RATE

    def rate_shape(self, z: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
        r"""
        Return the (trivial, constant) rate shape of kilonovae at a given redshift.

        Parameters
        ----------
        z : float or array-like
            Redshift(s) at which to evaluate the rate shape.

        Returns
        -------
        float or array-like
            Ones, since the kilonova rate is constant in :math:`z` (Fishbach et al. 2026).

        See Also
        --------
        rate : The (constant-in-:math:`z`) normalization this shape multiplies.
        """
        z = np.asarray(z)
        shape = np.ones_like(z, dtype=np.float64)
        return shape if z.ndim > 0 else shape.item()  # Return scalar if input was scalar.
