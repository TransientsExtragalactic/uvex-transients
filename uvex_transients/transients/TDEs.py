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


# ======================================= #
# Utility Functions                       #
# ======================================= #
def luminosity_function_yao23(log_L: np.ndarray) -> np.ndarray:
    r"""
    Evaluate the TDE blackbody luminosity function of :footcite:t:`yao2023`.

    The luminosity function :math:`\phi(L_\mathrm{bb})` is the volumetric rate per unit logarithmic
    (base 10) interval in the peak blackbody luminosity, a power law of the form

    .. math::

        \phi(L_\mathrm{bb}) = R \left(\frac{L_\mathrm{bb}}{L_0}\right)^{-\alpha},

    with :math:`R = 9.43\times10^{-7}\ \mathrm{Mpc^{-3}\,yr^{-1}\,dex^{-1}}`,
    :math:`L_0 = 10^{43}\ \mathrm{erg\,s^{-1}}`, and :math:`\alpha = 1.41`.

    Parameters
    ----------
    log_L : ~numpy.ndarray
        Natural logarithm of the blackbody luminosity :math:`L_\mathrm{bb}` in CGS units
        (:math:`\ln(L_\mathrm{bb}/\mathrm{erg\,s^{-1}})`). A base-10 logarithm will silently give
        wrong values, since :math:`\ln L_0 = 99.0112` is used internally.

    Returns
    -------
    ~numpy.ndarray
        The differential rate :math:`\phi(L_\mathrm{bb})` in units of
        :math:`\mathrm{Mpc^{-3}\,yr^{-1}\,dex^{-1}}`, with the same shape as `log_L`.

    Notes
    -----
    Because :math:`\alpha > 0`, this power law diverges toward low luminosity, so it cannot be
    integrated to a total rate without a lower luminosity cutoff. Above
    :math:`L_\mathrm{min} = 10^{43}\ \mathrm{erg\,s^{-1}}` the integral is
    :math:`R / (\alpha \ln 10) \approx 2.9\times10^{-7}\ \mathrm{Mpc^{-3}\,yr^{-1}}`, close to the
    quoted total TDE rate. The default TDE SEDs use this function, normalized to unit area above
    :math:`L_\mathrm{min}`, as their peak-luminosity prior (a power law of index
    :math:`\alpha + 1` per unit :math:`L`, since this function is per dex).
    """
    log_A = -13.87420
    alpha = 1.41
    log_x = log_L - 99.0112

    return np.exp(log_A - alpha * log_x)


class TidalDisruptionEvent(ExtragalacticTransient):
    r"""
    Optical/UV tidal disruption event (TDE) population.

    Modeled with `VanVelzenTDESED` (a Gaussian-rise/exponential-decay light curve with a
    constant-temperature blackbody photosphere, calibrated against the ZTF TDE sample of
    Van Velzen et al. 2021), a constant volumetric rate of
    :math:`3.1\times10^{-7}\ \mathrm{Mpc}^{-3}\,\mathrm{yr}^{-1}` out to :math:`z=2` (Yao et al.
    2023), and a 200-day duration window. The SED's peak-luminosity prior is the Yao et al. 2023
    luminosity function (see `luminosity_function_yao23`), normalized explicitly to unit area above
    :math:`10^{43}\ \mathrm{erg\,s^{-1}}`; the overall event count comes from the rate alone.

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
