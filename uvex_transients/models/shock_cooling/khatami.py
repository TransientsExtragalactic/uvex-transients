r"""Shock-breakout SED: the Khatami light curve with a luminosity-dependent blackbody photosphere."""

from typing import ClassVar

import numpy as np
from astropy import units as u

from uvex_transients.models._typing import CGSParameterValue, FloatArray
from uvex_transients.models._util_functions import planck_shape_log_cgs
from uvex_transients.models._utils import to_cgs_value
from uvex_transients.models.core.base import SpectralModel
from uvex_transients.models.core.parameters import Parameter
from uvex_transients.models.core.priors import ConstantPrior
from uvex_transients.models.lightcurves.generic import KhatamiSBOLightcurve

__all__ = ["KhatamiSED"]

_DAY = u.day.to(u.s)
_L_REF = 1e44
"""float: The luminosity (erg/s) at which the temperature law is normalized."""


class KhatamiSED(SpectralModel):
    r"""
    The :class:`~uvex_transients.models.lightcurves.generic.KhatamiSBOLightcurve` with a blackbody photosphere.

    The SED pairs the bolometric light curve with a time-dependent blackbody,

    .. math::

        L_\nu(\nu, t) = L_\mathrm{bol}(t)\, S_\mathrm{BB}\!\left[\nu, T(t)\right],

    where :math:`S_\mathrm{BB}` is the normalized blackbody shape of
    :class:`~uvex_transients.models.spectra.thermal.BlackbodySpectrum`, and the temperature is set by the
    instantaneous luminosity and the time since breakout,

    .. math::

        T(t) = T_0 \left[\frac{L_\mathrm{bol}(t)}{10^{44}\ \mathrm{erg\,s^{-1}}}\right]^{a}
        \left(\frac{t}{1\ \mathrm{day}}\right)^{-b}.

    Since :math:`L_\mathrm{bol}(t)` goes to zero faster than any power law as :math:`t \to 0` (the early-time cutoff
    of the light curve), so does :math:`T(t)`. By default :math:`T_0 = 112` kK, :math:`a = 0.5` and :math:`b = 0.5`.

    The shape of the light curve, and so of :math:`T(t)`, is set by the two light curve parameters
    :math:`(L_\mathrm{pk}, \bar{t})`. The three temperature parameters are fixed by default (a
    :class:`~uvex_transients.models.core.priors.ConstantPrior`), so that a population varies only through those two.
    They can be overridden like any other parameter.

    The bolometric luminosity is exact, with no integration over frequency, because the blackbody shape is normalized
    independently of the temperature.

    .. rubric:: Parameters

    The model parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``peak_luminosity``
         - :math:`L_\mathrm{pk}`
         - Peak bolometric luminosity.
       * - ``t_bar``
         - :math:`\bar{t}`
         - Luminosity-weighted duration, which sets the shape of the light curve.
       * - ``T0``
         - :math:`T_0`
         - Temperature at :math:`L_\mathrm{bol} = 10^{44}` erg/s and :math:`t = 1` day.
       * - ``L_index``
         - :math:`a`
         - Power-law index of the temperature in the luminosity.
       * - ``t_index``
         - :math:`b`
         - Power-law index of the temperature decline in time.

    See Also
    --------
    uvex_transients.models.lightcurves.generic.KhatamiSBOLightcurve :
        The bolometric light curve.
    """

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        **KhatamiSBOLightcurve._DEFAULT_PARAMETERS,
        "T0": Parameter(
            prior=ConstantPrior(value=1.12),
            scale=1e5 * u.K,
            description="Temperature at a luminosity of 1e44 erg/s and a time of 1 day. Fixed at 112 kK.",
            latex=r"T_0",
        ),
        "L_index": Parameter(
            prior=ConstantPrior(value=0.5),
            scale=1.0 * u.dimensionless_unscaled,
            description="Power-law index of the temperature in the luminosity. Fixed at 0.5.",
            latex=r"a",
        ),
        "t_index": Parameter(
            prior=ConstantPrior(value=0.5),
            scale=1.0 * u.dimensionless_unscaled,
            description="Power-law index of the temperature decline in time. Fixed at 0.5.",
            latex=r"b",
        ),
    }

    @classmethod
    def _temperature_cgs(cls, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        :math:`T(t) = T_0\,(L_\mathrm{bol}/10^{44})^{a}\,(t/\mathrm{day})^{-b}`.

        Parameters
        ----------
        t : numpy.ndarray
            Time since breakout, in seconds.
        **parameters
            This model's parameter values, in cgs units.

        Returns
        -------
        numpy.ndarray
            :math:`T(t)` in Kelvin, which is exactly zero where the luminosity is zero (including :math:`t \le 0`).
        """
        log_luminosity = cls._eval_bolometric(t, **parameters)

        with np.errstate(divide="ignore", invalid="ignore"):
            log_temperature = (
                np.log(parameters["T0"])
                + parameters["L_index"] * (log_luminosity - np.log(_L_REF))
                - parameters["t_index"] * np.log(t / _DAY)
            )

        # L = 0 (t -> 0) sends this to -inf, and t = 0 exactly to nan (-inf + inf): both mean T = 0.
        return np.where(np.isfinite(log_temperature), np.exp(log_temperature), 0.0)

    @classmethod
    def temperature(cls, t: u.Quantity, **parameters: CGSParameterValue) -> u.Quantity:
        r"""
        Photospheric temperature :math:`T(t)`.

        Parameters
        ----------
        t : ~astropy.units.Quantity
            Time since breakout.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`T(t)`, in Kelvin.
        """
        cgs_parameters = {name: np.asarray(to_cgs_value(value), dtype=np.float64) for name, value in parameters.items()}
        return cls._temperature_cgs(np.asarray(t.cgs.value, dtype=np.float64), **cgs_parameters) * u.K

    @classmethod
    def _eval_bolometric(cls, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""Natural log of :math:`L_\mathrm{bol}(t)` in erg/s, delegated to :class:`KhatamiSBOLightcurve`."""
        lightcurve_parameters = {name: parameters[name] for name in KhatamiSBOLightcurve._DEFAULT_PARAMETERS}
        return KhatamiSBOLightcurve._eval(t, **lightcurve_parameters)

    @classmethod
    def _eval_spectrum(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""Natural log of the normalized blackbody shape :math:`S(\nu, T(t))` in 1/Hz."""
        # T is exactly 0 where L is, which would give a nan shape. The 1 K floor keeps it finite there, and changes
        # nothing since L_bol is already zero wherever it applies.
        temperature = np.maximum(cls._temperature_cgs(t, **parameters), 1.0)
        return planck_shape_log_cgs(nu, temperature)

    @classmethod
    def _eval(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""Natural log of :math:`L_\nu(\nu, t) = L_\mathrm{bol}(t)\,S(\nu, T(t))` in erg/s/Hz."""
        return cls._eval_bolometric(t, **parameters) + cls._eval_spectrum(nu, t, **parameters)
