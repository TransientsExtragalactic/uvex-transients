r"""Shock-breakout SED: the inverse-gamma broken power-law light curve with a blackbody on an expanding photosphere."""

from typing import ClassVar

import numpy as np
from astropy import units as u

from uvex_transients.models._constants import C_CGS, SIGMA_SB_CGS
from uvex_transients.models._typing import CGSParameterValue, FloatArray
from uvex_transients.models._util_functions import planck_shape_log_cgs
from uvex_transients.models._utils import to_cgs_value
from uvex_transients.models.core.base import SpectralModel
from uvex_transients.models.core.parameters import Parameter
from uvex_transients.models.core.priors import ConstantPrior
from uvex_transients.models.lightcurves.generic import InverseGammaBrokenPowerLawLightcurve

__all__ = ["InverseGammaBlackbodySED"]

_FOUR_PI_SIGMA_SB = 4.0 * np.pi * SIGMA_SB_CGS
_SECONDS_PER_DAY = 86400.0
#: The cap on the breakout radius in cm, at ``t_0`` of a day and ``L_0`` of `_L_BO_CAP`, scaling as t_0^(2/3) L_0^(1/3).
_R_BO_CAP_CGS = 8.8e13
#: The luminosity, in erg/s, at which the cap on the breakout radius is `_R_BO_CAP_CGS` (at a ``t_0`` of a day).
_L_BO_CAP = 1e44


class InverseGammaBlackbodySED(SpectralModel):
    r"""
    The :class:`~uvex_transients.models.lightcurves.generic.InverseGammaBrokenPowerLawLightcurve` with a blackbody.

    The SED pairs the bolometric light curve with a time-dependent blackbody,

    .. math::

        L_\nu(\nu, t) = L_\mathrm{bol}(t)\, S_\mathrm{BB}\!\left[\nu, T(t)\right],

    where :math:`S_\mathrm{BB}` is the normalized blackbody shape of
    :class:`~uvex_transients.models.spectra.thermal.BlackbodySpectrum`. The temperature follows from the
    luminosity and a photospheric radius that grows linearly in time from a breakout radius,

    .. math::

        T(t) = \left[\frac{L_\mathrm{bol}(t)}{4\pi\sigma_\mathrm{SB} R_\mathrm{ph}^2(t)}\right]^{1/4},
        \qquad
        R_\mathrm{ph}(t) = R_\mathrm{bo}\left(1 + \frac{x}{x_s}\right),

    with :math:`x = t/t_0`. The radius doubles at :math:`x = x_s`. The breakout radius is

    .. math::

        R_\mathrm{bo} = \min\left[c\,t_0,\;
        8.8\times10^{13}\,\mathrm{cm}\left(\frac{t_0}{1\,\mathrm{d}}\right)^{2/3}
        \left(\frac{L_0}{10^{44}\,\mathrm{erg\,s^{-1}}}\right)^{1/3}\right],

    so it is set by :math:`t_0` and :math:`L_0` and is not a separate parameter.

    .. note::

        :math:`R_\mathrm{bo}` is an *effective* radius, the size implied by the duration of the pulse. The first term
        is the distance light travels in :math:`t_0`, which is the size of a breakout that is light-crossing limited
        (the observed pulse lasts :math:`\max(R/c, t_\mathrm{diff})`, with :math:`t_\mathrm{diff}` the photon diffusion
        time through the breakout layer). It overestimates the radius when diffusion sets the duration, and without a
        cap the photosphere of a long event would be implausibly large and cold. The second term is that cap. It grows
        as :math:`t_0^{2/3} L_0^{1/3}`, more slowly than :math:`c\,t_0`, and is the smaller of the two for all but the
        shortest time scales: at :math:`L_0 = 10^{44}` erg/s it takes over at :math:`t_0 \approx 3` s, and at
        :math:`10^{46}` erg/s at about :math:`3\times10^2` s. The two terms use :math:`t_0`, not the time of the peak
        of the light curve, which differs from it by under 1% for the default shape.

    The temperature goes to zero with the luminosity as :math:`t \to 0`, and the radius does not vanish, so the
    emission is a blackbody of finite radius at every time. The bolometric luminosity is exact, with no integration
    over frequency, because the blackbody shape is normalized independently of the temperature.

    By default only :math:`L_0` and :math:`t_0` vary. The four shape parameters of the light curve are fixed
    (a :class:`~uvex_transients.models.core.priors.ConstantPrior`) at :math:`p = 4/3`, :math:`q = 0.3`,
    :math:`x_s = 10` and :math:`k = 2`, and can be overridden like any other parameter.

    .. rubric:: Parameters

    The model parameters are those of the light curve, and are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude``
         - :math:`L_0`
         - Peak bolometric luminosity.
       * - ``t_scale``
         - :math:`t_0`
         - Time scale; the dimensionless time is :math:`x = t / t_0`.
       * - ``p``
         - :math:`p`
         - Index of the rise-and-decline factor, and of the decline before the break. Fixed by default.
       * - ``q``
         - :math:`q`
         - Index of the late-time decline after the break. Fixed by default.
       * - ``x_break``
         - :math:`x_s`
         - Location of the break, in units of :math:`t_0`. Also sets the growth of the photosphere. Fixed by default.
       * - ``smoothness``
         - :math:`k`
         - Sharpness of the break. Fixed by default.

    See Also
    --------
    uvex_transients.models.lightcurves.generic.InverseGammaBrokenPowerLawLightcurve :
        The bolometric light curve.
    """

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        **InverseGammaBrokenPowerLawLightcurve._DEFAULT_PARAMETERS,
        "p": Parameter(
            prior=ConstantPrior(value=4.0 / 3.0),
            scale=1.0 * u.dimensionless_unscaled,
            description="Index of the rise-and-decline factor, and of the decline before the break. Fixed at 4/3.",
            latex=r"p",
        ),
        "q": Parameter(
            prior=ConstantPrior(value=0.3),
            scale=1.0 * u.dimensionless_unscaled,
            description="Index of the late-time decline after the break. Fixed at 0.3.",
            latex=r"q",
        ),
        "x_break": Parameter(
            prior=ConstantPrior(value=10.0),
            scale=1.0 * u.dimensionless_unscaled,
            description="Location of the break, in units of t_scale. Fixed at 10.",
            latex=r"x_s",
        ),
        "smoothness": Parameter(
            prior=ConstantPrior(value=2.0),
            scale=1.0 * u.dimensionless_unscaled,
            description="Sharpness of the break. Fixed at 2.",
            latex=r"k",
        ),
    }

    @classmethod
    def _photospheric_radius_cgs(cls, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        :math:`R_\mathrm{ph}(t) = R_\mathrm{bo}\,(1 + t / (t_0 x_s))`, with :math:`R_\mathrm{bo}` the breakout radius.

        Parameters
        ----------
        t : numpy.ndarray
            Time since breakout, in seconds.
        **parameters
            This model's parameter values, in cgs units.

        Returns
        -------
        numpy.ndarray
            :math:`R_\mathrm{ph}(t)` in cm.
        """
        t_scale = parameters["t_scale"]
        light_crossing = C_CGS * t_scale
        capped = (
            _R_BO_CAP_CGS
            * (t_scale / _SECONDS_PER_DAY) ** (2.0 / 3.0)
            * (parameters["amplitude"] / _L_BO_CAP) ** (1.0 / 3.0)
        )
        breakout_radius = np.minimum(light_crossing, capped)
        return breakout_radius * (1.0 + t / (t_scale * parameters["x_break"]))

    @classmethod
    def _temperature_cgs(cls, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        :math:`T(t) = [L_\mathrm{bol}/(4\pi\sigma_\mathrm{SB} R_\mathrm{ph}^2)]^{1/4}`.

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
        radius = cls._photospheric_radius_cgs(t, **parameters)

        # L = 0 sends the log to -inf, which means T = 0.
        log_temperature = 0.25 * (log_luminosity - np.log(_FOUR_PI_SIGMA_SB) - 2.0 * np.log(radius))
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
    def photospheric_radius(cls, t: u.Quantity, **parameters: CGSParameterValue) -> u.Quantity:
        r"""
        Photospheric radius :math:`R_\mathrm{ph}(t)`.

        Parameters
        ----------
        t : ~astropy.units.Quantity
            Time since breakout.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`R_\mathrm{ph}(t)`, in cm.
        """
        cgs_parameters = {name: np.asarray(to_cgs_value(value), dtype=np.float64) for name, value in parameters.items()}
        return cls._photospheric_radius_cgs(np.asarray(t.cgs.value, dtype=np.float64), **cgs_parameters) * u.cm

    @classmethod
    def _eval_bolometric(cls, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""Natural log of :math:`L_\mathrm{bol}(t)` in erg/s, delegated to the light curve."""
        lightcurve_parameters = {
            name: parameters[name] for name in InverseGammaBrokenPowerLawLightcurve._DEFAULT_PARAMETERS
        }
        return InverseGammaBrokenPowerLawLightcurve._eval(t, **lightcurve_parameters)

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
