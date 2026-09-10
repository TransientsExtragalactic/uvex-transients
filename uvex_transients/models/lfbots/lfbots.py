r"""Luminous Fast Blue Optical Transient (LFBOT) SED model."""

from typing import ClassVar

import numpy as np
from astropy import units as u

from uvex_transients.models._typing import CGSParameterValue, FloatArray
from uvex_transients.models._utils import to_cgs_value
from uvex_transients.models.core._base import SpectralModel
from uvex_transients.models.core._parameters import Parameter
from uvex_transients.models.core.priors import LogNormalPrior, NormalPrior, UniformPrior
from uvex_transients.models.lightcurves.generic import GaussianRisePowerLawLightcurve
from uvex_transients.models.spectra.thermal import BlackbodySpectrum

__all__ = ["LFBOTCoolingBlackbodySED"]


class LFBOTCoolingBlackbodySED(SpectralModel):
    r"""
    A Gaussian-rise/power-law-decline light curve with a Villar-type cooling blackbody photosphere.

    This model pairs
    :class:`~uvex_transients.models.lightcurves.generic.GaussianRisePowerLawLightcurve`
    with a time-dependent blackbody spectral shape,

    .. math::

        L_\nu(\nu, t)
        =
        L_\mathrm{bol}(t)\,
        S_\mathrm{BB}\!\left[\nu, T(t)\right],

    where :math:`S_\mathrm{BB}` is the normalized blackbody spectrum provided
    by :class:`~uvex_transients.models.spectra.thermal.BlackbodySpectrum`.
    The photosphere follows the same cooling-law form as
    :class:`~uvex_transients.models.supernovae.VillarCoolingBlackbodySED`,
    but with the cooling timescale fixed to the light curve's own
    :math:`t_\mathrm{peak}` rather than a separate free parameter,

    .. math::

        T(t) = T_\mathrm{floor} + (T_0 - T_\mathrm{floor})(1 + t/t_\mathrm{peak})^{-\alpha_T},

    which has the correct :math:`T \to T_\mathrm{floor} + (T_0 -
    T_\mathrm{floor})(t/t_\mathrm{peak})^{-\alpha_T}` power-law asymptote at late times.

    .. rubric:: Parameters

    The model parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude``
         - :math:`A`
         - Peak bolometric luminosity.
       * - ``t_peak``
         - :math:`t_\mathrm{peak}`
         - Time of peak luminosity since explosion.
       * - ``decline_index``
         - :math:`\alpha_\mathrm{decline}`
         - Positive post-peak power-law decline index of the bolometric
           light curve.
       * - ``T0``
         - :math:`T_0`
         - Photospheric temperature at t=0 (the T(t) -> T0 limit, not
           literally T at peak).
       * - ``T_floor``
         - :math:`T_\mathrm{floor}`
         - Asymptotic late-time photospheric temperature (T(t) -> T_floor
           as t -> infinity).
       * - ``alpha_T``
         - :math:`\alpha_T`
         - Late-time photospheric cooling power-law index.
    """

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=NormalPrior(mean=44.5, sigma=0.5),
            scale=1.0 * u.erg / u.s,
            transform="log10",
            description="Peak bolometric luminosity. log10(L_0/[erg/s]) ~ N(44.5, 0.1^2), "
            "spanning the ~1e44 (CSS161010) to a few 1e45 erg/s range reported by Ho & Lu et al. 2026.",
            latex=r"L_0",
        ),
        "t_peak": Parameter(
            prior=NormalPrior(mean=np.log10(3), sigma=0.1),
            scale=1.0 * u.day,
            transform="log10",
            description="Time of peak bolometric luminosity since explosion. Log-normal around "
            "2 d with fairly small scatter (Ho & Lu et al. 2026).",
            latex=r"t_\mathrm{peak}",
        ),
        "decline_index": Parameter(
            prior=UniformPrior(lower=2, upper=4),
            scale=1 * u.dimensionless_unscaled,
            description="Positive post-peak power-law decline index; L_bol ~ t^-decline_index. "
            "Late-time light curves are broadly consistent with a t^-3 decline (Ho & Lu et al. 2026).",
            latex=r"\alpha_\mathrm{decline}",
        ),
        "T0": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.15),
            scale=2e4 * u.K,
            description="Photospheric temperature at t=0 (the T(t) -> T0 limit, not literally T at peak).",
            latex=r"T_0",
        ),
        "T_floor": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.01),
            scale=1e4 * u.K,
            description="Asymptotic late-time photospheric temperature (T(t) -> T_floor as t -> infinity). "
            "Log-normal, tightly centered on 1e4 K.",
            latex=r"T_\mathrm{floor}",
        ),
        "alpha_T": Parameter(
            prior=UniformPrior(lower=0, upper=1 / 3),
            scale=1,
            description="Late-time photospheric cooling power-law index; T ~ t^-alpha_T for t >> t_peak. "
            "~1/3 is a reasonable fit for most events (e.g. AT2018cow); CSS161010 was closer to "
            "constant temperature (Ho & Lu et al. 2026).",
            latex=r"\alpha_T",
        ),
    }

    # -------------------------------------- #
    # Cooling Law: T(t)                       #
    # -------------------------------------- #
    @classmethod
    def _temperature_cgs(
        cls,
        t: FloatArray,
        *,
        T0: CGSParameterValue,
        T_floor: CGSParameterValue,
        t_peak: CGSParameterValue,
        alpha_T: CGSParameterValue,
        **_ignored: CGSParameterValue,
    ) -> FloatArray:
        r""":math:`T(t) = T_\mathrm{floor} + (T_0 - T_\mathrm{floor})(1 + t/t_\mathrm{peak})^{-\alpha_T}`."""
        return T_floor + (T0 - T_floor) * (1.0 + t / t_peak) ** (-alpha_T)

    @classmethod
    def temperature(cls, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r""":math:`T(t)` in Kelvin."""
        cgs_parameters: dict[str, CGSParameterValue] = {name: to_cgs_value(value) for name, value in parameters.items()}

        return cls._temperature_cgs(t.cgs.value, **cgs_parameters) * u.K

    # -------------------------------------- #
    # Bolometric Luminosity: L_bol(t)         #
    # -------------------------------------- #
    @classmethod
    def _eval_bolometric(cls, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r""":math:`\log L_\mathrm{bol}(t)`, delegated directly to :class:`GaussianRisePowerLawLightcurve`.

        Exact -- no integration needed.
        """
        lightcurve_parameters = {name: parameters[name] for name in GaussianRisePowerLawLightcurve._DEFAULT_PARAMETERS}
        return GaussianRisePowerLawLightcurve._eval(t, **lightcurve_parameters)

    # -------------------------------------- #
    # Normalized Spectral Shape: S(nu, t)    #
    # -------------------------------------- #
    @classmethod
    def _eval_spectrum(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r""":math:`\log S(\nu, T(t))`, delegated to :class:`BlackbodySpectrum`.

        Evaluated at this ``t``'s own cooling-law temperature.
        """
        temperature = cls._temperature_cgs(t, **parameters)
        return BlackbodySpectrum._eval(nu, temperature=temperature)

    # -------------------------------------- #
    # Spectral Luminosity: L_nu(nu, t)        #
    # -------------------------------------- #
    @classmethod
    def _eval(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r""":math:`\log L_\nu(\nu, t) = \log L_\mathrm{bol}(t) + \log S(\nu, T(t))`."""
        return cls._eval_bolometric(t, **parameters) + cls._eval_spectrum(nu, t, **parameters)
