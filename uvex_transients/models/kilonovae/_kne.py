r"""Kilonova (BNS merger) SED model."""

from typing import ClassVar

from astropy import units as u

from uvex_transients.models._typing import CGSParameterValue, FloatArray
from uvex_transients.models._utils import to_cgs_value
from uvex_transients.models.core._base import SpectralModel
from uvex_transients.models.core._parameters import Parameter
from uvex_transients.models.core.priors import LogNormalPrior, NormalPrior, UniformPrior
from uvex_transients.models.lightcurves.generic import GaussianRiseBrokenPowerLawLightcurve
from uvex_transients.models.spectra.thermal import BlackbodySpectrum

__all__ = ["KilonovaCoolingBlackbodySED"]


class KilonovaCoolingBlackbodySED(SpectralModel):
    r"""
    A Gaussian-rise/broken-power-law-decline light curve with a cooling blackbody photosphere.

    .. rubric:: Parameters

    The model parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude``
         - :math:`L_0`
         - Peak bolometric luminosity.
       * - ``t_peak``
         - :math:`t_\mathrm{peak}`
         - Time of peak luminosity since merger. Also sets the Gaussian rise
           width (:math:`t_\mathrm{peak}/5`) and the temperature cooling
           timescale.
       * - ``decline_index_1``
         - :math:`\alpha_1`
         - Positive early-time post-peak power-law decline index.
       * - ``decline_index_2``
         - :math:`\alpha_2`
         - Positive late-time post-peak power-law decline index.
       * - ``t_break``
         - :math:`t_\mathrm{break}`
         - Time at which the decline steepens from :math:`\alpha_1` to :math:`\alpha_2`.
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
         - Early-time photospheric cooling power-law index.


    References
    ----------
    .. footbibliography::
    """

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=NormalPrior(mean=41.8, sigma=0.1),
            scale=1.0 * u.erg / u.s,
            transform="log10",
            description="Peak bolometric luminosity. log10(L_0/[erg/s]) ~ N(41.8, 0.1^2), "
            "anchored to Cowperthwaite et al. 2017's GW170817 measurement of "
            "L ~ 6.8e41 erg/s at 0.6 d.",
            latex=r"L_0",
        ),
        "t_peak": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.3),
            scale=0.6 * u.day,
            description="Time of peak luminosity since merger. Anchored near the 0.6 d epoch "
            "used to normalize amplitude/T0, with broad scatter since the rise itself was "
            "never observed for GW170817 (Cowperthwaite et al. 2017).",
            latex=r"t_\mathrm{peak}",
        ),
        "decline_index_1": Parameter(
            prior=UniformPrior(0.8, 1.2),
            scale=1.0 * u.dimensionless_unscaled,
            description="Positive early-time post-peak power-law decline index; "
            "L_bol ~ t^-decline_index_1 for t_peak < t <= t_break. ~1 (Waxman et al. 2018).",
            latex=r"\alpha_1",
        ),
        "decline_index_2": Parameter(
            prior=UniformPrior(3.0, 4.0),
            scale=1.0 * u.dimensionless_unscaled,
            description="Positive late-time post-peak power-law decline index; "
            "L_bol ~ t^-decline_index_2 for t > t_break. ~3 (Waxman et al. 2018).",
            latex=r"\alpha_2",
        ),
        "t_break": Parameter(
            prior=UniformPrior(5.0, 10.0),
            scale=1.0 * u.day,
            description="Time at which the bolometric decline steepens from decline_index_1 "
            "to decline_index_2, 5-10 d (Waxman et al. 2018).",
            latex=r"t_\mathrm{break}",
        ),
        "T0": Parameter(
            prior=NormalPrior(mean=3.9, sigma=0.1),
            scale=1.0 * u.K,
            transform="log10",
            description="Photospheric temperature at t=0 (the T(t) -> T0 limit, not literally "
            "T at peak). log10(T0/K) ~ N(3.9, 0.1^2), ~7900 K, anchored to Cowperthwaite et al. "
            "2017's GW170817 measurement of T ~ 8300 K at 0.6 d.",
            latex=r"T_0",
        ),
        "T_floor": Parameter(
            prior=NormalPrior(mean=3.4, sigma=0.08),
            scale=1.0 * u.K,
            transform="log10",
            description="Asymptotic late-time photospheric temperature (T(t) -> T_floor as "
            "t -> infinity). log10(T_floor/K) ~ N(3.4, 0.08^2), ~2500 K (Waxman et al. 2018).",
            latex=r"T_\mathrm{floor}",
        ),
        "alpha_T": Parameter(
            prior=UniformPrior(0.3, 0.7),
            scale=1.0 * u.dimensionless_unscaled,
            description="Early-time photospheric cooling power-law index; "
            "T ~ t^-alpha_T for t << t_peak/5. ~0.5 (Waxman et al. 2018).",
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
        r""":math:`T(t) = T_\mathrm{floor} + (T_0 - T_\mathrm{floor})(1 + t/(t_\mathrm{peak}/5))^{-\alpha_T}`."""
        sigma_rise = t_peak / 5
        return T_floor + (T0 - T_floor) * (1.0 + t / sigma_rise) ** (-alpha_T)

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
        r""":math:`\log L_\mathrm{bol}(t)`, delegated directly to :class:`GaussianRiseBrokenPowerLawLightcurve`.

        Exact -- no integration needed.
        """
        lightcurve_parameters = {
            name: parameters[name] for name in GaussianRiseBrokenPowerLawLightcurve._DEFAULT_PARAMETERS
        }
        return GaussianRiseBrokenPowerLawLightcurve._eval(t, **lightcurve_parameters)

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
