r"""SED models for Type IIP supernovae, with and without an early-time CSM-interaction excess."""

from typing import ClassVar

import numpy as np
from astropy import units as u

from uvex_transients.models._typing import CGSParameterValue, FloatArray
from uvex_transients.models._utils import to_cgs_value
from uvex_transients.models.core.base import SpectralModel
from uvex_transients.models.core.parameters import Parameter
from uvex_transients.models.core.priors import ConstantPrior, NormalPrior, UniformPrior
from uvex_transients.models.lightcurves.generic import VillarLightcurve
from uvex_transients.models.spectra.thermal import BlackbodySpectrum

__all__ = ["TypeIIPExcessSED", "TypeIIPSED"]


# ======================================== #
# Utility Functions                        #
# ======================================== #
def _bpl_floor_temperature(
    t: FloatArray,
    t0: CGSParameterValue,
    T_peak: CGSParameterValue,
    T_floor: CGSParameterValue,
    alpha_rise: CGSParameterValue,
    alpha_decay: CGSParameterValue,
    smoothing: CGSParameterValue,
) -> FloatArray:
    r""":math:`T(t) = T_f + (T_\mathrm{peak}-T_f)\left[(t/t_0)^{-s\alpha_r} + (t/t_0)^{s\alpha_d}\right]^{-1/s}`."""
    with np.errstate(divide="ignore"):
        log_x = np.log(t / t0)

    log_shape = -(1.0 / smoothing) * np.logaddexp(-smoothing * alpha_rise * log_x, smoothing * alpha_decay * log_x)

    return T_floor + 2 ** (1 / smoothing) * (T_peak - T_floor) * np.exp(log_shape)


# ======================================== #
# Model: no early excess                   #
# ======================================== #
class TypeIIPSED(SpectralModel):
    r"""
    Type IIP supernova SED: a Villar rise/plateau/decline light curve with a cooling blackbody photosphere.

    .. math::

        L_\mathrm{bol}(t) = A \times
        \begin{cases}
            \dfrac{1+\beta(t-t_0)}{1+\exp[-(t-t_0)/\tau_r]}, & t < t_1, \\[8pt]
            \dfrac{[1+\beta(t_1-t_0)]\exp[-(t-t_1)/\tau_f]}{1+\exp[-(t-t_0)/\tau_r]}, & t \ge t_1,
        \end{cases}

    delegated directly to
    :class:`~uvex_transients.models.lightcurves.generic.VillarLightcurve`, with
    :math:`\gamma = t_1 - t_0` computed on the fly rather than sampled directly.

    The photospheric temperature follows a smooth broken power law with a floor, referenced
    to the same :math:`t_0`:

    .. math::

        T(t) = T_f + (T_\mathrm{peak}-T_f)
        \left[(t/t_0)^{-s\alpha_r} + (t/t_0)^{s\alpha_d}\right]^{-1/s}.

    Because the blackbody shape is normalized independently of luminosity, the bolometric and
    spectral components remain exactly separable, so no numerical frequency integration is
    required to recover the bolometric luminosity.

    See :class:`TypeIIPExcessSED` for the variant that adds an early-time CSM-interaction
    excess on top of this same light curve and temperature law.

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
         - Luminosity normalization.
       * - ``t0``
         - :math:`t_0`
         - Peak/reference time -- shared by the light curve and the temperature law.
       * - ``t1``
         - :math:`t_1`
         - Plateau end time.
       * - ``beta``
         - :math:`\beta`
         - Plateau slope. Fixed.
       * - ``tau_rise``
         - :math:`\tau_r`
         - Logistic rise timescale. Fixed.
       * - ``tau_fall``
         - :math:`\tau_f`
         - Post-plateau exponential decline timescale. Fixed.
       * - ``T_peak``
         - :math:`T_\mathrm{peak}`
         - Approximate peak photospheric temperature.
       * - ``T_floor``
         - :math:`T_f`
         - Asymptotic photospheric temperature floor.
       * - ``alpha_rise``
         - :math:`\alpha_r`
         - Temperature-law rise power-law index. Fixed.
       * - ``alpha_decay``
         - :math:`\alpha_d`
         - Temperature-law decay power-law index. Fixed.
       * - ``smoothing``
         - :math:`s`
         - Sharpness of the temperature law's transition. Fixed.
    """

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=NormalPrior(mean=42, sigma=0.1),
            scale=1.0 * u.erg / u.s,
            transform="log10",
            description="Luminosity normalization.",
            latex=r"A",
        ),
        "t0": Parameter(
            prior=UniformPrior(lower=2.0, upper=5.0),
            scale=1.0 * u.day,
            description="Peak/reference time, shared by the light curve and the temperature law.",
            latex=r"t_0",
        ),
        "t1": Parameter(
            prior=UniformPrior(lower=50.0, upper=120.0),
            scale=1.0 * u.day,
            description="Plateau end time.",
            latex=r"t_1",
        ),
        "beta": Parameter(
            prior=ConstantPrior(value=1.0),
            scale=0.0 / u.day,
            description="Plateau slope. Fixed.",
            latex=r"\beta",
        ),
        "tau_rise": Parameter(
            prior=ConstantPrior(value=1.0),
            scale=1 * u.day,
            description="Logistic rise timescale. Fixed.",
            latex=r"\tau_r",
        ),
        "tau_fall": Parameter(
            prior=ConstantPrior(value=1.0),
            scale=50 * u.day,
            description="Post-plateau exponential decline timescale. Fixed.",
            latex=r"\tau_f",
        ),
        "T_peak": Parameter(
            prior=NormalPrior(mean=4.0, sigma=0.15),
            scale=1.0 * u.K,
            transform="log10",
            description="Approximate peak photospheric temperature.",
            latex=r"T_\mathrm{peak}",
        ),
        "T_floor": Parameter(
            prior=NormalPrior(mean=3.8, sigma=0.08),
            scale=1.0 * u.K,
            transform="log10",
            description="Asymptotic photospheric temperature floor.",
            latex=r"T_f",
        ),
        "alpha_rise": Parameter(
            prior=ConstantPrior(value=1.0),
            scale=0.8,
            description="Temperature-law rise power-law index. Fixed.",
            latex=r"\alpha_r",
        ),
        "alpha_decay": Parameter(
            prior=ConstantPrior(value=1.0),
            scale=1,
            description="Temperature-law decay power-law index. Fixed.",
            latex=r"\alpha_d",
        ),
        "smoothing": Parameter(
            prior=ConstantPrior(value=1.0),
            scale=4.0,
            description="Sharpness of the temperature law's transition. Fixed.",
            latex=r"s",
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
        t0: CGSParameterValue,
        T_peak: CGSParameterValue,
        T_floor: CGSParameterValue,
        alpha_rise: CGSParameterValue,
        alpha_decay: CGSParameterValue,
        smoothing: CGSParameterValue,
        **_ignored: CGSParameterValue,
    ) -> FloatArray:
        r"""Broken-power-law-with-floor photospheric temperature. See :func:`_bpl_floor_temperature`."""
        return _bpl_floor_temperature(t, t0, T_peak, T_floor, alpha_rise, alpha_decay, smoothing)

    @classmethod
    def temperature(cls, t: u.Quantity, **parameters: CGSParameterValue) -> u.Quantity:
        r""":math:`T(t)` in Kelvin."""
        cgs_parameters: dict[str, CGSParameterValue] = {name: to_cgs_value(value) for name, value in parameters.items()}
        return cls._temperature_cgs(t.cgs.value, **cgs_parameters) * u.K

    # -------------------------------------- #
    # Bolometric Luminosity: L_bol(t)         #
    # -------------------------------------- #
    @classmethod
    def _eval_bolometric(cls, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r""":math:`\log L_\mathrm{bol}(t)`, delegated directly to :class:`VillarLightcurve`."""
        return VillarLightcurve._eval(
            t,
            amplitude=parameters["amplitude"],
            t0=parameters["t0"],
            gamma=parameters["t1"] - parameters["t0"],
            beta=parameters["beta"],
            tau_rise=parameters["tau_rise"],
            tau_fall=parameters["tau_fall"],
        )

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


# ======================================== #
# Model: with early excess                 #
# ======================================== #
class TypeIIPExcessSED(TypeIIPSED):
    r"""
    :class:`TypeIIPSED`, plus an early-time sigmoid excess from CSM interaction.

    .. math::

        L_\mathrm{bol}(t) = L_\mathrm{Villar}(t) + L_\mathrm{excess}(t)

    where :math:`L_\mathrm{Villar}(t)` is exactly :class:`TypeIIPSED`'s own bolometric
    luminosity, and

    .. math::

        L_\mathrm{excess}(t) = A_e \times
        \begin{cases}
            0, & t < t_0, \\[4pt]
            \left[1-\exp\!\left(-\dfrac{t-t_0}{\tau_r}\right)\right]
            \exp\!\left(-\dfrac{t-t_0}{\tau_e}\right), & t \ge t_0,
        \end{cases}

    sharing the same rise timescale :math:`\tau_r` and reference time :math:`t_0` as the
    Villar component. The temperature law is unchanged from :class:`TypeIIPSED`.

    .. rubric:: Parameters

    In addition to all of :class:`TypeIIPSED`'s parameters:

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``excess_amplitude``
         - :math:`A_e`
         - Early-excess luminosity normalization.
       * - ``tau_excess``
         - :math:`\tau_e`
         - Excess decay timescale. Fixed.
    """

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        **TypeIIPSED._DEFAULT_PARAMETERS,
        "excess_amplitude": Parameter(
            prior=NormalPrior(mean=43.5, sigma=0.2),
            scale=1.0 * u.erg / u.s,
            transform="log10",
            description="Early-excess luminosity normalization.",
            latex=r"A_e",
        ),
        "tau_excess": Parameter(
            prior=NormalPrior(mean=1.0, sigma=0.1),
            scale=6 * u.day,
            description="Excess decay timescale. Fixed.",
            latex=r"\tau_e",
        ),
    }

    # -------------------------------------- #
    # Bolometric Luminosity: L_bol(t)         #
    # -------------------------------------- #
    @classmethod
    def _eval_bolometric(cls, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r""":math:`\log[L_\mathrm{Villar}(t) + L_\mathrm{excess}(t)]`."""
        log_L_villar = super()._eval_bolometric(t, **parameters)

        t0 = parameters["t0"]
        tau_rise = parameters["tau_rise"]
        excess_amplitude = parameters["excess_amplitude"]
        tau_excess = parameters["tau_excess"]

        # The `t < t0` branch is discarded by `np.where` below, so the invalid/overflowing
        # values it can produce there (e.g. `log1p` of a large negative number) are harmless.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            log_excess_shape = np.log1p(-np.exp(-(t - t0) / tau_rise)) - (t - t0) / tau_excess
        log_L_excess = np.log(excess_amplitude) + np.where(t < t0, -np.inf, log_excess_shape)

        return np.logaddexp(log_L_villar, log_L_excess)
