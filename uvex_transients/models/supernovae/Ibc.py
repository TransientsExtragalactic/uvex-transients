"""Models of SEDs from type Ib and Ic (stripped-envelope) supernovae."""

from typing import ClassVar

from astropy import units as u

from uvex_transients.models._typing import CGSParameterValue, FloatArray
from uvex_transients.models._util_functions import cooling_temperature_cgs
from uvex_transients.models._utils import to_cgs_value
from uvex_transients.models.core.base import SpectralModel
from uvex_transients.models.core.parameters import Parameter
from uvex_transients.models.core.priors import ConstantPrior, NormalPrior, UniformPrior
from uvex_transients.models.lightcurves.generic import BazinLightcurve
from uvex_transients.models.spectra.thermal import BlackbodySpectrum

__all__ = ["TypeIbSED", "TypeIcSED"]


class TypeIbSED(SpectralModel):
    r"""
    Phenomenological Type Ib supernova SED.

    A single Bazin pulse times a cooling blackbody photosphere:

    .. math::

        L_\mathrm{bol}(t) =
        A\,
        \frac{\exp[-(t-t_0)/\tau_\mathrm{fall}]}{1 + \exp[-(t-t_0)/\tau_\mathrm{rise}]},

    delegated to :class:`~uvex_transients.models.lightcurves.generic.BazinLightcurve` with the rise
    timescale tied to the transition time, :math:`\tau_\mathrm{rise} = t_0 / 2.5` (see
    ``_RISE_DIVISOR``), so ``t0`` alone sets the time to peak (:math:`t_p = t_0\,[1 + 0.4\ln(2.5\,
    \tau_\mathrm{fall}/t_0 - 1)]`, about :math:`1.9\,t_0`). The photospheric temperature follows the
    same single-power-law cooling law to a floor used elsewhere in this package
    (:func:`~uvex_transients.models._util_functions.cooling_temperature_cgs`):

    .. math::

        T(t) = T_\mathrm{floor} + (T_0 - T_\mathrm{floor})\left(1 + \frac{t}{\tau_T}\right)^{-\alpha_T}.

    Unlike :class:`~uvex_transients.models.supernovae.IIb.TypeIIbSED`, a Type Ib light curve has no
    separate early shock-cooling peak, so one Bazin pulse suffices.

    .. rubric:: Parameters

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude``
         - :math:`A`
         - Luminosity normalization of the Bazin pulse (not exactly the peak luminosity).
       * - ``t0``
         - :math:`t_0`
         - Characteristic transition time of the pulse; also sets the rise timescale, :math:`t_0/2.5`.
       * - ``fall``
         - :math:`\tau_\mathrm{fall}`
         - Exponential decline timescale.
       * - ``T0``
         - :math:`T_0`
         - Photospheric temperature at t=0 (the T(t) -> T0 limit, not literally T at peak).
       * - ``T_floor``
         - :math:`T_\mathrm{floor}`
         - Asymptotic late-time photospheric temperature (T(t) -> T_floor as t -> infinity).
       * - ``tau_T``
         - :math:`\tau_T`
         - Photospheric cooling timescale.
       * - ``alpha_T``
         - :math:`\alpha_T`
         - Photospheric cooling power-law index. Fixed.

    Notes
    -----
    The light-curve shape is set by ``fall`` and ``t0`` only: ``rise`` is not a free parameter but
    :math:`t_0/2.5`. With the default priors the time to peak is about 20 days (5th to 95th
    percentile 17.8--21.9 d), the peak luminosity is :math:`\log_{10}(L_\mathrm{peak}/\mathrm{erg\,s^{-1}})
    \approx 42.45 \pm 0.3`, and the luminosity at :math:`t=0` is about 14% of the peak. The priors are
    calibrated against the full UV-to-NIR bolometric light curves of Lyman et al. (2016) (13 Type Ib
    and 8 Type Ic SNe, aligned on the epoch of maximum) and the photospheric temperatures of
    Prentice et al. (2019) in ``test_data/transients/lightcurves.h5``. The temperature curves start too
    long after explosion to constrain :math:`T_0`, :math:`\tau_T` and :math:`\alpha_T` independently:
    the data prefer a fast, close to exponential relaxation to a floor, i.e. a large :math:`\alpha_T`
    with a long :math:`\tau_T`, which are strongly degenerate. ``alpha_T`` is therefore held fixed
    at 4 and the remaining temperature parameters carry the population scatter.
    """

    #: The Bazin logistic rise timescale is tied to the transition time: ``rise = t0 / _RISE_DIVISOR``. This fixes
    #: the shape of the rise (in particular how little luminosity is left at ``t = 0`` relative to the peak), so
    #: ``t0`` alone sets the time to peak.
    _RISE_DIVISOR = 2.5

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=NormalPrior(mean=42.6, sigma=0.3),
            scale=1 * u.erg / u.s,
            transform="log10",
            description="Luminosity normalization of the Bazin pulse (not exactly the peak luminosity).",
            latex=r"A",
        ),
        "t0": Parameter(
            prior=UniformPrior(lower=9.5, upper=12),
            scale=1 * u.day,
            description="Characteristic transition time of the pulse; also sets the rise timescale, t0 / 2.5.",
            latex=r"t_0",
        ),
        "fall": Parameter(
            prior=UniformPrior(lower=30, upper=50),
            scale=1 * u.day,
            description="Exponential decline timescale.",
            latex=r"\tau_\mathrm{fall}",
        ),
        "T0": Parameter(
            prior=NormalPrior(mean=4.2, sigma=0.1),
            scale=1 * u.K,
            transform="log10",
            description="Photospheric temperature at t=0 (the T(t) -> T0 limit, not literally T at peak).",
            latex=r"T_0",
        ),
        "T_floor": Parameter(
            prior=UniformPrior(lower=4000, upper=5200),
            scale=1 * u.K,
            description="Asymptotic late-time photospheric temperature (T(t) -> T_floor as t -> infinity).",
            latex=r"T_\mathrm{floor}",
        ),
        "tau_T": Parameter(
            prior=UniformPrior(lower=15, upper=45),
            scale=1 * u.day,
            description="Photospheric cooling timescale.",
            latex=r"\tau_T",
        ),
        "alpha_T": Parameter(
            prior=ConstantPrior(value=4.0),
            scale=1.0,
            description="Photospheric cooling power-law index. Fixed; degenerate with tau_T for the "
            "available data (see the class notes).",
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
        tau_T: CGSParameterValue,
        alpha_T: CGSParameterValue,
        **_ignored: CGSParameterValue,
    ) -> FloatArray:
        r""":math:`T(t) = T_\mathrm{floor} + (T_0 - T_\mathrm{floor})(1 + t/\tau_T)^{-\alpha_T}`."""
        return cooling_temperature_cgs(t, T0=T0, T_floor=T_floor, timescale=tau_T, alpha=alpha_T)

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
        r""":math:`\log L_\mathrm{bol}(t)`, delegated to :class:`BazinLightcurve` with ``rise = t0 / _RISE_DIVISOR``.

        Exact -- no integration needed.
        """
        t0 = parameters["t0"]
        return BazinLightcurve._eval(
            t, amplitude=parameters["amplitude"], t0=t0, rise=t0 / cls._RISE_DIVISOR, fall=parameters["fall"]
        )

    # -------------------------------------- #
    # Normalized Spectral Shape: S(nu, t)    #
    # -------------------------------------- #
    @classmethod
    def _eval_spectrum(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r""":math:`\log S(\nu, T(t))`, delegated to :class:`~uvex_transients.models.spectra.thermal.BlackbodySpectrum`.

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


class TypeIcSED(TypeIbSED):
    r"""
    Phenomenological Type Ic supernova SED.

    Same light-curve and temperature functional forms as :class:`TypeIbSED`. The two classes
    currently share the same parameter priors and differ only in their volumetric event rate (see
    :class:`~uvex_transients.transients.supernovae.TypeIcSNe`). See :class:`TypeIbSED` for the
    parameter table and for how the priors were calibrated.
    """

    #: The Bazin logistic rise timescale is tied to the transition time: ``rise = t0 / _RISE_DIVISOR``. This fixes
    #: the shape of the rise (in particular how little luminosity is left at ``t = 0`` relative to the peak), so
    #: ``t0`` alone sets the time to peak.
    _RISE_DIVISOR = 2.5

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=NormalPrior(mean=42.6, sigma=0.3),
            scale=1 * u.erg / u.s,
            transform="log10",
            description="Luminosity normalization of the Bazin pulse (not exactly the peak luminosity).",
            latex=r"A",
        ),
        "t0": Parameter(
            prior=UniformPrior(lower=9.5, upper=12),
            scale=1 * u.day,
            description="Characteristic transition time of the pulse; also sets the rise timescale, t0 / 2.5.",
            latex=r"t_0",
        ),
        "fall": Parameter(
            prior=UniformPrior(lower=30, upper=50),
            scale=1 * u.day,
            description="Exponential decline timescale.",
            latex=r"\tau_\mathrm{fall}",
        ),
        "T0": Parameter(
            prior=NormalPrior(mean=4.2, sigma=0.1),
            scale=1 * u.K,
            transform="log10",
            description="Photospheric temperature at t=0 (the T(t) -> T0 limit, not literally T at peak).",
            latex=r"T_0",
        ),
        "T_floor": Parameter(
            prior=UniformPrior(lower=4000, upper=5200),
            scale=1 * u.K,
            description="Asymptotic late-time photospheric temperature (T(t) -> T_floor as t -> infinity).",
            latex=r"T_\mathrm{floor}",
        ),
        "tau_T": Parameter(
            prior=UniformPrior(lower=15, upper=45),
            scale=1 * u.day,
            description="Photospheric cooling timescale.",
            latex=r"\tau_T",
        ),
        "alpha_T": Parameter(
            prior=ConstantPrior(value=4.0),
            scale=1.0,
            description="Photospheric cooling power-law index. Fixed; degenerate with tau_T for the "
            "available data (see the class notes).",
            latex=r"\alpha_T",
        ),
    }
