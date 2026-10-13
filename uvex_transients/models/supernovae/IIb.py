"""Models of SEDs from type IIb supernovae and associated scenarios."""

from typing import ClassVar

from astropy import units as u

from uvex_transients.models._typing import CGSParameterValue, FloatArray
from uvex_transients.models._util_functions import cooling_temperature_cgs
from uvex_transients.models._utils import to_cgs_value
from uvex_transients.models.core.base import SpectralModel
from uvex_transients.models.core.parameters import Parameter
from uvex_transients.models.core.priors import ConstantPrior, NormalPrior, UniformPrior
from uvex_transients.models.lightcurves.generic import TwoComponentBazinLightcurve
from uvex_transients.models.spectra.thermal import BlackbodySpectrum

__all__ = ["TypeIIbSED"]


# ======================================== #
# Model: Two-Component SED                 #
# ======================================== #
class TypeIIbSED(SpectralModel):
    r"""
    Phenomenological Type IIb supernova SED.

    This is a purely empirical light-curve shape intended to span a full Type IIb light curve.
    It fits real double- and single-peaked Type IIb events better than a single modulated pulse:

    .. math::

        L_\mathrm{bol}(t) =
        A_0\,
        \frac{\exp[-(t-t_0)/\tau_{\mathrm{fall},0}]}{1 + \exp[-(t-t_0)/\tau_{\mathrm{rise},0}]}
        +
        A_1\,
        \frac{\exp[-(t-t_1)/\tau_{\mathrm{fall},1}]}{1 + \exp[-(t-t_1)/\tau_{\mathrm{rise},1}]},

    The photospheric temperature follows the same single-power-law cooling law used elsewhere in this package
    (:func:`~uvex_transients.models._util_functions.cooling_temperature_cgs`):

    .. math::

        T(t) = T_\mathrm{floor} + (T_0 - T_\mathrm{floor})\left(1 + \frac{t}{\tau_T}\right)^{-\alpha_T}.

    .. rubric:: Parameters

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude_0``
         - :math:`A_0`
         - Luminosity normalization of the early, shock-cooling peak.
       * - ``t0``
         - :math:`t_0`
         - Characteristic transition time of the early peak.
       * - ``rise_0``
         - :math:`\tau_{\mathrm{rise},0}`
         - Logistic rise timescale of the early peak.
       * - ``fall_0``
         - :math:`\tau_{\mathrm{fall},0}`
         - Exponential decline timescale of the early peak.
       * - ``amplitude_1``
         - :math:`A_1`
         - Luminosity normalization of the main, radioactively powered peak.
       * - ``t1``
         - :math:`t_1`
         - Characteristic transition time of the main peak.
       * - ``rise_1``
         - :math:`\tau_{\mathrm{rise},1}`
         - Logistic rise timescale of the main peak.
       * - ``fall_1``
         - :math:`\tau_{\mathrm{fall},1}`
         - Exponential decline timescale of the main peak.
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
         - Photospheric cooling power-law index.

    Notes
    -----
    ``amplitude_0`` -- the early peak's normalization -- is uniform in
    :math:`\log_{10}(A_0/\mathrm{erg\,s^{-1}})` between 39 and 43, spanning several decades from
    far fainter than the main peak up to brighter than it. This single prior covers both
    populations at once: draws where the early peak ends up negligible are effectively
    single-peaked Type IIb light curves, and draws where it is comparable to or exceeds the main
    peak are double-peaked -- roughly a third of draws from the default priors below are
    double-peaked. ``t0`` and ``T_floor`` are held fixed; every other parameter is drawn from a
    broad Uniform (or, for ``amplitude_1``/``T0``, Normal-in-log) prior over an
    order-of-magnitude-motivated range, not yet a fit to any specific real Type IIb event (compare
    :class:`~uvex_transients.models.supernovae.IIp.TypeIIPSED`, whose light-curve priors do come
    from fitting real SNe).
    """

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude_0": Parameter(
            prior=UniformPrior(lower=39, upper=43),
            scale=1 * u.erg / u.s,
            transform="log10",
            description="Luminosity normalization of the early, shock-cooling peak; uniform in "
            "log10(amplitude_0 / erg/s) between 39 and 43, spanning far fainter to brighter than "
            "the main peak.",
            latex=r"A_0",
        ),
        "t0": Parameter(
            prior=ConstantPrior(value=1.0),
            scale=2.0 * u.day,
            description="Characteristic transition time of the early peak.",
            latex=r"t_0",
        ),
        "rise_0": Parameter(
            prior=UniformPrior(lower=0.3, upper=1),
            scale=1 * u.day,
            description="Logistic rise timescale of the early peak.",
            latex=r"\tau_{\mathrm{rise},0}",
        ),
        "fall_0": Parameter(
            prior=UniformPrior(lower=3, upper=10),
            scale=1 * u.day,
            description="Exponential decline timescale of the early peak.",
            latex=r"\tau_{\mathrm{fall},0}",
        ),
        "amplitude_1": Parameter(
            prior=NormalPrior(mean=42.5, sigma=0.1),
            scale=1 * u.erg / u.s,
            transform="log10",
            description="Luminosity normalization of the main, radioactively powered peak.",
            latex=r"A_1",
        ),
        "t1": Parameter(
            prior=UniformPrior(lower=10, upper=20),
            scale=1 * u.day,
            description="Characteristic transition time of the main peak.",
            latex=r"t_1",
        ),
        "rise_1": Parameter(
            prior=UniformPrior(lower=2, upper=4),
            scale=1 * u.day,
            description="Logistic rise timescale of the main peak.",
            latex=r"\tau_{\mathrm{rise},1}",
        ),
        "fall_1": Parameter(
            prior=UniformPrior(lower=30, upper=55),
            scale=1 * u.day,
            description="Exponential decline timescale of the main peak.",
            latex=r"\tau_{\mathrm{fall},1}",
        ),
        "T0": Parameter(
            prior=NormalPrior(mean=4.1, sigma=0.05),
            scale=1 * u.K,
            transform="log10",
            description="Photospheric temperature at t=0 (the T(t) -> T0 limit, not literally T at peak).",
            latex=r"T_0",
        ),
        "T_floor": Parameter(
            prior=ConstantPrior(value=1.0),
            scale=4e3 * u.K,
            description="Asymptotic late-time photospheric temperature (T(t) -> T_floor as t -> infinity).",
            latex=r"T_\mathrm{floor}",
        ),
        "tau_T": Parameter(
            prior=UniformPrior(lower=5, upper=10),
            scale=1 * u.day,
            description="Photospheric cooling timescale.",
            latex=r"\tau_T",
        ),
        "alpha_T": Parameter(
            prior=UniformPrior(lower=0.7, upper=1.3),
            scale=1.0,
            description="Photospheric cooling power-law index.",
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
        r"""
        :math:`T(t) = T_\mathrm{floor} + (T_0 - T_\mathrm{floor})(1 + t/\tau_T)^{-\alpha_T}`.

        Parameters
        ----------
        t : numpy.ndarray
            Time since explosion, in seconds.
        T0, T_floor, tau_T, alpha_T : float or numpy.ndarray
            This model's parameter values, in cgs units; see the class docstring.
        **_ignored
            Any other model parameter values, ignored.

        Returns
        -------
        numpy.ndarray
            :math:`T(t)`, in Kelvin.
        """
        return cooling_temperature_cgs(t, T0=T0, T_floor=T_floor, timescale=tau_T, alpha=alpha_T)

    @classmethod
    def temperature(cls, t: u.Quantity, **parameters: CGSParameterValue) -> u.Quantity:
        r"""
        :math:`T(t)` in Kelvin.

        Parameters
        ----------
        t : ~astropy.units.Quantity
            Time since explosion.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`T(t)`, in Kelvin.
        """
        cgs_parameters: dict[str, CGSParameterValue] = {name: to_cgs_value(value) for name, value in parameters.items()}
        return cls._temperature_cgs(t.cgs.value, **cgs_parameters) * u.K

    # -------------------------------------- #
    # Bolometric Luminosity: L_bol(t)         #
    # -------------------------------------- #
    @classmethod
    def _eval_bolometric(cls, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        :math:`\log L_\mathrm{bol}(t)`, delegated directly to :class:`TwoComponentBazinLightcurve`.

        Exact -- no integration needed.

        Parameters
        ----------
        t : numpy.ndarray
            Time since explosion, in seconds.
        **parameters
            This model's parameter values, in cgs units.

        Returns
        -------
        numpy.ndarray
            The natural log of :math:`L_\mathrm{bol}(t)`, in erg/s.
        """
        lightcurve_parameters = {name: parameters[name] for name in TwoComponentBazinLightcurve._DEFAULT_PARAMETERS}
        return TwoComponentBazinLightcurve._eval(t, **lightcurve_parameters)

    # -------------------------------------- #
    # Normalized Spectral Shape: S(nu, t)    #
    # -------------------------------------- #
    @classmethod
    def _eval_spectrum(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        :math:`\log S(\nu, T(t))`, delegated to :class:`~uvex_transients.models.spectra.thermal.BlackbodySpectrum`.

        Evaluated at this ``t``'s own cooling-law temperature.

        Parameters
        ----------
        nu : numpy.ndarray
            Frequency, in Hz.
        t : numpy.ndarray
            Time since explosion, in seconds.
        **parameters
            This model's parameter values, in cgs units.

        Returns
        -------
        numpy.ndarray
            The natural log of the normalized spectral shape, in 1/Hz.
        """
        temperature = cls._temperature_cgs(t, **parameters)
        return BlackbodySpectrum._eval(nu, temperature=temperature)

    # -------------------------------------- #
    # Spectral Luminosity: L_nu(nu, t)        #
    # -------------------------------------- #
    @classmethod
    def _eval(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        :math:`\log L_\nu(\nu, t) = \log L_\mathrm{bol}(t) + \log S(\nu, T(t))`.

        Parameters
        ----------
        nu : numpy.ndarray
            Frequency, in Hz.
        t : numpy.ndarray
            Time since explosion, in seconds.
        **parameters
            This model's parameter values, in cgs units.

        Returns
        -------
        numpy.ndarray
            The natural log of :math:`L_\nu(\nu, t)`, in erg/s/Hz.
        """
        return cls._eval_bolometric(t, **parameters) + cls._eval_spectrum(nu, t, **parameters)
