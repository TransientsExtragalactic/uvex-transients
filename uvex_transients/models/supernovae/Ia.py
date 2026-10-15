"""Model of the SED from type Ia (thermonuclear) supernovae."""

from typing import ClassVar

import numpy as np
from astropy import units as u

from uvex_transients.models._constants import SIGMA_SB_CGS
from uvex_transients.models._typing import CGSParameterValue, FloatArray
from uvex_transients.models._util_functions import (
    modified_blackbody_intensity_log_cgs,
    modified_blackbody_log_norm_cgs,
)
from uvex_transients.models.arnett import ArnettDecaySED
from uvex_transients.models.core.parameters import Parameter
from uvex_transients.models.core.priors import (
    ConstantPrior,
    NormalPrior,
    TruncatedNormalPrior,
    UniformPrior,
)

__all__ = ["TypeIaSED"]


class TypeIaSED(ArnettDecaySED):
    r"""
    Type Ia supernova SED, with priors fit to a ZTF sample of SNe Ia.

    Reuses :class:`~uvex_transients.models.arnett.ArnettDecaySED`'s radioactive-decay Arnett diffusion and
    floored photospheric temperature, but replaces its blackbody with a line-blanketed one
    (see :class:`~uvex_transients.models.spectra.thermal.ModifiedBlackbodySpectrum`), which suppresses the
    UV blueward of ``break_frequency`` as :math:`\nu^{-\beta}`. The shape is divided by its integral
    (from the shared table of
    :func:`~uvex_transients.models._util_functions.get_modified_blackbody_norm_interpolator`), so
    :math:`L_\mathrm{bol}(t)` is unchanged and the flux removed from the UV is returned redward of the
    break. The priors follow
    :footcite:t:`sarin2026` (Sarin et al. 2026, a sample of 2205 SNe Ia from ZTF): ``M_Ni`` and ``M_ej``
    are the population-level Gaussians from that paper's hierarchical Arnett-model fit
    (:math:`\mu_\mathrm{Ni}=0.64\,M_\odot`, :math:`\sigma_\mathrm{Ni}=0.42\,M_\odot`;
    :math:`\mu_\mathrm{ej}=1.26\,M_\odot`, :math:`\sigma_\mathrm{ej}=0.33\,M_\odot`), ``kappa_gamma`` is
    fixed at the value adopted by :footcite:t:`scalzo2014` (Scalzo et al. 2014), and ``kappa`` is uniform
    over that paper's marginalization range.

    .. rubric:: Parameters

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``M_Ni``
         - :math:`M_\mathrm{Ni}`
         - Nickel-56 mass synthesized in the explosion.
       * - ``M_ej``
         - :math:`M_\mathrm{ej}`
         - Ejecta mass.
       * - ``v_ej``
         - :math:`v_\mathrm{ej}`
         - Ejecta velocity, taken to be constant and equal to the photospheric velocity.
       * - ``kappa``
         - :math:`\kappa`
         - Grey optical opacity.
       * - ``kappa_gamma``
         - :math:`\kappa_\gamma`
         - Opacity to high-energy photons. Fixed.
       * - ``T_floor``
         - :math:`T_\mathrm{floor}`
         - Minimum photospheric temperature.
       * - ``break_frequency``
         - :math:`\nu_\mathrm{break}`
         - Frequency blueward of which line blanketing suppresses the flux.
       * - ``beta``
         - :math:`\beta`
         - Power-law index of the suppression above the break.

    References
    ----------
    .. footbibliography::
    """

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "M_Ni": Parameter(
            prior=TruncatedNormalPrior(mean=0.64, sigma=0.42, lower=0.05, upper=3.0),
            scale=1 * u.Msun,
            description="Nickel-56 mass synthesized in the explosion.",
            latex=r"M_\mathrm{Ni}",
        ),
        "M_ej": Parameter(
            prior=TruncatedNormalPrior(mean=1.26, sigma=0.33, lower=0.05, upper=3.0),
            scale=1 * u.Msun,
            description="Ejecta mass.",
            latex=r"M_\mathrm{ej}",
        ),
        "v_ej": Parameter(
            prior=NormalPrior(mean=11e3, sigma=1e3),
            scale=1.0 * u.km / u.s,
            description="Constant ejecta velocity, identified with the photospheric velocity.",
            latex=r"v_\mathrm{ej}",
        ),
        "kappa": Parameter(
            prior=UniformPrior(lower=0.05, upper=0.15),
            scale=1 * u.cm**2 / u.g,
            description="Grey optical opacity.",
            latex=r"\kappa",
        ),
        "kappa_gamma": Parameter(
            prior=ConstantPrior(value=0.03),
            scale=1 * u.cm**2 / u.g,
            description="Opacity to high-energy photons. Fixed (Scalzo+14).",
            latex=r"\kappa_\gamma",
        ),
        "T_floor": Parameter(
            prior=TruncatedNormalPrior(mean=6000.0, sigma=1000.0, lower=3000.0, upper=10000.0),
            scale=1 * u.K,
            description="Minimum photospheric temperature.",
            latex=r"T_\mathrm{floor}",
        ),
        "break_frequency": Parameter(
            prior=ConstantPrior(value=0.75),
            scale=1e15 * u.Hz,
            description="Frequency above which the line-blanketing suppression sets in.",
            latex=r"\nu_\mathrm{break}",
        ),
        "beta": Parameter(
            prior=ConstantPrior(value=5.0),
            scale=1.0,
            description="Power-law index of the suppression above the break.",
            latex=r"\beta",
        ),
    }

    SMOOTHING: ClassVar[float] = 10.0
    """float: The smoothing parameter :math:`s` of the break, fixed for every instance."""

    @classmethod
    def _luminosity_and_temperature_cgs(  # type: ignore[override]
        cls,
        t: FloatArray,
        *,
        break_frequency: CGSParameterValue,
        beta: CGSParameterValue,
        **parameters: CGSParameterValue,
    ) -> tuple[FloatArray, FloatArray]:
        r"""
        :math:`L(t)` and :math:`T(t)`, which do not depend on the line-blanketing parameters.

        Parameters
        ----------
        t : numpy.ndarray
            Time since explosion, in seconds.
        break_frequency, beta : float or numpy.ndarray
            Line-blanketing parameters; accepted and ignored here.
        **parameters
            The remaining parameters, passed to :meth:`ArnettDecaySED._luminosity_and_temperature_cgs`.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray)
            :math:`L(t)`, in erg/s, and :math:`T(t)`, in Kelvin.
        """
        return super()._luminosity_and_temperature_cgs(t, **parameters)

    @classmethod
    def _log_shape(
        cls,
        nu: FloatArray,
        temperature: FloatArray,
        break_frequency: CGSParameterValue,
        beta: CGSParameterValue,
    ) -> FloatArray:
        r"""
        :math:`\log S(\nu)` of the line-blanketed blackbody, divided by its integral.

        Parameters
        ----------
        nu : numpy.ndarray
            Frequency, in Hz.
        temperature : numpy.ndarray
            Photospheric temperature, in K.
        break_frequency, beta : float or numpy.ndarray
            Line-blanketing parameters, in cgs units.

        Returns
        -------
        numpy.ndarray
            The natural log of the unit-integral spectral shape, in 1/Hz.
        """
        shape = modified_blackbody_intensity_log_cgs(
            nu, temperature, break_frequency, beta, np.pi / (SIGMA_SB_CGS * temperature**4), cls.SMOOTHING
        )
        return shape - modified_blackbody_log_norm_cgs(temperature, break_frequency, beta, cls.SMOOTHING)

    @classmethod
    def _eval_spectrum(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        :math:`\log S(\nu, t)`, the unit-integral line-blanketed blackbody at the floored photospheric temperature.

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
        _, temperature = cls._luminosity_and_temperature_cgs(t, **parameters)
        return cls._log_shape(nu, temperature, parameters["break_frequency"], parameters["beta"])

    @classmethod
    def _eval(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        :math:`\log L_\nu(\nu, t) = \log L(t) + \log S(\nu, t)`, with :math:`L` integrated only once.

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
        luminosity, temperature = cls._luminosity_and_temperature_cgs(t, **parameters)
        with np.errstate(divide="ignore"):
            log_luminosity = np.log(luminosity)
        return log_luminosity + cls._log_shape(nu, temperature, parameters["break_frequency"], parameters["beta"])
