r"""Magnetar-powered (Arnett-style) superluminous supernova SEDs."""

from typing import ClassVar

import numpy as np
from astropy import units as u

from uvex_transients.models._typing import CGSParameterValue, FloatArray
from uvex_transients.models._util_functions import photospheric_temperature_with_floor_cgs
from uvex_transients.models._utils import to_cgs_value
from uvex_transients.models.arnett import (
    GAUSS_CGS,
    _arnett_luminosity_cgs,
    _diffusion_time_cgs,
    _leakage_parameter_cgs,
    _magnetar_energy_cgs,
    _magnetar_luminosity_cgs,
    _magnetar_timescale_cgs,
)
from uvex_transients.models.core.base import SpectralModel
from uvex_transients.models.core.parameters import Parameter
from uvex_transients.models.core.priors import TruncatedNormalPrior, UniformPrior
from uvex_transients.models.spectra.thermal import BlackbodySpectrum

__all__ = ["ArnettMagnetarSpindownSED"]


class ArnettMagnetarSpindownSED(SpectralModel):
    r"""
    Magnetar spin-down powered SED, with the light curve from Arnett-style diffusion.

    This model follows the formalism of :footcite:t:`2017ApJ...850...55N` (the MOSFiT magnetar model of Type I
    superluminous
    supernovae) and the leakage treatment of :footcite:t:`2015ApJ...799..107W`.

    A newly born magnetar injects

    .. math::

        F_\mathrm{mag}(t) = \frac{E_\mathrm{mag}/t_\mathrm{mag}}\frac{1}{(1 + t/t_\mathrm{mag})^{2}}

    into homologously expanding, grey-opacity ejecta, which diffuses out and leaks high-energy photons according to
    :func:`~uvex_transients.models.arnett.compute_arnett_luminosity`.

    The temperature is computed using a floored photospheric temperature model:

    .. math::

        T(t) = \max\left\{\left[\frac{L(t)}{4\pi\sigma_\mathrm{SB}(v_\mathrm{ej}t)^2}\right]^{1/4},
        T_\mathrm{floor}\right\}, \qquad
        L_\nu(\nu, t) = L(t)\,\frac{\pi B_\nu(\nu, T(t))}{\sigma_\mathrm{SB}T(t)^4}.

    The light curve :math:`L(t)` is integrated once per evaluation and reused for both the temperature and the
    spectral luminosity.

    .. rubric:: Parameters

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``spin_period``
         - :math:`P`
         - Initial magnetar spin period.
       * - ``B_perp``
         - :math:`B_\perp`
         - Perpendicular component of the dipole field, in Gaussian-cgs gauss
           (:data:`~uvex_transients.models.arnett.GAUSS_CGS`).
       * - ``M_ej``
         - :math:`M_\mathrm{ej}`
         - Ejecta mass.
       * - ``v_ej``
         - :math:`v_\mathrm{ej}`
         - Ejecta velocity, taken to be constant and equal to the photospheric velocity.
       * - ``M_ns``
         - :math:`M_\mathrm{NS}`
         - Neutron star mass.
       * - ``kappa``
         - :math:`\kappa`
         - Grey optical opacity.
       * - ``kappa_gamma``
         - :math:`\kappa_\gamma`
         - Opacity to high-energy photons; sets how much of the late-time input energy leaks out.
       * - ``T_floor``
         - :math:`T_\mathrm{floor}`
         - Minimum photospheric temperature.

    References
    ----------
    .. footbibliography::
    """

    _N_GRID: ClassVar[int] = 1000
    """int: Grid points in each half of the diffusion-integral grid; the error falls as its inverse square."""

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "spin_period": Parameter(
            prior=TruncatedNormalPrior(mean=np.log10(3.0), sigma=0.26 * 0.4, lower=np.log10(0.7), upper=np.log10(20)),
            scale=1 * u.ms,
            transform="log10",
            description="Initial magnetar spin period.",
            latex=r"P",
        ),
        "B_perp": Parameter(
            prior=TruncatedNormalPrior(mean=np.log10(0.8), sigma=0.48 * 0.4, lower=np.log10(0.01), upper=np.log10(10)),
            scale=1e14 * GAUSS_CGS,
            transform="log10",
            description="Perpendicular component of the magnetar dipole field.",
            latex=r"B_\perp",
        ),
        "M_ej": Parameter(
            prior=TruncatedNormalPrior(mean=np.log10(4.8), sigma=0.38 * 0.4, lower=np.log10(0.1), upper=np.log10(100)),
            scale=1 * u.Msun,
            transform="log10",
            description="Ejecta mass.",
            latex=r"M_\mathrm{ej}",
        ),
        "v_ej": Parameter(
            prior=TruncatedNormalPrior(mean=0.9, sigma=0.3, lower=0.1, upper=3.0),
            scale=1e4 * u.km / u.s,
            description="Constant ejecta velocity, identified with the photospheric velocity.",
            latex=r"v_\mathrm{ej}",
        ),
        "M_ns": Parameter(
            prior=UniformPrior(lower=1.4, upper=2.2),
            scale=1 * u.Msun,
            description="Neutron star mass.",
            latex=r"M_\mathrm{NS}",
        ),
        "kappa": Parameter(
            prior=UniformPrior(lower=0.05, upper=0.2),
            scale=1 * u.cm**2 / u.g,
            description="Grey optical opacity.",
            latex=r"\kappa",
        ),
        "kappa_gamma": Parameter(
            prior=UniformPrior(lower=-2.0, upper=0.0),
            scale=1 * u.cm**2 / u.g,
            transform="log10",
            description="Opacity to high-energy photons.",
            latex=r"\kappa_\gamma",
        ),
        "T_floor": Parameter(
            prior=TruncatedNormalPrior(mean=6000.0, sigma=1000.0, lower=3000.0, upper=10000.0),
            scale=1 * u.K,
            description="Minimum photospheric temperature.",
            latex=r"T_\mathrm{floor}",
        ),
    }

    # -------------------------------------- #
    # L(t) and T(t), from one integral        #
    # -------------------------------------- #
    @classmethod
    def _luminosity_and_temperature_cgs(
        cls,
        t: FloatArray,
        *,
        spin_period: CGSParameterValue,
        B_perp: CGSParameterValue,
        M_ej: CGSParameterValue,
        v_ej: CGSParameterValue,
        M_ns: CGSParameterValue,
        kappa: CGSParameterValue,
        kappa_gamma: CGSParameterValue,
        T_floor: CGSParameterValue,
    ) -> tuple[FloatArray, FloatArray]:
        r"""
        :math:`L(t)` in erg/s and :math:`T(t)` in K, from a single diffusion integral.

        The luminosity is the only expensive step; the temperature is derived from that same array.

        Parameters
        ----------
        t : numpy.ndarray
            Time since explosion, in seconds.
        spin_period, B_perp, M_ej, v_ej, M_ns, kappa, kappa_gamma, T_floor : float or numpy.ndarray
            This model's parameter values, in cgs units; see the class docstring.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray)
            :math:`L(t)`, in erg/s, and :math:`T(t)`, in Kelvin.
        """
        t = np.asarray(t, dtype=np.float64)
        luminosity = _arnett_luminosity_cgs(
            t,
            _diffusion_time_cgs(kappa, M_ej, v_ej),
            _leakage_parameter_cgs(kappa_gamma, M_ej, v_ej),
            _magnetar_luminosity_cgs,
            {
                "energy": _magnetar_energy_cgs(spin_period, M_ns),
                "timescale": _magnetar_timescale_cgs(spin_period, B_perp, M_ns),
            },
            n_grid=cls._N_GRID,
        )

        # L vanishes at t = 0, where the photosphere has no radius: T is 0/0 there, and irrelevant since L_nu = 0.
        positive = t > 0
        temperature = photospheric_temperature_with_floor_cgs(
            np.where(positive, t, 1.0), L=luminosity, v_phot=v_ej, T_floor=T_floor
        )
        return luminosity, np.where(positive, temperature, T_floor)

    @classmethod
    def temperature(cls, t: u.Quantity, **parameters: u.Quantity) -> u.Quantity:
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
        cgs_parameters = {name: to_cgs_value(value) for name, value in parameters.items()}
        return cls._luminosity_and_temperature_cgs(t.cgs.value, **cgs_parameters)[1] * u.K

    # -------------------------------------- #
    # Bolometric Luminosity: L_bol(t)         #
    # -------------------------------------- #
    @classmethod
    def _eval_bolometric(cls, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        :math:`\log L_\mathrm{bol}(t)`, the diffusion-integral luminosity.

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
        luminosity, _ = cls._luminosity_and_temperature_cgs(t, **parameters)
        with np.errstate(divide="ignore"):
            return np.log(luminosity)

    # -------------------------------------- #
    # Normalized Spectral Shape: S(nu, t)    #
    # -------------------------------------- #
    @classmethod
    def _eval_spectrum(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        :math:`\log S(\nu, T(t))`, a blackbody at the floored photospheric temperature.

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
        return BlackbodySpectrum._eval(nu, temperature=temperature)

    # -------------------------------------- #
    # Spectral Luminosity: L_nu(nu, t)        #
    # -------------------------------------- #
    @classmethod
    def _eval(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        :math:`\log L_\nu(\nu, t) = \log L(t) + \log S(\nu, T(t))`, with :math:`L` integrated only once.

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
        return log_luminosity + BlackbodySpectrum._eval(nu, temperature=temperature)
