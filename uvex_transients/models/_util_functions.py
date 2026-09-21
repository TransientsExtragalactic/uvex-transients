import numpy as np

from uvex_transients.models._constants import C_CGS, H_CGS, K_B_CGS, LOG_SIGMA_SB_CGS, SIGMA_SB_CGS
from uvex_transients.models._typing import CGSParameterValue, FloatArray, NumericalInput


def _softplus(x: FloatArray) -> FloatArray:
    r"""Numerically stable :math:`\sigma(x) = \ln(1+e^x)`."""
    return np.logaddexp(0.0, x)


def _log_sigmoid(x: FloatArray) -> FloatArray:
    r"""Numerically stable :math:`\ln[\mathrm{sigmoid}(x)] = -\sigma(-x)`."""
    return -_softplus(-x)


def log_expm1(x: FloatArray) -> FloatArray:
    r"""
    Compute :math:`\log(\exp(x) - 1)`, stable for both small and large ``x``.

    Used to evaluate the natural log of the Planck function's occupation
    factor :math:`(\exp(h\nu/k_BT) - 1)^{-1}` without overflowing for large
    arguments or losing precision for small ones.

    Parameters
    ----------
    x : array_like
        Input value(s). Assumed non-negative.

    Returns
    -------
    numpy.ndarray
        :math:`\log(\exp(x) - 1)`, using the asymptotic form :math:`\approx x`
        for ``x > 30`` (where :func:`numpy.expm1` would otherwise overflow).
    """
    x = np.asarray(x, dtype=np.float64)
    large = x > 30.0
    safe_x = np.where(large, 0.0, x)

    # `safe_x` is 0 on the `large` branch, purely to keep `expm1`/`log` from
    # overflowing there -- that branch's `log(expm1(0)) = -inf` is discarded
    # by `np.where` below, not a real result, so the warning it would
    # otherwise raise is suppressed.
    with np.errstate(divide="ignore"):
        return np.where(large, x, np.log(np.expm1(safe_x)))


def powerlaw_shape_integral_cgs(
    spectral_index: NumericalInput,
    reference_frequency: NumericalInput,
    frequency_min: NumericalInput,
    frequency_max: NumericalInput,
) -> FloatArray:
    r"""
    Integral of an unnormalized power-law shape over a finite frequency range.

    For a shape :math:`S(\nu) = \nu_0^{-1}(\nu/\nu_0)^\alpha`,

    .. math::

        \int_{\nu_\min}^{\nu_\max} S(\nu)\,d\nu =
        \begin{cases}
        \dfrac{x_{\max}^{\alpha+1} - x_{\min}^{\alpha+1}}{\alpha+1}, & \alpha \ne -1, \\
        \ln(x_{\max}/x_{\min}), & \alpha = -1,
        \end{cases}

    where :math:`x = \nu/\nu_0`. Broadcasts over its arguments -- used both
    for :class:`~uvex_transients.models.spectra.powerlaw.PowerLawSpectrum`'s own
    normalization and, applied twice with a shared reference frequency, for
    :class:`~uvex_transients.models.spectra.powerlaw.BrokenPowerLawSpectrum`'s two
    segments.

    Parameters
    ----------
    spectral_index : array_like
        Power-law index :math:`\alpha` in :math:`S(\nu) \propto \nu^\alpha`.
    reference_frequency : array_like
        Reference frequency :math:`\nu_0`, in Hz.
    frequency_min : array_like
        Lower integration limit, in Hz.
    frequency_max : array_like
        Upper integration limit, in Hz.

    Returns
    -------
    numpy.ndarray
        The dimensionless integral, broadcast over the input arrays.
    """
    alpha = np.asarray(spectral_index, dtype=np.float64)
    nu_0 = np.asarray(reference_frequency, dtype=np.float64)
    x_min = np.asarray(frequency_min, dtype=np.float64) / nu_0
    x_max = np.asarray(frequency_max, dtype=np.float64) / nu_0
    exponent = alpha + 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
        power_law_integral = (x_max**exponent - x_min**exponent) / exponent
        log_integral = np.log(x_max / x_min)

    return np.where(np.isclose(exponent, 0.0), log_integral, power_law_integral)


def planck_Bnu_log_cgs(nu: FloatArray, temperature: CGSParameterValue) -> FloatArray:
    r"""
    Natural log of the (un-normalized) Planck function :math:`B_\nu(\nu, T)`, in cgs.

    .. math::

        \log B_\nu(\nu, T) = \log(2h/c^2) + 3\log\nu - \log\left(e^{h\nu/k_BT} - 1\right)

    The primal Planck-function implementation -- :func:`planck_shape_log_cgs` is
    derived from this by undoing its Stefan-Boltzmann normalization.

    Parameters
    ----------
    nu : array_like
        Frequency, in Hz.
    temperature : array_like
        Temperature, in K.

    Returns
    -------
    numpy.ndarray
        :math:`\log B_\nu(\nu, T)`, in :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}\,sr^{-1}}`.
    """
    x = H_CGS * np.asarray(nu, dtype=np.float64) / (K_B_CGS * np.asarray(temperature, dtype=np.float64))
    return np.log(2.0 * H_CGS / C_CGS**2) + 3.0 * np.log(nu) - log_expm1(x)


def planck_shape_log_cgs(nu: FloatArray, temperature: CGSParameterValue) -> FloatArray:
    r"""
    Natural log of the normalized, Lambertian-emergent Planck function :math:`S(\nu, T)`.

    .. math::

        S(\nu, T) = \frac{\pi B_\nu(\nu, T)}{\sigma_\mathrm{SB} T^4},
        \qquad \int_0^\infty S(\nu, T)\,d\nu = 1 \ \forall\, T.

    Parameters
    ----------
    nu : array_like
        Frequency, in Hz.
    temperature : array_like
        Temperature, in K.

    Returns
    -------
    numpy.ndarray
        :math:`\log S(\nu, T)`, in :math:`\mathrm{Hz^{-1}}`.
    """
    temperature = np.asarray(temperature, dtype=np.float64)
    return planck_Bnu_log_cgs(nu, temperature) + np.log(np.pi) - LOG_SIGMA_SB_CGS - 4.0 * np.log(temperature)


def cooling_temperature_cgs(
    t: FloatArray,
    *,
    T0: CGSParameterValue,
    T_floor: CGSParameterValue,
    timescale: CGSParameterValue,
    alpha: CGSParameterValue,
) -> FloatArray:
    r"""
    Single-power-law cooling photospheric temperature.

    .. math::

        T(t) = T_\mathrm{floor} + (T_0 - T_\mathrm{floor})\left(1 + \frac{t}{\tau}\right)^{-\alpha}

    :math:`T(t) \to T_0` as :math:`t \to 0` and :math:`T(t) \to T_\mathrm{floor}` as
    :math:`t \to \infty`. Shared by models whose photosphere cools according to this
    single power law; a doubly-broken cooling law (e.g. a Type IIP's early/cooling/plateau
    phases) is a genuinely different functional form and is not expressed in terms of this.

    Parameters
    ----------
    t : array_like
        Time since explosion, in s.
    T0 : array_like
        Photospheric temperature as :math:`t \to 0`.
    T_floor : array_like
        Asymptotic late-time photospheric temperature.
    timescale : array_like
        Cooling timescale :math:`\tau`, in s.
    alpha : array_like
        Cooling power-law index.

    Returns
    -------
    numpy.ndarray
        :math:`T(t)`, in K.
    """
    return T_floor + (T0 - T_floor) * (1.0 + t / timescale) ** (-alpha)


def photospheric_temperature_cgs(
    t: FloatArray,
    *,
    L: CGSParameterValue,
    v_phot: CGSParameterValue,
) -> FloatArray:
    r"""
    Blackbody temperature of a photosphere expanding at constant velocity.

    .. math::

        T(t) = \left[\frac{L}{4\pi\sigma_\mathrm{SB}\,(v_\mathrm{phot}\,t)^2}\right]^{1/4}

    Parameters
    ----------
    t : array_like
        Time since explosion, in s.
    L : array_like
        Bolometric luminosity, in erg/s.
    v_phot : array_like
        Photospheric velocity, in cm/s.

    Returns
    -------
    numpy.ndarray
        :math:`T(t)`, in K. Diverges as :math:`t \to 0`.
    """
    return (L / (4.0 * np.pi * SIGMA_SB_CGS * (v_phot * t) ** 2)) ** 0.25


def photospheric_temperature_with_floor_cgs(
    t: FloatArray,
    *,
    L: CGSParameterValue,
    v_phot: CGSParameterValue,
    T_floor: CGSParameterValue,
) -> FloatArray:
    r"""
    Photospheric temperature, clipped from below at a floor temperature.

    .. math::

        T(t) = \max\left[T_\mathrm{phot}(t),\, T_\mathrm{floor}\right]

    where :math:`T_\mathrm{phot}` is given by :func:`photospheric_temperature_cgs`.
    Once the expanding blackbody photosphere cools to :math:`T_\mathrm{floor}`,
    the temperature stays fixed there.

    Parameters
    ----------
    t : array_like
        Time since explosion, in s.
    L : array_like
        Bolometric luminosity, in erg/s.
    v_phot : array_like
        Photospheric velocity, in cm/s.
    T_floor : array_like
        Minimum photospheric temperature, in K.

    Returns
    -------
    numpy.ndarray
        :math:`T(t)`, in K.
    """
    return np.maximum(photospheric_temperature_cgs(t, L=L, v_phot=v_phot), T_floor)
