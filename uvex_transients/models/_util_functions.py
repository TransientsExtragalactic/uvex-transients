"""Utility functions for working with models."""

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
    single power law. Models with an additional temperature rebrightening add
    :func:`temperature_bump_cgs` on top of it.

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


def sigmoid_temperature_cgs(
    t: FloatArray,
    *,
    T0: CGSParameterValue,
    T_floor: CGSParameterValue,
    t_transition: CGSParameterValue,
    steepness: CGSParameterValue,
) -> FloatArray:
    r"""
    Photospheric temperature that falls from a hot start to a cool floor as a logistic in :math:`\ln t`.

    .. math::

        T(t) = T_\mathrm{floor} + (T_0 - T_\mathrm{floor})\,
        \frac{1}{1 + (t/t_\mathrm{tr})^{k}}

    :math:`T(t) \to T_0` as :math:`t \to 0` and :math:`T(t) \to T_\mathrm{floor}` as :math:`t \to \infty`, with
    the excess over the floor halved at :math:`t_\mathrm{tr}` and falling as :math:`t^{-k}` well after it. Unlike
    :func:`cooling_temperature_cgs`, whose early plateau gives way to a pure power law, this has two distinct
    plateaus joined by a transition whose width in :math:`\ln t` is :math:`\sim1/k`.

    Parameters
    ----------
    t : array_like
        Time since explosion, in s.
    T0 : array_like
        Photospheric temperature as :math:`t \to 0`.
    T_floor : array_like
        Asymptotic late-time photospheric temperature.
    t_transition : array_like
        Epoch :math:`t_\mathrm{tr}` at which the temperature is halfway between :math:`T_0` and
        :math:`T_\mathrm{floor}`, in s.
    steepness : array_like
        Steepness :math:`k` of the transition.

    Returns
    -------
    numpy.ndarray
        :math:`T(t)`, in K.
    """
    # The logistic of -k ln(t/t_tr), through the stable log-sigmoid; t = 0 gives the T0 limit.
    with np.errstate(divide="ignore"):
        log_ratio = np.log(t) - np.log(t_transition)
    return T_floor + (T0 - T_floor) * np.exp(_log_sigmoid(-steepness * log_ratio))


def temperature_bump_cgs(
    t: FloatArray,
    *,
    amplitude: CGSParameterValue,
    t_peak: CGSParameterValue,
    beta: CGSParameterValue,
    alpha: CGSParameterValue,
) -> FloatArray:
    r"""
    Temperature excess with a power-law rise and power-law decline, parametrized by its peak.

    With :math:`x = t/\tau` and :math:`\tau = t_\mathrm{peak}\,\alpha/\beta`,

    .. math::

        \Delta T(t) = \Delta T_\mathrm{pk}\,\frac{g(x)}{g(\beta/\alpha)}, \qquad
        g(x) = \frac{x^{\beta}}{(1 + x)^{\alpha + \beta}}.

    :math:`g` rises as :math:`x^\beta` for :math:`x \ll 1`, falls as :math:`x^{-\alpha}` for
    :math:`x \gg 1`, and peaks at :math:`x = \beta/\alpha`, so :math:`\Delta T` is zero at
    :math:`t = 0`, reaches exactly ``amplitude`` at ``t = t_peak``, and decays to zero at late times.

    Parameters
    ----------
    t : array_like
        Time since explosion, in s.
    amplitude : array_like
        Peak value of the excess :math:`\Delta T_\mathrm{pk}`, in K.
    t_peak : array_like
        Time of the peak, in s.
    beta : array_like
        Rise index (:math:`\Delta T \propto t^\beta` at early times).
    alpha : array_like
        Decay index (:math:`\Delta T \propto t^{-\alpha}` at late times).

    Returns
    -------
    numpy.ndarray
        :math:`\Delta T(t)`, in K.
    """
    x_peak = beta / alpha
    x = t / (t_peak / x_peak)
    with np.errstate(divide="ignore"):
        ln_g = beta * np.log(x) - (alpha + beta) * np.log1p(x)
    ln_g_peak = beta * np.log(x_peak) - (alpha + beta) * np.log1p(x_peak)
    return amplitude * np.exp(ln_g - ln_g_peak)


def blackbody_temperature_cgs(L: CGSParameterValue, radius: CGSParameterValue) -> FloatArray:
    r"""
    Temperature of a blackbody sphere with a given luminosity and radius.

    .. math::

        T = \left[\frac{L}{4\pi\sigma_\mathrm{SB}\,r^2}\right]^{1/4}

    Parameters
    ----------
    L : array_like
        Bolometric luminosity, in erg/s.
    radius : array_like
        Photospheric radius, in cm.

    Returns
    -------
    numpy.ndarray
        :math:`T`, in K.
    """
    return (L / (4.0 * np.pi * SIGMA_SB_CGS * radius**2)) ** 0.25


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
    return blackbody_temperature_cgs(L, v_phot * t)


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
