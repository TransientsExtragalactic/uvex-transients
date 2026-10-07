r"""
Analytic shock cooling emission (SCE) from extended, low-mass material.

Implements the model of :footcite:t:`2021ApJ...909..209P`. An envelope of mass :math:`M_e` and initial radius
:math:`R_e` is given an energy :math:`E_e` by a passing shock and then expands homologously, :math:`r = vt`. Its
density is a broken power law in the scaled coordinate :math:`x = r / (v_t t)`,

.. math::

    \rho(r, t) = \frac{K M_e}{v_t^3 t^3} \begin{cases} x^{-n} & x \ge 1 \\ x^{-\delta} & x < 1, \end{cases}

with transition velocity :math:`v_t` (Eq. 4), and :math:`K` set by mass conservation (Eq. 3). A constant (electron
scattering) opacity :math:`\kappa` is assumed throughout. The model has three characteristic outputs:

* the bolometric luminosity :math:`L(t)` (:func:`compute_luminosity`), a power law before the diffusion time
  :math:`t_d` and an exponential decay after it (Eqs. 17 and 20),
* the photospheric radius :math:`r_\mathrm{ph}(t)` (:func:`compute_photospheric_radius`), which breaks at the
  photospheric time :math:`t_\mathrm{ph}` (Eqs. 7 and 9), and
* the blackbody temperature :math:`T(t) = [L / (4\pi r_\mathrm{ph}^2 \sigma_\mathrm{SB})]^{1/4}`
  (:func:`compute_temperature`, Eq. 22).

Everything private here works on bare CGS floats/arrays. The public functions accept unit-aware
:class:`~astropy.units.Quantity` inputs (unitless inputs are assumed to already be in CGS), convert at the
boundary, and return :class:`~astropy.units.Quantity` outputs.

The model assumes :math:`n > 5` (finite energy in the outer envelope) and :math:`1 < \delta < 3`. As discussed in
the paper, it ignores recombination and any interior heating (e.g. radioactive), both of which matter before
:math:`t_\mathrm{ph}`, so the early power-law scalings and the break in :math:`r_\mathrm{ph}` at :math:`t_d` are
the most robust predictions.

References
----------
.. footbibliography::
"""

import numpy as np
from astropy import units as u

from uvex_transients.models._constants import C_CGS, SIGMA_SB_CGS
from uvex_transients.models._typing import FloatArray, NumericalInput, PhysicalInput
from uvex_transients.models._utils import ensure_in_units

__all__ = [
    "compute_ejecta_normalization",
    "compute_transition_velocity",
    "compute_diffusion_time",
    "compute_photospheric_time",
    "compute_luminosity",
    "compute_photospheric_radius",
    "compute_temperature",
]

DEFAULT_N: float = 10.0
"""float: Typical outer density power-law index (Chevalier & Soker 1989; see the paper's Appendix A)."""

DEFAULT_DELTA: float = 1.1
"""float: Typical inner density power-law index (Chevalier & Soker 1989)."""


# ================================================= #
# Physics (CGS)                                     #
# ================================================= #
def _ejecta_normalization(n: FloatArray, delta: FloatArray) -> FloatArray:
    r"""Density normalization :math:`K = (n - 3)(3 - \delta) / [4\pi (n - \delta)]` (Eq. 3)."""
    return (n - 3.0) * (3.0 - delta) / (4.0 * np.pi * (n - delta))


def _transition_velocity_cgs(
    E_envelope: FloatArray, M_envelope: FloatArray, n: FloatArray, delta: FloatArray
) -> FloatArray:
    r"""
    Transition velocity between the outer and inner envelope, in cm/s (Eq. 4).

    .. math::

        v_t = \left[\frac{(n - 5)(5 - \delta)}{(n - 3)(3 - \delta)}\right]^{1/2}
              \left(\frac{2 E_e}{M_e}\right)^{1/2}
    """
    coefficient = ((n - 5.0) * (5.0 - delta)) / ((n - 3.0) * (3.0 - delta))
    return np.sqrt(coefficient) * np.sqrt(2.0 * E_envelope / M_envelope)


def _diffusion_time_cgs(
    E_envelope: FloatArray, M_envelope: FloatArray, kappa: FloatArray, n: FloatArray, delta: FloatArray
) -> FloatArray:
    r"""
    Time at which the diffusion depth reaches the transition velocity, in s (Eq. 14).

    .. math::

        t_d = \left[\frac{3\kappa K M_e}{(n - 1) v_t c}\right]^{1/2}
    """
    K = _ejecta_normalization(n, delta)
    v_t = _transition_velocity_cgs(E_envelope, M_envelope, n, delta)
    return np.sqrt(3.0 * kappa * K * M_envelope / ((n - 1.0) * v_t * C_CGS))


def _photospheric_time_cgs(
    E_envelope: FloatArray, M_envelope: FloatArray, kappa: FloatArray, n: FloatArray, delta: FloatArray
) -> FloatArray:
    r"""
    Time at which the photosphere reaches the transition velocity, in s (Eq. 8).

    .. math::

        t_\mathrm{ph} = \left[\frac{3\kappa K M_e}{2 (n - 1) v_t^2}\right]^{1/2}
    """
    K = _ejecta_normalization(n, delta)
    v_t = _transition_velocity_cgs(E_envelope, M_envelope, n, delta)
    return np.sqrt(3.0 * kappa * K * M_envelope / (2.0 * (n - 1.0) * v_t * v_t))


def _luminosity_cgs(
    t: FloatArray,
    E_envelope: FloatArray,
    R_envelope: FloatArray,
    M_envelope: FloatArray,
    kappa: FloatArray,
    n: FloatArray,
    delta: FloatArray,
) -> FloatArray:
    r"""
    Bolometric luminosity in erg/s (Eqs. 17, 20, and 21).

    .. math::

        L(t) = \frac{\pi (n - 1)}{3 (n - 5)} \frac{c R_e v_t^2}{\kappa}
        \begin{cases}
            (t_d / t)^{4/(n - 2)} & t \le t_d \\
            \exp\left[-\frac{1}{2}\left(t^2 / t_d^2 - 1\right)\right] & t \ge t_d.
        \end{cases}
    """
    v_t = _transition_velocity_cgs(E_envelope, M_envelope, n, delta)
    t_d = _diffusion_time_cgs(E_envelope, M_envelope, kappa, n, delta)

    prefactor = np.pi * (n - 1.0) / (3.0 * (n - 5.0)) * C_CGS * R_envelope * v_t**2 / kappa
    x = t / t_d

    # Both branches are finite for every t > 0 (the unused one at worst under or overflows harmlessly to 0 or a
    # large number), so np.where is safe and broadcasts over arbitrary parameter and time shapes.
    shape = np.where(x <= 1.0, x ** (-4.0 / (n - 2.0)), np.exp(-0.5 * (x**2 - 1.0)))
    return prefactor * shape


def _photospheric_radius_cgs(
    t: FloatArray,
    E_envelope: FloatArray,
    M_envelope: FloatArray,
    kappa: FloatArray,
    n: FloatArray,
    delta: FloatArray,
) -> FloatArray:
    r"""
    Photospheric radius in cm (Eqs. 7 and 9).

    .. math::

        r_\mathrm{ph}(t) = v_t t
        \begin{cases}
            (t_\mathrm{ph} / t)^{2/(n - 1)} & t \le t_\mathrm{ph} \\
            \left[\frac{\delta - 1}{n - 1}\left(t^2 / t_\mathrm{ph}^2 - 1\right) + 1\right]^{-1/(\delta - 1)}
            & t \ge t_\mathrm{ph}.
        \end{cases}
    """
    v_t = _transition_velocity_cgs(E_envelope, M_envelope, n, delta)
    t_ph = _photospheric_time_cgs(E_envelope, M_envelope, kappa, n, delta)

    x = t / t_ph

    # As in the luminosity, both branches are finite for every t > 0: the late-time base is at least
    # 1 - (delta - 1) / (n - 1) > 0 for x < 1.
    shape = np.where(
        x <= 1.0,
        x ** (-2.0 / (n - 1.0)),
        ((delta - 1.0) / (n - 1.0) * (x**2 - 1.0) + 1.0) ** (-1.0 / (delta - 1.0)),
    )
    return shape * v_t * t


def _temperature_cgs(
    t: FloatArray,
    E_envelope: FloatArray,
    R_envelope: FloatArray,
    M_envelope: FloatArray,
    kappa: FloatArray,
    n: FloatArray,
    delta: FloatArray,
) -> FloatArray:
    r"""Blackbody temperature in K, :math:`T = [L / (4\pi r_\mathrm{ph}^2 \sigma_\mathrm{SB})]^{1/4}` (Eq. 22)."""
    L = _luminosity_cgs(t, E_envelope, R_envelope, M_envelope, kappa, n, delta)
    r_ph = _photospheric_radius_cgs(t, E_envelope, M_envelope, kappa, n, delta)
    return (L / (4.0 * np.pi * SIGMA_SB_CGS * r_ph**2)) ** 0.25


# ================================================= #
# Input handling                                    #
# ================================================= #
def _check_profile(n: FloatArray, delta: FloatArray) -> None:
    """Raise if the density profile indices are outside the range where the model is defined."""
    if not (np.all(np.isfinite(n)) and np.all(n > 5.0)):
        raise ValueError("n must be finite and greater than 5")
    if not (np.all(np.isfinite(delta)) and np.all(delta > 1.0) and np.all(delta < 3.0)):
        raise ValueError("delta must be finite and in the open interval (1, 3)")


def _check_positive(**values: FloatArray) -> None:
    """Raise if any of the named arrays is non-finite or not strictly positive."""
    for name, value in values.items():
        if not (np.all(np.isfinite(value)) and np.all(value > 0.0)):
            raise ValueError(f"{name} must be finite and positive")


def _prepare_envelope(
    E_envelope: PhysicalInput, M_envelope: PhysicalInput, n: NumericalInput, delta: NumericalInput
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Coerce and validate the parameters shared by every public function (energy in erg, mass in g)."""
    E = ensure_in_units(E_envelope, u.erg)
    M = ensure_in_units(M_envelope, u.g)
    n = ensure_in_units(n, None)
    delta = ensure_in_units(delta, None)
    _check_positive(E_envelope=E, M_envelope=M)
    _check_profile(n, delta)
    return E, M, n, delta


def _prepare_time(t: PhysicalInput) -> FloatArray:
    """Coerce and validate a time since explosion (in s), which must be strictly positive."""
    t = ensure_in_units(t, u.s)
    _check_positive(t=t)
    return t


# ================================================= #
# Public API                                        #
# ================================================= #
def compute_ejecta_normalization(n: NumericalInput = DEFAULT_N, delta: NumericalInput = DEFAULT_DELTA) -> np.ndarray:
    r"""
    Density normalization :math:`K` of the two-component envelope.

    Set by mass conservation (Eq. 3 of :footcite:t:`2021ApJ...909..209P`),

    .. math::

        K = \frac{(n - 3)(3 - \delta)}{4\pi (n - \delta)}.

    Parameters
    ----------
    n : array_like, optional
        Outer density power-law index, :math:`\rho \propto r^{-n}`. Must exceed 5. Default 10.
    delta : array_like, optional
        Inner density power-law index, :math:`\rho \propto r^{-\delta}`. Must lie in (1, 3). Default 1.1.

    Returns
    -------
    numpy.ndarray
        Dimensionless :math:`K` (about 0.119 for the defaults).

    References
    ----------
    .. footbibliography::
    """
    n = ensure_in_units(n, None)
    delta = ensure_in_units(delta, None)
    _check_profile(n, delta)
    return _ejecta_normalization(n, delta)


def compute_transition_velocity(
    E_envelope: PhysicalInput,
    M_envelope: PhysicalInput,
    n: NumericalInput = DEFAULT_N,
    delta: NumericalInput = DEFAULT_DELTA,
) -> u.Quantity:
    r"""
    Transition velocity :math:`v_t` between the outer and inner envelope.

    From energy conservation (Eq. 4 of :footcite:t:`2021ApJ...909..209P`),

    .. math::

        v_t = \left[\frac{(n - 5)(5 - \delta)}{(n - 3)(3 - \delta)}\right]^{1/2}
              \left(\frac{2 E_e}{M_e}\right)^{1/2}.

    Parameters
    ----------
    E_envelope : ~astropy.units.Quantity or array_like
        Energy :math:`E_e` imparted to the envelope (floats are taken to be in erg).
    M_envelope : ~astropy.units.Quantity or array_like
        Envelope mass :math:`M_e` (floats are taken to be in g).
    n, delta : array_like, optional
        Density power-law indices; see :func:`compute_ejecta_normalization`.

    Returns
    -------
    ~astropy.units.Quantity
        :math:`v_t` in cm/s.

    References
    ----------
    .. footbibliography::
    """
    E, M, n, delta = _prepare_envelope(E_envelope, M_envelope, n, delta)
    return _transition_velocity_cgs(E, M, n, delta) * (u.cm / u.s)


def compute_diffusion_time(
    E_envelope: PhysicalInput,
    M_envelope: PhysicalInput,
    kappa: PhysicalInput,
    n: NumericalInput = DEFAULT_N,
    delta: NumericalInput = DEFAULT_DELTA,
) -> u.Quantity:
    r"""
    Diffusion time :math:`t_d`, when the diffusion depth reaches the transition velocity.

    Eq. 14 of :footcite:t:`2021ApJ...909..209P`,

    .. math::

        t_d = \left[\frac{3\kappa K M_e}{(n - 1) v_t c}\right]^{1/2}.

    The luminosity switches from a power law to an exponential decay here.

    Parameters
    ----------
    E_envelope : ~astropy.units.Quantity or array_like
        Energy :math:`E_e` imparted to the envelope (floats are taken to be in erg).
    M_envelope : ~astropy.units.Quantity or array_like
        Envelope mass :math:`M_e` (floats are taken to be in g).
    kappa : ~astropy.units.Quantity or array_like
        Constant opacity (floats are taken to be in cm^2/g).
    n, delta : array_like, optional
        Density power-law indices; see :func:`compute_ejecta_normalization`.

    Returns
    -------
    ~astropy.units.Quantity
        :math:`t_d` in s.

    References
    ----------
    .. footbibliography::
    """
    E, M, n, delta = _prepare_envelope(E_envelope, M_envelope, n, delta)
    kappa = ensure_in_units(kappa, u.cm**2 / u.g)
    _check_positive(kappa=kappa)
    return _diffusion_time_cgs(E, M, kappa, n, delta) * u.s


def compute_photospheric_time(
    E_envelope: PhysicalInput,
    M_envelope: PhysicalInput,
    kappa: PhysicalInput,
    n: NumericalInput = DEFAULT_N,
    delta: NumericalInput = DEFAULT_DELTA,
) -> u.Quantity:
    r"""
    Photospheric time :math:`t_\mathrm{ph}`, when the photosphere reaches the transition velocity.

    Eq. 8 of :footcite:t:`2021ApJ...909..209P`,

    .. math::

        t_\mathrm{ph} = \left[\frac{3\kappa K M_e}{2 (n - 1) v_t^2}\right]^{1/2}
                      = \left(\frac{c}{2 v_t}\right)^{1/2} t_d.

    The photospheric radius shows a break here.

    Parameters
    ----------
    E_envelope : ~astropy.units.Quantity or array_like
        Energy :math:`E_e` imparted to the envelope (floats are taken to be in erg).
    M_envelope : ~astropy.units.Quantity or array_like
        Envelope mass :math:`M_e` (floats are taken to be in g).
    kappa : ~astropy.units.Quantity or array_like
        Constant opacity (floats are taken to be in cm^2/g).
    n, delta : array_like, optional
        Density power-law indices; see :func:`compute_ejecta_normalization`.

    Returns
    -------
    ~astropy.units.Quantity
        :math:`t_\mathrm{ph}` in s.

    References
    ----------
    .. footbibliography::
    """
    E, M, n, delta = _prepare_envelope(E_envelope, M_envelope, n, delta)
    kappa = ensure_in_units(kappa, u.cm**2 / u.g)
    _check_positive(kappa=kappa)
    return _photospheric_time_cgs(E, M, kappa, n, delta) * u.s


def compute_luminosity(
    t: PhysicalInput,
    E_envelope: PhysicalInput,
    R_envelope: PhysicalInput,
    M_envelope: PhysicalInput,
    kappa: PhysicalInput,
    n: NumericalInput = DEFAULT_N,
    delta: NumericalInput = DEFAULT_DELTA,
) -> u.Quantity:
    r"""
    Bolometric shock cooling luminosity :math:`L(t)`.

    Eqs. 17, 20, and 21 of :footcite:t:`2021ApJ...909..209P`,

    .. math::

        L(t) = \frac{\pi (n - 1)}{3 (n - 5)} \frac{c R_e v_t^2}{\kappa}
        \begin{cases}
            (t_d / t)^{4/(n - 2)} & t \le t_d \\
            \exp\left[-\frac{1}{2}\left(t^2 / t_d^2 - 1\right)\right] & t \ge t_d.
        \end{cases}

    Parameters
    ----------
    t : ~astropy.units.Quantity or array_like
        Times since explosion (floats are taken to be in s). Must be positive.
    E_envelope : ~astropy.units.Quantity or array_like
        Energy :math:`E_e` imparted to the envelope (floats are taken to be in erg).
    R_envelope : ~astropy.units.Quantity or array_like
        Initial envelope radius :math:`R_e` (floats are taken to be in cm).
    M_envelope : ~astropy.units.Quantity or array_like
        Envelope mass :math:`M_e` (floats are taken to be in g).
    kappa : ~astropy.units.Quantity or array_like
        Constant opacity (floats are taken to be in cm^2/g).
    n, delta : array_like, optional
        Density power-law indices; see :func:`compute_ejecta_normalization`.

    Returns
    -------
    ~astropy.units.Quantity
        Luminosity in erg/s. Inputs are broadcast against one another, so e.g. a time grid of shape ``(T,)``
        against parameters of shape ``(N, 1)`` gives an ``(N, T)`` result.

    Raises
    ------
    ValueError
        If ``t`` or a physical parameter is non-finite or not positive, if ``n <= 5``, or if ``delta`` is outside
        (1, 3).

    References
    ----------
    .. footbibliography::
    """
    t = _prepare_time(t)
    E, M, n, delta = _prepare_envelope(E_envelope, M_envelope, n, delta)
    R = ensure_in_units(R_envelope, u.cm)
    kappa = ensure_in_units(kappa, u.cm**2 / u.g)
    _check_positive(R_envelope=R, kappa=kappa)
    return _luminosity_cgs(t, E, R, M, kappa, n, delta) * (u.erg / u.s)


def compute_photospheric_radius(
    t: PhysicalInput,
    E_envelope: PhysicalInput,
    M_envelope: PhysicalInput,
    kappa: PhysicalInput,
    n: NumericalInput = DEFAULT_N,
    delta: NumericalInput = DEFAULT_DELTA,
) -> u.Quantity:
    r"""
    Photospheric radius :math:`r_\mathrm{ph}(t)`, where the optical depth is 2/3.

    Eqs. 7 and 9 of :footcite:t:`2021ApJ...909..209P`,

    .. math::

        r_\mathrm{ph}(t) = v_t t
        \begin{cases}
            (t_\mathrm{ph} / t)^{2/(n - 1)} & t \le t_\mathrm{ph} \\
            \left[\frac{\delta - 1}{n - 1}\left(t^2 / t_\mathrm{ph}^2 - 1\right) + 1\right]^{-1/(\delta - 1)}
            & t \ge t_\mathrm{ph}.
        \end{cases}

    Unlike the luminosity, this does not depend on the initial radius :math:`R_e`. Since :math:`\delta` is close to
    1, the late-time branch is very sensitive to :math:`\delta`, which the paper cautions is hard to predict
    without simulations.

    Parameters
    ----------
    t : ~astropy.units.Quantity or array_like
        Times since explosion (floats are taken to be in s). Must be positive.
    E_envelope : ~astropy.units.Quantity or array_like
        Energy :math:`E_e` imparted to the envelope (floats are taken to be in erg).
    M_envelope : ~astropy.units.Quantity or array_like
        Envelope mass :math:`M_e` (floats are taken to be in g).
    kappa : ~astropy.units.Quantity or array_like
        Constant opacity (floats are taken to be in cm^2/g).
    n, delta : array_like, optional
        Density power-law indices; see :func:`compute_ejecta_normalization`.

    Returns
    -------
    ~astropy.units.Quantity
        Photospheric radius in cm, broadcast as in :func:`compute_luminosity`.

    Raises
    ------
    ValueError
        If ``t`` or a physical parameter is non-finite or not positive, if ``n <= 5``, or if ``delta`` is outside
        (1, 3).

    References
    ----------
    .. footbibliography::
    """
    t = _prepare_time(t)
    E, M, n, delta = _prepare_envelope(E_envelope, M_envelope, n, delta)
    kappa = ensure_in_units(kappa, u.cm**2 / u.g)
    _check_positive(kappa=kappa)
    return _photospheric_radius_cgs(t, E, M, kappa, n, delta) * u.cm


def compute_temperature(
    t: PhysicalInput,
    E_envelope: PhysicalInput,
    R_envelope: PhysicalInput,
    M_envelope: PhysicalInput,
    kappa: PhysicalInput,
    n: NumericalInput = DEFAULT_N,
    delta: NumericalInput = DEFAULT_DELTA,
) -> u.Quantity:
    r"""
    Blackbody temperature :math:`T(t)` of the photosphere.

    Eq. 22 of :footcite:t:`2021ApJ...909..209P`,

    .. math::

        T(t) = \left[\frac{L(t)}{4\pi r_\mathrm{ph}^2(t)\,\sigma_\mathrm{SB}}\right]^{1/4},

    with :math:`L` from :func:`compute_luminosity` and :math:`r_\mathrm{ph}` from
    :func:`compute_photospheric_radius`.

    Parameters
    ----------
    t : ~astropy.units.Quantity or array_like
        Times since explosion (floats are taken to be in s). Must be positive.
    E_envelope : ~astropy.units.Quantity or array_like
        Energy :math:`E_e` imparted to the envelope (floats are taken to be in erg).
    R_envelope : ~astropy.units.Quantity or array_like
        Initial envelope radius :math:`R_e` (floats are taken to be in cm).
    M_envelope : ~astropy.units.Quantity or array_like
        Envelope mass :math:`M_e` (floats are taken to be in g).
    kappa : ~astropy.units.Quantity or array_like
        Constant opacity (floats are taken to be in cm^2/g).
    n, delta : array_like, optional
        Density power-law indices; see :func:`compute_ejecta_normalization`.

    Returns
    -------
    ~astropy.units.Quantity
        Temperature in K, broadcast as in :func:`compute_luminosity`.

    Raises
    ------
    ValueError
        If ``t`` or a physical parameter is non-finite or not positive, if ``n <= 5``, or if ``delta`` is outside
        (1, 3).

    References
    ----------
    .. footbibliography::
    """
    t = _prepare_time(t)
    E, M, n, delta = _prepare_envelope(E_envelope, M_envelope, n, delta)
    R = ensure_in_units(R_envelope, u.cm)
    kappa = ensure_in_units(kappa, u.cm**2 / u.g)
    _check_positive(R_envelope=R, kappa=kappa)
    return _temperature_cgs(t, E, R, M, kappa, n, delta) * u.K
