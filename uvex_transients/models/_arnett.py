r"""
Utilities for computing Arnett-style (diffusion-limited) bolometric lightcurves.

Implements the semi-analytic model of `Nicholl, Guillochon & Berger (2017)
<https://arxiv.org/abs/1706.00825>`__ (Sec. 2; the MOSFiT magnetar model), building on Arnett (1982) and the
gamma-ray leakage treatment of Wang et al. (2015): an internal energy-injection rate :math:`F_\mathrm{in}(t)` is
diffused through a homologously expanding, grey-opacity ejecta,

.. math::

    L(t) = e^{-(t/t_d)^2}\left(1 - e^{-A/t^2}\right)
           \int_0^t 2 F_\mathrm{in}(t')\,\frac{t'}{t_d}\,e^{(t'/t_d)^2}\,\frac{dt'}{t_d},

with diffusion time :math:`t_d = \sqrt{2\kappa M_\mathrm{ej} / (\beta c v_\mathrm{ej})}` and high-energy leakage
parameter :math:`A = 3\kappa_\gamma M_\mathrm{ej} / (4\pi v_\mathrm{ej}^2)`.

Everything private here works on bare CGS floats/arrays. The two public functions,
:func:`get_magnetar_engine` and :func:`compute_arnett_luminosity`, accept unit-aware
:class:`~astropy.units.Quantity` inputs (unitless inputs are assumed to already be in CGS) and convert at the
boundary.
"""

from collections.abc import Callable
from math import exp, expm1

import numpy as np
from astropy import units as u
from numba import njit

from uvex_transients.models._constants import C_CGS, MSUN_G
from uvex_transients.models._typing import FloatArray, PhysicalInput
from uvex_transients.models._utils import ensure_in_units

__all__ = [
    "get_magnetar_engine",
    "compute_arnett_luminosity",
]


# ================================================= #
# Physics (CGS)                                     #
# ================================================= #
def _diffusion_time_cgs(kappa: float, ejecta_mass: float, ejecta_velocity: float) -> float:
    r"""
    Diffusion time :math:`t_d = \sqrt{2\kappa M / (\beta c v)}`, in s (Nicholl+17 Sec. 2).

    ``kappa`` is in cm^2/g, ``ejecta_mass`` in g and ``ejecta_velocity`` in cm/s. The paper leaves the
    density-profile constant :math:`\beta` unstated; we use the standard value 13.8 (Arnett 1982).
    """
    return float(np.sqrt(2.0 * kappa * ejecta_mass / (13.8 * C_CGS * ejecta_velocity)))


def _leakage_parameter_cgs(kappa_gamma: float, ejecta_mass: float, ejecta_velocity: float) -> float:
    r"""Leakage parameter :math:`A = 3\kappa_\gamma M / (4\pi v^2)`, in s^2 (Wang+15; Nicholl+17 Sec. 2)."""
    return 3.0 * kappa_gamma * ejecta_mass / (4.0 * np.pi * ejecta_velocity**2)


def _magnetar_energy_cgs(spin_period: float, ns_mass: float) -> float:
    """Magnetar rotational energy in erg, from the spin period (s) and NS mass (g) (Nicholl+17 Sec. 2)."""
    return 2.6e52 * (ns_mass / (1.4 * MSUN_G)) ** 1.5 * (spin_period / 1e-3) ** -2


def _magnetar_timescale_cgs(spin_period: float, magnetic_field: float, ns_mass: float) -> float:
    """Magnetar spin-down time in s, from the period (s), perpendicular field (G) and NS mass (g) (Nicholl+17)."""
    return 1.3e5 * (ns_mass / (1.4 * MSUN_G)) ** 1.5 * (spin_period / 1e-3) ** 2 * (magnetic_field / 1e14) ** -2


def _magnetar_luminosity_cgs(t: FloatArray, energy: float, timescale: float) -> FloatArray:
    r"""Magnetar spin-down power :math:`F(t) = (E/t_m)(1 + t/t_m)^{-2}`, in erg/s."""
    return energy / timescale / (1.0 + np.asarray(t, dtype=np.float64) / timescale) ** 2


# ================================================= #
# Numerics                                          #
# ================================================= #
def _build_grid(
    t_eval: FloatArray,
    t_min: float | None,
    t_max: float | None,
    t_grid: FloatArray | None,
    n: int,
) -> FloatArray:
    """
    Build the strictly increasing integration grid (in s) starting at 0.

    ``t_eval`` is always folded into the grid so results can be read off exactly rather than interpolated.
    """
    if t_grid is not None:
        extra = [np.asarray(t_grid, dtype=np.float64).ravel()]
        if not np.all(np.isfinite(extra[0])) or np.any(extra[0] < 0):
            raise ValueError("t_grid must be finite and non-negative")
    else:
        t_max = float(t_eval.max()) if t_max is None else t_max
        if t_max > 0:
            t_min = 1e-4 * t_max if t_min is None else t_min
            if not 0 < t_min < t_max:
                raise ValueError("t_min must satisfy 0 < t_min < t_max")
            extra = [np.geomspace(t_min, t_max, n), np.linspace(0.0, t_max, n)]
        else:
            extra = []

    return np.unique(np.concatenate([[0.0], t_eval, *extra]))


@njit(cache=True)
def _diffusion_integral(dx: FloatArray, source: FloatArray) -> FloatArray:
    r"""
    Evaluate :math:`e^{-u}\int_0^u F\,e^{u'}\,du'` with :math:`u = (t/t_d)^2` on a grid.

    Uses the exact-decay recurrence :math:`I_{i+1} = e^{-\Delta u_i} I_i + \bar F_i (1 - e^{-\Delta u_i})`, with
    :math:`\bar F_i` the trapezoid mean of the source across the step. Since :math:`e^{-\Delta u} \in (0, 1]`,
    it cannot overflow however late the times, unlike evaluating :math:`e^{+u}` directly.

    Parameters
    ----------
    dx : numpy.ndarray
        Steps :math:`\Delta u_i = (t_{i+1}^2 - t_i^2)/t_d^2`, length ``n - 1``.
    source : numpy.ndarray
        Source :math:`F` evaluated on the grid, length ``n``.
    """
    n = source.size
    out = np.empty(n)
    out[0] = 0.0
    for i in range(n - 1):
        out[i + 1] = exp(-dx[i]) * out[i] - 0.5 * (source[i] + source[i + 1]) * expm1(-dx[i])
    return out


# ================================================= #
# Public API                                        #
# ================================================= #
def get_magnetar_engine(
    spin_period: PhysicalInput,
    magnetic_field: PhysicalInput,
    ns_mass: PhysicalInput = 1.4 * u.Msun,
) -> Callable[[PhysicalInput], FloatArray]:
    r"""
    Build a magnetar spin-down energy source for :func:`compute_arnett_luminosity`.

    .. math::

        F_\mathrm{mag}(t) = \frac{E_\mathrm{mag}}{t_\mathrm{mag}}\frac{1}{(1 + t/t_\mathrm{mag})^2},

    with :math:`E_\mathrm{mag} = 2.6\times10^{52}\,(M_\mathrm{NS}/1.4\,M_\odot)^{3/2}(P/1\,\mathrm{ms})^{-2}` erg and
    :math:`t_\mathrm{mag} = 1.3\times10^{5}\,(M_\mathrm{NS}/1.4\,M_\odot)^{3/2}(P/1\,\mathrm{ms})^{2}
    (B_\perp/10^{14}\,\mathrm{G})^{-2}` s (Nicholl+17, Sec. 2).

    Parameters
    ----------
    spin_period : ~astropy.units.Quantity or float
        Initial spin period :math:`P` (a float is taken to be in s).
    magnetic_field : ~astropy.units.Quantity or float
        Perpendicular dipole field :math:`B_\perp` (a float is taken to be in G).
    ns_mass : ~astropy.units.Quantity or float, optional
        Neutron star mass (a float is taken to be in g). Default ``1.4 Msun``.

    Returns
    -------
    callable
        ``engine(t)`` returning the injected power as a bare ``float64`` array in erg/s. ``t`` may be a Quantity
        (a float is taken to be in s).
    """
    period = float(ensure_in_units(spin_period, u.s))
    field = float(ensure_in_units(magnetic_field, u.G))
    mass = float(ensure_in_units(ns_mass, u.g))
    if period <= 0 or field <= 0 or mass <= 0:
        raise ValueError("spin_period, magnetic_field and ns_mass must be positive")

    energy = _magnetar_energy_cgs(period, mass)
    timescale = _magnetar_timescale_cgs(period, field, mass)

    def engine(t: PhysicalInput) -> FloatArray:
        return _magnetar_luminosity_cgs(ensure_in_units(t, u.s), energy, timescale)

    return engine


def compute_arnett_luminosity(
    t_eval: PhysicalInput,
    energy_function: Callable[[FloatArray], FloatArray],
    ejecta_mass: PhysicalInput,
    ejecta_velocity: PhysicalInput,
    kappa: PhysicalInput = 0.2 * u.cm**2 / u.g,
    kappa_gamma: PhysicalInput = np.inf,
    t_min: PhysicalInput | None = None,
    t_max: PhysicalInput | None = None,
    t_grid: PhysicalInput | None = None,
    n_grid: int = 2000,
) -> u.Quantity:
    r"""
    Bolometric luminosity of an energy source diffusing through expanding ejecta (Nicholl+17, Sec. 2).

    Parameters
    ----------
    t_eval : ~astropy.units.Quantity or array_like
        Times since explosion at which to evaluate :math:`L` (floats are taken to be in s). Must be
        non-negative.
    energy_function : callable
        Injected power :math:`F_\mathrm{in}(t)`. Called as ``energy_function(t)`` with ``t`` a bare CGS array in
        s, and must return an array of the same shape in erg/s -- e.g. the callable from
        :func:`get_magnetar_engine`.
    ejecta_mass : ~astropy.units.Quantity or float
        Ejecta mass :math:`M_\mathrm{ej}` (a float is taken to be in g).
    ejecta_velocity : ~astropy.units.Quantity or float
        Constant ejecta velocity :math:`v_\mathrm{ej}` (a float is taken to be in cm/s).
    kappa : ~astropy.units.Quantity or float, optional
        Grey optical opacity (a float is taken to be in cm^2/g). Default 0.2, electron scattering in H-free
        material.
    kappa_gamma : ~astropy.units.Quantity or float, optional
        Opacity to high-energy photons (a float is taken to be in cm^2/g). Smaller values let more of the
        injected energy leak out un-thermalized at late times. The default ``inf`` means full trapping (no
        leakage).
    t_min, t_max : ~astropy.units.Quantity or float, optional
        Range of the auto-generated grid (geometric + linear). Default to ``1e-4 * t_max`` and ``max(t_eval)``.
        Ignored if ``t_grid`` is given.
    t_grid : ~astropy.units.Quantity or array_like, optional
        Explicit integration grid, replacing the automatic one. ``0`` and ``t_eval`` are always added to it.
    n_grid : int, optional
        Number of points in each of the geometric and linear parts of the automatic grid.

    Returns
    -------
    ~astropy.units.Quantity
        Luminosity in erg/s, with the shape of ``t_eval``.

    Raises
    ------
    ValueError
        If ``t_eval`` or the grid is negative or non-finite, if a physical parameter is not positive, or if
        ``energy_function`` returns a non-finite or wrongly shaped result.
    """
    t_eval_cgs = ensure_in_units(t_eval, u.s)
    if not np.all(np.isfinite(t_eval_cgs)) or np.any(t_eval_cgs < 0):
        raise ValueError("t_eval must be finite and non-negative")

    mass = float(ensure_in_units(ejecta_mass, u.g))
    velocity = float(ensure_in_units(ejecta_velocity, u.cm / u.s))
    kappa_cgs = float(ensure_in_units(kappa, u.cm**2 / u.g))
    kappa_gamma_cgs = float(ensure_in_units(kappa_gamma, u.cm**2 / u.g))
    if mass <= 0 or velocity <= 0 or kappa_cgs <= 0 or kappa_gamma_cgs < 0:
        raise ValueError("ejecta_mass, ejecta_velocity and kappa must be positive, and kappa_gamma non-negative")

    diffusion_time = _diffusion_time_cgs(kappa_cgs, mass, velocity)
    leakage = _leakage_parameter_cgs(kappa_gamma_cgs, mass, velocity)

    flat_eval = t_eval_cgs.ravel()
    grid = _build_grid(
        flat_eval,
        None if t_min is None else float(ensure_in_units(t_min, u.s)),
        None if t_max is None else float(ensure_in_units(t_max, u.s)),
        None if t_grid is None else ensure_in_units(t_grid, u.s),
        n_grid,
    )

    source = np.asarray(energy_function(grid), dtype=np.float64)
    if source.shape != grid.shape or not np.all(np.isfinite(source)):
        raise ValueError("energy_function must return a finite array with the same shape as its input")

    integral = _diffusion_integral(np.diff(grid**2) / diffusion_time**2, source)
    integral_at_eval = integral[np.searchsorted(grid, flat_eval)]

    # (1 - exp(-A/t^2)) -> 1 as t -> 0 (or A -> inf); the integral vanishes at t = 0 regardless.
    with np.errstate(divide="ignore", invalid="ignore"):
        trapped = np.where(flat_eval > 0, -np.expm1(-leakage / flat_eval**2), 1.0)

    return (trapped * integral_at_eval).reshape(t_eval_cgs.shape) * (u.erg / u.s)
