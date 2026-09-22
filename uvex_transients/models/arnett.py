r"""
Utilities for computing Arnett-style (diffusion-limited) bolometric lightcurves.

Implements the semi-analytic model of :footcite:t:`2017ApJ...850...55N` (Sec. 2; the MOSFiT magnetar model),
building on Arnett (1982) and the gamma-ray leakage treatment of :footcite:t:`2015ApJ...799..107W`: an internal
energy-injection rate :math:`F_\mathrm{in}(t)` is diffused through a homologously expanding, grey-opacity ejecta,

.. math::

    L(t) = e^{-(t/t_d)^2}\left(1 - e^{-A/t^2}\right)
           \int_0^t 2 F_\mathrm{in}(t')\,\frac{t'}{t_d}\,e^{(t'/t_d)^2}\,\frac{dt'}{t_d},

with diffusion time :math:`t_d = \sqrt{2\kappa M_\mathrm{ej} / (\beta c v_\mathrm{ej})}` and high-energy leakage
parameter :math:`A = 3\kappa_\gamma M_\mathrm{ej} / (4\pi v_\mathrm{ej}^2)`.

Everything private here works on bare CGS floats/arrays. The two public functions,
:func:`get_magnetar_engine` and :func:`compute_arnett_luminosity`, accept unit-aware
:class:`~astropy.units.Quantity` inputs (unitless inputs are assumed to already be in CGS) and convert at the
boundary.

References
----------
.. footbibliography::
"""

from collections.abc import Callable
from math import exp, expm1

import numpy as np
from astropy import units as u
from numba import njit

from uvex_transients.models._constants import (
    C_CGS,
    MSUN_G,
    cobalt_decay_time,
    cobalt_decay_yield,
    nickel_decay_time,
    nickel_decay_yield,
)
from uvex_transients.models._typing import FloatArray, FloatValue, PhysicalInput
from uvex_transients.models._utils import ensure_in_units

__all__ = [
    "GAUSS_CGS",
    "get_magnetar_engine",
    "compute_arnett_luminosity",
]


GAUSS_CGS = u.g**0.5 * u.cm**-0.5 / u.s
"""astropy.units.Unit: The Gaussian-cgs gauss (numerically 1 G). Unlike astropy's SI-based ``u.G``, it decomposes
into cgs base units, so it survives the ``.cgs`` conversion applied to every model parameter."""


def _magnetic_field_to_gauss(magnetic_field: PhysicalInput) -> FloatArray:
    """Strip a magnetic field to gauss, accepting ``u.G``/``u.T`` as well as :data:`GAUSS_CGS`."""
    if isinstance(magnetic_field, u.Quantity):
        try:
            return np.asarray(magnetic_field.to_value(u.G), dtype=np.float64)
        except u.UnitConversionError:
            return np.asarray(magnetic_field.to_value(GAUSS_CGS), dtype=np.float64)
    return np.asarray(magnetic_field, dtype=np.float64)


# ================================================= #
# Physics (CGS)                                     #
# ================================================= #
def _diffusion_time_cgs(kappa: FloatValue, ejecta_mass: FloatValue, ejecta_velocity: FloatValue) -> FloatValue:
    r"""
    Diffusion time :math:`t_d = \sqrt{2\kappa M / (\beta c v)}`, in s (Nicholl+17 Sec. 2, ``2017ApJ...850...55N``).

    ``kappa`` is in cm^2/g, ``ejecta_mass`` in g and ``ejecta_velocity`` in cm/s. The paper leaves the
    density-profile constant :math:`\beta` unstated; we use the standard value 13.8 (Arnett 1982).
    """
    return np.sqrt(2.0 * kappa * ejecta_mass / (13.8 * C_CGS * ejecta_velocity))


def _leakage_parameter_cgs(kappa_gamma: FloatValue, ejecta_mass: FloatValue, ejecta_velocity: FloatValue) -> FloatValue:
    r"""
    Leakage parameter :math:`A = 3\kappa_\gamma M / (4\pi v^2)`, in s^2.

    From Wang+15 (``2015ApJ...799..107W``), as adopted in Nicholl+17 Sec. 2.
    """
    return 3.0 * kappa_gamma * ejecta_mass / (4.0 * np.pi * ejecta_velocity**2)


def _magnetar_energy_cgs(spin_period: FloatValue, ns_mass: FloatValue) -> FloatValue:
    """Magnetar rotational energy in erg, from the spin period (s) and NS mass (g) (Nicholl+17 Sec. 2)."""
    return 2.6e52 * (ns_mass / (1.4 * MSUN_G)) ** 1.5 * (spin_period / 1e-3) ** -2


def _magnetar_timescale_cgs(spin_period: FloatValue, magnetic_field: FloatValue, ns_mass: FloatValue) -> FloatValue:
    """Magnetar spin-down time in s, from the period (s), perpendicular field (G) and NS mass (g) (Nicholl+17)."""
    return 1.3e5 * (ns_mass / (1.4 * MSUN_G)) ** 1.5 * (spin_period / 1e-3) ** 2 * (magnetic_field / 1e14) ** -2


def _magnetar_luminosity_cgs(t: FloatArray, energy: FloatValue, timescale: FloatValue) -> FloatArray:
    r"""Magnetar spin-down power :math:`F(t) = (E/t_m)(1 + t/t_m)^{-2}`, in erg/s."""
    return energy / timescale / (1.0 + np.asarray(t, dtype=np.float64) / timescale) ** 2


def _radioactive_luminosity_cgs(
    t: FloatArray,
    nickel_mass: FloatValue,
    nickel_yield: FloatValue = nickel_decay_yield.cgs.value,
    nickel_decay: FloatValue = nickel_decay_time.cgs.value,
    cobalt_yield: FloatValue = cobalt_decay_yield.cgs.value,
    cobalt_decay: FloatValue = cobalt_decay_time.cgs.value,
) -> FloatArray:
    """Compute the radioactive heating luminosity from the Ni-56 decay chain.

    The heating rate includes energy released by the decay chain

        Ni-56 -> Co-56 -> Fe-56,

    assuming that all radioactive decay energy is deposited locally. Gamma-ray
    leakage is therefore not included.

    Parameters
    ----------
    t:
        Time since explosion in seconds.
    nickel_mass:
        Initial Ni-56 mass in grams.
    nickel_yield:
        Specific heating rate from Ni-56 decay in erg s^-1 g^-1.
    nickel_decay:
        Ni-56 e-folding decay time in seconds.
    cobalt_yield:
        Specific heating rate from Co-56 decay in erg s^-1 g^-1.
    cobalt_decay:
        Co-56 e-folding decay time in seconds.

    Returns
    -------
    FloatArray
        Radioactive heating luminosity in erg s^-1.

    """
    nickel_heating = nickel_yield * np.exp(-t / nickel_decay)
    cobalt_heating = cobalt_yield * (np.exp(-t / cobalt_decay) - np.exp(-t / nickel_decay))

    return nickel_mass * (nickel_heating + cobalt_heating)


# ================================================= #
# Numerics                                          #
# ================================================= #
_MAX_GRID_CELLS = 2_000_000
"""int: Upper bound on (groups x grid points) held in memory at once; larger batches are processed in chunks."""


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


@njit(cache=True)
def _diffusion_integral_rows(dx: FloatArray, source: FloatArray) -> FloatArray:
    """Apply :func:`_diffusion_integral` to every row of ``dx`` (``(g, n-1)``) and ``source`` (``(g, n)``)."""
    out = np.empty(source.shape)
    for r in range(source.shape[0]):
        out[r] = _diffusion_integral(dx[r], source[r])
    return out


@njit(cache=True)
def _interp_rows(grid: FloatArray, values: FloatArray, row: np.ndarray, t: FloatArray) -> FloatArray:
    """Linearly interpolate ``values[row[i]]`` (sampled on ``grid[row[i]]``) at ``t[i]``, for every ``i``."""
    n = grid.shape[1]
    out = np.empty(t.size)
    for i in range(t.size):
        r = row[i]
        j = np.searchsorted(grid[r], t[i], side="right") - 1
        j = min(max(j, 0), n - 2)
        x0 = grid[r, j]
        x1 = grid[r, j + 1]
        if x1 > x0:
            w = (t[i] - x0) / (x1 - x0)
            out[i] = (1.0 - w) * values[r, j] + w * values[r, j + 1]
        else:
            out[i] = values[r, j]
    return out


def _arnett_luminosity_cgs(
    t: FloatArray,
    diffusion_time: FloatValue,
    leakage: FloatValue,
    source: Callable[..., FloatArray],
    source_params: dict[str, FloatValue] | None = None,
    *,
    n_grid: int = 1000,
    t_grid: FloatArray | None = None,
    t_end: float = 0.0,
    min_fraction: float = 1e-4,
) -> FloatArray:
    r"""
    Bolometric luminosity :math:`L(t)` of a diffusing energy source, for arbitrarily broadcast inputs.

    ``t``, ``diffusion_time``, ``leakage`` and every value of ``source_params`` are broadcast together against
    plain NumPy rules; there is no privileged time axis, so one element may be a single event at one time
    (per-event parameters) or one light curve of many times (shared parameters). Elements sharing identical
    ``(diffusion_time, leakage, source_params)`` are grouped, and each group's diffusion integral is computed
    *once* on a grid reaching its latest time, then interpolated to that group's times.

    Parameters
    ----------
    t : numpy.ndarray
        Time since explosion, in s, ``>= 0``.
    diffusion_time : float or numpy.ndarray
        Diffusion time :math:`t_d`, in s.
    leakage : float or numpy.ndarray
        Leakage parameter :math:`A`, in s^2 (``inf`` for no leakage).
    source : callable
        ``source(grid, **params)``: injected power in erg/s on ``grid`` (shape ``(g, m)``, in s), with each
        ``source_params`` value shaped ``(g, 1)``. Must return shape ``(g, m)``.
    source_params : dict, optional
        Per-element parameters of ``source``, broadcast against ``t``.
    n_grid : int, optional
        Points in each of the geometric and linear halves of the automatic grid.
    t_grid : numpy.ndarray, optional
        Absolute grid (s) shared by every group, replacing the automatic one. It is extended to reach each group's
        latest time.
    t_end : float, optional
        Minimum extent (s) of each group's grid, in place of its latest time.
    min_fraction : float, optional
        Start of the geometric grid half, as a fraction of the grid extent.

    Returns
    -------
    numpy.ndarray
        :math:`L(t)` in erg/s, with the broadcast shape. ``NaN`` in, ``NaN`` out.
    """
    source_params = source_params or {}
    t, diffusion_time, leakage, *extras = np.broadcast_arrays(
        np.asarray(t, dtype=np.float64),
        np.asarray(diffusion_time, dtype=np.float64),
        np.asarray(leakage, dtype=np.float64),
        *(np.asarray(v, dtype=np.float64) for v in source_params.values()),
    )
    shape = t.shape
    t, diffusion_time, leakage = t.ravel(), diffusion_time.ravel(), leakage.ravel()
    if t.size == 0:
        return np.empty(shape)

    keys = np.column_stack([diffusion_time, leakage, *(e.ravel() for e in extras)])
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    n_groups = unique_keys.shape[0]

    latest = np.zeros(n_groups)
    np.maximum.at(latest, inverse, np.nan_to_num(t, nan=0.0))
    extent = np.maximum(latest, t_end)

    if t_grid is None:
        fractions = np.unique(
            np.concatenate([[0.0], np.geomspace(min_fraction, 1.0, n_grid), np.linspace(0, 1, n_grid)])
        )
        n_points = fractions.size
    else:
        n_points = t_grid.size + 1

    order = np.argsort(inverse, kind="stable")
    sorted_groups = inverse[order]
    chunk = max(1, _MAX_GRID_CELLS // n_points)
    integral_at_t = np.empty(t.size)

    for start in range(0, n_groups, chunk):
        stop = min(n_groups, start + chunk)
        first, last = np.searchsorted(sorted_groups, [start, stop])
        members = order[first:last]
        keys_chunk = unique_keys[start:stop]

        if t_grid is None:
            grid = extent[start:stop, None] * fractions[None, :]
        else:
            tail = np.maximum(extent[start:stop], t_grid[-1])[:, None]
            grid = np.concatenate([np.broadcast_to(t_grid, (stop - start, t_grid.size)), tail], axis=1)
        grid = np.ascontiguousarray(grid)

        params = {name: keys_chunk[:, 2 + i, None] for i, name in enumerate(source_params)}
        power = np.ascontiguousarray(source(grid, **params), dtype=np.float64)
        if power.shape != grid.shape:
            raise ValueError(f"source returned shape {power.shape}, expected {grid.shape}")

        dx = np.ascontiguousarray(np.diff(grid**2, axis=1) / keys_chunk[:, 0, None] ** 2)
        integral = _diffusion_integral_rows(dx, power)
        integral_at_t[members] = _interp_rows(grid, integral, inverse[members] - start, t[members])

    # (1 - exp(-A/t^2)) -> 1 as t -> 0 (or A -> inf); the integral vanishes at t = 0 regardless.
    with np.errstate(divide="ignore", invalid="ignore"):
        trapped = np.where(t > 0, -np.expm1(-leakage / t**2), 1.0)

    return (trapped * integral_at_t).reshape(shape)


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
    (B_\perp/10^{14}\,\mathrm{G})^{-2}` s :footcite:p:`2017ApJ...850...55N`.

    Parameters
    ----------
    spin_period : ~astropy.units.Quantity or float
        Initial spin period :math:`P` (a float is taken to be in s).
    magnetic_field : ~astropy.units.Quantity or float
        Perpendicular dipole field :math:`B_\perp` (a float is taken to be in G). Quantities may be in ``u.G``,
        ``u.T`` or the Gaussian-cgs base unit :data:`GAUSS_CGS`.
    ns_mass : ~astropy.units.Quantity or float, optional
        Neutron star mass (a float is taken to be in g). Default ``1.4 Msun``.

    Returns
    -------
    callable
        ``engine(t)`` returning the injected power as a bare ``float64`` array in erg/s. ``t`` may be a Quantity
        (a float is taken to be in s).

    References
    ----------
    .. footbibliography::
    """
    period = float(ensure_in_units(spin_period, u.s))
    field = float(_magnetic_field_to_gauss(magnetic_field))
    mass = float(ensure_in_units(ns_mass, u.g))
    if period <= 0 or field <= 0 or mass <= 0:
        raise ValueError("spin_period, magnetic_field and ns_mass must be positive")

    energy = _magnetar_energy_cgs(period, mass)
    timescale = _magnetar_timescale_cgs(period, field, mass)

    def _engine(t: PhysicalInput) -> FloatArray:
        return _magnetar_luminosity_cgs(ensure_in_units(t, u.s), energy, timescale)

    return _engine


def get_nickel_engine(
    nickel_mass: PhysicalInput,
    nickel_yield: PhysicalInput = nickel_decay_yield,
    nickel_decay: PhysicalInput = nickel_decay_time,
    cobalt_yield: PhysicalInput = cobalt_decay_yield,
    cobalt_decay: PhysicalInput = cobalt_decay_time,
) -> Callable[[PhysicalInput], FloatArray]:
    r"""Construct a radioactive Ni-56/Co-56 heating engine.

    The returned function evaluates the instantaneous radioactive heating
    luminosity produced by the decay chain

    .. math::

        {}^{56}{\rm Ni} \to {}^{56}{\rm Co} \to {}^{56}{\rm Fe}

    assuming complete local deposition of the radioactive decay energy.
    Gamma-ray leakage is not included.

    Parameters
    ----------
    nickel_mass : ~astropy.units.Quantity or float
        Initial Ni-56 mass. Must have dimensions of mass.
    nickel_yield : ~astropy.units.Quantity or float, optional
        Specific heating rate from Ni-56 decay. Must have dimensions of
        energy per unit mass per unit time.
    nickel_decay : ~astropy.units.Quantity or float, optional
        Ni-56 e-folding decay time. Must have dimensions of time.
    cobalt_yield : ~astropy.units.Quantity or float, optional
        Specific heating rate from Co-56 decay. Must have dimensions of
        energy per unit mass per unit time.
    cobalt_decay : ~astropy.units.Quantity or float, optional
        Co-56 e-folding decay time. Must have dimensions of time.

    Returns
    -------
    Callable[[PhysicalInput], FloatArray]
        Function that accepts the time since explosion and returns the
        radioactive heating luminosity in erg s^-1.
    """
    # Coerce model parameters to CGS units.
    nickel_mass_cgs = float(ensure_in_units(nickel_mass, u.g))
    nickel_yield_cgs = float(ensure_in_units(nickel_yield, u.erg / (u.g * u.s)))
    cobalt_yield_cgs = float(ensure_in_units(cobalt_yield, u.erg / (u.g * u.s)))
    nickel_decay_cgs = float(ensure_in_units(nickel_decay, u.s))
    cobalt_decay_cgs = float(ensure_in_units(cobalt_decay, u.s))

    def _engine(t: PhysicalInput) -> FloatArray:
        t_cgs = np.asarray(ensure_in_units(t, u.s), dtype=float)

        return _radioactive_luminosity_cgs(
            t=t_cgs,
            nickel_mass=nickel_mass_cgs,
            nickel_yield=nickel_yield_cgs,
            nickel_decay=nickel_decay_cgs,
            cobalt_yield=cobalt_yield_cgs,
            cobalt_decay=cobalt_decay_cgs,
        )

    return _engine


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
    Bolometric luminosity of an energy source diffusing through expanding ejecta.

    Follows :footcite:t:`2017ApJ...850...55N`, including the high-energy leakage term of
    :footcite:t:`2015ApJ...799..107W`.

    Parameters
    ----------
    t_eval : ~astropy.units.Quantity or array_like
        Times since explosion at which to evaluate :math:`L` (floats are taken to be in s). Must be
        non-negative.
    energy_function : callable
        Injected power :math:`F_\mathrm{in}(t)`. Called as ``energy_function(t)`` with ``t`` a bare 1D CGS array
        in s, and must return an array of the same shape in erg/s -- e.g. the callable from
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
        Explicit integration grid, replacing the automatic one; it is extended to ``max(t_eval)`` if it stops
        short, so cover ``t_eval`` for accuracy.
    n_grid : int, optional
        Number of points in each of the geometric and linear parts of the automatic grid. The result is
        second-order accurate in the grid spacing: the default 2000 gives about 3e-5 relative error.

    Returns
    -------
    ~astropy.units.Quantity
        Luminosity in erg/s, with the shape of ``t_eval``.

    Raises
    ------
    ValueError
        If ``t_eval`` or the grid is negative or non-finite, if a physical parameter is not positive, or if
        ``energy_function`` returns a non-finite or wrongly shaped result.

    References
    ----------
    .. footbibliography::
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

    grid_cgs = None
    if t_grid is not None:
        grid_cgs = np.unique(np.concatenate([[0.0], ensure_in_units(t_grid, u.s).ravel()]))
        if not np.all(np.isfinite(grid_cgs)) or np.any(grid_cgs < 0):
            raise ValueError("t_grid must be finite and non-negative")

    t_last = float(t_eval_cgs.max()) if t_eval_cgs.size else 0.0
    t_end = t_last if t_max is None else float(ensure_in_units(t_max, u.s))
    min_fraction = 1e-4
    if t_min is not None and t_end > 0:
        min_fraction = float(ensure_in_units(t_min, u.s)) / max(t_end, t_last)
        if not 0 < min_fraction < 1:
            raise ValueError("t_min must satisfy 0 < t_min < t_max")

    def _source(grid: FloatArray) -> FloatArray:
        power = np.asarray(energy_function(grid[0]), dtype=np.float64)
        if power.shape != grid[0].shape or not np.all(np.isfinite(power)):
            raise ValueError("energy_function must return a finite array with the same shape as its input")
        return power[np.newaxis, :]

    luminosity = _arnett_luminosity_cgs(
        t_eval_cgs,
        _diffusion_time_cgs(kappa_cgs, mass, velocity),
        _leakage_parameter_cgs(kappa_gamma_cgs, mass, velocity),
        _source,
        n_grid=n_grid,
        t_grid=grid_cgs,
        t_end=t_end,
        min_fraction=min_fraction,
    )
    return luminosity * (u.erg / u.s)
