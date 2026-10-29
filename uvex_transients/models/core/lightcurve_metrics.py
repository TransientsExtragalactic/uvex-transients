r"""
Reduce sampled light curves to numbers: a peak, and the time to decline from it.

The functions here work on arrays, a time grid ``t`` of shape ``(T,)`` and the luminosities ``L`` of ``N`` draws on it,
shape ``(N, T)``, from whatever source: an SED evaluated over parameter draws (see
`~uvex_transients.models.core.base.SpectralModel.eval_band_luminosity`), a bolometric curve, or data. They know nothing
about SEDs or bands.

The helpers at the end, `first_true_index` and `crossing_coordinate`, are the shared core of finding where a curve first
crosses a level and interpolating the crossing between two samples. The detection-horizon calculation uses them too, to
find the redshift at which a population's peak magnitude crosses a limit.
"""

import warnings

import numpy as np
from astropy import units as u
from astropy.units import Quantity
from numpy.typing import NDArray

__all__ = [
    "crossing_coordinate",
    "decline_time",
    "first_true_index",
    "peak_luminosity",
    "resolve_time_grid",
]


def resolve_time_grid(
    t_rest: Quantity | None, t_min: Quantity | None, t_max: Quantity | None, n_time: int
) -> NDArray[np.float64]:
    """
    Turn the user's time-grid arguments into rest-frame times in seconds.

    Parameters
    ----------
    t_rest : ~astropy.units.Quantity or None
        An explicit grid. If given, `t_min`, `t_max` and `n_time` are ignored.
    t_min, t_max : ~astropy.units.Quantity or None
        Ends of a logarithmically spaced grid, used when `t_rest` is `None`. Logarithmic spacing
        puts the resolution where light curves change fastest, near rise and peak.
    n_time : int
        Number of points of the logarithmic grid.

    Returns
    -------
    numpy.ndarray
        Times in seconds, shape ``(T,)``.

    Raises
    ------
    ValueError
        If neither an explicit grid nor both ends are given, or if ``0 < t_min < t_max`` fails.
        A model has no notion of its own duration, so there is deliberately no default.
    """
    if t_rest is not None:
        return np.atleast_1d(t_rest.to_value(u.s)).astype(np.float64)
    if t_min is None or t_max is None:
        raise ValueError("Provide either 't_rest' or both 't_min' and 't_max'.")
    t_lo, t_hi = t_min.to_value(u.s), t_max.to_value(u.s)
    if not 0 < t_lo < t_hi:
        raise ValueError("Need 0 < 't_min' < 't_max'.")
    return np.geomspace(t_lo, t_hi, n_time)


# --------------------------------------------------------------------------- #
# Peak and decline                                                            #
# --------------------------------------------------------------------------- #
def _peak_index(luminosity: Quantity) -> tuple[NDArray[np.intp], NDArray[np.bool_]]:
    """Return the grid index of each draw's brightest point, and whether the draw has any valid epoch."""
    values = np.asarray(luminosity.value)
    valid = ~np.all(np.isnan(values), axis=1)
    index = np.zeros(len(values), dtype=int)
    index[valid] = np.nanargmax(values[valid], axis=1)
    return index, valid


def peak_luminosity(t: Quantity, luminosity: Quantity) -> tuple[Quantity, Quantity]:
    r"""
    Reduce each light curve to its peak.

    The peak is found on the grid and then refined by the parabola through the grid point and its two neighbours, in
    :math:`\ln L` against :math:`\ln t`, which removes most of the error from the spacing of the grid. A grid too
    coarse to resolve the peak still underestimates it. Where the parabola is not a maximum (a peak at the end of the
    grid, a neighbour with no light), the grid point itself is returned.

    Parameters
    ----------
    t : ~astropy.units.Quantity
        Times, shape ``(T,)``, increasing and positive.
    luminosity : ~astropy.units.Quantity
        Luminosities, shape ``(N, T)``.

    Returns
    -------
    peak : ~astropy.units.Quantity
        The peak luminosity of each draw, shape ``(N,)``. ``NaN`` for a draw with no valid epoch.
    t_peak : ~astropy.units.Quantity
        The time of the peak, shape ``(N,)``.
    """
    index, valid = _peak_index(luminosity)
    values = np.asarray(luminosity.value, dtype=np.float64)
    time = np.asarray(t.value, dtype=np.float64)
    rows = np.arange(len(values))
    peak = np.where(valid, values[rows, index], np.nan)
    t_peak = np.where(valid, time[index], np.nan)

    interior = valid & (index > 0) & (index < time.size - 1)
    lo, hi = np.where(interior, index - 1, 0), np.where(interior, index + 1, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log(time)
        y0, y1, y2 = (np.log(values[rows, j]) for j in (lo, index, hi))
        x0, x1, x2 = x[lo], x[index], x[hi]
        # The parabola through three points: y = y1 + b (x - x1) + c (x - x1)^2.
        d0, d2 = x0 - x1, x2 - x1
        c = ((y0 - y1) / d0 - (y2 - y1) / d2) / (d0 - d2)
        b = (y2 - y1) / d2 - c * d2
        x_peak = x1 - b / (2.0 * c)
        y_peak = y1 - b**2 / (4.0 * c)
        refine = interior & np.isfinite(c) & (c < 0) & (x_peak >= x0) & (x_peak <= x2) & np.isfinite(y_peak)
    peak = np.where(refine, np.maximum(np.exp(np.where(refine, y_peak, 0.0)), peak), peak)
    t_peak = np.where(refine, np.exp(np.where(refine, x_peak, 0.0)), t_peak)
    return peak * luminosity.unit, t_peak * t.unit


def decline_time(t: Quantity, luminosity: Quantity, fraction: float = 0.5) -> Quantity:
    r"""
    Measure the time from the peak until the luminosity first falls to a fraction of it.

    The crossing is interpolated linearly in :math:`\ln L` against :math:`\ln t` between the two grid points that
    bracket it.

    Parameters
    ----------
    t : ~astropy.units.Quantity
        Times, shape ``(T,)``, increasing.
    luminosity : ~astropy.units.Quantity
        Luminosities, shape ``(N, T)``.
    fraction : float, optional
        The fraction of the peak, between 0 and 1. The default of 0.5 gives the half-decline time :math:`t_{1/2}`.

    Returns
    -------
    ~astropy.units.Quantity
        The decline time of each draw, shape ``(N,)``. ``NaN`` where the luminosity never falls that far within the
        grid (the draw is censored), or the draw has no valid epoch.

    Raises
    ------
    ValueError
        If `fraction` is not between 0 and 1.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"`fraction` must be between 0 and 1, got {fraction!r}.")
    values = np.asarray(luminosity.value, dtype=np.float64)
    time = np.asarray(t.value, dtype=np.float64)
    peak, t_peak = (quantity.value for quantity in peak_luminosity(t, luminosity))
    index, valid = _peak_index(luminosity)
    rows = np.arange(len(values))

    # The first grid point after the brightest one at which the light has fallen to the threshold.
    after = np.arange(time.size)[np.newaxis, :] > index[:, np.newaxis]
    with np.errstate(invalid="ignore"):
        below = after & (values <= fraction * peak[:, np.newaxis])
    j, crossed = first_true_index(below)
    crossed &= valid
    j_before = np.maximum(j - 1, 0)

    tiny = np.finfo(np.float64).tiny
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        log_t = crossing_coordinate(
            np.log(time[j_before]),
            np.log(time[j]),
            np.log(np.maximum(values[rows, j_before], tiny)),
            np.log(np.maximum(values[rows, j], tiny)),
            np.log(fraction * peak),
            clip=True,
        )
    result = np.where(crossed, np.exp(log_t) - t_peak, np.nan)
    return result * t.unit


# --------------------------------------------------------------------------- #
# Crossings                                                                   #
# --------------------------------------------------------------------------- #
def first_true_index(condition: NDArray[np.bool_]) -> tuple[NDArray[np.intp], NDArray[np.bool_]]:
    """
    Find the first sample of each row at which a condition holds.

    Parameters
    ----------
    condition : numpy.ndarray
        Boolean array, shape ``(M, K)``.

    Returns
    -------
    index : numpy.ndarray
        Shape ``(M,)``: the first column of each row where `condition` is true, and 0 for a row where it never is.
    found : numpy.ndarray
        Shape ``(M,)``: whether the row has such a column.
    """
    return condition.argmax(axis=1), condition.any(axis=1)


def crossing_coordinate(
    x_before: NDArray,
    x_after: NDArray,
    y_before: NDArray,
    y_after: NDArray,
    target: float | NDArray,
    *,
    clip: bool = False,
) -> NDArray:
    """
    Interpolate, linearly, where a curve reaches a level between two samples.

    Parameters
    ----------
    x_before, x_after : numpy.ndarray
        The coordinate at the two samples that bracket the crossing.
    y_before, y_after : numpy.ndarray
        The curve at those samples.
    target : float or numpy.ndarray
        The level the curve crosses.
    clip : bool, optional
        Whether to keep the result between the two samples. The default is `False`.

    Returns
    -------
    numpy.ndarray
        The coordinate at which the straight line between the samples reaches `target`.
    """
    weight = (target - y_before) / (y_after - y_before)
    if clip:
        weight = np.clip(weight, 0.0, 1.0)
    return x_before + weight * (x_after - x_before)
