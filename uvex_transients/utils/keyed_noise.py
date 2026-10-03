"""Stateless, key-addressed standard-normal draws, for noise that must not depend on draw order."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtri

__all__ = ["keyed_standard_normal", "time_key"]

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MUL_1 = np.uint64(0xBF58476D1CE4E5B9)
_MUL_2 = np.uint64(0x94D049BB133111EB)
_TWO_POW_53 = float(2**53)


def _mix(x: NDArray[np.uint64]) -> NDArray[np.uint64]:
    """
    Apply the splitmix64 finalizer, a bijection on ``uint64`` with strong avalanche behavior.

    Parameters
    ----------
    x : numpy.ndarray
        ``uint64`` array.

    Returns
    -------
    numpy.ndarray
        ``uint64`` array of the same shape.
    """
    x = x + _GOLDEN
    x = (x ^ (x >> np.uint64(30))) * _MUL_1
    x = (x ^ (x >> np.uint64(27))) * _MUL_2
    return x ^ (x >> np.uint64(31))


def keyed_standard_normal(
    seed: ArrayLike,
    observation_key: ArrayLike,
    band_index: ArrayLike = 0,
) -> NDArray[np.float64]:
    """
    Draw standard-normal noise that depends only on a key, not on draw order.

    Each output element is a pure function of its own ``(seed, observation_key,
    band_index)`` triple. Unlike a draw from a sequential `numpy.random.Generator`, the
    value for one key is the same no matter which other keys are requested alongside it,
    in what order, or in how many batches. That lets a chunked iterator and a per-event
    photometry simulation reproduce the exact same noise for the same measurement.

    The three integers are combined by chained splitmix64 finalizers (not by sums or
    products, which would collide for swapped keys), the result is turned into a uniform
    on the open interval (0, 1), and `scipy.special.ndtri` maps it to a normal draw.

    Parameters
    ----------
    seed : int or array-like of int
        Per-event seed (a non-negative integer below ``2**64``).
    observation_key : int or array-like of int
        Integer identifying the observation, e.g. `time_key` of its start time.
    band_index : int or array-like of int, optional
        Integer id of the band (its position in the detector's bandpass list). Distinct
        bands of one observation get independent draws. Defaults to 0.

    Returns
    -------
    numpy.ndarray
        Standard-normal draws, with the shape the three inputs broadcast to.

    Raises
    ------
    OverflowError
        If any key is negative or does not fit in ``uint64``.
    """
    seed, observation_key, band_index = np.broadcast_arrays(
        np.asarray(seed, dtype=np.uint64),
        np.asarray(observation_key, dtype=np.uint64),
        np.asarray(band_index, dtype=np.uint64),
    )
    with np.errstate(over="ignore"):
        h = _mix(seed)
        h = _mix(h + observation_key)
        h = _mix(h + band_index)

    # Top 53 bits, offset by half a step, so the uniform is never exactly 0 or 1.
    uniform = ((h >> np.uint64(11)).astype(np.float64) + 0.5) / _TWO_POW_53
    return ndtri(uniform)


def time_key(time) -> NDArray[np.uint64]:
    """
    Turn observation start times into integer keys for `keyed_standard_normal`.

    The key is the bit pattern of each time's Julian date as a ``float64``. Any two code
    paths that read the same schedule row get the same key, with no need to share a row
    index, so the survey-wide cuts and per-event photometry draw identical noise for the
    same observation. Two observations of one spacecraft never share a start time, so
    keys are unique within a schedule.

    Parameters
    ----------
    time : ~astropy.time.Time
        Observation start time(s).

    Returns
    -------
    numpy.ndarray
        ``uint64`` keys, one per time.
    """
    return np.ascontiguousarray(np.atleast_1d(time.jd), dtype=np.float64).view(np.uint64)
