r"""Helpers for the detection-horizon calculation behind ``SpectralModel.get_observability_curve``."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from astropy import units as u
from astropy.units import Quantity
from scipy.stats import binom
from synphot import SpectralElement

__all__ = [
    "EffectivePeakGrid",
    "first_crossing",
    "min_samples",
    "reduce_bands",
    "resolve_bandpasses",
    "resolve_time_grid",
    "tolerance_limit",
    "wilks_rank",
]

Criterion = Literal["any", "all", "k_of_n"]


@dataclass(frozen=True)
class EffectivePeakGrid:
    """
    Effective peak magnitudes of a sampled population on a redshift grid.

    Returned by
    :meth:`~uvex_transients.models.core.base.SpectralModel.get_effective_peak_magnitudes` and
    consumed by :meth:`~uvex_transients.models.core.base.SpectralModel.get_observability_curve`.
    It is the expensive intermediate product of the horizon calculation, and does not depend on
    any magnitude limit, so one grid can be reused for as many limits, confidences and tolerances
    as needed.

    "Effective peak" means the brightest magnitude a draw ever reaches (the minimum over the
    rest-frame time grid) at a given redshift, combined over bands as `criterion` says.

    Attributes
    ----------
    m_eff : numpy.ndarray
        Effective peak AB magnitudes, shape ``(M, Z)``: one row per parameter draw, one column
        per redshift. ``inf`` means the draw emits no flux at all at that redshift. ``nan`` means
        the model has no valid epoch on the time grid for that draw (some models return ``nan``
        outside their validity window), and counts as undetectable.
    z_grid : numpy.ndarray
        The redshift grid, shape ``(Z,)``, strictly increasing.
    criterion : {"any", "all", "k_of_n"}
        How bands were combined: detected in at least one band, in every band, or in at
        least `k` bands.
    k : int or None
        The ``k`` of ``"k_of_n"``; `None` for the other criteria.
    bands : tuple of str
        Names of the bands combined.
    t_rest : numpy.ndarray
        The rest-frame time grid, in seconds.
    parameters : dict
        The ``M`` sampled SED parameter values, as ``{name: array of shape (M,)}`` in physical
        units. Row ``i`` of `m_eff` belongs to element ``i`` of each array, which makes it
        possible to ask which parameter values set the horizon.
    """

    m_eff: np.ndarray
    z_grid: np.ndarray
    criterion: str
    k: int | None
    bands: tuple[str, ...]
    t_rest: np.ndarray
    parameters: dict

    @property
    def n_samples(self) -> int:
        """int: Number of parameter draws, ``M``."""
        return self.m_eff.shape[0]


def reduce_bands(peak_mags: np.ndarray, criterion: Criterion = "any", k: int | None = None) -> np.ndarray:
    """
    Combine per-band peak magnitudes into one effective peak magnitude.

    Magnitudes are backwards (smaller is brighter), so the transient counts as detected at a
    given limit when the returned value is at most that limit.

    Parameters
    ----------
    peak_mags : numpy.ndarray
        Per-band peak magnitudes, shape ``(..., B)``.
    criterion : {"any", "all", "k_of_n"}
        ``"any"``: detected in at least one band (minimum over bands). ``"all"``: detected in
        every band (maximum over bands). ``"k_of_n"``: detected in at least `k` bands (the
        `k`-th smallest).
    k : int, optional
        Required for ``"k_of_n"``, with ``1 <= k <= B``.

    Returns
    -------
    numpy.ndarray
        Shape ``(...)``.
    """
    n_bands = peak_mags.shape[-1]
    if criterion == "any":
        return peak_mags.min(axis=-1)
    if criterion == "all":
        return peak_mags.max(axis=-1)
    if criterion == "k_of_n":
        if k is None or not 1 <= k <= n_bands:
            raise ValueError(f"'k' must be an integer in [1, {n_bands}] for criterion 'k_of_n', got {k!r}.")
        return np.partition(peak_mags, k - 1, axis=-1)[..., k - 1]
    raise ValueError(f"Unknown criterion {criterion!r}; expected 'any', 'all' or 'k_of_n'.")


def resolve_time_grid(
    t_rest: Quantity | None, t_min: Quantity | None, t_max: Quantity | None, n_time: int
) -> np.ndarray:
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


def resolve_bandpasses(
    bandpasses: Mapping[str, SpectralElement] | Sequence[SpectralElement],
) -> tuple[tuple[str, ...], list[tuple[np.ndarray, np.ndarray]]]:
    """
    Reduce bandpasses to the ``(frequency, throughput)`` arrays that band integration needs.

    Parameters
    ----------
    bandpasses : Mapping[str, ~synphot.SpectralElement] or sequence of ~synphot.SpectralElement
        The bands. A sequence is named ``band0``, ``band1``, and so on.

    Returns
    -------
    names : tuple of str
        Band names, in the order of `grids`.
    grids : list of tuple of numpy.ndarray
        For each band, its frequency samples in Hz and the dimensionless throughput at each.

    Raises
    ------
    ValueError
        If `bandpasses` is empty.
    """
    if isinstance(bandpasses, Mapping):
        names = tuple(bandpasses)
        elements = list(bandpasses.values())
    else:
        elements = list(bandpasses)
        names = tuple(f"band{i}" for i in range(len(elements)))
    if not elements:
        raise ValueError("'bandpasses' must contain at least one bandpass.")

    grids = []
    for element in elements:
        wave = element.waveset
        grids.append(
            (wave.to_value(u.Hz, equivalencies=u.spectral()), element(wave).to_value(u.dimensionless_unscaled))
        )
    return names, grids


def min_samples(confidence: float, tolerance: float) -> int:
    """
    Smallest number of draws for which the sample maximum is a valid tolerance limit.

    Parameters
    ----------
    confidence : float
        Required confidence, in (0, 1).
    tolerance : float
        Allowed population fraction beyond the limit, in (0, 1).

    Returns
    -------
    int
        ``ceil(ln(1 - confidence) / ln(1 - tolerance))``.
    """
    _check_probabilities(confidence, tolerance)
    return int(np.ceil(np.log1p(-confidence) / np.log1p(-tolerance)))


def wilks_rank(n_samples: int, confidence: float, tolerance: float) -> int:
    r"""
    Largest rank ``r`` whose ``r``-th largest draw is a valid tolerance limit.

    The ``r``-th largest of ``M`` draws bounds the population from above, leaving at most a
    fraction `tolerance` beyond it, with confidence :math:`P(\mathrm{Binomial}(M, \delta)
    \geq r)`. A larger ``r`` gives a tighter (smaller) limit but a lower confidence, so this
    returns the largest ``r`` that still meets `confidence`.

    Parameters
    ----------
    n_samples : int
        Number of draws, ``M``.
    confidence : float
        Required confidence, in (0, 1).
    tolerance : float
        Allowed population fraction beyond the limit, in (0, 1).

    Returns
    -------
    int
        The rank ``r >= 1``.

    Raises
    ------
    ValueError
        If even the sample maximum (``r = 1``) does not reach `confidence`, i.e. ``n_samples <
        min_samples(confidence, tolerance)``.
    """
    _check_probabilities(confidence, tolerance)
    n_min = min_samples(confidence, tolerance)
    if n_samples < n_min:
        raise ValueError(
            f"{n_samples} samples cannot support confidence={confidence} at tolerance={tolerance}; "
            f"at least {n_min} are needed."
        )
    # P(Bin >= r) >= confidence  <=>  cdf(r - 1) <= 1 - confidence, and the cdf is increasing.
    j = np.arange(n_samples + 1)
    return int(np.count_nonzero(binom.cdf(j, n_samples, tolerance) <= 1.0 - confidence))


def first_crossing(m_eff: np.ndarray, z_grid: np.ndarray, mag_limit: float, *, interpolate: bool = False) -> np.ndarray:
    r"""
    Find the detection redshift of each draw: the first grid redshift where it is fainter than the limit.

    Only the first crossing counts. A K-correction can make an event dip back below the limit at
    higher redshift, and that is deliberately ignored.

    Parameters
    ----------
    m_eff : numpy.ndarray
        Effective peak magnitudes, shape ``(M, Z)``.
    z_grid : numpy.ndarray
        Redshift grid, shape ``(Z,)``, strictly increasing.
    mag_limit : float
        AB magnitude limit.
    interpolate : bool, optional
        If `False` (the default), return the grid redshift ``z_grid[k]`` of the first
        non-detection. That is an upper bound on the true crossing, so the tolerance limit
        built from it stays conservative (a ceiling on the horizon). If `True`, interpolate
        linearly in magnitude against :math:`\log(1+z)` between ``k - 1`` and ``k``. That is a
        tighter estimate of the crossing but no longer guaranteed to be an upper bound.

    Returns
    -------
    numpy.ndarray
        Shape ``(M,)``. ``inf`` for a draw that never drops below the limit inside the grid
        (censored); ``z_grid[0]`` for a draw already fainter than the limit at ``z_grid[0]``.
    """
    faint = ~(m_eff <= mag_limit)  # NaN (no valid epoch) counts as not detected
    crosses = faint.any(axis=1)
    k = faint.argmax(axis=1)
    z_det = np.where(crosses, z_grid[k], np.inf)
    if not interpolate:
        return z_det

    rows = np.flatnonzero(crosses & (k > 0))
    lo, hi = k[rows] - 1, k[rows]
    m_lo, m_hi = m_eff[rows, lo], m_eff[rows, hi]
    frac = (mag_limit - m_lo) / (m_hi - m_lo)
    log_z = np.log1p(z_grid[lo]) + frac * (np.log1p(z_grid[hi]) - np.log1p(z_grid[lo]))
    z_det[rows] = np.expm1(log_z)
    return z_det


def tolerance_limit(z_det: np.ndarray, confidence: float, tolerance: float) -> tuple[float, int, int]:
    """
    Upper tolerance limit on the horizon from a sample of detection redshifts.

    Parameters
    ----------
    z_det : numpy.ndarray
        Detection redshift of each draw, shape ``(M,)``; ``inf`` marks a censored draw.
    confidence : float
        Required confidence, in (0, 1).
    tolerance : float
        Allowed population fraction beyond the limit, in (0, 1).

    Returns
    -------
    z_limit : float
        The ``r``-th largest value of `z_det`. ``inf`` if that draw is censored, meaning the
        redshift grid was too short to bracket the limit.
    rank : int
        The rank ``r`` used. See :func:`wilks_rank`.
    n_censored : int
        Number of draws that never crossed the limit inside the grid.
    """
    z_det = np.asarray(z_det, dtype=np.float64)
    rank = wilks_rank(z_det.size, confidence, tolerance)
    z_limit = float(np.partition(z_det, z_det.size - rank)[z_det.size - rank])
    return z_limit, rank, int(np.isinf(z_det).sum())


def _check_probabilities(confidence: float, tolerance: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"'confidence' must be in (0, 1), got {confidence}.")
    if not 0.0 < tolerance < 1.0:
        raise ValueError(f"'tolerance' must be in (0, 1), got {tolerance}.")
