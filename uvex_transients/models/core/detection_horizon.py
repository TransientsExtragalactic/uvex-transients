r"""
Detection horizons: the redshift beyond which a population is too faint to detect.

A transient class is a population: its SED's parameter priors describe the spread of luminosities, timescales and
colours. For a magnitude limit :math:`m`, the *horizon* :math:`z_L(m)` is the redshift beyond which, with probability
``confidence``, at most a fraction ``tolerance`` of that population is brighter than :math:`m` in the detecting bands
(95% and 1% by default). Sampling events beyond it wastes effort on events that are, to that confidence, undetectable.

The calculation has three steps, one function each:

1. `effective_peak_grid` draws parameters from the priors and finds, for every draw at every redshift of a grid, its
   *effective peak magnitude*: the brightest magnitude it reaches, combined over bands (`reduce_bands`). This is the
   expensive step. It does not depend on the magnitude limit, and its result is an `EffectivePeakGrid`.
2. `first_crossing` finds each draw's *detection redshift* for a limit: the first grid redshift at which it is
   fainter than the limit.
3. `observability_curve` takes the upper tolerance limit of those redshifts
   (`~uvex_transients.utils.stat_utils.tolerance_limit`), one for each magnitude limit.

`detection_horizon` runs all three and widens the redshift grid until every limit is bracketed. The SED and transient
classes expose these steps as `~uvex_transients.models.core.base.SpectralModel.get_effective_peak_magnitudes`,
`~uvex_transients.models.core.base.SpectralModel.get_observability_curve` and
`~uvex_transients.transients.base.ExtragalacticTransient.get_detection_horizon`. The horizons of the shipped classes
are stored as `~uvex_transients.transients.horizon.HorizonCurve` files.

The guarantee covers the SED's parameter priors only. Milky Way extinction and sky position are not included.
"""

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from astropy import units as u
from astropy.cosmology import FLRW
from astropy.table import QTable
from astropy.units import Quantity
from numpy.typing import NDArray
from synphot import SpectralElement
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from uvex_transients.models._utils import to_cgs_value
from uvex_transients.utils import get_rng, logger
from uvex_transients.utils.cosmology import resolve_cosmological_distances
from uvex_transients.utils.phot_utils import (
    D_LN_NU,
    band_weight_matrix,
    default_chunk_size,
    flux_to_ab_mag,
    integrate_chunk,
    resolve_bandpasses,
    restframe_frequency_grid,
)
from uvex_transients.utils.stat_utils import tolerance_limit, tolerance_rank

from .._typing import NumericalInput, RNGInput
from .lightcurve_metrics import crossing_coordinate, first_true_index, resolve_time_grid

if TYPE_CHECKING:
    from .base import SpectralModel

__all__ = [
    "EffectivePeakGrid",
    "check_sample_count",
    "detection_horizon",
    "effective_peak_grid",
    "first_crossing",
    "observability_curve",
    "reduce_bands",
]

Criterion = Literal["any", "all", "k_of_n"]
"""How bands combine into one effective peak magnitude; see `reduce_bands`."""


# --------------------------------------------------------------------------- #
# The grid                                                                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EffectivePeakGrid:
    """
    Effective peak magnitudes of a sampled population on a redshift grid.

    Returned by `effective_peak_grid` and read by `observability_curve`. It does not depend on any magnitude limit, so
    one grid serves any number of limits, confidences and tolerances.

    Attributes
    ----------
    m_eff : numpy.ndarray
        Effective peak AB magnitudes, shape ``(M, Z)``: one row per draw, one column per redshift. ``inf`` means the
        draw emits no flux at that redshift. ``nan`` means the model has no valid epoch for it on the time grid, and
        counts as undetectable.
    z_grid : numpy.ndarray
        The redshifts, shape ``(Z,)``, strictly increasing.
    criterion : {"any", "all", "k_of_n"}
        How bands were combined; see `reduce_bands`.
    k : int or None
        The ``k`` of ``"k_of_n"``, otherwise `None`.
    bands : tuple of str
        Names of the bands combined.
    t_rest : numpy.ndarray
        The rest-frame time grid in seconds.
    parameters : dict
        The ``M`` sampled parameter values, ``{name: array of shape (M,)}`` in physical units. Row ``i`` of `m_eff`
        belongs to element ``i`` of each, so one can ask which parameters set the horizon.
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
        """int: Number of draws, ``M``."""
        return self.m_eff.shape[0]


def reduce_bands(peak_mags: np.ndarray, criterion: Criterion = "any", k: int | None = None) -> np.ndarray:
    """
    Combine per-band peak magnitudes into one effective peak magnitude.

    Magnitudes run backwards, so an event is detected at a limit when the result is at most the limit.

    Parameters
    ----------
    peak_mags : numpy.ndarray
        Per-band peak magnitudes, shape ``(..., B)``.
    criterion : {"any", "all", "k_of_n"}
        ``"any"``: detected in at least one band (the minimum over bands). ``"all"``: in every band (the maximum).
        ``"k_of_n"``: in at least `k` bands (the `k`-th smallest).
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


def peak_band_magnitudes(
    eval_log_luminosity: Callable[..., NDArray],
    parameters: Mapping[str, NDArray],
    z_grid: NDArray,
    luminosity_distance: NDArray,
    band_grids: Sequence[tuple[NDArray, NDArray]],
    t_rest: NDArray,
    *,
    chunk_size: int | None = None,
    d_ln_nu: float = D_LN_NU,
    progress: bool = True,
) -> NDArray:
    r"""
    Brightest AB magnitude over time of each draw, in each band, at each redshift.

    Parameters
    ----------
    eval_log_luminosity : callable
        ``eval_log_luminosity(nu, t, **parameters)``: the natural log of the rest-frame :math:`L_\nu` in cgs, for
        frequencies ``nu`` (Hz) and times ``t`` (s) that broadcast against the parameters. For a
        `~uvex_transients.models.core.base.SpectralModel` this is its ``_eval``.
    parameters : Mapping[str, numpy.ndarray]
        The model parameters in cgs units, each of shape ``(N,)``, one entry per draw.
    z_grid : numpy.ndarray
        Redshifts, shape ``(Z,)``, increasing and positive.
    luminosity_distance : numpy.ndarray
        Luminosity distance at each redshift in cm, shape ``(Z,)``.
    band_grids : sequence of tuple of numpy.ndarray
        Each band's observed frequency samples (Hz) and throughput.
    t_rest : numpy.ndarray
        Rest-frame times since explosion in seconds, shape ``(T,)``.
    chunk_size : int, optional
        Draws evaluated per step. By default it is chosen so that the largest intermediate array is about 32 MB.
    d_ln_nu : float, optional
        Spacing of the rest-frame frequency grid in :math:`\ln\nu`; see
        `~uvex_transients.utils.phot_utils.restframe_frequency_grid`.
    progress : bool, optional
        Whether to show a progress bar.

    Returns
    -------
    numpy.ndarray
        Shape ``(N, B, Z)``. ``NaN`` where the model has no valid epoch for the draw, redshift and band, which is
        any epoch where the model returns ``NaN`` at a frequency the band receives light from.
    """
    n_draws = len(next(iter(parameters.values())))
    nu_rest, _ = restframe_frequency_grid(band_grids, z_grid, d_ln_nu)
    n_bands, n_z = len(band_grids), z_grid.size
    weights = band_weight_matrix(nu_rest, band_grids, z_grid).reshape(nu_rest.size, n_bands * n_z)
    support = weights > 0  # which frequencies each (band, redshift) integrates over
    # (1 + z) / (4 pi D_L^2), and the AB zero point, are applied after the sum over frequency.
    prefactor = (1.0 + z_grid) / (4.0 * np.pi * np.asarray(luminosity_distance) ** 2)

    if chunk_size is None:
        chunk_size = default_chunk_size(n_draws, t_rest.size, nu_rest.size)
    n_chunks = -(-n_draws // chunk_size)

    nu_axis, t_axis = nu_rest[np.newaxis, np.newaxis, :], t_rest[np.newaxis, :, np.newaxis]
    peak_flux = np.empty((n_draws, n_bands, n_z))
    with (
        tqdm(total=n_chunks, desc="Evaluating peak magnitudes", unit="chunk", disable=not progress) as pbar,
        logging_redirect_tqdm(loggers=[logger]),
    ):
        for start in range(0, n_draws, chunk_size):
            stop = min(start + chunk_size, n_draws)
            flux = integrate_chunk(eval_log_luminosity, parameters, start, stop, nu_axis, t_axis, weights, support)

            # The brightest valid epoch; a draw with no valid epoch at all stays NaN.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # "All-NaN slice"
                peak_flux[start:stop] = np.nanmax(flux, axis=1).reshape(stop - start, n_bands, n_z)
            pbar.update(1)

    return flux_to_ab_mag(peak_flux * prefactor)


def effective_peak_grid(
    sed: "SpectralModel",
    z_grid: NumericalInput,
    bandpasses: Mapping[str, SpectralElement] | Sequence[SpectralElement],
    *,
    t_rest: Quantity | None = None,
    t_min: Quantity | None = None,
    t_max: Quantity | None = None,
    n_time: int = 200,
    criterion: Criterion = "any",
    k: int | None = None,
    n_samples: int = 1000,
    cosmology: FLRW | None = None,
    chunk_size: int | None = None,
    progress: bool = True,
    rng: RNGInput = None,
) -> EffectivePeakGrid:
    r"""
    Evaluate a sampled population's effective peak magnitude at every redshift of a grid.

    Draws `n_samples` parameter sets from the SED's priors and evaluates each at every redshift: its light curve is
    computed on a rest-frame time grid in each band, reduced to its brightest point, and the bands are combined by
    `criterion`. The SED is evaluated once per draw, and every redshift and band is then a weighted sum over
    frequency (see `uvex_transients.utils.phot_utils`), so the cost is set by ``n_samples * n_time``.

    Parameters
    ----------
    sed : ~uvex_transients.models.core.base.SpectralModel
        The model whose priors are sampled.
    z_grid : array-like
        Redshifts, shape ``(Z,)``, strictly increasing and positive. The grid has to reach past the horizon, or the
        horizon cannot be bracketed.
    bandpasses : Mapping[str, ~synphot.SpectralElement] or sequence of ~synphot.SpectralElement
        The bands, for example ``detector.bandpasses``. A sequence is named ``band0``, ``band1``, and so on.
    t_rest : ~astropy.units.Quantity, optional
        Rest-frame times since explosion. Rest-frame times are the right choice because time dilation moves the peak
        in observed time but does not change its magnitude, so any observing cadence is irrelevant.
    t_min, t_max : ~astropy.units.Quantity, optional
        If `t_rest` is omitted, the ends of a logarithmic grid of `n_time` points. A model has no notion of its own
        duration, so one of the two ways of giving the times is required.
    n_time : int, optional
        Number of points of the logarithmic grid. It must resolve the peak: a grid that misses it makes events look
        fainter than they are, and so puts the horizon too close.
    criterion : {"any", "all", "k_of_n"}, optional
        How bands combine; see `reduce_bands`. The default is ``"any"``.
    k : int, optional
        The number of bands required for ``"k_of_n"``.
    n_samples : int, optional
        Number of draws, ``M``. `observability_curve` needs at least 299 for 95% confidence at 1% tolerance.
    cosmology : ~astropy.cosmology.FLRW, optional
        For the luminosity distances. Defaults to the configured cosmology.
    chunk_size : int, optional
        Draws evaluated per step. By default the largest intermediate array is about 32 MB. It does not change the
        result.
    progress : bool, optional
        Whether to show a progress bar. The default is `True`.
    rng : numpy.random.Generator, int or None, optional
        Random-number source for the draws.

    Returns
    -------
    EffectivePeakGrid
        The ``(n_samples, Z)`` effective peak magnitudes with the grids and the parameters that produced them.

    Raises
    ------
    ValueError
        If `z_grid` is not a 1-d, strictly increasing array of positive redshifts, if no time grid is given, or if
        `criterion` and `k` are inconsistent.

    Notes
    -----
    The redshift is a grid every draw is evaluated on, not a random draw: a draw has a horizon only if it is followed
    across redshift. The time of the peak is found separately at each redshift, because the K-correction changes which
    epoch is brightest in a band. Zero is not allowed in `z_grid`, where the luminosity distance vanishes.
    """
    z_grid = np.asarray(z_grid, dtype=np.float64)
    if z_grid.ndim != 1 or z_grid.size == 0 or z_grid[0] <= 0 or np.any(np.diff(z_grid) <= 0):
        raise ValueError("'z_grid' must be a 1-d, strictly increasing array of positive redshifts.")
    t_rest_s = resolve_time_grid(t_rest, t_min, t_max, n_time)
    band_names, band_grids = resolve_bandpasses(bandpasses)
    reduce_bands(np.zeros(len(band_grids)), criterion, k)  # fail now on a bad criterion or k

    # Everything the loop below needs, converted once: distances for every grid redshift,
    # and the sampled parameters as plain cgs numbers.
    luminosity_distance = resolve_cosmological_distances(redshift=z_grid, cosmology=cosmology)[
        "luminosity_distance"
    ].cgs.value
    samples = sed.sample_parameters(size=n_samples, rng=rng)
    cgs_samples = {name: to_cgs_value(value) for name, value in samples.items()}

    # The SED is evaluated once per draw on a rest-frame time and frequency grid, and every redshift is then a
    # change of band weights; see `uvex_transients.utils.phot_utils`.
    peak_mags = peak_band_magnitudes(
        sed._eval,
        cgs_samples,
        z_grid,
        luminosity_distance,
        band_grids,
        t_rest_s,
        chunk_size=chunk_size,
        progress=progress,
    )  # (n_samples, B, Z)
    m_eff = reduce_bands(peak_mags.transpose(0, 2, 1), criterion, k)

    if np.isnan(m_eff).any():
        logger.warning(
            "%d of %d effective peak magnitudes are NaN because the model has no valid epoch on the time "
            "grid for that draw, redshift and band. They are treated as undetectable.",
            int(np.isnan(m_eff).sum()),
            m_eff.size,
        )

    return EffectivePeakGrid(
        m_eff=m_eff,
        z_grid=z_grid,
        criterion=criterion,
        k=k,
        bands=band_names,
        t_rest=t_rest_s,
        parameters=samples,
    )


# --------------------------------------------------------------------------- #
# From the grid to a horizon                                                  #
# --------------------------------------------------------------------------- #
def first_crossing(m_eff: np.ndarray, z_grid: np.ndarray, mag_limit: float, *, interpolate: bool = False) -> np.ndarray:
    r"""
    Find each draw's detection redshift: the first grid redshift at which it is fainter than the limit.

    Only the first crossing counts. A K-correction can make an event dip back below the limit at a higher redshift,
    and that is deliberately ignored.

    Parameters
    ----------
    m_eff : numpy.ndarray
        Effective peak magnitudes, shape ``(M, Z)``.
    z_grid : numpy.ndarray
        Redshifts, shape ``(Z,)``, strictly increasing.
    mag_limit : float
        AB magnitude limit.
    interpolate : bool, optional
        If `False` (the default), return the grid redshift of the first non-detection, which is an upper bound on the
        true crossing. If `True`, interpolate linearly in magnitude against :math:`\log(1+z)` between that grid point
        and the one before. This is tighter but no longer an upper bound.

    Returns
    -------
    numpy.ndarray
        Shape ``(M,)``. ``inf`` for a draw that never gets fainter than the limit inside the grid (censored);
        ``z_grid[0]`` for a draw already fainter at ``z_grid[0]``.
    """
    faint = ~(m_eff <= mag_limit)  # NaN (no valid epoch) counts as not detected
    k, crosses = first_true_index(faint)
    z_det = np.where(crosses, z_grid[k], np.inf)
    if not interpolate:
        return z_det

    rows = np.flatnonzero(crosses & (k > 0))
    lo, hi = k[rows] - 1, k[rows]
    log_z = crossing_coordinate(np.log1p(z_grid[lo]), np.log1p(z_grid[hi]), m_eff[rows, lo], m_eff[rows, hi], mag_limit)
    z_det[rows] = np.expm1(log_z)
    return z_det


def _curve_table(
    grid: EffectivePeakGrid, limits: np.ndarray, confidence: float, tolerance: float, interpolate: bool
) -> QTable:
    """Tabulate the tolerance limit of the detection redshifts at each magnitude limit, without warning."""
    z_limit = np.empty(limits.size)
    n_censored = np.empty(limits.size, dtype=int)
    for i, mag_limit in enumerate(limits):
        z_det = first_crossing(grid.m_eff, grid.z_grid, mag_limit, interpolate=interpolate)
        z_limit[i], rank, n_censored[i] = tolerance_limit(z_det, confidence, tolerance)
    return QTable(
        {"mag_limit": limits * u.ABmag, "z_limit": z_limit, "n_censored": n_censored},
        meta={
            "confidence": confidence,
            "tolerance": tolerance,
            "n_samples": grid.n_samples,
            "rank": rank,
            "criterion": grid.criterion,
            "bands": grid.bands,
        },
    )


def _warn_if_unbracketed(table: QTable, grid: EffectivePeakGrid) -> None:
    """Warn that the redshift grid is too short for the magnitude limits whose redshift limit is infinite."""
    unbracketed = np.isinf(table["z_limit"])
    if np.any(unbracketed):
        logger.warning(
            "The redshift grid (z <= %.3g) is too short to bracket the limit for mag_limit = %s; extend 'z_grid'.",
            grid.z_grid[-1],
            np.asarray(table["mag_limit"].value)[unbracketed],
        )


def observability_curve(
    grid: EffectivePeakGrid,
    mag_limits: NumericalInput,
    *,
    confidence: float = 0.95,
    tolerance: float = 0.01,
    interpolate: bool = False,
) -> QTable:
    r"""
    Bound the redshift beyond which a population is undetectable, for each magnitude limit.

    For each limit, finds every draw's detection redshift (`first_crossing`) and takes the upper tolerance limit of
    them (`~uvex_transients.utils.stat_utils.tolerance_limit`): with probability `confidence`, at most a fraction
    `tolerance` of the population has a detection redshift above it.

    Parameters
    ----------
    grid : EffectivePeakGrid
        From `effective_peak_grid`. It needs at least ``ln(1 - confidence) / ln(1 - tolerance)`` draws, 299 for the
        defaults.
    mag_limits : float or array-like
        AB magnitude limit(s). Tabulating many costs almost nothing once the grid exists.
    confidence : float, optional
        Probability that the guarantee holds, in ``(0, 1)``. The default is 0.95.
    tolerance : float, optional
        Fraction of the population allowed beyond the limit, in ``(0, 1)``. The default is 0.01.
    interpolate : bool, optional
        If `False` (the default), each detection redshift is rounded up to a grid point, so the limit is never an
        underestimate but is coarse by up to one grid step. If `True`, the crossing is interpolated, which is tighter
        but not guaranteed to be an overestimate. See `first_crossing`.

    Returns
    -------
    ~astropy.table.QTable
        One row per magnitude limit, with columns ``mag_limit``, ``z_limit`` (``inf``, with a warning, where the
        grid does not reach the limit) and ``n_censored`` (draws that never got fainter than the limit inside the
        grid). Its ``meta`` records ``confidence``, ``tolerance``, ``n_samples``, ``rank``, ``criterion`` and
        ``bands``.

    Raises
    ------
    ValueError
        If the grid has too few draws for `confidence` and `tolerance`.
    """
    limits = np.atleast_1d(np.asarray(getattr(mag_limits, "value", mag_limits), dtype=np.float64))
    table = _curve_table(grid, limits, confidence, tolerance, interpolate)
    _warn_if_unbracketed(table, grid)
    return table


def detection_horizon(
    sed: "SpectralModel",
    mag_limits: NumericalInput,
    bandpasses: Mapping[str, SpectralElement] | Sequence[SpectralElement],
    *,
    t_max: Quantity,
    t_min: Quantity = 1e-3 * u.day,
    z_min: float = 0.003,
    z_max: float = 1.0,
    confidence: float = 0.95,
    tolerance: float = 0.01,
    n_samples: int = 1000,
    n_z: int = 30,
    n_time: int = 60,
    rng: RNGInput = None,
    progress: bool = True,
    interpolate: bool = False,
    max_extensions: int = 5,
    **kwargs,
) -> tuple[QTable, EffectivePeakGrid]:
    """
    Find the detection horizon of a population for each magnitude limit.

    Evaluates the population on a logarithmic redshift grid from `z_min` to `z_max` and doubles `z_max` until every
    limit is bracketed, with one seed for every attempt so that widening the grid does not redraw the population.

    Parameters
    ----------
    sed : ~uvex_transients.models.core.base.SpectralModel
        The model whose priors define the population.
    mag_limits : float or array-like
        AB magnitude limit(s).
    bandpasses : Mapping[str, ~synphot.SpectralElement] or sequence of ~synphot.SpectralElement
        The bands in which an event may be detected.
    t_max : ~astropy.units.Quantity
        Latest rest-frame time, usually the population's duration.
    t_min : ~astropy.units.Quantity, optional
        Earliest rest-frame time. The default is 0.001 day.
    z_min : float, optional
        Smallest redshift of the grid. A limit cannot be reported below it. The default is 0.003.
    z_max : float, optional
        Largest redshift of the first attempt. The default is 1.
    confidence, tolerance : float, optional
        The guarantee; see `observability_curve`. The defaults are 0.95 and 0.01.
    n_samples : int, optional
        Number of draws. The default is 1000.
    n_z : int, optional
        Number of logarithmically spaced redshifts. The default is 30.
    n_time : int, optional
        Number of logarithmically spaced times from `t_min` to `t_max`. It must resolve the peak. The default is 60.
    rng : numpy.random.Generator, int or None, optional
        Random-number source; give an integer for reproducible output.
    progress : bool, optional
        Whether to show a progress bar. The default is `True`.
    interpolate : bool, optional
        Interpolate the crossings; see `observability_curve`.
    max_extensions : int, optional
        How many times `z_max` may be doubled. The default is 5.
    **kwargs
        Further arguments of `effective_peak_grid`, such as ``criterion``.

    Returns
    -------
    curve : ~astropy.table.QTable
        As returned by `observability_curve`.
    grid : EffectivePeakGrid
        The grid the curve was read from, which can be reused for further limits.
    """
    limits = np.atleast_1d(np.asarray(getattr(mag_limits, "value", mag_limits), dtype=np.float64))
    seed = int(get_rng(rng).integers(2**32))

    for _ in range(max_extensions + 1):
        grid = effective_peak_grid(
            sed,
            np.geomspace(z_min, z_max, n_z),
            bandpasses,
            t_min=t_min,
            t_max=t_max,
            n_time=n_time,
            n_samples=n_samples,
            progress=progress,
            rng=seed,
            **kwargs,
        )
        table = _curve_table(grid, limits, confidence, tolerance, interpolate)
        if np.all(np.isfinite(table["z_limit"])):
            break
        z_max *= 2.0

    _warn_if_unbracketed(table, grid)
    return table, grid


def check_sample_count(n_samples: int, confidence: float, tolerance: float) -> None:
    """
    Check that a number of draws can support a confidence and tolerance, before an expensive grid is built.

    Parameters
    ----------
    n_samples : int
        Number of draws.
    confidence, tolerance : float
        The guarantee; see `observability_curve`.

    Raises
    ------
    ValueError
        If `n_samples` is below ``ln(1 - confidence) / ln(1 - tolerance)``; the message names the minimum.
    """
    tolerance_rank(n_samples, confidence, tolerance)
