r"""
Reusable statistical operations.

Exact binomial confidence intervals, distribution-free tolerance limits, a logit transform of binomial counts, and
propagation of fractional errors. Everything here is a plain function of arrays and numbers.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logit
from scipy.stats import beta, binom

__all__ = [
    "binomial_logit",
    "clopper_pearson_interval",
    "combine_fractional_errors",
    "convert_CI_to_fractional",
    "fractional_error_to_bounds",
    "tolerance_limit",
    "tolerance_min_samples",
    "tolerance_rank",
]


# --------------------------------------------------------------------------- #
# Binomial proportions                                                        #
# --------------------------------------------------------------------------- #
def clopper_pearson_interval(k: int, n: int, confidence: float) -> tuple[float, float]:
    r"""
    Compute the exact (Clopper-Pearson) confidence interval on a binomial proportion.

    The interval is central, with probability :math:`(1-C)/2` in each tail. Its limits are quantiles of Beta
    distributions, except at the boundaries where they are fixed: the lower limit is 0 for :math:`k=0` and the upper
    limit is 1 for :math:`k=n`.

    Parameters
    ----------
    k : int
        Number of successes, ``0 <= k <= n``.
    n : int
        Number of trials. For ``n = 0`` the proportion is unconstrained and the interval is ``(0, 1)``.
    confidence : float
        Confidence level :math:`C`, in ``(0, 1)``.

    Returns
    -------
    tuple of float
        The ``(lower, upper)`` limits.
    """
    if n == 0:
        return (0.0, 1.0)

    alpha = 1.0 - confidence
    lower = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    upper = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lower), float(upper))


def binomial_logit(k: ArrayLike, n: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""
    Transform binomial counts to the logit scale, with the variance of the transformed value.

    The proportion is estimated as :math:`\hat p=(k+1/2)/(n+1)`, which stays strictly inside ``(0, 1)`` for any
    ``k`` and ``n``, and the variance is the delta-method one, :math:`1/[(n+1)\hat p(1-\hat p)]`.

    Parameters
    ----------
    k, n : array-like
        Successes and trials.

    Returns
    -------
    logit : numpy.ndarray
        :math:`\mathrm{logit}\,\hat p`.
    variance : numpy.ndarray
        Its variance.
    """
    k = np.asarray(k, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    p_hat = (k + 0.5) / (n + 1.0)
    return logit(p_hat), 1.0 / ((n + 1.0) * p_hat * (1.0 - p_hat))


# --------------------------------------------------------------------------- #
# Tolerance limits                                                            #
# --------------------------------------------------------------------------- #
def _check_probabilities(confidence: float, tolerance: float) -> None:
    """Raise `ValueError` unless both probabilities are in ``(0, 1)``."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"'confidence' must be in (0, 1), got {confidence}.")
    if not 0.0 < tolerance < 1.0:
        raise ValueError(f"'tolerance' must be in (0, 1), got {tolerance}.")


def tolerance_min_samples(confidence: float, tolerance: float) -> int:
    r"""
    Find the fewest samples for which the sample maximum is a valid upper tolerance limit.

    Parameters
    ----------
    confidence : float
        Required confidence, in ``(0, 1)``.
    tolerance : float
        Largest allowed fraction of the population beyond the limit, in ``(0, 1)``.

    Returns
    -------
    int
        :math:`\lceil\ln(1-C)/\ln(1-\delta)\rceil` for confidence :math:`C` and tolerance :math:`\delta`: 299 for 95%
        and 1%.
    """
    _check_probabilities(confidence, tolerance)
    return int(np.ceil(np.log1p(-confidence) / np.log1p(-tolerance)))


def tolerance_rank(n_samples: int, confidence: float, tolerance: float) -> int:
    r"""
    Find the rank of the sample value that is the tightest valid upper tolerance limit (Wilks 1941).

    The ``r``-th largest of ``M`` samples leaves at most a fraction :math:`\delta` of the population above it with
    confidence :math:`P[\mathrm{Binomial}(M, \delta)\geq r]`. This returns the largest ``r`` for which that still
    reaches the required confidence.

    Parameters
    ----------
    n_samples : int
        Number of samples, ``M``.
    confidence : float
        Required confidence, in ``(0, 1)``.
    tolerance : float
        Largest allowed fraction of the population beyond the limit, in ``(0, 1)``.

    Returns
    -------
    int
        The rank ``r >= 1``, counted from the largest sample.

    Raises
    ------
    ValueError
        If ``n_samples`` is below `tolerance_min_samples`.
    """
    _check_probabilities(confidence, tolerance)
    n_min = tolerance_min_samples(confidence, tolerance)
    if n_samples < n_min:
        raise ValueError(
            f"{n_samples} samples cannot support confidence={confidence} at tolerance={tolerance}; "
            f"at least {n_min} are needed."
        )
    # P(Binomial >= r) >= confidence  <=>  cdf(r - 1) <= 1 - confidence, and the cdf is increasing.
    j = np.arange(n_samples + 1)
    return int(np.count_nonzero(binom.cdf(j, n_samples, tolerance) <= 1.0 - confidence))


def tolerance_limit(values: ArrayLike, confidence: float, tolerance: float) -> tuple[float, int, int]:
    """
    Compute a distribution-free upper tolerance limit from a sample.

    With probability `confidence`, at most a fraction `tolerance` of the population lies above the limit.

    Parameters
    ----------
    values : array-like
        The sample. An infinite value is one that lies beyond any finite limit (a censored measurement).
    confidence : float
        Required confidence, in ``(0, 1)``.
    tolerance : float
        Largest allowed fraction of the population beyond the limit, in ``(0, 1)``.

    Returns
    -------
    limit : float
        The sample value at rank `tolerance_rank`. It is infinite if that value is, which means the limit lies
        beyond what the sample measured.
    rank : int
        The rank used, counted from the largest value.
    n_infinite : int
        Number of infinite values in the sample.

    Raises
    ------
    ValueError
        If the sample is smaller than `tolerance_min_samples`.
    """
    values = np.asarray(values, dtype=np.float64)
    rank = tolerance_rank(values.size, confidence, tolerance)
    limit = float(np.partition(values, values.size - rank)[values.size - rank])
    return limit, rank, int(np.isinf(values).sum())


# --------------------------------------------------------------------------- #
# Fractional errors                                                           #
# --------------------------------------------------------------------------- #
def convert_CI_to_fractional(value: ArrayLike, upper: ArrayLike, lower: ArrayLike) -> tuple:
    """
    Convert an asymmetric confidence interval to fractional errors.

    Parameters
    ----------
    value : array-like
        The central value.
    upper, lower : array-like
        The upper and lower limits of the interval.

    Returns
    -------
    tuple
        The ``(lower, upper)`` fractional errors, ``(value - lower) / value`` and ``(upper - value) / value``.
    """
    return (value - lower) / value, (upper - value) / value


def combine_fractional_errors(*errors: tuple[float, float]) -> tuple[float, float]:
    r"""
    Combine the fractional errors of independent factors of a product.

    This is linear error propagation of :math:`\ln\prod_i x_i=\sum_i\ln x_i`, so the lower and upper errors each add
    in quadrature.

    Parameters
    ----------
    *errors : tuple of float
        A ``(lower, upper)`` fractional error for each factor.

    Returns
    -------
    tuple of float
        The ``(lower, upper)`` fractional error of the product.
    """
    lower = float(np.sqrt(sum(lo**2 for lo, _ in errors)))
    upper = float(np.sqrt(sum(hi**2 for _, hi in errors)))
    return lower, upper


def fractional_error_to_bounds(error: tuple[float, float]) -> tuple[float, float]:
    """
    Convert a fractional error of a positive quantity to multiplicative bounds on it.

    Parameters
    ----------
    error : tuple of float
        The ``(lower, upper)`` fractional error.

    Returns
    -------
    tuple of float
        The factors ``(max(0, 1 - lower), 1 + upper)`` that the value is multiplied by to give its lower and upper
        bounds. The lower factor is floored at zero, since a positive quantity cannot have a negative bound.
    """
    lower, upper = error
    return max(0.0, 1.0 - lower), 1.0 + upper
