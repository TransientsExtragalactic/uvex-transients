"""Shared statistical helpers for `EventCatalog`/`PhotometryCatalog`'s yield/detection estimators."""

from scipy.stats import beta as _beta_dist


def clopper_pearson_interval(k: int, n: int, confidence: float) -> tuple[float, float]:
    r"""
    Central Clopper-Pearson binomial confidence interval on :math:`k/n`.

    Implements :ref:`yield-statistics`'s "Confidence bounds from the simulated
    catalog" section exactly, including its boundary conventions -- ``k=0``, ``k=n``,
    and ``n=0`` are each handled as an explicit special case rather than left to the
    general Beta-quantile formula, which is singular at those points:

    .. math::

        \epsilon_{\mathrm L}=
        \begin{cases}
          0, & k=0,\\
          Q_{\rm B}(\alpha/2;k,n-k+1), & k>0,
        \end{cases}
        \qquad
        \epsilon_{\mathrm U}=
        \begin{cases}
          1, & k=n,\\
          Q_{\rm B}(1-\alpha/2;k+1,n-k), & k<n,
        \end{cases}

    where :math:`Q_{\rm B}` is the Beta-distribution quantile function. For an empty
    catalog (:math:`n=0`), the efficiency is unidentified; per the doc, this returns
    ``(0.0, 1.0)`` -- the widest possible interval, not a degenerate point.

    Parameters
    ----------
    k : int
        Number of "successes" (detections), ``0 <= k <= n``.
    n : int
        Number of trials (feasible Monte Carlo draws).
    confidence : float
        Confidence level :math:`C=1-\alpha`, in ``(0, 1)``.

    Returns
    -------
    tuple of float
        ``(lower, upper)`` bounds on the true binomial proportion.
    """
    if n == 0:
        return (0.0, 1.0)

    alpha = 1.0 - confidence
    lower = 0.0 if k == 0 else _beta_dist.ppf(alpha / 2, k, n - k + 1)
    upper = 1.0 if k == n else _beta_dist.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lower), float(upper))
