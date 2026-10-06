"""
Expected event counts, with confidence bounds, from an event summary table.

`estimate_yield` turns any selection of rows of the table made by
`~uvex_transients.simulation.core.SurveySimulator.run_event_summary_action` into, per transient
type, the fraction of generated events selected and the number of real UVEX events that implies.
Everything it needs is in the table's ``meta``, so a sliced table can be passed straight in.
"""

import numpy as np
from astropy.table import QTable

from uvex_transients.utils import config

from ._stats import clopper_pearson_interval

_COLUMNS = (
    "transient_type",
    "n_pre_cut",
    "n_selected",
    "fraction",
    "fraction_lower",
    "fraction_upper",
    "intrinsic_events",
    "expected_events",
    "expected_events_binom_lower",
    "expected_events_binom_upper",
    "expected_events_rate_lower",
    "expected_events_rate_upper",
)


def estimate_yield(
    events: QTable,
    mask=None,
    confidence: float | None = None,
    transient_types: list[str] | None = None,
) -> QTable:
    r"""
    Estimate the expected number of real UVEX events in a selection of `events`, per transient type.

    For each type, with :math:`n` the number of events generated, :math:`k` the number of selected
    rows and :math:`\mu_0` the intrinsic number of events the survey could see, the fraction is
    :math:`\hat\epsilon=k/n` and the expected number of real events is
    :math:`\hat\lambda=\mu_0\hat\epsilon`. Two uncertainties are reported separately: a
    Clopper-Pearson interval on :math:`k/n` for the finite Monte Carlo sample (``binom``), and
    the type's rate-normalization factors applied to :math:`\hat\lambda` (``rate``).

    Parameters
    ----------
    events : ~astropy.table.QTable
        A table from `~uvex_transients.simulation.core.SurveySimulator.run_event_summary_action`,
        or a row slice of one. Its ``meta`` must hold ``n_pre_cut`` (:math:`n`),
        ``expected_events`` (:math:`\mu_0`) and ``rate_ci``, each keyed by transient type.
    mask : array-like of bool, optional
        True for the rows to count, one per row of `events`. The default counts every row, which
        gives the efficiency of the cuts the table was built from.
    confidence : float, optional
        Confidence level of the Clopper-Pearson interval. The default is
        ``config["simulation.default_confidence"]``.
    transient_types : list of str, optional
        Report only these types. The default is every type generated, including any with no
        selected rows.

    Returns
    -------
    ~astropy.table.QTable
        One row per transient type, with columns ``transient_type``, ``n_pre_cut``,
        ``n_selected``, ``fraction``, ``fraction_lower``, ``fraction_upper``,
        ``intrinsic_events``, ``expected_events``, ``expected_events_binom_lower``/``_upper``
        and ``expected_events_rate_lower``/``_upper``. Types are not summed, since that would
        mix independent Monte Carlo errors with a shared rate uncertainty.

    Raises
    ------
    ValueError
        If `events` lacks a required ``meta`` entry, `mask` is not a boolean array with one
        entry per row, a requested type was not generated, or a type has more selected rows than
        generated events.

    Examples
    --------
    .. code-block:: python

        estimate_yield(
            events
        )  # the cuts the table was built from, per type

        delay = (
            events["t_first_det"]
            - events["t_last_nondet"]
        ).to_value("hr")
        estimate_yield(
            events, mask=delay <= 6
        )  # first detected within 6 h of a non-detection
    """
    confidence = config["simulation.default_confidence"] if confidence is None else float(confidence)

    # The table carries its own denominators and normalizations, so a missing entry means it did
    # not come from run_event_summary_action.
    for key in ("n_pre_cut", "expected_events", "rate_ci"):
        if key not in events.meta:
            raise ValueError(f"events.meta has no {key!r}; the table must come from run_event_summary_action.")
    n_pre_cut = events.meta["n_pre_cut"]
    intrinsic = events.meta["expected_events"]
    rate_ci = events.meta["rate_ci"]

    if mask is None:
        mask = np.ones(len(events), dtype=bool)
    else:
        mask = np.asarray(mask)
        if mask.dtype != bool or mask.shape != (len(events),):
            raise ValueError(
                f"'mask' must be a boolean array of shape ({len(events)},); got {mask.dtype} with shape {mask.shape}."
            )

    names = sorted(n_pre_cut)
    if transient_types is not None:
        unknown = sorted(set(transient_types) - set(names))
        if unknown:
            raise ValueError(f"Unknown transient type(s) {unknown}; generated: {names}.")
        names = sorted(set(transient_types))

    types = np.asarray(events["transient_type"]).astype(str)
    rows = []
    for name in names:
        # n is fixed at generation, however the table was cut or sliced; only k depends on the mask.
        n = int(n_pre_cut[name])
        k = int(np.sum(mask & (types == name)))
        if k > n:
            raise ValueError(
                f"Type {name!r} has {k} selected rows but only {n} were generated, "
                "so the table does not match its meta['n_pre_cut']."
            )

        # k = 0 and k = n are handled inside the interval; n = 0 gives (0, 1) and a NaN fraction.
        fraction = k / n if n > 0 else np.nan
        fraction_lower, fraction_upper = clopper_pearson_interval(k, n, confidence)

        # The Monte Carlo interval moves the fraction at fixed mu0. The rate interval instead
        # rescales mu0 (and so the point estimate) at fixed fraction.
        mu0 = float(intrinsic[name])
        rate_lower, rate_upper = (float(factor) for factor in rate_ci[name])
        expected = mu0 * fraction
        rows.append(
            (
                name,
                n,
                k,
                fraction,
                fraction_lower,
                fraction_upper,
                mu0,
                expected,
                mu0 * fraction_lower,
                mu0 * fraction_upper,
                expected * rate_lower,
                expected * rate_upper,
            )
        )

    result = QTable(rows=rows, names=_COLUMNS) if rows else QTable(names=_COLUMNS)
    result.meta["confidence"] = confidence
    return result
