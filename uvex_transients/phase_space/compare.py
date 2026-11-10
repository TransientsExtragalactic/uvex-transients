r"""
Compare two recovery maps.

`difference_map` takes the fitted maps of two runs, each the ``map.ecsv`` table of
`~uvex_transients.phase_space.sampler.PhaseSpaceSampler.map_table`, and returns how much more (or less) one recovers
than the other at every point of the plane, in dex of the recovered volume-time :math:`f\,\mathcal V T`. That is the
factor by which the expected number of recoveries changes for a population of a given peak luminosity and half-decline
time, whatever its rate (for a rate constant in redshift). It does not depend on the number of explosion-time bins, as
the recovery fraction :math:`f` alone does. Nothing is simulated, so it needs neither the schedule nor scikit-learn.
"""

import numpy as np
from astropy.table import Table

__all__ = ["COMPARED_META", "difference_map"]

#: The metadata of a map that gives its axes their meaning. Two maps that differ in one of these (when both record it)
#: are not maps of the same plane, so `difference_map` refuses to compare them.
COMPARED_META = ("band", "spectral_index", "rise_ratio", "fade_mag")

#: The columns of a map that `difference_map` reads, besides its two coordinates.
_REQUIRED = ("f", "f_lo", "f_hi", "volume_time")

#: Smallest recovery fraction taken to be non-zero, so that the logarithm of an underflowed interval is finite.
_TINY = 1e-300

#: The resolution assumed of a map that does not record its ``n_max``: about twice that of the default ``n_max`` of
#: 20000 events, the most a point of the map simulates.
_DEFAULT_RESOLUTION = 1e-4


def _resolution(table: Table) -> float:
    """Return the smallest recovery fraction a map can resolve, one recovery in the most events a point simulates."""
    n_max = table.meta.get("n_max")
    return 1.0 / float(n_max) if n_max else _DEFAULT_RESOLUTION


def _check_grids(baseline: Table, comparison: Table) -> tuple[str, str]:
    """
    Check that two maps hold the columns `difference_map` needs and sit on the same grid; return the coordinate names.

    Raises
    ------
    ValueError
        If a column is missing, or the coordinates differ in name, number of points or value.
    """
    for label, table in (("baseline", baseline), ("comparison", comparison)):
        missing = [name for name in _REQUIRED if name not in table.colnames]
        if len(table.colnames) < 2 or missing:
            raise ValueError(f"The {label} map is missing the column(s) {missing or 'of its coordinates'}.")

    names = tuple(baseline.colnames[:2])
    if tuple(comparison.colnames[:2]) != names:
        raise ValueError(f"The maps have different coordinates: {names} and {tuple(comparison.colnames[:2])}.")
    if len(baseline) != len(comparison) or not all(
        np.allclose(baseline[name], comparison[name], rtol=0.0, atol=1e-9) for name in names
    ):
        raise ValueError("The maps are not on the same grid; evaluate both with the same bounds and grid size.")
    return names


def _check_meta(baseline: Table, comparison: Table) -> None:
    """
    Check that two maps measure the same axes.

    Raises
    ------
    ValueError
        If a quantity of `COMPARED_META` that both record differs. A map written before one was recorded is not
        held to it.
    """
    differ = []
    for key in COMPARED_META:
        if key not in baseline.meta or key not in comparison.meta:
            continue
        a, b = baseline.meta[key], comparison.meta[key]
        if not (a == b if isinstance(a, str) or isinstance(b, str) else np.isclose(a, b)):
            differ.append(f"{key} ({a!r} vs {b!r})")
    if differ:
        raise ValueError(
            f"The maps do not measure the same axes: {', '.join(differ)}. Pass check_meta=False to compare them anyway."
        )


def _log_half_width(table: Table) -> np.ndarray:
    """Return half the width of the 1 sigma interval of the recovered volume-time of a map, in dex."""
    lo, hi = (np.log10(np.clip(np.asarray(table[name], dtype=np.float64), _TINY, None)) for name in ("f_lo", "f_hi"))
    return 0.5 * (hi - lo)


def difference_map(baseline: Table, comparison: Table, *, floor: float | None = None, check_meta: bool = True) -> Table:
    r"""
    Difference two recovery maps, in dex of the recovered volume-time.

    At every point of the plane the difference is :math:`\Delta = \log_{10}(R_\mathrm{c} / R_\mathrm{b})`, with
    :math:`R = f\,\mathcal V T` the recovered volume-time of the comparison and the baseline map. It is positive
    where the comparison recovers more. The maps are fitted independently, so the uncertainties of the two combine in
    quadrature, each taken from the half-width of the 1 sigma interval of the map.

    A point of a map cannot tell a recovery fraction much below one recovery in its ``n_max`` events from none at all,
    so a map resolves :math:`f` only down to about :math:`1/n_\mathrm{max}`. That is a limit of the measurement, not of
    what matters: a rare event can still be recovered often, where the volume-time is large, which is why the floor is
    on :math:`f` and the rest of the difference is not cut. Wherever a map has :math:`f` below `floor`, its :math:`f` is
    replaced by `floor`: if only one map is below it, :math:`\Delta` is then a bound (``limit``), and if both are, it is
    undefined (NaN).

    Parameters
    ----------
    baseline, comparison : astropy.table.Table
        The two maps, as made by `~uvex_transients.phase_space.sampler.PhaseSpaceSampler.map_table` or read from
        ``map.ecsv``. They need the columns ``f``, ``f_lo``, ``f_hi`` and ``volume_time``, and must be on the same grid.
    floor : float, optional
        The recovery fraction below which a map is taken not to have recovered anything. If `None` (the default), the
        coarser of the resolutions of the two maps, :math:`1/n_\mathrm{max}` from the ``n_max`` that each records. A map
        that does not (those made before it was recorded) is taken to resolve 1e-4.
    check_meta : bool, optional
        Whether to require that the two maps agree on the quantities of `COMPARED_META` that they both record. A
        luminosity in one band is not one in another, so a comparison across them is only meaningful when it is
        deliberate. The default is `True`. The depth, the schedule and the definition of a detection are not checked, as
        they are what maps are most often compared for, and a map does not record them.

    Returns
    -------
    astropy.table.Table
        One row per grid point, in the order of the maps, with their two coordinates and

        ``delta_dex``
            :math:`\Delta`, in dex.
        ``sigma_dex``
            Its 1 sigma uncertainty, in dex.
        ``delta_dex_lo``, ``delta_dex_hi``
            :math:`\Delta \mp \sigma`.
        ``significance``
            :math:`\Delta / \sigma`.
        ``limit``
            0 where :math:`\Delta` is a measurement, +1 where the baseline is below `floor` so that :math:`\Delta` is a
            lower bound, and -1 where the comparison is, so that it is an upper bound.

        Its ``meta`` holds the band, the unit of the luminosity and the zero point of the absolute magnitude (from the
        baseline), `floor`, and the metadata of each map as ``baseline_*`` and ``comparison_*``.

    Raises
    ------
    ValueError
        If a map lacks a column, the maps are not on the same grid, or (with `check_meta`) they measure different axes.
    """
    names = _check_grids(baseline, comparison)
    if check_meta:
        _check_meta(baseline, comparison)

    if floor is None:
        floor = max(_resolution(baseline), _resolution(comparison))

    f_b, f_c = (np.asarray(table["f"], dtype=np.float64) for table in (baseline, comparison))
    below_b, below_c = f_b < floor, f_c < floor

    with np.errstate(divide="ignore", invalid="ignore"):
        log_b, log_c = (
            np.log10(np.maximum(f, floor) * np.asarray(table["volume_time"], dtype=np.float64))
            for f, table in ((f_b, baseline), (f_c, comparison))
        )
        delta = log_c - log_b
        sigma = np.hypot(_log_half_width(baseline), _log_half_width(comparison))

        # Neither map recovers anything here, or the volume-time is zero
        undefined = (below_b & below_c) | ~np.isfinite(delta) | ~np.isfinite(sigma)
        delta, sigma = np.where(undefined, np.nan, delta), np.where(undefined, np.nan, sigma)
        significance = delta / sigma

    table = Table()
    for name in names:
        table[name] = np.asarray(baseline[name])
    table["delta_dex"] = delta
    table["sigma_dex"] = sigma
    table["delta_dex_lo"] = delta - sigma
    table["delta_dex_hi"] = delta + sigma
    table["significance"] = significance
    table["limit"] = np.where(
        undefined, 0, np.where(below_b & ~below_c, 1, np.where(below_c & ~below_b, -1, 0))
    ).astype(np.int8)

    for key in ("band", "luminosity_unit", "absolute_magnitude_zero_point"):
        if key in baseline.meta:
            table.meta[key] = baseline.meta[key]
    table.meta["floor"] = float(floor)
    for label, source in (("baseline", baseline), ("comparison", comparison)):
        table.meta.update({f"{label}_{key}": value for key, value in source.meta.items()})
    return table
