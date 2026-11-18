"""
What the figures of a recovery map share: contour labels, the shaded regions of an overlay, and axis labels.

The figures themselves are in :mod:`~uvex_transients.phase_space.plotting.map` (the canonical map),
:mod:`~uvex_transients.phase_space.plotting.difference` (the difference of two maps) and
:mod:`~uvex_transients.phase_space.plotting.grid` (a map over any two axes).
"""

import numpy as np
from astropy.table import Table

from uvex_transients.phase_space.space import Axis

__all__ = [
    "CANONICAL_NAMES",
    "F_LEVELS",
    "F_RANGE",
    "OVERLAY_COLORS",
    "SAVE_DPI",
    "axis_label",
    "draw_overlay",
    "label_once",
]

# ============================================ #
# Constants                                    #
# ============================================ #
#: Resolution of a saved figure.
SAVE_DPI = 150

#: Colour scale and contour levels of the recovery fraction.
F_RANGE = (1e-3, 1.0)
F_LEVELS = (0.01, 0.1, 0.5)

#: The axes of the canonical map, which the overlays (measurements of a class) are of.
CANONICAL_NAMES = ("log_luminosity", "log_half_time")

#: Distinct colours for the shaded classes, which a legend tells apart, not names on the map.
OVERLAY_COLORS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#17becf", "#bcbd22", "#ffd700", "#ff1493", "#7f7f7f",
)  # fmt: skip

#: Shaded region of a class: the fraction of its draws it holds, the size of the grid its density is estimated on, how
#: far past the draws the grid reaches (as a fraction of their range), the fewest draws that estimate is attempted
#: with, and the opacity of the fill.
_REGION_MASS = 0.68
_REGION_GRID = 240
_REGION_PADDING = 0.4
_REGION_MIN_DRAWS = 5
_REGION_ALPHA = 0.35


# ============================================ #
# Helpers                                      #
# ============================================ #
def label_once(ax, contours, fmt) -> None:
    """
    Label each contour level once, at the middle of its longest segment.

    The default placement repeats a label along a line, and lines that run almost vertically through a log-scaled
    panel then collect overlapping labels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes the contours are on.
    contours : matplotlib.contour.QuadContourSet
        The contours to label.
    fmt : str or callable
        The format of a label, as for `matplotlib.axes.Axes.clabel`.
    """
    levels, positions = [], []
    for level, segments in zip(contours.levels, contours.allsegs):
        segments = [segment for segment in segments if len(segment)]
        if segments:
            longest = max(segments, key=len)
            levels.append(level)
            positions.append(tuple(longest[len(longest) // 2]))
    if levels:
        ax.clabel(contours, levels=levels, fmt=fmt, fontsize=9, manual=positions)


def _density_region(log_t: np.ndarray, log_l: np.ndarray, mass: float = _REGION_MASS):
    """
    Return the smallest region of the plane that holds `mass` of a population, as a gridded density and its level.

    Parameters
    ----------
    log_t, log_l : numpy.ndarray
        The population's two coordinates.
    mass : float, optional
        The fraction of the population the region holds.

    Returns
    -------
    t_grid, l_grid, density : numpy.ndarray
        A grid covering the population, and its kernel density estimate on it.
    level : float
        The density above which the region lies.

    Raises
    ------
    ValueError
        If the population is too small or too degenerate for a density estimate.
    """
    from scipy.stats import gaussian_kde

    if len(log_t) < _REGION_MIN_DRAWS:
        raise ValueError("Too few draws for a density estimate.")
    try:
        kde = gaussian_kde(np.vstack([log_t, log_l]))
    except np.linalg.LinAlgError as error:
        raise ValueError("The draws are degenerate.") from error

    # The grid reaches past the draws, so that the region closes inside it
    pad_t, pad_l = (_REGION_PADDING * np.ptp(v) + 1e-3 for v in (log_t, log_l))
    t_grid, l_grid = np.meshgrid(
        np.linspace(log_t.min() - pad_t, log_t.max() + pad_t, _REGION_GRID),
        np.linspace(log_l.min() - pad_l, log_l.max() + pad_l, _REGION_GRID),
    )
    density = kde(np.vstack([t_grid.ravel(), l_grid.ravel()])).reshape(t_grid.shape)

    # The level above which the density holds the requested mass
    ordered = np.sort(density.ravel())[::-1]
    level = ordered[np.searchsorted(np.cumsum(ordered) / ordered.sum(), mass)]
    return t_grid, l_grid, density, level


def draw_overlay(axes, overlay: Table, color):
    """
    Shade the region of the map that one measured population occupies, on every panel.

    The region is the smallest one that holds :data:`~uvex_transients.phase_space.plotting.core._REGION_MASS` of the
    draws, from a kernel density estimate. A population too small or degenerate for one is shown as the box between the
    16th and 84th percentiles of its two quantities.

    Parameters
    ----------
    axes : numpy.ndarray
        The axes to shade on, such as the four panels of
        :func:`~uvex_transients.phase_space.plotting.map.plot_map`.
    overlay : astropy.table.Table
        The measurements, with ``log_luminosity``, ``log_half_time`` and ``measured`` columns, and the class name in
        ``meta["class"]``.
    color : str or tuple
        The colour of the region.

    Returns
    -------
    matplotlib.patches.Patch or None
        A legend handle for the region, or `None` if the population has no measured draw.
    """
    from matplotlib.patches import Patch, Rectangle

    keep = np.asarray(overlay["measured"], dtype=bool)
    if not keep.any():
        return None
    log_l, log_t = np.asarray(overlay["log_luminosity"])[keep], np.asarray(overlay["log_half_time"])[keep]
    (l_lo, _, l_hi), (t_lo, _, t_hi) = (np.percentile(v, [16, 50, 84]) for v in (log_l, log_t))

    # The density region, or the percentile box where there is too little to estimate one
    try:
        t_grid, l_grid, density, level = _density_region(log_t, log_l)
    except ValueError:
        density = None

    for ax in axes.ravel():
        if density is None:
            ax.add_patch(
                Rectangle(
                    (10.0**t_lo, 10.0**l_lo),
                    10.0**t_hi - 10.0**t_lo,
                    10.0**l_hi - 10.0**l_lo,
                    facecolor=color,
                    alpha=_REGION_ALPHA,
                    edgecolor=color,
                )
            )
        else:
            x, y = 10.0**t_grid, 10.0**l_grid
            ax.contourf(x, y, density, levels=[level, np.inf], colors=[color], alpha=_REGION_ALPHA)
            ax.contour(x, y, density, levels=[level], colors=[color], linewidths=1.2)
    return Patch(facecolor=color, edgecolor=color, alpha=0.6, label=overlay.meta.get("class", ""))


def axis_label(axis: Axis) -> str:
    """
    Return what a figure writes on an axis.

    Parameters
    ----------
    axis : Axis
        The axis of the map.

    Returns
    -------
    str
        Its label (or name) and, if it has one, its unit.
    """
    label = axis.label or axis.name
    return f"{label} [{axis.unit}]" if axis.unit else label
