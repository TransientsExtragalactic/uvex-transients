"""The figure of the difference of two recovery maps."""

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from astropy.table import Table

from uvex_transients.phase_space.plotting.core import (
    CANONICAL_NAMES,
    OVERLAY_COLORS,
    SAVE_DPI,
    axis_label,
    draw_overlay,
    label_once,
)
from uvex_transients.phase_space.space import Space
from uvex_transients.utils.plotting import set_plot_style

__all__ = ["plot_difference"]

# ============================================ #
# Constants                                    #
# ============================================ #
#: The figure of a difference of two maps: its size, the largest difference (in dex) the colour scale reaches, the
#: contour levels, the significance (in sigma) below which a cell is hatched, and the hatches of an insignificant
#: difference and of a lower and an upper bound.
_DIFFERENCE_FIGSIZE = (9, 7)
_DIFFERENCE_RANGE = 2.0
_DIFFERENCE_LEVELS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
_SIGNIFICANCE = 2.0
_HATCH_INSIGNIFICANT, _HATCH_LOWER, _HATCH_UPPER = "..", "\\\\", "//"


# ============================================ #
# Figure                                       #
# ============================================ #
def plot_difference(
    difference: Table,
    path: str | Path | None = None,
    title: str | None = None,
    overlays: Sequence[Table] | None = None,
    annotate: Callable | None = None,
):
    r"""
    Draw the difference of two recovery maps, in dex of the recovered yield (the volume-time, for a survey).

    The axes come from the ``space`` in the table's ``meta`` when it has one (as a table of
    :func:`~uvex_transients.phase_space.analysis.compare.difference_map` of two maps made with
    :meth:`~uvex_transients.phase_space.experiment.Experiment.grid_table` does), so the figure does not know what they
    are, and from the canonical luminosity and half-decline time otherwise.

    The colour is :math:`\Delta = \log_{10}(R_\mathrm{c}/R_\mathrm{b})`, positive where the comparison recovers more,
    with contours at fixed numbers of dex. Cells where :math:`|\Delta|` is less than two standard deviations are dotted,
    and cells where it is only a bound, because one map recovers essentially nothing there, are hatched by the direction
    of the bound. Cells where neither map recovers anything are left blank.

    Parameters
    ----------
    difference : astropy.table.Table
        The output of :func:`~uvex_transients.phase_space.analysis.compare.difference_map`.
    path : str or pathlib.Path, optional
        Where to write the figure. If omitted, the figure is not saved.
    title : str, optional
        A title.
    overlays : sequence of astropy.table.Table, optional
        Measurements of populations to shade, as in `plot_map`. They are of the canonical axes, so a map over others
        takes `annotate` instead.
    annotate : callable, optional
        ``annotate(ax)``, called on the axes to draw anything else, as in `plot_grid`. The legend handles it returns
        are added to the legend of the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure, left open so that it can be adjusted or shown.

    Raises
    ------
    ValueError
        If the table records a space that does not have two axes, is not a regular grid, or (with `overlays`) is not
        over the canonical axes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch

    set_plot_style()

    # The axes: the first is drawn vertically and the second horizontally, as in `plot_map` and `plot_grid`
    if "space" in difference.meta:
        space = Space.from_dict(difference.meta["space"])
        if space.ndim != 2:
            raise ValueError(f"A figure is of a map over two axes, and this one has {space.ndim}.")
        names = space.names
        vertical, horizontal = space.axes
        x_log, y_log = horizontal.scale == "log10", vertical.scale == "log10"
        x_label, y_label = axis_label(horizontal), axis_label(vertical)
        if overlays and names != CANONICAL_NAMES:
            raise ValueError(f"The overlays are of the axes {CANONICAL_NAMES}, and this map is over {names}.")
    else:
        names = tuple(difference.colnames[:2])
        x_log = y_log = True
        x_label = r"rest-frame half-decline time $t_{1/2}$ [day]"
        band = difference.meta.get("band")
        y_label = f"{rf'$L_\mathrm{{{band}}}$' if band else r'$L$'} [erg s$^{{-1}}$]"

    shape = tuple(len(np.unique(difference[name])) for name in names)
    if int(np.prod(shape)) != len(difference):
        raise ValueError("The table is not a regular grid of its coordinates.")
    duration = np.asarray(difference[names[1]]).reshape(shape)
    luminosity = np.asarray(difference[names[0]]).reshape(shape)
    duration, luminosity = (10.0**duration if x_log else duration), (10.0**luminosity if y_log else luminosity)
    delta = np.ma.masked_invalid(np.asarray(difference["delta_dex"]).reshape(shape))
    significance = np.nan_to_num(np.asarray(difference["significance"]).reshape(shape))
    limit = np.asarray(difference["limit"]).reshape(shape)

    # Symmetric about no difference, and not stretched by one extreme corner
    reach = min(float(np.abs(delta).max()) if delta.count() else 1.0, _DIFFERENCE_RANGE)
    reach = max(reach, 0.1)

    fig, ax = plt.subplots(figsize=_DIFFERENCE_FIGSIZE, constrained_layout=True)
    ax.set_facecolor("0.88")
    mesh = ax.pcolormesh(duration, luminosity, delta, norm=Normalize(-reach, reach), cmap="RdBu", shading="auto")
    weight = difference.meta.get("weight", "volume_time")
    what = {"volume_time": "recovered volume-time", "": "recovery fraction"}.get(
        weight, f"recovery fraction x {weight}"
    )
    fig.colorbar(mesh, ax=ax, label=rf"$\log_{{10}}$(comparison / baseline) {what} [dex]")
    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")

    # Contours of the difference, where it reaches them
    levels = [level for level in _DIFFERENCE_LEVELS if delta.min() < level < delta.max()] if delta.count() else []
    if levels:
        contours = ax.contour(duration, luminosity, delta, levels=levels, colors="k", linewidths=0.8)
        label_once(ax, contours, lambda level: f"{level:+g}" if level else "0")

    # What not to read as a measurement
    handles = []
    insignificant = (np.abs(significance) < _SIGNIFICANCE) & (limit == 0) & ~np.ma.getmaskarray(delta)
    for mask, hatch, label in (
        (insignificant, _HATCH_INSIGNIFICANT, rf"$|\Delta| < {_SIGNIFICANCE:g}\sigma$"),
        (limit > 0, _HATCH_LOWER, r"lower bound on $\Delta$"),
        (limit < 0, _HATCH_UPPER, r"upper bound on $\Delta$"),
    ):
        if mask.any():
            ax.contourf(duration, luminosity, mask.astype(float), levels=[0.5, 1.5], colors="none", hatches=[hatch])
            handles.append(Patch(facecolor="none", edgecolor="k", hatch=hatch, label=label))

    for k, overlay in enumerate(overlays or []):
        if (handle := draw_overlay(np.array([ax]), overlay, OVERLAY_COLORS[k % len(OVERLAY_COLORS)])) is not None:
            handles.append(handle)
    if annotate is not None:
        handles.extend(annotate(ax) or [])
    if handles:
        fig.legend(handles=handles, loc="outside lower center", ncol=min(4, len(handles)), fontsize=9, frameon=False)

    # An overlay that reaches past the map is cut off by it, not allowed to stretch the axes
    ax.set_xlim(duration.min(), duration.max())
    ax.set_ylim(luminosity.min(), luminosity.max())

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Magnitude is linear in log L, so a twin axis with the limits of the luminosity axis converted lines up exactly
    zero_point = difference.meta.get("absolute_magnitude_zero_point")
    if zero_point is not None:
        magnitude = ax.twinx()
        magnitude.set_ylim(zero_point - 2.5 * difference[names[0]].min(), zero_point - 2.5 * difference[names[0]].max())
        magnitude.set_ylabel(r"absolute AB magnitude $M$")

    if title:
        ax.set_title(title)
    if path is not None:
        fig.savefig(path, dpi=SAVE_DPI)
    return fig
