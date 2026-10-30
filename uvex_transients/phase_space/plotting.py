"""Figures of a recovery map from `~uvex_transients.phase_space.sampler.PhaseSpaceSampler`."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.units import Quantity

from uvex_transients.utils.plotting import set_plot_style

__all__ = ["plot_difference", "plot_map"]

# ============================================ #
# Constants                                    #
# ============================================ #
#: Size and resolution of the saved figure of four panels.
_FIGSIZE = (14, 11)
_SAVE_DPI = 150

#: Colour scale and contour levels of the recovery fraction, and of the all-sky efficiency (whose scale tops out at its
#: largest value, but never below `_EFFICIENCY_CEILING`).
_F_RANGE = (1e-3, 1.0)
_F_LEVELS = (0.01, 0.1, 0.5)
_EFFICIENCY_FLOOR = 1e-4
_EFFICIENCY_CEILING = 2e-4
_EFFICIENCY_LEVELS = (1e-3, 1e-2, 1e-1)

#: Levels, in recoveries, of the contours drawn when the table has expected recoveries.
_RECOVERY_LEVELS = (1, 10, 100, 1000, 1e4)

#: Shaded region of a class: the fraction of its draws it holds, the size of the grid its density is estimated on, how
#: far past the draws the grid reaches (as a fraction of their range), the fewest draws that estimate is attempted
#: with, and the opacity of the fill.
_REGION_MASS = 0.68
_REGION_GRID = 240
_REGION_PADDING = 0.4
_REGION_MIN_DRAWS = 5
_REGION_ALPHA = 0.35

#: The figure of a difference of two maps: its size, the largest difference (in dex) the colour scale reaches, the
#: contour levels, the significance (in sigma) below which a cell is hatched, and the hatches of an insignificant
#: difference and of a lower and an upper bound.
_DIFFERENCE_FIGSIZE = (9, 7)
_DIFFERENCE_RANGE = 2.0
_DIFFERENCE_LEVELS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
_SIGNIFICANCE = 2.0
_HATCH_INSIGNIFICANT, _HATCH_LOWER, _HATCH_UPPER = "..", "\\\\", "//"

#: Distinct colours for the shaded classes, which a legend tells apart, not names on the map.
_OVERLAY_COLORS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#17becf", "#bcbd22", "#ffd700", "#ff1493", "#7f7f7f",
)  # fmt: skip


# ============================================ #
# Helpers                                      #
# ============================================ #
def _label_once(ax, contours, fmt) -> None:
    """
    Label each contour level once, at the middle of its longest segment.

    The default placement repeats a label along a line, and lines that run almost vertically through a log-scaled
    panel then collect overlapping labels.
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


def _draw_overlay(axes, overlay: Table, color):
    """
    Shade the region of the map that one measured population occupies, on every panel.

    The region is the smallest one that holds `_REGION_MASS` of the draws, from a kernel density estimate. A population
    too small or degenerate for one is shown as the box between the 16th and 84th percentiles of its two quantities.

    Parameters
    ----------
    axes : numpy.ndarray
        The ``(2, 2)`` array of axes of `plot_map`.
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


def _panel_specs(fields: dict, reference_label: str) -> list[tuple]:
    """
    Describe the four panels of `plot_map`.

    Each is a tuple of the column drawn, its colour bar label, colour scale and colour map, and the levels of its
    contours.
    """
    from matplotlib.colors import LogNorm

    required = fields["required_rate"]

    # Contours of the required rate, every second decade it spans
    decades = range(int(np.ceil(np.log10(required.min()))), int(np.floor(np.log10(required.max()))) + 1)
    required_levels = [10.0**k for k in decades if k % 2 == 0]

    return [
        # name, colour bar label, norm, cmap, contour levels
        ("f", "recovery fraction in the sampled footprint, f", LogNorm(*_F_RANGE), "viridis", list(_F_LEVELS)),
        (
            "efficiency",
            r"all-sky efficiency $\eta = f \times$ coverage",
            LogNorm(vmin=_EFFICIENCY_FLOOR, vmax=max(fields["efficiency"].max(), _EFFICIENCY_CEILING)),
            "viridis",
            list(_EFFICIENCY_LEVELS),
        ),
        (
            "required_rate",
            f"rate for one recovery / {reference_label}",
            LogNorm(vmin=required.min(), vmax=required.max()),
            "magma_r",
            required_levels,
        ),
        ("sigma_log", r"uncertainty of $\ln(f+\epsilon)$", None, "cividis", []),
    ]


# ============================================ #
# Figure                                       #
# ============================================ #
def plot_map(
    grid: Table,
    points: Table,
    path: str | Path | None = None,
    title: str | None = None,
    overlays: Sequence[Table] | None = None,
    show_points: bool = False,
    reference_rate: Quantity | None = None,
    reference_label: str = "local CCSN rate",
):
    r"""
    Draw the recovery fraction, the all-sky efficiency, the rate needed to recover one event, and the uncertainty.

    The third panel is the volumetric rate at which the survey would recover one event on average, as a fraction of a
    reference rate, :math:`1/(f\,\mathcal V T\,R_\mathrm{ref})`: a class at least that fraction as common as the
    reference yields an event, so the lower the number the better the survey does there. If the table has
    ``expected_recoveries``, its contours are drawn on that panel.

    Parameters
    ----------
    grid : astropy.table.Table
        The output of `PhaseSpaceSampler.map_table`.
    points : astropy.table.Table
        The simulated points (`PhaseSpaceSampler.table`).
    path : str or pathlib.Path, optional
        Where to write the figure. If omitted, the figure is not saved.
    title : str, optional
        A suptitle.
    overlays : sequence of astropy.table.Table, optional
        Measurements of populations to shade on the map, as returned by
        `~uvex_transients.phase_space.overlay.measure_class`. Each is the smallest region holding 68% of its draws, in
        its own colour, named in a legend.
    show_points : bool, optional
        Whether to mark the simulated points, as circles that grow with the number of events simulated.
    reference_rate : ~astropy.units.Quantity, optional
        The rate the third panel is relative to, in events per volume per time. By default the local core-collapse
        supernova rate, `~uvex_transients.utils.cosmology.core_collapse_rate_coefficient`.
    reference_label : str, optional
        What to call the reference rate in the colour bar.

    Returns
    -------
    matplotlib.figure.Figure
        The figure, left open so that it can be adjusted or shown.
    """
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt

    set_plot_style()

    # ---------------------------------------------- #
    # The data                                       #
    # ---------------------------------------------- #
    names = grid.colnames[:2]
    n = int(np.sqrt(len(grid)))
    duration = 10.0 ** np.asarray(grid[names[1]]).reshape(n, n)
    luminosity = 10.0 ** np.asarray(grid[names[0]]).reshape(n, n)
    fields = {name: np.asarray(grid[name]).reshape(n, n) for name in grid.colnames}

    # The rate at which one recovery is expected is 1 / (f V T), here in units of the reference rate
    if reference_rate is None:
        from uvex_transients.utils.cosmology import core_collapse_rate_coefficient

        reference_rate = core_collapse_rate_coefficient()
    recovered = np.clip(fields["recovered_volume_time"], 1e-300, None)
    fields["required_rate"] = 1.0 / (recovered * reference_rate.to_value(u.Mpc**-3 / u.yr))

    # ---------------------------------------------- #
    # The panels                                     #
    # ---------------------------------------------- #
    fig, axes = plt.subplots(2, 2, figsize=_FIGSIZE, sharex=True, sharey=True, constrained_layout=True)
    for ax, (name, label, norm, cmap, levels) in zip(axes.ravel(), _panel_specs(fields, reference_label)):
        values = np.clip(fields[name], 1e-12, None)
        mesh = ax.pcolormesh(duration, luminosity, values, norm=norm, cmap=cmap, shading="auto")
        fig.colorbar(mesh, ax=ax, label=label)

        if show_points:
            ax.scatter(
                10.0 ** points[names[1]],
                10.0 ** points[names[0]],
                s=8 + 30 * np.log10(points["n"] / points["n"].min() + 1.0),
                facecolor="none",
                edgecolor="white",
                linewidth=0.8,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")

        if levels:
            contours = ax.contour(duration, luminosity, fields[name], levels=levels, colors="white", linewidths=1)
            if name == "required_rate":
                # Light and dark regions both: a white line needs an outline to show on the pale end of the scale
                contours.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground="black")])
            _label_once(ax, contours, "%g")

    # ---------------------------------------------- #
    # Classes                                        #
    # ---------------------------------------------- #
    handles = [
        handle
        for k, overlay in enumerate(overlays or [])
        if (handle := _draw_overlay(axes, overlay, _OVERLAY_COLORS[k % len(_OVERLAY_COLORS)])) is not None
    ]
    if handles:
        fig.legend(handles=handles, loc="outside lower center", ncol=min(6, len(handles)), fontsize=9, frameon=False)

    # An overlay that reaches past the map is cut off by it, not allowed to stretch the axes
    axes[0, 0].set_xlim(duration.min(), duration.max())
    axes[0, 0].set_ylim(luminosity.min(), luminosity.max())

    # ---------------------------------------------- #
    # Expected recoveries                            #
    # ---------------------------------------------- #
    if "expected_recoveries" in fields:
        expected = fields["expected_recoveries"]
        levels = [level for level in _RECOVERY_LEVELS if expected.min() < level < expected.max()]
        if levels:
            contours = axes[1, 0].contour(duration, luminosity, expected, levels=levels, colors="cyan", linewidths=1)
            _label_once(
                axes[1, 0], contours, lambda level: f"{level:g} recovered" if level == 1 else f"{level:g} recoveries"
            )

    # ---------------------------------------------- #
    # Labels                                         #
    # ---------------------------------------------- #
    for ax in axes[1]:
        ax.set_xlabel(r"rest-frame half-decline time $t_{1/2}$ [day]")
    band = grid.meta.get("band")
    luminosity_label = rf"$L_\mathrm{{{band}}}$" if band else r"$L$"
    for ax in axes[:, 0]:
        ax.set_ylabel(f"{luminosity_label} [erg s$^{{-1}}$]")

    # Magnitude is linear in log L, so a twin axis with the limits of the luminosity axis converted lines up exactly
    zero_point = grid.meta.get("absolute_magnitude_zero_point")
    if zero_point is not None:
        log_l_lo, log_l_hi = grid[names[0]].min(), grid[names[0]].max()
        for ax in axes[:, 1]:
            magnitude = ax.twinx()
            magnitude.set_ylim(zero_point - 2.5 * log_l_lo, zero_point - 2.5 * log_l_hi)
            magnitude.set_ylabel(r"absolute AB magnitude $M$")

    if title:
        fig.suptitle(title)
    if path is not None:
        fig.savefig(path, dpi=_SAVE_DPI)
    return fig


def plot_difference(
    difference: Table,
    path: str | Path | None = None,
    title: str | None = None,
    overlays: Sequence[Table] | None = None,
):
    r"""
    Draw the difference of two recovery maps, in dex of the recovered volume-time.

    The colour is :math:`\Delta = \log_{10}(R_\mathrm{c}/R_\mathrm{b})`, positive where the comparison recovers more,
    with contours at fixed numbers of dex. Cells where :math:`|\Delta|` is less than two standard deviations are dotted,
    and cells where it is only a bound, because one map recovers essentially nothing there, are hatched by the direction
    of the bound. Cells where neither map recovers anything are left blank.

    Parameters
    ----------
    difference : astropy.table.Table
        The output of `~uvex_transients.phase_space.compare.difference_map`.
    path : str or pathlib.Path, optional
        Where to write the figure. If omitted, the figure is not saved.
    title : str, optional
        A title.
    overlays : sequence of astropy.table.Table, optional
        Measurements of populations to shade, as in `plot_map`.

    Returns
    -------
    matplotlib.figure.Figure
        The figure, left open so that it can be adjusted or shown.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch

    set_plot_style()

    names = difference.colnames[:2]
    n = int(np.sqrt(len(difference)))
    duration = 10.0 ** np.asarray(difference[names[1]]).reshape(n, n)
    luminosity = 10.0 ** np.asarray(difference[names[0]]).reshape(n, n)
    delta = np.ma.masked_invalid(np.asarray(difference["delta_dex"]).reshape(n, n))
    significance = np.nan_to_num(np.asarray(difference["significance"]).reshape(n, n))
    limit = np.asarray(difference["limit"]).reshape(n, n)

    # Symmetric about no difference, and not stretched by one extreme corner
    reach = min(float(np.abs(delta).max()) if delta.count() else 1.0, _DIFFERENCE_RANGE)
    reach = max(reach, 0.1)

    fig, ax = plt.subplots(figsize=_DIFFERENCE_FIGSIZE, constrained_layout=True)
    ax.set_facecolor("0.88")
    mesh = ax.pcolormesh(duration, luminosity, delta, norm=Normalize(-reach, reach), cmap="RdBu", shading="auto")
    fig.colorbar(mesh, ax=ax, label=r"$\log_{10}$(comparison / baseline) recovered volume-time [dex]")
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Contours of the difference, where it reaches them
    levels = [level for level in _DIFFERENCE_LEVELS if delta.min() < level < delta.max()] if delta.count() else []
    if levels:
        contours = ax.contour(duration, luminosity, delta, levels=levels, colors="k", linewidths=0.8)
        _label_once(ax, contours, lambda level: f"{level:+g}" if level else "0")

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
        if (handle := _draw_overlay(np.array([ax]), overlay, _OVERLAY_COLORS[k % len(_OVERLAY_COLORS)])) is not None:
            handles.append(handle)
    if handles:
        fig.legend(handles=handles, loc="outside lower center", ncol=min(4, len(handles)), fontsize=9, frameon=False)

    # An overlay that reaches past the map is cut off by it, not allowed to stretch the axes
    ax.set_xlim(duration.min(), duration.max())
    ax.set_ylim(luminosity.min(), luminosity.max())

    ax.set_xlabel(r"rest-frame half-decline time $t_{1/2}$ [day]")
    band = difference.meta.get("band")
    ax.set_ylabel(f"{rf'$L_\mathrm{{{band}}}$' if band else r'$L$'} [erg s$^{{-1}}$]")

    # Magnitude is linear in log L, so a twin axis with the limits of the luminosity axis converted lines up exactly
    zero_point = difference.meta.get("absolute_magnitude_zero_point")
    if zero_point is not None:
        magnitude = ax.twinx()
        magnitude.set_ylim(zero_point - 2.5 * difference[names[0]].min(), zero_point - 2.5 * difference[names[0]].max())
        magnitude.set_ylabel(r"absolute AB magnitude $M$")

    if title:
        ax.set_title(title)
    if path is not None:
        fig.savefig(path, dpi=_SAVE_DPI)
    return fig
