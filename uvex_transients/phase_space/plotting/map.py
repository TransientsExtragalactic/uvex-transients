"""The figure of the canonical recovery map, over peak luminosity and half-decline time."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.units import Quantity

from uvex_transients.phase_space.plotting.core import (
    F_LEVELS,
    F_RANGE,
    OVERLAY_COLORS,
    SAVE_DPI,
    draw_overlay,
    label_once,
)
from uvex_transients.utils.plotting import set_plot_style

__all__ = ["plot_map"]

# ============================================ #
# Constants                                    #
# ============================================ #
#: Size of the saved figure of four panels.
_FIGSIZE = (14, 11)

#: Colour scale and contour levels of the all-sky efficiency (whose scale tops out at its largest value, but never
#: below `_EFFICIENCY_CEILING`).
_EFFICIENCY_FLOOR = 1e-4
_EFFICIENCY_CEILING = 2e-4
_EFFICIENCY_LEVELS = (1e-3, 1e-2, 1e-1)

#: Levels, in recoveries, of the contours drawn when the table has expected recoveries.
_RECOVERY_LEVELS = (1, 10, 100, 1000, 1e4)


# ============================================ #
# Figure                                       #
# ============================================ #
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
        ("f", "recovery fraction in the sampled footprint, f", LogNorm(*F_RANGE), "viridis", list(F_LEVELS)),
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
        The output of `~uvex_transients.phase_space.experiment.Experiment.grid_table`.
    points : astropy.table.Table
        The measured points (`~uvex_transients.phase_space.map.PhaseSpaceMap.table`).
    path : str or pathlib.Path, optional
        Where to write the figure. If omitted, the figure is not saved.
    title : str, optional
        A suptitle.
    overlays : sequence of astropy.table.Table, optional
        Measurements of populations to shade on the map, as returned by
        `~uvex_transients.phase_space.analysis.overlay.measure_class`. Each is the smallest region holding 68% of its
        draws, in its own colour, named in a legend.
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
            label_once(ax, contours, "%g")

    # ---------------------------------------------- #
    # Classes                                        #
    # ---------------------------------------------- #
    handles = [
        handle
        for k, overlay in enumerate(overlays or [])
        if (handle := draw_overlay(axes, overlay, OVERLAY_COLORS[k % len(OVERLAY_COLORS)])) is not None
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
            label_once(
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
        fig.savefig(path, dpi=SAVE_DPI)
    return fig
