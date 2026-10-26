"""Figures of a recovery map from `~uvex_transients.phase_space.sampler.PhaseSpaceSampler`."""

from pathlib import Path

import numpy as np
from astropy.table import Table

from uvex_transients.utils.plotting import set_plot_style

__all__ = ["plot_map"]


def _label_once(ax, contours, fmt) -> None:
    """Label each contour level a single time, at the middle of its longest segment.

    The default placement repeats a label along a line, and the lines that run almost vertically through a
    log-scaled panel then collect overlapping labels.
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


def plot_map(grid: Table, points: Table, path: str | Path | None = None, title: str | None = None):
    """
    Draw the recovery fraction, the all-sky efficiency, the recovered volume-time and the uncertainty of the map.

    A dashed line marks the luminosity above which the redshift limit is capped, since above it the sample is
    truncated by the redshift cap and not by depth. If the table has ``expected_recoveries``, its contours are drawn
    on the recovered volume-time panel.

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

    Returns
    -------
    matplotlib.figure.Figure
        The figure, left open so that it can be adjusted or shown.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    set_plot_style()
    names = grid.colnames[:2]
    n = int(np.sqrt(len(grid)))

    # The dashed line sits between the highest uncapped and lowest capped grid luminosity.
    capped = np.asarray(grid["z_limit_capped"])
    clip_log_l = None
    if capped.any() and not capped.all():
        luminosity_values = np.asarray(grid[names[0]])
        clip_log_l = 0.5 * (luminosity_values[~capped].max() + luminosity_values[capped].min())
    duration = 10.0 ** np.asarray(grid[names[1]]).reshape(n, n)
    luminosity = 10.0 ** np.asarray(grid[names[0]]).reshape(n, n)

    def field(name):
        """
        Reshape a column of the grid onto the square map.

        Parameters
        ----------
        name : str
            The column of ``grid``.

        Returns
        -------
        numpy.ndarray
            The column as an ``(n, n)`` array.
        """
        return np.asarray(grid[name]).reshape(n, n)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharex=True, sharey=True, constrained_layout=True)
    recovered = field("recovered_volume_time")
    panels = [
        # name, colorbar label, norm, cmap, contour levels
        (
            "f",
            "recovery fraction in the sampled footprint, f",
            LogNorm(vmin=1e-3, vmax=1.0),
            "viridis",
            [0.01, 0.1, 0.5],
        ),
        (
            "efficiency",
            r"all-sky efficiency $\eta = f \times$ coverage",
            LogNorm(vmin=1e-4, vmax=max(field("efficiency").max(), 2e-4)),
            "viridis",
            [1e-3, 1e-2, 1e-1],
        ),
        (
            "recovered_volume_time",
            r"recovered volume-time $f\,\mathcal{V}T$ [Mpc$^3$ yr]",
            LogNorm(vmin=max(recovered.max() * 1e-4, 1.0)),
            "magma",
            [],
        ),
        ("sigma_log", r"uncertainty of $\ln(f+\epsilon)$", None, "cividis", []),
    ]
    for ax, (name, label, norm, cmap, levels) in zip(axes.ravel(), panels):
        values = np.clip(field(name), 1e-12, None)
        mesh = ax.pcolormesh(duration, luminosity, values, norm=norm, cmap=cmap, shading="auto")
        fig.colorbar(mesh, ax=ax, label=label)
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
            contours = ax.contour(duration, luminosity, field(name), levels=levels, colors="white", linewidths=1)
            _label_once(ax, contours, "%g")
        if clip_log_l is not None and grid[names[0]].min() < clip_log_l < grid[names[0]].max():
            ax.axhline(10.0**clip_log_l, color="white", linestyle="--", linewidth=1)

    if "expected_recoveries" in grid.colnames:
        expected = field("expected_recoveries")
        levels = [level for level in (1, 10, 100, 1000, 1e4) if expected.min() < level < expected.max()]
        if levels:
            contours = axes[1, 0].contour(duration, luminosity, expected, levels=levels, colors="cyan", linewidths=1)
            _label_once(
                axes[1, 0], contours, lambda level: f"{level:g} recovered" if level == 1 else f"{level:g} recoveries"
            )

    for ax in axes[1]:
        ax.set_xlabel("rest-frame duration [day]")
    band = grid.meta.get("band")
    luminosity_label = rf"$L_\mathrm{{{band}}}$" if band else r"$L$"
    for ax in axes[:, 0]:
        ax.set_ylabel(f"{luminosity_label} [erg s$^{{-1}}$]")

    # Magnitude is linear in log L, so a twin axis with the limits of the luminosity axis converted lines up exactly.
    zero_point = grid.meta.get("absolute_magnitude_zero_point")
    if zero_point is not None:
        log_l_lo, log_l_hi = grid[names[0]].min(), grid[names[0]].max()
        for ax in axes[:, 1]:
            magnitude = ax.twinx()
            magnitude.set_ylim(zero_point - 2.5 * log_l_lo, zero_point - 2.5 * log_l_hi)
            magnitude.set_ylabel(r"absolute AB magnitude $M$")
    if clip_log_l is not None:
        axes[0, 0].annotate(
            r"$z_\mathrm{lim}$ capped above",
            (axes[0, 0].get_xlim()[0] * 1.1, 10.0**clip_log_l * 1.15),
            color="white",
            fontsize=9,
        )
    if title:
        fig.suptitle(title)
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig
