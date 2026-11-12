"""The figure of a recovery map over any two axes."""

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from astropy.table import Table

from uvex_transients.phase_space.plotting.core import F_LEVELS, F_RANGE, SAVE_DPI, axis_label, label_once
from uvex_transients.phase_space.space import Axis, Space
from uvex_transients.utils.plotting import set_plot_style

__all__ = ["plot_grid"]


# ============================================ #
# Figure                                       #
# ============================================ #
def _grid_panels(fields: dict, products: dict, selected: Sequence[str] | None) -> list[tuple]:
    """
    Describe the panels of `plot_grid`.

    Each is a tuple of the column drawn, its colour bar label, colour scale and colour map, and the levels of its
    contours.
    """
    from matplotlib.colors import LogNorm

    if selected is None:
        # The success fraction, every yield (a product with an interval), and the uncertainty
        yields = [name for name in products if f"{name}_lo" in fields]
        selected = ["f", *yields, "sigma_log"]

    panels = []
    for name in selected:
        if name not in fields:
            raise ValueError(f"The table has no column {name!r}; it has {sorted(fields)}.")
        if name == "f":
            panels.append((name, "success fraction f", LogNorm(*F_RANGE), "viridis", list(F_LEVELS)))
        elif name == "sigma_log":
            panels.append((name, r"uncertainty of $\ln(f+\epsilon)$", None, "cividis", []))
        else:
            product = products.get(name, {})
            label = product.get("label") or name
            label = f"{label} [{product['unit']}]" if product.get("unit") else label
            positive = fields[name][np.isfinite(fields[name]) & (fields[name] > 0)]
            if positive.size == 0:
                panels.append((name, label, None, "viridis", []))
                continue
            vmax = positive.max()
            vmin = max(positive.min(), vmax * 1e-6)
            decades = range(int(np.ceil(np.log10(vmin))), int(np.floor(np.log10(vmax))) + 1)
            panels.append((name, label, LogNorm(vmin, vmax), "viridis", [10.0**k for k in decades][::2]))
    return panels


def plot_grid(
    grid: Table,
    points: Table | None = None,
    path: str | Path | None = None,
    title: str | None = None,
    *,
    fields: Sequence[str] | None = None,
    show_points: bool = False,
    annotate: Callable | None = None,
):
    r"""
    Draw a two-axis map: the success fraction, its yields and its uncertainty, from the metadata of the table alone.

    The axes, their labels and units, and whether each is logarithmic come from the ``space`` in the table's ``meta``
    (`~uvex_transients.phase_space.map.PhaseSpaceMap.grid_table` records it), so nothing here knows what the axes
    are. The first axis is drawn vertically and the second horizontally, as in `plot_map`. A panel is made of the
    success fraction ``f``, of every product that carries an interval (a yield, ``<name>``, ``<name>_lo``,
    ``<name>_hi``), and of ``sigma_log``, unless `fields` says otherwise.

    Parameters
    ----------
    grid : astropy.table.Table
        The output of `~uvex_transients.phase_space.map.PhaseSpaceMap.grid_table` of a map over two axes.
    points : astropy.table.Table, optional
        The measured points (``PhaseSpaceMap.table``), drawn when `show_points` is set.
    path : str or pathlib.Path, optional
        Where to write the figure. If omitted, the figure is not saved.
    title : str, optional
        A suptitle.
    fields : sequence of str, optional
        The columns to draw, one panel each.
    show_points : bool, optional
        Whether to mark the measured points, as circles that grow with the number of tries.
    annotate : callable, optional
        ``annotate(ax)``, called on every panel to draw anything else, such as the region a class of transients
        occupies. The coordinates are the plotted ones (the values of a logarithmic axis, not their logarithms).

    Returns
    -------
    matplotlib.figure.Figure
        The figure, left open so that it can be adjusted or shown.

    Raises
    ------
    ValueError
        If the table does not record its space, the space does not have two axes, or a column asked for is missing.
    """
    import matplotlib.pyplot as plt

    if "space" not in grid.meta:
        raise ValueError("The table does not record its space; make it with PhaseSpaceMap.grid_table.")
    space = Space.from_dict(grid.meta["space"])
    if space.ndim != 2:
        raise ValueError(f"A figure is of a map over two axes, and this one has {space.ndim}.")
    set_plot_style()

    # The grid, with the first axis varying slowest
    shape = tuple(len(np.unique(grid[name])) for name in space.names)
    if int(np.prod(shape)) != len(grid):
        raise ValueError("The table is not a regular grid of its space.")
    fields_by_name = {name: np.asarray(grid[name]).reshape(shape) for name in grid.colnames}

    def plotted(axis: Axis) -> np.ndarray:
        """
        Return the grid of an axis as it is drawn: the values of a logarithmic axis, not their logarithms.

        Parameters
        ----------
        axis : Axis
            The axis.

        Returns
        -------
        numpy.ndarray
            The coordinate at each point of the grid.
        """
        values = fields_by_name[axis.name]
        return 10.0**values if axis.scale == "log10" else values

    vertical, horizontal = space.axes
    y, x = plotted(vertical), plotted(horizontal)
    products = {product["name"]: product for product in grid.meta.get("products", [])}
    panels = _grid_panels(fields_by_name, products, fields)

    n_rows = int(np.ceil(len(panels) / 2))
    n_columns = 2 if len(panels) > 1 else 1
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(7 * n_columns, 5.5 * n_rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    for ax, (name, label, norm, cmap, levels) in zip(axes.ravel(), panels):
        values = np.clip(fields_by_name[name], 1e-12, None) if norm is not None else fields_by_name[name]
        mesh = ax.pcolormesh(x, y, values, norm=norm, cmap=cmap, shading="auto")
        fig.colorbar(mesh, ax=ax, label=label)
        if show_points and points is not None:
            px, py = np.asarray(points[horizontal.name]), np.asarray(points[vertical.name])
            ax.scatter(
                10.0**px if horizontal.scale == "log10" else px,
                10.0**py if vertical.scale == "log10" else py,
                s=8 + 30 * np.log10(points["n"] / points["n"].min() + 1.0),
                facecolor="none",
                edgecolor="white",
                linewidth=0.8,
            )
        if horizontal.scale == "log10":
            ax.set_xscale("log")
        if vertical.scale == "log10":
            ax.set_yscale("log")
        levels = [
            level for level in levels if np.nanmin(fields_by_name[name]) < level < np.nanmax(fields_by_name[name])
        ]
        if levels:
            label_once(ax, ax.contour(x, y, fields_by_name[name], levels=levels, colors="white", linewidths=1), "%g")
        if annotate is not None:
            annotate(ax)

    for ax in axes.ravel()[len(panels) :]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(axis_label(horizontal))
    for ax in axes[:, 0]:
        ax.set_ylabel(axis_label(vertical))
    axes[0, 0].set_xlim(x.min(), x.max())
    axes[0, 0].set_ylim(y.min(), y.max())

    if title:
        fig.suptitle(title)
    if path is not None:
        fig.savefig(path, dpi=SAVE_DPI)
    return fig
