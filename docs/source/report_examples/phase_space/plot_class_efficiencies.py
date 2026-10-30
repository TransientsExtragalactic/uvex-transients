r"""
Classes on the Maps
===================

Where does each transient class sit on the detection maps, and what is the map's detection probability there?

The draws that measure a class's region on a map (see the :ref:`maps <report_phase_space_gallery>`) also place each one
at a point of the map, so the fitted map can be read off at every draw. This page does that for the **all-sky
efficiency** :math:`\eta`, the probability that an event of that peak luminosity and half-decline time, exploding at a
random position and time during the survey within its redshift limit, is detected in the band of the map.

This is a reading of the map, not the yield of the class. The map holds the rise, the tail and the colour of the event
fixed, and a class does not, so the Standard Run, which simulates each class, is the number to quote. The comparison is
a check, and shows how much of each class's spread in detectability comes from its luminosity and decline time alone.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from scipy.interpolate import LinearNDInterpolator

from uvex_transients.phase_space.overlay import split_measurements
from uvex_transients.utils import get_phase_space_map
from uvex_transients.utils.plotting import get_band_color, set_plot_style

set_plot_style()

# %%
# Load the maps
# --------------

MAPS = {"FUV": "fuv_detection", "NUV": "nuv_detection"}

try:
    maps = {}
    for band, map_id in MAPS.items():
        paths = get_phase_space_map(map_id, ["map", "overlay"])
        maps[band] = {key: Table.read(path, format="ascii.ecsv") for key, path in paths.items()}
except (ConnectionError, LookupError, OSError, ValueError) as err:
    maps = None
    print(f"Skipping this page's analysis, the release data is not available: {err}")

# %%
# Read the map at every draw
# ---------------------------
#
# The fitted map is on a grid, so the efficiency at a draw is interpolated linearly between the grid points around it
# in the log luminosity and the log half-decline time. A draw outside the map has no value, and is counted as off the
# map.


def efficiency_at_draws(grid, measurements):
    """Return the map's efficiency at each draw of a class, NaN where the draw is off the map or not measured."""
    names = grid.colnames[:2]
    interpolate = LinearNDInterpolator(
        np.column_stack([grid[names[0]], grid[names[1]]]), np.asarray(grid["efficiency"])
    )
    efficiency = interpolate(np.asarray(measurements["log_luminosity"]), np.asarray(measurements["log_half_time"]))
    return np.where(np.asarray(measurements["measured"]), efficiency, np.nan)


if maps is not None:
    rows = {}
    for band, tables in maps.items():
        for measurements in split_measurements(tables["overlay"]):
            eta = efficiency_at_draws(tables["map"], measurements)
            on_map = np.isfinite(eta)
            low, median, high = np.percentile(eta[on_map], [16, 50, 84]) if on_map.any() else (np.nan,) * 3
            rows.setdefault(measurements.meta["class"], {})[band] = (median, low, high, on_map.mean())

    classes = sorted(rows)
    print(f"{'class':36s}" + "".join(f"{band + ' eta (16-50-84%)':>34s}{'on map':>9s}" for band in maps))
    for name in classes:
        line = f"{name:36s}"
        for band in maps:
            median, low, high, on_map = rows[name][band]
            line += f"{low:11.2g}{median:11.2g}{high:11.2g}{on_map:9.0%}"
        print(line)

# %%
# The efficiency of each class
# -----------------------------
#
# The bar is the median over the draws of a class that are on the map, and the whiskers the 16th and 84th percentiles.
# A class whose draws mostly lie off the map is not shown for that band, because the map does not reach it.

if maps is not None:
    fig, axes = plt.subplots(1, len(maps), figsize=(11, 5), sharey=True, constrained_layout=True)
    for ax, band in zip(np.atleast_1d(axes), maps):
        shown = [name for name in classes if rows[name][band][3] >= 0.5]
        median, low, high, _ = (np.array([rows[name][band][k] for name in shown]) for k in range(4))
        ax.barh(shown, median, color=get_band_color(band), alpha=0.8)
        ax.errorbar(median, shown, xerr=[median - low, high - median], fmt="none", ecolor="black", capsize=3)
        ax.set_xscale("log")
        ax.set_xlabel(rf"all-sky efficiency $\eta$ in the {band}")
        ax.invert_yaxis()
    plt.show()
