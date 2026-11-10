r"""
Detection Maps
==============

How detectable is a transient of a given peak luminosity and decline time, in the FUV and in the NUV?

A transient is **detected** if at least two observations reach :math:`5\sigma` in the band of the map, with an event
that is only seen on the first visit to its field not counted. The pages around this one explain how a map is made and
how to read it. Here are the maps of the latest release, one for each band, with the shipped transient classes shaded
on them.

The data is read through :func:`~uvex_transients.utils.results.get_phase_space_map`. Redrawing a map needs no
simulation: the figure is made from the fitted map and the simulated points.
"""

import matplotlib.pyplot as plt
from astropy.table import Table

from uvex_transients.phase_space.overlay import split_measurements
from uvex_transients.phase_space.plotting import plot_map
from uvex_transients.utils import get_phase_space_map
from uvex_transients.utils.plotting import set_plot_style

set_plot_style()

# %%
# Load the maps
# --------------
#
# Each map is one band: ``map.ecsv`` is the fitted map on a grid, ``points.ecsv`` the simulated points it was fitted to
# and ``overlay.ecsv`` the measured transient classes.

MAPS = {"FUV": "fuv_detection", "NUV": "nuv_detection"}

try:
    maps = {}
    for band, map_id in MAPS.items():
        paths = get_phase_space_map(map_id, ["map", "points", "overlay"])
        maps[band] = {key: Table.read(path, format="ascii.ecsv") for key, path in paths.items()}
except (ConnectionError, LookupError, OSError, ValueError) as err:
    maps = None
    print(f"Skipping this page's analysis, the release data is not available: {err}")
else:
    for band, tables in maps.items():
        print(f"{band}: {len(tables['points'])} simulated points, {int(tables['points']['n'].sum()):,} events.")

# %%
# The FUV map
# ------------
#
# The luminosity is the one emitted in the FUV band, :math:`L_\mathrm{FUV}`, so the shaded classes are measured in
# the same band.

if maps is not None:
    tables = maps["FUV"]
    plot_map(
        tables["map"],
        tables["points"],
        title="FUV detection map",
        overlays=split_measurements(tables["overlay"]),
    )
    plt.show()

# %%
# The NUV map
# ------------
#
# The same map in the NUV band, with the luminosity :math:`L_\mathrm{NUV}` emitted in that band.

if maps is not None:
    tables = maps["NUV"]
    plot_map(
        tables["map"],
        tables["points"],
        title="NUV detection map",
        overlays=split_measurements(tables["overlay"]),
    )
    plt.show()
