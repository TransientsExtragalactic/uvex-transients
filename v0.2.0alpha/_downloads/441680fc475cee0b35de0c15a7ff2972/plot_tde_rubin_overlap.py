"""
UVEX TDEs in the Rubin Footprint
================================

How many of the TDEs UVEX detects also fall where Rubin LSST observes? Those are the ones
that can have optical follow-up or pre-discovery data from LSST. We'll:

1. **Sample TDEs from the default UVEX schedule** and keep the ones UVEX detects.
2. **Mask them against the LSST footprints** with
   :meth:`~uvex_transients.simulation.event_catalog.EventCatalog.in_footprint`.
3. **Count and plot** the overlap.

.. note::

   Generating the LSST footprints needs the optional ``rubin_scheduler`` package
   (``pip install "uvex-transients[rubin]"``). The finished MOCs are cached, so later runs don't.
"""

# %%
# Sample and detect TDEs with UVEX
# ------------------------------------
#
# This follows :ref:`the end-to-end TDE example <sphx_glr_auto_examples_simulating_plot_tde_end_to_end.py>`:
# draw a downsampled Monte Carlo population, then keep the events detected above
# :math:`\mathrm{SNR}=5` at an observation the schedule actually made.

from astropy import units as u
from matplotlib import pyplot as plt

from uvex_transients.missions import uvex_fast as uvex
from uvex_transients.simulation.core import SurveySimulator
from uvex_transients.surveys import get_schedule
from uvex_transients.surveys.footprints import lsst_combined_footprint
from uvex_transients.transients.TDEs import TidalDisruptionEvent
from uvex_transients.utils.plotting import get_categorical_colors, plot_footprints, resolve_fig_axes, set_plot_style

set_plot_style()

schedule = get_schedule()
tde = TidalDisruptionEvent()
simulator = SurveySimulator(schedule, transients={"tde": tde}, simulation_seed=42)

TIME_BINS = 20
NSIDE = 64
DOWNSAMPLE = 20
SNR_THRESHOLD = 5.0

catalog = simulator.generate_events(time_bins=TIME_BINS, nside=NSIDE, downsample=DOWNSAMPLE)
detected = simulator.filter_by_snr(catalog, uvex, snr_threshold=SNR_THRESHOLD)
print(f"{len(catalog) * DOWNSAMPLE} TDEs sampled, {len(detected) * DOWNSAMPLE} detected by UVEX.")

# %%
# Mask against the Rubin footprint
# ------------------------------------
#
# :meth:`~uvex_transients.simulation.event_catalog.EventCatalog.in_footprint` takes a footprint
# or its registered name and returns one boolean per event, from a single vectorized MOC lookup.
# ``lsst:combined`` is everything LSST observes: the main survey plus the deep drilling fields.

in_lsst = detected.in_footprint(lsst_combined_footprint)
n_detected = len(detected) * DOWNSAMPLE
n_overlap = int(in_lsst.sum()) * DOWNSAMPLE
print(f"{n_overlap} of {n_detected} UVEX-detected TDEs ({n_overlap / n_detected:.0%}) are in the LSST footprint.")

# %%
# Which LSST region?
# ----------------------
#
# The regions overlap, so the per-region counts need not add up to the total. The wide-fast-deep
# area carries the best LSST cadence and depth.

regions = {
    "Combined": "lsst:combined",
    "Wide-fast-deep": "lsst:wfd",
    "Galactic plane and bulge": "lsst:galplane",
    "North ecliptic spur": "lsst:nes",
    "South celestial pole": "lsst:scp",
    "Deep drilling fields": "lsst:ddf",
}
counts = {label: int(detected.in_footprint(name).sum()) * DOWNSAMPLE for label, name in regions.items()}
for label, count in counts.items():
    print(f"{label:>26}: {count}")

fig, ax = resolve_fig_axes(fig_size=(8, 4))
labels = list(counts)
ax.barh(labels, list(counts.values()), color=get_categorical_colors(1)[0])
ax.invert_yaxis()
ax.set_xlabel("UVEX-detected TDEs")
ax.set_title("UVEX-detected TDEs by LSST region")
for i, count in enumerate(counts.values()):
    ax.text(count, i, f" {count:,}", va="center")
fig.tight_layout()

# %%
# Sky distribution
# ---------------------
#
# The LSST footprint is drawn first, with the detected TDEs on top: those inside it are colored,
# and those outside are gray.

# sphinx_gallery_thumbnail_number = 2
fig, ax = plot_footprints({"LSST (combined)": lsst_combined_footprint}, colors=["#cfe0f5"], fig_size=(10, 6.5))
for mask, color, label in (
    (~in_lsst, "#888888", "Outside LSST"),
    (in_lsst, "#eb6834", "Inside LSST"),
):
    coord = detected.coord[mask]
    ax.scatter(
        coord.ra.wrap_at(180 * u.deg).radian,
        coord.dec.radian,
        s=6,
        color=color,
        label=f"{label} ({int(mask.sum()) * DOWNSAMPLE})",
    )
ax.legend(loc="lower right", markerscale=3)
ax.set_title("UVEX-detected TDEs and the Rubin LSST footprint")
plt.show()
