"""
Cut: Baseline
=================

:meth:`~uvex_transients.simulation.core.SurveySimulator.filter_by_baseline` cuts on the
spacing (days) between each event's own SNR-qualifying detection epochs: `min_baseline`
requires some pair of detections closer together than that; `max_baseline` requires the
full detected span to exceed it. Together they pick out events with both a dense
sub-cadence and a long overall baseline, e.g. for light-curve-shape science.
"""

from m4opt.missions import uvex
from matplotlib import pyplot as plt

from uvex_transients.simulation.core import SurveySimulator
from uvex_transients.surveys import get_schedule
from uvex_transients.transients.supernovae import MagnetarSLSNe
from uvex_transients.utils.plotting import resolve_fig_axes, set_plot_style

set_plot_style()

schedule = get_schedule()
slsn = MagnetarSLSNe()
simulator = SurveySimulator(schedule, transients={"slsn": slsn}, simulation_seed=42)

TIME_BINS = 10
NSIDE = 32
DOWNSAMPLE = 50
MAG_LIMIT = 25.0
SNR_THRESHOLD = 5.0

catalog = simulator.generate_events(time_bins=TIME_BINS, nside=NSIDE, downsample=DOWNSAMPLE)
mag_filtered = simulator.filter_by_limiting_magnitude(catalog, uvex, mag_limit=MAG_LIMIT)
detected = simulator.filter_by_snr(mag_filtered, uvex, snr_threshold=SNR_THRESHOLD)
print(f"{len(detected) * DOWNSAMPLE} events detected at all.")

# %%
# Requiring a long, well-sampled baseline
# ----------------------------------------------
#
# An event with fewer than 2 qualifying epochs never satisfies `min_baseline` (there is
# no gap to compare); one with 0 or 1 never satisfies `max_baseline` (there is no span).

well_sampled = simulator.filter_by_baseline(
    mag_filtered, uvex, snr_threshold=SNR_THRESHOLD, min_baseline=15.0, max_baseline=100.0
)
print(f"{len(well_sampled) * DOWNSAMPLE} events have a gap < 15 d and a span > 100 d.")

# %%
# Before and after
# ---------------------

fig, ax = resolve_fig_axes(fig_size=(5, 4))
ax.bar(
    ["Detected", "min<15 d, max>100 d"],
    [len(detected) * DOWNSAMPLE, len(well_sampled) * DOWNSAMPLE],
    color=["#888888", "#4C72B0"],
)
ax.set_yscale("log")
ax.set_ylabel("Events surviving")
ax.set_title("Effect of the baseline cut")
fig.tight_layout()
plt.show()
