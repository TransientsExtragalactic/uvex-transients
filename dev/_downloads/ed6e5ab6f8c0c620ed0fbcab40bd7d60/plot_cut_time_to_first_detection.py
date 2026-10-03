"""
Cut: Time to First Detection
=================================

:meth:`~uvex_transients.simulation.core.SurveySimulator.filter_by_time_to_first_detection`
cuts on the time (days) from explosion to each event's first SNR-qualifying detection.
It is schedule-aware, like
:meth:`~uvex_transients.simulation.core.SurveySimulator.filter_by_snr`, and is useful for
selecting events caught early, e.g. while a rise is still being resolved.
"""

from matplotlib import pyplot as plt

from uvex_transients.missions import uvex_fast as uvex
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
print(f"{len(mag_filtered) * DOWNSAMPLE} events available before the delay cut.")

# %%
# Keeping only events caught early
# --------------------------------------
#
# An event with zero SNR-qualifying detections never survives, regardless of
# `min_delay`/`max_delay`.

early = simulator.filter_by_time_to_first_detection(mag_filtered, uvex, snr_threshold=SNR_THRESHOLD, max_delay=14.0)
late = simulator.filter_by_time_to_first_detection(mag_filtered, uvex, snr_threshold=SNR_THRESHOLD, min_delay=100.0)
print(f"First detected within 14 d: {len(early) * DOWNSAMPLE} events.")
print(f"First detected after 100 d: {len(late) * DOWNSAMPLE} events.")

# %%
# Early vs. late counts
# ---------------------------

fig, ax = resolve_fig_axes(fig_size=(5, 4))
ax.bar(
    ["max_delay=14 d", "min_delay=100 d"],
    [len(early) * DOWNSAMPLE, len(late) * DOWNSAMPLE],
    color=["#4C72B0", "#C44E52"],
)
ax.set_ylabel("Events surviving")
ax.set_title("Effect of the time-to-first-detection cut")
fig.tight_layout()
plt.show()
