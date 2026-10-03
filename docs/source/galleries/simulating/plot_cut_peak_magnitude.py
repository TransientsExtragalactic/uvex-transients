"""
Cut: Peak Magnitude
========================

:meth:`~uvex_transients.simulation.core.SurveySimulator.filter_by_peak_magnitude` cuts on
each event's own peak (brightest) apparent AB magnitude, ignoring Milky Way dust
attenuation and the survey schedule entirely. It is a cheap, purely
intrinsic-plus-distance screen, distinct from
:meth:`~uvex_transients.simulation.core.SurveySimulator.filter_by_limiting_magnitude`
(which folds dust in and checks a shared phase grid rather than each event's own peak).
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

catalog = simulator.generate_events(time_bins=TIME_BINS, nside=NSIDE, downsample=DOWNSAMPLE)
print(f"{len(catalog) * DOWNSAMPLE} events sampled.")

# %%
# Comparing peak-magnitude limits
# -------------------------------------
#
# `min_mag`/`max_mag` bound how bright (numerically small) an event's own peak apparent
# magnitude is allowed to be.

max_mags = [21.0, 22.0, 23.5]
counts = []
for max_mag in max_mags:
    bright = simulator.filter_by_peak_magnitude(catalog, uvex, max_mag=max_mag)
    counts.append(len(bright) * DOWNSAMPLE)
    print(f"max_mag={max_mag}: {len(bright) * DOWNSAMPLE} events peak this bright.")

# %%
# The rising survival count
# -------------------------------

fig, ax = resolve_fig_axes(fig_size=(6, 4))
ax.bar([str(m) for m in max_mags], counts, color="#4C72B0")
ax.set_yscale("log")
ax.set_xlabel("Peak-magnitude limit (AB)")
ax.set_ylabel("Events surviving")
ax.set_title("Effect of the peak-magnitude cut")
fig.tight_layout()
plt.show()
