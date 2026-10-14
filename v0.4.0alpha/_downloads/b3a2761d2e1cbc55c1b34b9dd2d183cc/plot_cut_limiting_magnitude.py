"""
Cut: Limiting Magnitude
===========================

:meth:`~uvex_transients.simulation.core.SurveySimulator.filter_by_limiting_magnitude` is the
cheapest of the cuts: no schedule, no noise model, just "could this event, at its brightest
sampled phase, ever clear a fixed magnitude limit?" It is meant to run first, on a freshly
sampled catalog, before anything schedule-aware.
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

# %%
# Sampling a population
# --------------------------
#
# A freshly sampled catalog, with no screening applied yet.

TIME_BINS = 10
NSIDE = 32
DOWNSAMPLE = 50

catalog = simulator.generate_events(time_bins=TIME_BINS, nside=NSIDE, downsample=DOWNSAMPLE)
print(f"Sampled {len(catalog) * DOWNSAMPLE} SLSNe across {TIME_BINS} time bin(s) at NSIDE={NSIDE}.")

# %%
# Comparing magnitude limits
# -------------------------------
#
# Stricter limits keep fewer events. ``n_visits`` raises the bar further: an event must
# clear the limit at more than one sampled phase, not just momentarily.

mag_limits = [22.0, 23.5, 25.0]
counts = []
for mag_limit in mag_limits:
    kept = simulator.filter_by_limiting_magnitude(catalog, uvex, mag_limit=mag_limit)
    counts.append(len(kept) * DOWNSAMPLE)
    print(f"mag_limit={mag_limit}: {len(kept) * DOWNSAMPLE} survive.")

strict_kept = simulator.filter_by_limiting_magnitude(catalog, uvex, mag_limit=25.0, n_visits=3)
print(f"mag_limit=25.0, n_visits=3: {len(strict_kept) * DOWNSAMPLE} survive.")

# %%
# The falling survival count
# -------------------------------

fig, ax = resolve_fig_axes(fig_size=(6, 4))
ax.bar([str(m) for m in mag_limits], counts, color="#4C72B0")
ax.set_yscale("log")
ax.set_xlabel("Magnitude limit (AB)")
ax.set_ylabel("Events surviving")
ax.set_title("Effect of the limiting-magnitude cut")
fig.tight_layout()
plt.show()
