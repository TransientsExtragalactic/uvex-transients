"""
Cut: Redshift
=================

:meth:`~uvex_transients.simulation.core.SurveySimulator.filter_by_redshift` cuts on an
`~uvex_transients.simulation.event_catalog.EventCatalog`'s own ``redshift`` column, no
schedule or photometry involved. It is a cheap way to restrict a population to a
redshift range of interest, e.g. a nearby subsample for individual light-curve
follow-up.
"""

import numpy as np
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

catalog = simulator.generate_events(time_bins=TIME_BINS, nside=NSIDE, downsample=DOWNSAMPLE)
mag_filtered = simulator.filter_by_limiting_magnitude(catalog, uvex, mag_limit=MAG_LIMIT)
print(f"{len(mag_filtered) * DOWNSAMPLE} events available before the redshift cut.")

# %%
# Keeping only a nearby subsample
# -------------------------------------
#
# `min_redshift`/`max_redshift` can be given independently or together.

nearby = simulator.filter_by_redshift(mag_filtered, uvex, max_redshift=0.5)
mid_shell = simulator.filter_by_redshift(mag_filtered, uvex, min_redshift=0.5, max_redshift=1.5)
print(f"z <= 0.5: {len(nearby) * DOWNSAMPLE} events.")
print(f"0.5 < z <= 1.5: {len(mid_shell) * DOWNSAMPLE} events.")

# %%
# Before and after
# ---------------------

fig, ax = resolve_fig_axes(fig_size=(6, 4))
bins = np.linspace(0, mag_filtered.table["redshift"].max(), 30)
ax.hist(mag_filtered.table["redshift"], bins=bins, alpha=0.5, label="Before cut", color="#888888")
ax.hist(nearby.table["redshift"], bins=bins, alpha=0.8, label="z <= 0.5", color="#4C72B0")
ax.set_xlabel("Redshift")
ax.set_ylabel("Events (simulated)")
ax.set_title("Effect of the redshift cut")
ax.legend()
fig.tight_layout()
plt.show()
