"""
Cut: Peak Flux
==================

:meth:`~uvex_transients.simulation.core.SurveySimulator.filter_by_peak_flux` is the same
evaluation as :meth:`~uvex_transients.simulation.core.SurveySimulator.filter_by_peak_magnitude`,
just compared in flux (erg/s/cm^2/Hz) rather than AB magnitude, for a threshold already
in hand as a flux.
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
# Comparing peak-flux limits
# --------------------------------
#
# `min_flux` keeps only events whose peak observed flux clears a threshold, in
# erg/s/cm^2/Hz.

min_fluxes = [1e-27, 1e-28, 1e-29]
counts = []
for min_flux in min_fluxes:
    bright = simulator.filter_by_peak_flux(catalog, uvex, min_flux=min_flux)
    counts.append(len(bright) * DOWNSAMPLE)
    print(f"min_flux={min_flux:.0e}: {len(bright) * DOWNSAMPLE} events clear it.")

# %%
# The falling survival count
# -------------------------------

fig, ax = resolve_fig_axes(fig_size=(6, 4))
ax.bar([f"{f:.0e}" for f in min_fluxes], counts, color="#55A868")
ax.set_yscale("log")
ax.set_xlabel("Peak-flux limit (erg/s/cm^2/Hz)")
ax.set_ylabel("Events surviving")
ax.set_title("Effect of the peak-flux cut")
fig.tight_layout()
plt.show()
