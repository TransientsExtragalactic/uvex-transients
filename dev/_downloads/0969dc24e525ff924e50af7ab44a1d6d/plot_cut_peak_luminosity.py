"""
Cut: Peak Luminosity
=========================

:meth:`~uvex_transients.simulation.core.SurveySimulator.filter_by_peak_luminosity` cuts on
each event's own peak bolometric luminosity (erg/s): purely intrinsic, with no bands, no
distance, no dust, and no schedule involved at all. It is the distance-independent
analog of :meth:`~uvex_transients.simulation.core.SurveySimulator.filter_by_peak_magnitude`/
:meth:`~uvex_transients.simulation.core.SurveySimulator.filter_by_peak_flux`.
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

catalog = simulator.generate_events(time_bins=TIME_BINS, nside=NSIDE, downsample=DOWNSAMPLE)
print(f"{len(catalog) * DOWNSAMPLE} events sampled.")

# %%
# Comparing peak-luminosity limits
# --------------------------------------
#
# Unlike `filter_by_peak_magnitude`/`filter_by_peak_flux`, this cut doesn't depend on
# where an event happens to fall on the sky, or how far away it is: it is purely a
# property of the sampled SED parameters themselves.

min_luminosities = [1e43, 3e43, 1e44]
counts = []
for min_luminosity in min_luminosities:
    luminous = simulator.filter_by_peak_luminosity(catalog, uvex, min_luminosity=min_luminosity)
    counts.append(len(luminous) * DOWNSAMPLE)
    print(f"min_luminosity={min_luminosity:.0e}: {len(luminous) * DOWNSAMPLE} events clear it.")

# %%
# The falling survival count
# -------------------------------

fig, ax = resolve_fig_axes(fig_size=(6, 4))
ax.bar([f"{val:.0e}" for val in min_luminosities], counts, color="#8172B2")
ax.set_yscale("log")
ax.set_xlabel("Peak-luminosity limit (erg/s)")
ax.set_ylabel("Events surviving")
ax.set_title("Effect of the peak-luminosity cut")
fig.tight_layout()
plt.show()
