"""
Cut: Signal-to-Noise Ratio
===============================

:meth:`~uvex_transients.simulation.core.SurveySimulator.filter_by_snr` asks the real
question a magnitude-limit screen can only approximate: over every observation the
schedule actually made of an event's position while it was active, is it ever detected
above a given SNR? It is schedule-aware, so it is run after the cheaper
:meth:`~uvex_transients.simulation.core.SurveySimulator.filter_by_limiting_magnitude`,
not instead of it.
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

catalog = simulator.generate_events(time_bins=TIME_BINS, nside=NSIDE, downsample=DOWNSAMPLE)
mag_filtered = simulator.filter_by_limiting_magnitude(catalog, uvex, mag_limit=MAG_LIMIT)
print(f"{len(mag_filtered) * DOWNSAMPLE} could ever clear {MAG_LIMIT} AB mag.")

# %%
# Comparing SNR thresholds
# ----------------------------
#
# Raising `n_visits` on top of `snr_threshold` requires the event to be detected more
# than once, which is a much stricter, and schedule-real, requirement than
# `filter_by_limiting_magnitude`'s own `n_visits` ever could be.

snr_thresholds = [3.0, 5.0, 10.0]
counts = []
for snr_threshold in snr_thresholds:
    detected = simulator.filter_by_snr(mag_filtered, uvex, snr_threshold=snr_threshold)
    counts.append(len(detected) * DOWNSAMPLE)
    print(f"snr_threshold={snr_threshold}: {len(detected) * DOWNSAMPLE} detected.")

twice_detected = simulator.filter_by_snr(mag_filtered, uvex, snr_threshold=5.0, n_visits=2)
print(f"snr_threshold=5.0, n_visits=2: {len(twice_detected) * DOWNSAMPLE} detected on 2+ visits.")

# %%
# The detected count vs. threshold
# -------------------------------------

fig, ax = resolve_fig_axes(fig_size=(6, 4))
ax.bar([str(s) for s in snr_thresholds], counts, color="#55A868")
ax.set_yscale("log")
ax.set_xlabel("SNR threshold")
ax.set_ylabel("Events detected")
ax.set_title("Effect of the SNR cut")
fig.tight_layout()
plt.show()
