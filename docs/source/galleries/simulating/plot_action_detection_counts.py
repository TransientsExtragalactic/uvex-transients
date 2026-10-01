"""
Action: Detection Counts
=============================

:meth:`~uvex_transients.simulation.core.SurveySimulator.run_detection_counts_action`
estimates, per transient type, how many events show :math:`N_{\\rm det}\\ge k` detected
epochs, for every :math:`k`. It generalizes
:meth:`~uvex_transients.simulation.core.SurveySimulator.filter_by_snr`'s own
detected-or-not split (its :math:`k=1` row) to the full distribution of visit counts.
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
MAG_LIMIT = 22.0
SNR_THRESHOLD = 5.0

catalog = simulator.generate_events(time_bins=TIME_BINS, nside=NSIDE, downsample=DOWNSAMPLE)
mag_filtered = simulator.filter_by_limiting_magnitude(catalog, uvex, mag_limit=MAG_LIMIT)
exposure = simulator.compute_effective_exposure(time_bins=TIME_BINS, nside=NSIDE)
print(f"{len(mag_filtered)} events clear {MAG_LIMIT} AB mag (a stricter limit, to keep photometry fast here).")

# %%
# Photometry over the magnitude-filtered catalog
# -----------------------------------------------------
#
# `~uvex_transients.simulation.core.SurveySimulator.run_detection_counts_action` needs
# real synthetic photometry (not just a keep/discard cut) to know how many epochs each
# event was actually detected at, so photometry is run first, over the whole
# magnitude-filtered catalog.

photometry = simulator.run_photometry_action(catalog=mag_filtered, mission=uvex)

# %%
# The detection-count table
# -------------------------------

counts = simulator.run_detection_counts_action(
    catalog=mag_filtered,
    exposure=exposure,
    photometry=photometry,
    mission=uvex,
    snr_threshold=SNR_THRESHOLD,
)
print(counts["transient_type", "n_detections", "n_at_least", "fraction", "expected_events"])

# %%
# Survival fraction vs. required visit count
# -------------------------------------------------

fig, ax = resolve_fig_axes(fig_size=(6, 4))
ax.bar(counts["n_detections"], counts["fraction"], color="#4C72B0")
ax.set_xlabel(r"Minimum detected epochs, $k$")
ax.set_ylabel(r"Fraction with $N_{\rm det} \geq k$")
ax.set_title("SLSNe detection-count distribution")
fig.tight_layout()
plt.show()
