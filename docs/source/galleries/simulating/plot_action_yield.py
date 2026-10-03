"""
Action: Yield Summary
==========================

:meth:`~uvex_transients.simulation.core.SurveySimulator.run_yield_action` combines a raw
(pre-cut) catalog, a detected (post-cut) catalog, and an
`~uvex_transients.simulation.exposure_catalog.ExposureCatalog` into a per-transient-type
`~uvex_transients.simulation.yield_table.YieldTable`: expected detection counts, with
both Monte Carlo and rate-normalization uncertainty kept separate.
"""

from uvex_transients.missions import uvex_fast as uvex
from uvex_transients.simulation.core import SurveySimulator
from uvex_transients.surveys import get_schedule
from uvex_transients.transients.supernovae import MagnetarSLSNe
from uvex_transients.utils.plotting import set_plot_style

set_plot_style()

schedule = get_schedule()
slsn = MagnetarSLSNe()
simulator = SurveySimulator(schedule, transients={"slsn": slsn}, simulation_seed=42)

TIME_BINS = 10
NSIDE = 32
DOWNSAMPLE = 50
MAG_LIMIT = 25.0
SNR_THRESHOLD = 5.0

# %%
# Building the three inputs
# -------------------------------
#
# `raw` is the catalog before any schedule-aware cut; `detected` is after
# :meth:`~uvex_transients.simulation.core.SurveySimulator.filter_by_snr`; `exposure` comes
# from :meth:`~uvex_transients.simulation.core.SurveySimulator.compute_effective_exposure`,
# not from either catalog.

raw = simulator.generate_events(time_bins=TIME_BINS, nside=NSIDE, downsample=DOWNSAMPLE)
mag_filtered = simulator.filter_by_limiting_magnitude(raw, uvex, mag_limit=MAG_LIMIT)
detected = simulator.filter_by_snr(mag_filtered, uvex, snr_threshold=SNR_THRESHOLD)
exposure = simulator.compute_effective_exposure(time_bins=TIME_BINS, nside=NSIDE)

# %%
# The yield table
# --------------------

yields = simulator.run_yield_action(raw=raw, detected=detected, exposure=exposure, mission=uvex)
print(yields.table["transient_type", "uvex_intrinsic_events", "detected_events", "expected_detections"])

for row in yields.table:
    lower = row["expected_detections"] - row["expected_detections_binom_lower"]
    upper = row["expected_detections_binom_upper"] - row["expected_detections"]
    print(
        f"{row['transient_type']}: expected {row['expected_detections']:.0f} "
        f"(+{upper:.0f}/-{lower:.0f}) detections over the whole survey."
    )
