"""
TDE End-to-End Simulation
===========================

A minimal example of an end-to-end simulation of transient yields from a UVEX survey schedule. In this example,
we'll determine the anticipated yield of TDEs with the default UVEX schedule. To do this, we'll take the following
steps:

1. **Configure our transient class**, including changing any relevant priors, durations, etc.
2. **Draw events from the schedule**, and then
3. **Filter events by observability**.
4. **Generate synthetic photometry**.
"""

# %%
# Load the schedule and set up the population
# -----------------------------------------------
#
# We can start by loading the default schedule using
# :func:`~uvex_transients.surveys.utils.get_schedule`.
# We'll also want to get the TDE class configured and load the simulator.
#
# .. note::
#
#   At this stage, you can make modifications to priors as needed.

import numpy as np
from astropy import units as u
from m4opt.missions import uvex
from matplotlib import pyplot as plt

from uvex_transients.simulation.core import SurveySimulator
from uvex_transients.surveys import get_schedule
from uvex_transients.transients.TDEs import TidalDisruptionEvent

schedule = get_schedule()
tde = TidalDisruptionEvent()
simulator = SurveySimulator(schedule, transients={"tde": tde}, simulation_seed=42)

# %%
# Sample a Monte Carlo population
# -----------------------------------
#
# We now want to draw synthetic events for the schedule. This uses a **windowed sampling** technique
# so that only events which fall within the instrument's footprint at some point during their duration are
# sampled.
#
# .. note::
#
#   For events with very high intrinsic rates, it can be useful to provide a ``downsample`` value to
#   reduce the number of simulated events. This can then be used to rescale to the total number of events
#   at a later point in the analysis.
TIME_BINS = 20
NSIDE = 64
DOWNSAMPLE = 20

catalog = simulator.generate_events(time_bins=TIME_BINS, nside=NSIDE, downsample=DOWNSAMPLE)
print(f"Sampled {len(catalog) * DOWNSAMPLE} TDEs across {TIME_BINS} time bin(s) at NSIDE={NSIDE}.")

# %%
# Screen the population
# --------------------------
#
# Two progressively more expensive passes narrow the freshly-sampled catalog down to the
# events that matter.
#
# 1. :meth:`~uvex_transients.simulation.core.SurveySimulator.filter_by_limiting_magnitude` asks whether an event
#    could ever clear a fixed magnitude limit.
# 2. :meth:`~uvex_transients.simulation.core.SurveySimulator.filter_by_snr` asks the real question: is the event ever
#    detected above a given SNR at an observation the schedule actually made?

MAG_LIMIT = 25.0
SNR_THRESHOLD = 5.0

mag_filtered = simulator.filter_by_limiting_magnitude(catalog, uvex, mag_limit=MAG_LIMIT)
print(f"{len(mag_filtered) * DOWNSAMPLE} could ever clear {MAG_LIMIT} AB mag.")

detected = simulator.filter_by_snr(mag_filtered, uvex, snr_threshold=SNR_THRESHOLD)
print(f"{len(detected) * DOWNSAMPLE} were detected above SNR={SNR_THRESHOLD}.")

# %%
# Detection funnel
# --------------------

stages = ["Sampled", f"Mag < {MAG_LIMIT}", f"SNR > {SNR_THRESHOLD}"]
counts = [len(catalog) * DOWNSAMPLE, len(mag_filtered) * DOWNSAMPLE, len(detected) * DOWNSAMPLE]

fig, ax = plt.subplots()
ax.bar(stages, counts, color=["#888888", "#4C72B0", "#55A868"])
for i, count in enumerate(counts):
    ax.text(i, count, f"{count:,}", ha="center", va="bottom")
ax.set_ylabel("Number of TDEs")
ax.set_title("TDE detection funnel")
fig.tight_layout()

# %%
# Sky distribution
# ---------------------
#
# The sampled population traces the schedule's own footprint; the detected subset is
# whatever fraction of it UVEX actually caught above :math:`\mathrm{SNR}=5`.

# sphinx_gallery_thumbnail_number = 2
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(111, projection="aitoff")
ax.grid(True)

ra_sampled = catalog.coord.ra.wrap_at(180 * u.deg).radian
ra_detected = detected.coord.ra.wrap_at(180 * u.deg).radian

ax.scatter(
    ra_sampled,
    catalog.coord.dec.radian,
    s=2,
    alpha=0.2,
    color="#888888",
    label=f"Sampled ({DOWNSAMPLE * len(catalog)})",
)
ax.scatter(
    ra_detected, detected.coord.dec.radian, s=4, color="#55A868", label=f"Detected ({DOWNSAMPLE * (len(detected))})"
)
ax.legend(loc="lower right", markerscale=4)
ax.set_title("TDE sky distribution")
fig.tight_layout()

# %%
# An example light curve
# ---------------------------
#
# Reconstruct one detected event as a real :class:`~uvex_transients.simulation.event.Event`
# (:meth:`~uvex_transients.simulation.event_catalog.EventCatalog.get_events`) and run full
# synthetic photometry (:meth:`~uvex_transients.simulation.event.Event.simulate_photometry`)
# against every observation the schedule actually made of it.

rng = np.random.default_rng(1)
example_id = rng.choice(detected.event_id)
event = detected.get_events(int(example_id), {"tde": tde}, schedule)
print(event)

phot = event.simulate_photometry(uvex)
t_since_explosion = (phot["obs_time"] - event.t_explosion).to(u.day)
t_theory = np.linspace(0, tde.duration_limit.to_value(u.day), 300) * u.day

fig, ax = plt.subplots(figsize=(7, 4))
for band, color in {"FUV": "#4C72B0", "NUV": "#DD8452"}.items():
    ax.plot(t_theory.value, event.mag(t_theory, uvex, band=band).value, color=color, lw=1.5, alpha=0.6)

    in_band = np.isfinite(phot["ab_mag"]) & (phot["band"] == band)
    detected_pts = in_band & (phot["snr"] > SNR_THRESHOLD)
    upper_limits = in_band & (phot["snr"] <= SNR_THRESHOLD)

    if np.any(detected_pts):
        ax.errorbar(
            t_since_explosion[detected_pts].value,
            phot["ab_mag"][detected_pts],
            yerr=5 * phot["mag_err"][detected_pts],
            marker="s",
            mfc=color,
            mec="k",
            ecolor=color,
            linestyle="none",
            label=band,
        )
    if np.any(upper_limits):
        ax.errorbar(
            t_since_explosion[upper_limits].value,
            phot["ab_mag"][upper_limits],
            yerr=[
                phot["mag_upper"][upper_limits] - phot["ab_mag"][upper_limits],
                np.abs(phot["mag_lower"][upper_limits] - phot["ab_mag"][upper_limits]),
            ],
            marker="v",
            mfc="w",
            mec=color,
            ecolor=color,
            linestyle="none",
        )

ax.invert_yaxis()
ax.set_xlabel("Days since explosion")
ax.set_ylabel("AB magnitude")
ax.set_title(f"Event {event.event_id} (z={event.redshift:.3f}, {event.n_observations} observations)")
ax.legend()
fig.tight_layout()
plt.show()
