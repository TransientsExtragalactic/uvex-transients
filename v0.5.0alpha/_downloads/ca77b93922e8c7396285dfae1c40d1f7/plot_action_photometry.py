"""
Action: Synthetic Photometry
=================================

:meth:`~uvex_transients.simulation.core.SurveySimulator.run_photometry_action` is a thin
wrapper over `~uvex_transients.simulation.event_catalog.EventCatalog.simulate_photometry`,
supplying the simulator's own `transient_collection`/`survey_schedule`. It runs real
synthetic photometry, band by band and observation by observation, for every event in a
catalog, and is normally run only after the cuts have already narrowed things down to
the events worth the cost.
"""

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

from uvex_transients.missions import uvex_fast as uvex
from uvex_transients.simulation.core import SurveySimulator
from uvex_transients.surveys import get_schedule
from uvex_transients.transients.supernovae import MagnetarSLSNe
from uvex_transients.utils.plotting import get_band_color, plot_band_light_curve, resolve_fig_axes, set_plot_style

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
detected = simulator.filter_by_snr(mag_filtered, uvex, snr_threshold=SNR_THRESHOLD)
print(f"{len(detected)} events detected (before scaling by downsample).")

# %%
# Running photometry on a handful of events
# -------------------------------------------------
#
# The action itself takes the catalog straight from a cut and a mission; only the events
# actually present in `catalog` are simulated.

rng = np.random.default_rng(0)
sample_ids = [int(eid) for eid in rng.choice(detected.event_id, size=min(5, len(detected)), replace=False)]
sample = simulator.filter_by_query(detected, uvex, expr=f"np.isin(event_id, {sample_ids})")

photometry = simulator.run_photometry_action(catalog=sample, mission=uvex)
print(f"{len(photometry)} (event, observation, band) rows for {len(sample)} events.")
print(photometry["event_id", "obs_time", "band", "snr", "ab_mag"][:6])

# %%
# One event's light curve
# -----------------------------

event_id = int(sample_ids[0])
event = detected.get_events(event_id, {"slsn": slsn}, schedule)
phot = photometry[photometry["event_id"] == event_id]
t_since_explosion = (phot["obs_time"] - event.t_explosion).to(u.day)
t_theory = np.linspace(0, slsn.duration_limit.to_value(u.day), 300) * u.day

fig, ax = resolve_fig_axes(fig_size=(7, 4))
for band in ("FUV", "NUV"):
    plot_band_light_curve(
        ax,
        band,
        t_since_explosion,
        phot,
        t_theory=t_theory,
        theory_mag=event.mag(t_theory, uvex, band=band),
        snr_threshold=SNR_THRESHOLD,
        color=get_band_color(band),
        err_scale=5.0,
        label=band,
    )

ax.invert_yaxis()
ax.set_xlabel("Days since explosion")
ax.set_ylabel("AB magnitude")
ax.set_title(f"Event {event.event_id} (z={event.redshift:.3f})")
ax.legend()
fig.tight_layout()
plt.show()
