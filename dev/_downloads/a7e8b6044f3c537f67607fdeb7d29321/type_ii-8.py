import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from m4opt.missions import uvex
from uvex_transients.transients.supernovae import TypeIIbSNe
from uvex_transients.models.supernovae import TypeIIbSED

rng = np.random.default_rng(20260918)
n_samples = 1000

sn = TypeIIbSNe()
z = sn.sample_event_redshift(n_samples, rng=rng)
params = TypeIIbSED().sample_parameters(size=n_samples, rng=rng)
params_grid = {name: value[:, None] for name, value in params.items()}

# Numerically search each event's own light curve for its brightest (peak) apparent
# magnitude, since neither the early nor the main peak sits at a single named parameter.
t_grid_rest = np.geomspace(0.1, 200, 300) * u.day
t_obs_grid = t_grid_rest[None, :] * (1.0 + z)[:, None]
z_grid_bcast = np.broadcast_to(z[:, None], t_obs_grid.shape)

bandpasses = uvex.detector.bandpasses
band_names = list(bandpasses)

fig, axes = plt.subplots(1, len(band_names), figsize=(10.5, 4.8), sharey=True)

for ax, band_name in zip(axes, band_names):
    mag_curve = TypeIIbSED.mag_bandpass(
        bandpasses[band_name], t_obs_grid, redshift=z_grid_bcast, **params_grid
    ).to_value(u.ABmag)
    mag = np.nanmin(mag_curve, axis=1)
    finite = np.isfinite(mag)

    ax.scatter(z[finite], mag[finite], s=5, ec="k", fc="k", alpha=0.5, label="Simulated events")
    ax.axhline(24.5, color="firebrick", ls="--", lw=1.2, label="UVEX limit (1 Dwell)")

    ax.invert_yaxis()
    ax.set_xlabel("Redshift")
    ax.set_title(f"UVEX {band_name}")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_ylim([35, 15])

axes[0].set_ylabel("Peak apparent AB magnitude")
fig.suptitle(f"Type IIb: peak apparent magnitude vs. redshift (n={n_samples})")
fig.tight_layout()