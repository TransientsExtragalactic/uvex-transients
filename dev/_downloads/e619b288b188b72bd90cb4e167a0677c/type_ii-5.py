import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from scipy.stats import gaussian_kde

from m4opt.missions import uvex
from uvex_transients.transients.supernovae import TypeIIPExcessSNe
from uvex_transients.models.supernovae import TypeIIPExcessSED

rng = np.random.default_rng(20260911)
n_samples = 3000

sn = TypeIIPExcessSNe()
z = sn.sample_event_redshift(n_samples, rng=rng)
params = TypeIIPExcessSED().sample_parameters(size=n_samples, rng=rng)
params_grid = {name: value[:, None] for name, value in params.items()}

# Numerically search each event's own light curve for its brightest (peak) apparent
# magnitude, rather than assuming a single named parameter marks the true peak.
t_grid_rest = np.geomspace(0.2, 200, 200) * u.day
t_obs_grid = t_grid_rest[None, :] * (1.0 + z)[:, None]
z_grid_bcast = np.broadcast_to(z[:, None], t_obs_grid.shape)

bandpasses = uvex.detector.bandpasses
band_names = list(bandpasses)

fig, axes = plt.subplots(1, len(band_names), figsize=(10.5, 4.8), sharey=True)

for ax, band_name in zip(axes, band_names):
    mag_curve = TypeIIPExcessSED.mag_bandpass(
        bandpasses[band_name], t_obs_grid, redshift=z_grid_bcast, **params_grid
    ).to_value(u.ABmag)
    mag = np.nanmin(mag_curve, axis=1)
    finite = np.isfinite(mag)
    z_finite, mag_finite = z[finite], mag[finite]

    ax.scatter(z_finite, mag_finite, s=5, ec='k', fc='k', alpha=0.5, label="Simulated events")

    kde = gaussian_kde(np.vstack([z_finite, mag_finite]))
    z_kde_grid = np.linspace(z_finite.min(), z_finite.max(), 150)
    mag_kde_grid = np.linspace(mag_finite.min(), mag_finite.max(), 150)
    Z_grid, Mag_grid = np.meshgrid(z_kde_grid, mag_kde_grid)
    density = kde(np.vstack([Z_grid.ravel(), Mag_grid.ravel()])).reshape(Z_grid.shape)
    ax.contour(Z_grid, Mag_grid, density, levels=6, colors="k", linewidths=0.7)

    ax.axhline(24.5, color="firebrick", ls="--", lw=1.2, label="UVEX limit (1 Dwell)")

    ax.invert_yaxis()
    ax.set_xlabel("Redshift")
    ax.set_title(f"UVEX {band_name}")
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    ax.invert_yaxis()
    ax.set_ylim([35, 15])

axes[0].set_ylabel("Peak apparent AB magnitude")
fig.suptitle(f"Type IIP + Excess SNe: peak apparent magnitude vs. redshift (n={n_samples})")
fig.tight_layout()
plt.show()