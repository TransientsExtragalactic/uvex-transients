import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from scipy.stats import gaussian_kde

from m4opt.missions import uvex
from uvex_transients.transients.supernovae import MagnetarSLSNe

rng = np.random.default_rng(20260921)
n_samples = 2000

slsne = MagnetarSLSNe()
z = slsne.sample_event_redshift(n_samples, rng=rng)
params = slsne.sed.sample_parameters(size=n_samples, rng=rng)
params_grid = {name: value[:, None] for name, value in params.items()}

t_grid_rest = np.geomspace(1, 600, 250) * u.day
t_obs_grid = t_grid_rest[None, :] * (1.0 + z)[:, None]
z_grid_bcast = np.broadcast_to(z[:, None], t_obs_grid.shape)

bandpasses = uvex.detector.bandpasses
band_names = list(bandpasses)

fig, axes = plt.subplots(1, len(band_names), figsize=(10.5, 4.8), sharey=True)

for ax, band_name in zip(axes, band_names):
    mag_curve = slsne.sed.mag_bandpass(
        bandpasses[band_name], t_obs_grid, redshift=z_grid_bcast, **params_grid
    ).to_value(u.ABmag)
    mag = np.nanmin(mag_curve, axis=1)
    finite = np.isfinite(mag)
    z_finite, mag_finite = z[finite], mag[finite]

    ax.scatter(z_finite, mag_finite, s=5, ec="k", fc="k", alpha=0.4, label="Simulated events")

    kde = gaussian_kde(np.vstack([z_finite, mag_finite]))
    z_kde = np.linspace(z_finite.min(), z_finite.max(), 150)
    mag_kde = np.linspace(mag_finite.min(), mag_finite.max(), 150)
    Z_grid, Mag_grid = np.meshgrid(z_kde, mag_kde)
    density = kde(np.vstack([Z_grid.ravel(), Mag_grid.ravel()])).reshape(Z_grid.shape)
    ax.contour(Z_grid, Mag_grid, density, levels=6, colors="k", linewidths=0.7)

    ax.axhline(24.5, color="firebrick", ls="--", lw=1.2, label="UVEX limit (1 Dwell)")

    ax.invert_yaxis()
    ax.set_xlabel("Redshift")
    ax.set_title(f"UVEX {band_name}")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_ylim([32, 16])

axes[0].set_ylabel("Peak apparent AB magnitude")
fig.suptitle(f"SLSN-I: peak apparent magnitude vs. redshift (n={n_samples})")
fig.tight_layout()