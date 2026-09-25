import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from scipy.stats import gaussian_kde

from m4opt.missions import uvex
from uvex_transients.transients.LFBOTs import LuminousFastBlueOpticalTransient
from uvex_transients.models.lfbots import LFBOTCoolingBlackbodySED as SEDClass

rng = np.random.default_rng(20260911)
n_samples = 3000

lfbot = LuminousFastBlueOpticalTransient()
z = lfbot.sample_event_redshift(n_samples, rng=rng)
params = SEDClass().sample_parameters(size=n_samples, rng=rng)

# Observed-frame time of rest-frame peak, i.e. where each event is brightest as seen by UVEX.
t_obs_peak = params["t_peak"] * (1.0 + z)

bandpasses = uvex.detector.bandpasses
band_names = list(bandpasses)

fig, axes = plt.subplots(1, len(band_names), figsize=(10.5, 4.8), sharey=True)

for ax, band_name in zip(axes, band_names):
    mag = SEDClass.mag_bandpass(bandpasses[band_name], t_obs_peak, redshift=z, **params).to_value(u.ABmag)
    finite = np.isfinite(mag)
    z_finite, mag_finite = z[finite], mag[finite]

    ax.scatter(z_finite, mag_finite, s=5, ec='k',fc='k',alpha=0.5, label="Simulated events")

    kde = gaussian_kde(np.vstack([z_finite, mag_finite]))
    z_grid = np.linspace(z_finite.min(), z_finite.max(), 150)
    mag_grid = np.linspace(mag_finite.min(), mag_finite.max(), 150)
    Z_grid, Mag_grid = np.meshgrid(z_grid, mag_grid)
    density = kde(np.vstack([Z_grid.ravel(), Mag_grid.ravel()])).reshape(Z_grid.shape)
    ax.contour(Z_grid, Mag_grid, density, levels=6, colors="k", linewidths=0.7)

    ax.axhline(24.5, color="firebrick", ls="--", lw=1.2, label="UVEX limit (1 Dwell)")

    ax.invert_yaxis()
    ax.set_xlabel("Redshift")
    ax.set_title(f"UVEX {band_name}")
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    ax.invert_yaxis()
    ax.set_ylim([50, 15])

axes[0].set_ylabel("Peak apparent AB magnitude")
fig.suptitle(f"LFBOTs: peak apparent magnitude vs. redshift (n={n_samples})")
fig.tight_layout()
plt.show()