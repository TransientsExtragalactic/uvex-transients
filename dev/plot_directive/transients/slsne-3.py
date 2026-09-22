import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from m4opt.missions import uvex
from uvex_transients.transients.supernovae import MagnetarSLSNe

rng = np.random.default_rng(20260921)
n_samples = 2000

slsne = MagnetarSLSNe()
redshift = slsne.sample_event_redshift(n_samples, rng=rng)
params = slsne.sed.sample_parameters(size=n_samples, rng=rng)
params_grid = {name: value[:, None] for name, value in params.items()}

# Convert the integrated rate per steradian to an all-sky rate.
all_sky_rate = 4 * np.pi * slsne.integrated_event_rate * u.sr

t_grid_rest = np.geomspace(1, 600, 250) * u.day
t_obs_grid = t_grid_rest[None, :] * (1.0 + redshift)[:, None]
z_grid_bcast = np.broadcast_to(redshift[:, None], t_obs_grid.shape)

visible_rates = {}
for band_name, bandpass in uvex.detector.bandpasses.items():
    mag_curve = slsne.sed.mag_bandpass(
        bandpass, t_obs_grid, redshift=z_grid_bcast, **params_grid
    ).to_value(u.ABmag)
    magnitudes = np.nanmin(mag_curve, axis=1)

    visible = magnitudes < 24.5
    visible_fraction = np.mean(visible)
    visible_rate = visible_fraction * all_sky_rate

    visible_rates[band_name] = visible_rate.to_value(1 / u.yr)

    print(
        f"{band_name}: {visible_rate:.2f} "
        f"({visible_fraction:.1%} of events visible)"
    )

fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(list(visible_rates), list(visible_rates.values()), color=["C0", "C1"])
ax.set_yscale("log")
ax.set_ylabel(r"All-sky rate [yr$^{-1}$]")
ax.set_title("Peak-visible SLSN-I rate")

fig.tight_layout()