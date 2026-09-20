import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from m4opt.missions import uvex
from uvex_transients.transients.supernovae import TypeIIPExcessSNe

rng = np.random.default_rng(20260911)
n_samples = 3000

sn = TypeIIPExcessSNe()
redshift = sn.sample_event_redshift(n_samples, rng=rng)
params = sn.sed.sample_parameters(size=n_samples, rng=rng)
params_grid = {pname: value[:, None] for pname, value in params.items()}

# Convert the integrated rate per steradian to an all-sky rate.
all_sky_rate = 4 * np.pi * sn.integrated_event_rate * u.sr

# Numerically search each event's own light curve for its brightest (peak)
# apparent magnitude -- see the discussion above for why the peak
# doesn't sit at a single named parameter.
t_grid_rest = np.geomspace(0.2, 200, 200) * u.day
t_obs_grid = t_grid_rest[None, :] * (1.0 + redshift)[:, None]
z_grid_bcast = np.broadcast_to(redshift[:, None], t_obs_grid.shape)

visible_rates = {}
for band_name, bandpass in uvex.detector.bandpasses.items():
    mag_curve = sn.sed.mag_bandpass(
        bandpass,
        t_obs_grid,
        redshift=z_grid_bcast,
        **params_grid,
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

# Plot all-sky visible rates, by band.
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(list(visible_rates), list(visible_rates.values()), color=["C0", "C1"])
ax.set_yscale("log")
ax.set_ylabel(r"All-sky rate [yr$^{-1}$]")
ax.set_title("Peak-visible Type IIP + Excess SNe rate")

fig.tight_layout()
plt.show()