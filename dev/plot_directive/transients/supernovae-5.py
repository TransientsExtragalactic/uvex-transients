import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from m4opt.missions import uvex
from uvex_transients.transients.supernovae import TypeIIPSNe, TypeIIPExcessSNe


rng = np.random.default_rng(20260911)
n_samples = 3000

detection_limits = {
    "FUV": 24.5 * u.ABmag,
    "NUV": 24.5 * u.ABmag,
}

populations = {
    "Type IIP": TypeIIPSNe(),
    "Type IIP + excess": TypeIIPExcessSNe(),
}

# Compute peak-visible rates for each population.
visible_rates = {band_name: [] for band_name in uvex.detector.bandpasses}
population_names = list(populations)

for name, sn in populations.items():
    redshift = sn.sample_event_redshift(n_samples, rng=rng)
    params = sn.sed.sample_parameters(size=n_samples, rng=rng)
    params_grid = {pname: value[:, None] for pname, value in params.items()}

    # Convert the integrated rate per steradian to an all-sky rate.
    all_sky_rate = 4 * np.pi * sn.integrated_event_rate * u.sr

    # Numerically search each event's own light curve for its brightest (peak)
    # apparent magnitude -- see the discussion above for why neither model's peak
    # sits at a single named parameter.
    t_grid_rest = np.geomspace(0.1, 200, 200) * u.day
    t_obs_grid = t_grid_rest[None, :] * (1.0 + redshift)[:, None]
    z_grid_bcast = np.broadcast_to(redshift[:, None], t_obs_grid.shape)

    for band_name, bandpass in uvex.detector.bandpasses.items():
        mag_curve = sn.sed.mag_bandpass(
            bandpass,
            t_obs_grid,
            redshift=z_grid_bcast,
            **params_grid,
        ).to_value(u.ABmag)
        magnitudes = np.nanmin(mag_curve, axis=1)

        visible = magnitudes < detection_limits[band_name].to_value(u.ABmag)
        visible_fraction = np.mean(visible)
        visible_rate = visible_fraction * all_sky_rate

        visible_rates[band_name].append(visible_rate.to_value(1 / u.yr))

        print(
            f"{name} {band_name}: {visible_rate:.2f} "
            f"({visible_fraction:.1%} of events visible)"
        )

# Plot all-sky visible rates, grouped by band.
band_names = list(visible_rates)
x = np.arrange(len(population_names))
width = 0.35

fig, ax = plt.subplots(figsize=(6, 4))

for offset, band_name in zip((-width / 2, width / 2), band_names):
    ax.bar(x + offset, visible_rates[band_name], width, label=band_name)

ax.set_xticks(x)
ax.set_xticklabels(population_names)
ax.set_yscale("log")
ax.set_ylabel(r"All-sky rate [yr$^{-1}$]")
ax.set_title("Peak-visible Type IIP SNe rate")
ax.legend(loc="upper right", fontsize=8, frameon=False)

fig.tight_layout()
plt.show()