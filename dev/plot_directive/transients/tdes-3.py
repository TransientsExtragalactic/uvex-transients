import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from m4opt.missions import uvex
from uvex_transients.transients.TDEs import TidalDisruptionEvent


rng = np.random.default_rng(20260911)
n_samples = 3000

# Sample the TDE population.
tde = TidalDisruptionEvent()
redshift = tde.sample_event_redshift(n_samples, rng=rng)
params = tde.sed.sample_parameters(size=n_samples, rng=rng)

# Convert the integrated rate per steradian to an all-sky rate.
all_sky_rate = 4 * np.pi * tde.integrated_event_rate * u.sr

# Observed-frame time corresponding to the rest-frame peak.
t_peak_obs = 5 * params["sigma_rise"] * (1 + redshift)

detection_limits = {
    "FUV": 24.5 * u.ABmag,
    "NUV": 24.5 * u.ABmag,
}

# Compute peak-visible rates.
visible_rates = {}

for band_name, bandpass in uvex.detector.bandpasses.items():
    magnitudes = tde.sed.mag_bandpass(
        bandpass,
        t_peak_obs,
        redshift=redshift,
        **params,
    ).to_value(u.ABmag)

    visible = magnitudes < detection_limits[band_name].to_value(u.ABmag)
    visible_fraction = np.mean(visible)
    visible_rate = visible_fraction * all_sky_rate

    visible_rates[band_name] = visible_rate

    print(
        f"{band_name}: {visible_rate:.2f} "
        f"({visible_fraction:.1%} of events visible)"
    )

# Plot all-sky visible rates.
band_names = list(visible_rates)
rates = [visible_rates[band].to_value(1 / u.yr) for band in band_names]

fig, ax = plt.subplots(figsize=(5, 4))

ax.bar(band_names, rates)

ax.set_yscale("log")
ax.set_ylabel(r"All-sky rate [yr$^{-1}$]")
ax.set_title("Peak-visible TDE rate")

fig.tight_layout()
plt.show()