import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from m4opt.missions import uvex
from uvex_transients.transients.kilonovae import Kilonova
from uvex_transients.utils.plotting import add_funnel_legend, get_band_color, plot_rate_bars


rng = np.random.default_rng(20260911)
n_samples = 3000

# Sample the kilonova population.
kilonova = Kilonova()
redshift = kilonova.sample_event_redshift(n_samples, rng=rng)
params = kilonova.sed.sample_parameters(size=n_samples, rng=rng)

# The all-sky rate, with no survey footprint applied.
all_sky_rate = kilonova.all_sky_rate

# Observed-frame time corresponding to the rest-frame peak.
t_peak_obs = params["t_peak"] * (1 + redshift)

detection_limits = {
    "FUV": 24.5 * u.ABmag,
    "NUV": 24.5 * u.ABmag,
}

# Compute the number of the n_samples draws visible in each band.
visible_counts = {}

for band_name, bandpass in uvex.detector.bandpasses.items():
    magnitudes = kilonova.sed.mag_bandpass(
        bandpass,
        t_peak_obs,
        redshift=redshift,
        **params,
    ).to_value(u.ABmag)

    visible = magnitudes < detection_limits[band_name].to_value(u.ABmag)
    visible_counts[band_name] = int(np.count_nonzero(visible))

    print(
        f"{band_name}: {visible_counts[band_name] / n_samples * all_sky_rate:.2f} "
        f"({visible_counts[band_name] / n_samples:.1%} of events visible)"
    )

# Plot all-sky visible rates, with MC (statistical) and rate (systematic) uncertainty --
# see uvex_transients.utils.plotting.plot_rate_bars.
band_names = list(visible_counts)

fig, ax = plt.subplots(figsize=(5, 4))

plot_rate_bars(
    ax,
    band_names,
    [visible_counts[band] for band in band_names],
    n_samples,
    all_sky_rate,
    rate_ci=kilonova.RATE_CI,
    color=[get_band_color(band) for band in band_names],
)

ax.set_yscale("log")
ax.set_ylabel(r"All-sky rate [yr$^{-1}$]")
ax.set_title("Peak-visible kilonova rate")
add_funnel_legend(ax, loc="lower right")

fig.tight_layout()
plt.show()