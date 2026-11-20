import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from uvex_transients.missions import uvex_fast as uvex
from uvex_transients.transients.llgrb import LowLuminosityGRB
from uvex_transients.utils.plotting import add_funnel_legend, get_band_color, plot_rate_bars

rng = np.random.default_rng(20261022)
n_samples = 2000

# Sample the llGRB population.
llgrb = LowLuminosityGRB()
redshift = llgrb.sample_event_redshift(n_samples, rng=rng)
params = llgrb.sed.sample_parameters(size=n_samples, rng=rng)

# The all-sky rate, with no survey footprint applied.
all_sky_rate = llgrb.all_sky_rate

# Observer-frame epochs at which each event's brightness is evaluated, to find its peak.
t_obs = np.geomspace(0.03, 30, 50) * u.day

detection_limits = {
    "FUV": 24.5 * u.ABmag,
    "NUV": 24.5 * u.ABmag,
}

# Compute the number of the n_samples draws visible in each band at their brightest epoch.
visible_counts = {}

for band_name, bandpass in uvex.detector.bandpasses.items():
    magnitudes = llgrb.sed.mag_bandpass(
        bandpass,
        t_obs[None, :],
        redshift=redshift[:, None],
        **{name: value[:, None] for name, value in params.items()},
    ).to_value(u.ABmag)

    visible = magnitudes.min(axis=1) < detection_limits[band_name].to_value(u.ABmag)
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
    rate_ci=llgrb.RATE_CI,
    color=[get_band_color(band) for band in band_names],
)

ax.set_yscale("log")
ax.set_ylabel(r"All-sky rate [yr$^{-1}$]")
ax.set_title("Peak-visible llGRB rate")
add_funnel_legend(ax, loc="lower right")

fig.tight_layout()
plt.show()