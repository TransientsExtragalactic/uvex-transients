import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from m4opt.missions import uvex
from uvex_transients.transients.supernovae import MagnetarSLSNe
from uvex_transients.utils.plotting import add_funnel_legend, get_band_color, plot_rate_bars

rng = np.random.default_rng(20260921)
n_samples = 2000

slsne = MagnetarSLSNe()
redshift = slsne.sample_event_redshift(n_samples, rng=rng)
params = slsne.sed.sample_parameters(size=n_samples, rng=rng)
params_grid = {name: value[:, None] for name, value in params.items()}

# The all-sky rate, with no survey footprint applied.
all_sky_rate = slsne.all_sky_rate

t_grid_rest = np.geomspace(1, 600, 250) * u.day
t_obs_grid = t_grid_rest[None, :] * (1.0 + redshift)[:, None]
z_grid_bcast = np.broadcast_to(redshift[:, None], t_obs_grid.shape)

visible_counts = {}
for band_name, bandpass in uvex.detector.bandpasses.items():
    mag_curve = slsne.sed.mag_bandpass(
        bandpass, t_obs_grid, redshift=z_grid_bcast, **params_grid
    ).to_value(u.ABmag)
    magnitudes = np.nanmin(mag_curve, axis=1)

    visible = magnitudes < 24.5
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
    rate_ci=slsne.RATE_CI,
    color=[get_band_color(band) for band in band_names],
)
ax.set_yscale("log")
ax.set_ylabel(r"All-sky rate [yr$^{-1}$]")
ax.set_title("Peak-visible SLSN-I rate")
add_funnel_legend(ax, loc="lower right")

fig.tight_layout()