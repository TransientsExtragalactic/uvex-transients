import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from uvex_transients.transients.TDEs import TidalDisruptionEvent
from uvex_transients.models.lightcurves.generic import GREDLightcurve
from uvex_transients.utils.lightcurve_archive import LightcurveArchive

rng = np.random.default_rng(20260910)
n_samples = 1000

TDEs = TidalDisruptionEvent()
params = TDEs.sed.sample_parameters(size=n_samples, rng=rng)
params_grid = {name: value[:, None] for name, value in params.items()}

# Recenter each realization on its own peak time (t_peak = 5 sigma_rise) so the
# simulated curves line up with the peak-relative observed TDE data below.
t_rel = np.linspace(-30, 200, 400) * u.day
t_peak = 5 * params_grid["sigma_rise"]
t = t_rel + t_peak

L_bol = TDEs.sed.eval_bolometric(t, **params_grid)

archive = LightcurveArchive()
observed_tdes = [
    ("2018hyz_vanvelzen", "AT2018hyz (van Velzen+2021)", "o", "k"),
    ("2019qiz_vanvelzen", "AT2019qiz (van Velzen+2021)", "s", "firebrick"),
    ("2018lna_vanvelzen", "AT2018lna (van Velzen+2021)", "^", "darkorange"),
    ("2018iih_vanvelzen", "AT2018iih (van Velzen+2021)", "D", "seagreen"),
    ("2019mha_vanvelzen", "AT2019mha (van Velzen+2021)", "v", "mediumpurple"),
]

fig, ax_L = plt.subplots(figsize=(6.4, 4.8))

for row in range(n_samples):
    ax_L.plot(t_rel.to_value(u.day), L_bol[row].to_value(u.erg / u.s), color="C0", lw=0.4, alpha=0.06)

for suffix, label, marker, color in observed_tdes:
    lbol_obs = archive.table("tdes", suffix, "L_bol")
    ax_L.scatter(
        lbol_obs["time"].to_value(u.day), lbol_obs["L_bol"].to_value(u.erg / u.s),
        marker=marker, s=28, color=color, edgecolor="white", linewidth=0.5, zorder=5,
        label=label,
    )

ax_L.set_xlim(-30, 200)
ax_L.set_yscale("log")
ax_L.set_ylim(1e41, 1e45)
ax_L.set_xlabel("Time since peak [days]")
ax_L.set_ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
ax_L.set_title("Tidal disruption events: simulated bolometric light curves (n=1000)")
ax_L.legend(loc="upper right", fontsize=8, frameon=False)

fig.tight_layout()
plt.show()