from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.table import Table

import uvex_transients
from uvex_transients.models.kilonovae import KilonovaCoolingBlackbodySED as SEDClass

rng = np.random.default_rng(20260910)
n_samples = 1000

params = SEDClass().sample_parameters(size=n_samples, rng=rng)
params_grid = {name: value[:, None] for name, value in params.items()}

t = np.geomspace(0.02, 30, 200) * u.day
L_bol = SEDClass.eval_bolometric(t, **params_grid)
T = SEDClass.temperature(t, **params_grid)

data_dir = Path(uvex_transients.__file__).parent.parent / "test_data" / "transients"
lbol_cowperthwaite = Table.read(data_dir / "lbol_gw170817_cowperthwaite.txt", format="ascii")
lbol_waxman = Table.read(data_dir / "lbol_gw170817_waxman.txt", format="ascii")
tphot_waxman = Table.read(data_dir / "Tphot_gw170817_waxman.txt", format="ascii")

fig, (ax_L, ax_T) = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True)

for row in range(n_samples):
    ax_L.plot(t.to_value(u.day), L_bol[row].to_value(u.erg / u.s), color="C0", lw=0.4, alpha=0.06)
    ax_T.plot(t.to_value(u.day), T[row].to_value(u.K), color="C3", lw=0.4, alpha=0.06)

ax_L.scatter(
    lbol_cowperthwaite["time"], lbol_cowperthwaite["L_bol"],
    marker="o", s=28, color="k", edgecolor="white", linewidth=0.5, zorder=5,
    label="Cowperthwaite+2017",
)
ax_L.scatter(
    lbol_waxman["time"], lbol_waxman["L_bol"],
    marker="s", s=28, color="firebrick", edgecolor="white", linewidth=0.5, zorder=5,
    label="Waxman+2018",
)

ax_T.scatter(
    tphot_waxman["time"], tphot_waxman["T_phot"],
    marker="s", s=28, color="firebrick", edgecolor="white", linewidth=0.5, zorder=5,
    label="Waxman+2018",
)

ax_L.set_xscale("log")
ax_L.set_yscale("log")
ax_L.set_ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
ax_L.set_title("Kilonova: simulated bolometric light curves (n=1000)")
ax_L.legend(loc="upper right", fontsize=8, frameon=False)

ax_T.set_yscale("log")
ax_T.set_xlabel("Time since merger [days]")
ax_T.set_ylabel("Photospheric temperature [K]")
ax_T.legend(loc="upper right", fontsize=8, frameon=False)

fig.tight_layout()