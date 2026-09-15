from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.table import Table

import uvex_transients
from uvex_transients.models.lfbots import LFBOTCoolingBlackbodySED as SEDClass

rng = np.random.default_rng(20260910)
n_samples = 1000

params = SEDClass().sample_parameters(size=n_samples, rng=rng)
params_grid = {name: value[:, None] for name, value in params.items()}

t = np.geomspace(0.05, 100, 200) * u.day
L_bol = SEDClass.eval_bolometric(t, **params_grid)
T = SEDClass.temperature(t, **params_grid)

fig, (ax_L, ax_T) = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True)

for row in range(n_samples):
    ax_L.plot(t.to_value(u.day), L_bol[row].to_value(u.erg / u.s), color="C0", lw=0.4, alpha=0.06)
    ax_T.plot(t.to_value(u.day), T[row].to_value(u.K), color="C3", lw=0.4, alpha=0.06)

data_dir = Path(uvex_transients.__file__).parent.parent / "test_data" / "transients"
observed_lfbots = [
    ("2018cow_holu2026.txt", "AT2018cow (Ho & Lu+2026)", "o", "k"),
    ("css161010_holu2026.txt", "CSS161010 (Ho & Lu+2026)", "s", "firebrick"),
    ("2024wpp_holu2026.txt", "AT2024wpp (Ho & Lu+2026)", "^", "darkorange"),
    ("2024puz_holu2026.txt", "AT2024puz (Ho & Lu+2026)", "D", "seagreen"),
]

for suffix, label, marker, color in observed_lfbots:
    lbol_obs = Table.read(data_dir / f"lbol_{suffix}", format="ascii")
    Tphot_obs = Table.read(data_dir / f"Tphot_{suffix}", format="ascii")
    ax_L.scatter(
        lbol_obs["time"], lbol_obs["L_bol"],
        marker=marker, s=28, color=color, edgecolor="white", linewidth=0.5, zorder=5,
        label=label,
    )
    ax_T.scatter(
        Tphot_obs["time"], Tphot_obs["T_phot"],
        marker=marker, s=28, color=color, edgecolor="white", linewidth=0.5, zorder=5,
        label=label,
    )

ax_L.set_xscale("log")
ax_L.set_yscale("log")
ax_L.set_ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
ax_L.set_title("LFBOTs: simulated bolometric light curves (n=1000)")
ax_L.legend(loc="upper right", fontsize=8, frameon=False)

ax_T.set_yscale("log")
ax_T.set_xlabel("Time since explosion [days]")
ax_T.set_ylabel("Photospheric temperature [K]")

fig.tight_layout()