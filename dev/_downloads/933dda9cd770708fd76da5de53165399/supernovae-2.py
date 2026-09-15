from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.table import Table

import uvex_transients
from uvex_transients.models.supernovae import TypeIIPExcessSED as SEDClass

rng = np.random.default_rng(20260910)
n_samples = 1000

params = SEDClass().sample_parameters(size=n_samples, rng=rng)
params_grid = {name: value[:, None] for name, value in params.items()}

t = np.geomspace(0.1, 200, 200) * u.day
L_bol = SEDClass.eval_bolometric(t, **params_grid)
T = SEDClass.temperature(t, **params_grid)

fig, (ax_L, ax_T) = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True)

for row in range(n_samples):
    ax_L.plot(t.to_value(u.day), L_bol[row].to_value(u.erg / u.s), color="C0", lw=0.4, alpha=0.06)
    ax_T.plot(t.to_value(u.day), T[row].to_value(u.K), color="C3", lw=0.4, alpha=0.06)

data_dir = Path(uvex_transients.__file__).parent.parent / "test_data" / "transients"
observed_excess_sne = [
    ("2023ixf_hsu2025.txt", "SN 2023ixf (Hsu+2025)", "o", "k"),
    ("2024ggi_chen2024.txt", "SN 2024ggi (Chen+2024)", "s", "firebrick"),
]

for suffix, label, marker, color in observed_excess_sne:
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
ax_L.set_title("Type IIP + early excess SNe: simulated bolometric light curves (n=1000)")
ax_L.legend(loc="upper right", fontsize=8, frameon=False)

ax_T.set_yscale("log")
ax_T.set_xlabel("Time since explosion [days]")
ax_T.set_ylabel("Photospheric temperature [K]")

fig.tight_layout()