import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from uvex_transients.models.supernovae import TypeIIPExcessSED
from uvex_transients.utils.lightcurve_archive import LightcurveArchive

rng = np.random.default_rng(20260910)
n_samples = 300

params = TypeIIPExcessSED().sample_parameters(size=n_samples, rng=rng)
params_grid = {name: value[:, None] for name, value in params.items()}

t = np.geomspace(0.2, 200, 400) * u.day
L_bol = TypeIIPExcessSED.eval_bolometric(t, **params_grid)
T = TypeIIPExcessSED.temperature(t, **params_grid)

fig, (ax_L, ax_T) = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True)

for row in range(n_samples):
    ax_L.plot(t.to_value(u.day), L_bol[row].to_value(u.erg / u.s), color="C0", lw=0.4, alpha=0.2)
    ax_T.plot(t.to_value(u.day), T[row].to_value(u.K), color="C3", lw=0.4, alpha=0.2)

archive = LightcurveArchive()
observed_excess_sne = [
    ("2023ixf_hsu2025", "SN 2023ixf", "o", "k"),
    ("2024ggi_chen2024", "SN 2024ggi", "s", "firebrick"),
]

for suffix, label, marker, color in observed_excess_sne:
    lbol_obs = archive.table("supernovae/II", suffix, "L_bol")
    Tphot_obs = archive.table("supernovae/II", suffix, "T_phot")
    ax_L.scatter(
        lbol_obs["time"].to_value(u.day), lbol_obs["L_bol"].to_value(u.erg / u.s),
        marker=marker, s=28, color=color, edgecolor="white", linewidth=0.5, zorder=5,
        label=label,
    )
    ax_T.scatter(
        Tphot_obs["time"].to_value(u.day), Tphot_obs["T_phot"].to_value(u.K),
        marker=marker, s=28, color=color, edgecolor="white", linewidth=0.5, zorder=5,
        label=label,
    )

ax_L.set_xscale("log")
ax_L.set_yscale("log")
ax_L.set_ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
ax_L.set_title("Type IIP + Excess SNe: simulated bolometric light curves (n=300)")
ax_L.legend(loc="lower left", fontsize=8, frameon=False)
ax_L.set_ylim([1e40,None])
ax_T.set_yscale("log")
ax_T.set_xlabel("Time since explosion [days]")
ax_T.set_ylabel("Photospheric temperature [K]")
ax_T.set_ylim([1e3,None])

fig.tight_layout()