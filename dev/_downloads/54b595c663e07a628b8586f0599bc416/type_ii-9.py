import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from uvex_transients.models.supernovae import TypeIIbSED as SEDClass
from uvex_transients.utils.lightcurve_archive import LightcurveArchive

rng = np.random.default_rng(20260918)
n_samples = 300

params = SEDClass().sample_parameters(size=n_samples, rng=rng)
params_grid = {name: value[:, None] for name, value in params.items()}

t = np.geomspace(0.1, 200, 400) * u.day
L_bol = SEDClass.eval_bolometric(t, **params_grid)
T = SEDClass.temperature(t, **params_grid)

fig, (ax_L, ax_T) = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True)

for row in range(n_samples):
    ax_L.plot(t.to_value(u.day), L_bol[row].to_value(u.erg / u.s), color="C0", lw=0.4, alpha=0.15)
    ax_T.plot(t.to_value(u.day), T[row].to_value(u.K), color="C3", lw=0.4, alpha=0.15)

archive = LightcurveArchive()
observed_iib_sne = [
    ("2011fu_moralesgaroffolo2015", "SN 2011fu", "o", "k"),
    ("2013df_moralesgaroffolo2014", "SN 2013df", "s", "firebrick"),
]
# Temperature-only comparison objects (photospheric temperatures from Prentice+19, shifted to
# time since explosion using the tabulated peak time; no bolometric light curve available).
tphot_only_iib_sne = [
    ("2013bb_prentice2019", "SN 2013bb", "P", "C1"),
    ("2016gkg_prentice2019", "SN 2016gkg", "h", "C5"),
    ("2017ixz_prentice2019", "SN 2017ixz", "<", "C6"),
]
# Light-curve-only comparison objects (no photospheric temperature sequence available).
lbol_only_iib_sne = [
    ("1993J_richmond1994", "SN 1993J", "^", "C2"),
    ("2008ax_taubenberger2011", "SN 2008ax", "X", "C7"),
    ("2024iss_yamanaka2025", "SN 2024iss", "*", "C8"),
]

for suffix, label, marker, color in observed_iib_sne:
    lbol_obs = archive.table("supernovae/IIb", suffix, "L_bol")
    Tphot_obs = archive.table("supernovae/IIb", suffix, "T_phot")
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

for suffix, label, marker, color in tphot_only_iib_sne:
    Tphot_obs = archive.table("supernovae/IIb", suffix, "T_phot")
    ax_T.scatter(
        Tphot_obs["time"].to_value(u.day), Tphot_obs["T_phot"].to_value(u.K),
        marker=marker, s=28, color=color, edgecolor="white", linewidth=0.5, zorder=5,
        label=label,
    )

for suffix, label, marker, color in lbol_only_iib_sne:
    lbol_obs = archive.table("supernovae/IIb", suffix, "L_bol")
    ax_L.scatter(
        lbol_obs["time"].to_value(u.day), lbol_obs["L_bol"].to_value(u.erg / u.s),
        marker=marker, s=28, color=color, edgecolor="white", linewidth=0.5, zorder=5,
        label=label,
    )

ax_L.set_xscale("log")
ax_L.set_yscale("log")
ax_L.set_ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
ax_L.set_title("Type IIb: simulated bolometric light curves (n=300)")
ax_L.legend(loc="lower left", fontsize=8, frameon=False)
ax_L.set_ylim([1e39, None])

ax_T.set_yscale("log")
ax_T.set_xlabel("Time since explosion [days]")
ax_T.set_ylabel("Photospheric temperature [K]")
ax_T.set_ylim([1e3, None])

fig.tight_layout()