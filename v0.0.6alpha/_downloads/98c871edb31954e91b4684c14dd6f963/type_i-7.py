import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from uvex_transients.models.supernovae import TypeIcSED as SEDClass
from uvex_transients.utils.lightcurve_archive import LightcurveArchive

rng = np.random.default_rng(20260918)
n_samples = 300

params = SEDClass().sample_parameters(size=n_samples, rng=rng)
params_grid = {name: value[:, None] for name, value in params.items()}
archive = LightcurveArchive()

fig, (ax_P, ax_T) = plt.subplots(2, 1, figsize=(6.4, 7.2))

# Top panel: simulated light curves aligned on their own peak, against Lyman+16
# (whose times are already relative to maximum light).
t_lin = np.linspace(0.05, 100, 2000) * u.day
L_lin = SEDClass.eval_bolometric(t_lin, **params_grid).to_value(u.erg / u.s)
t_peak = t_lin.to_value(u.day)[np.argmax(L_lin, axis=1)]
for row in range(n_samples):
    ax_P.plot(t_lin.to_value(u.day) - t_peak[row], L_lin[row], color="C0", lw=0.4, alpha=0.15)

lyman_events = [name for name in archive.events("supernovae/Ic") if name.endswith("lyman2016")]
for i, name in enumerate(lyman_events):
    lbol_obs = archive.table("supernovae/Ic", name, "L_bol")
    ax_P.plot(
        lbol_obs["time"].to_value(u.day), lbol_obs["L_bol"].to_value(u.erg / u.s),
        color="k", lw=0.8, marker="o", ms=2.5, alpha=0.7,
        label="Lyman+16 (n=%d)" % len(lyman_events) if i == 0 else None,
    )
ax_P.set_yscale("log")
ax_P.set_xlim([-25, 70])
ax_P.set_ylim([1e40, 10**43.5])
ax_P.set_xlabel("Time since peak [days]")
ax_P.set_ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
ax_P.set_title("Type Ic: simulated bolometric light curves (n=300)")
ax_P.legend(loc="lower right", fontsize=8, frameon=False)

# Bottom panel: photospheric temperature against time since explosion.
t = np.geomspace(0.5, 100, 400) * u.day
T = SEDClass.temperature(t, **params_grid)
for row in range(n_samples):
    ax_T.plot(t.to_value(u.day), T[row].to_value(u.K), color="C3", lw=0.4, alpha=0.15)

# (archive key, label, marker, colour) from Prentice+19.
observed_sne = [
    ("2013F_prentice2019", "SN 2013F", "o", "k"),
    ("2016P_prentice2019", "SN 2016P", "s", "firebrick"),
    ("2016iae_prentice2019", "SN 2016iae", "^", "C2"),
    ("2017dcc_prentice2019", "SN 2017dcc", "X", "C7"),
    ("2017ifh_prentice2019", "SN 2017ifh", "*", "C8"),
    ("2018ie_prentice2019", "SN 2018ie", "D", "C4"),
]
for suffix, label, marker, color in observed_sne:
    Tphot_obs = archive.table("supernovae/Ic", suffix, "T_phot")
    ax_T.scatter(
        Tphot_obs["time"].to_value(u.day), Tphot_obs["T_phot"].to_value(u.K),
        marker=marker, s=28, color=color, edgecolor="white", linewidth=0.5, zorder=5,
        label=label,
    )
ax_T.set_xscale("log")
ax_T.set_yscale("log")
ax_T.set_xlabel("Time since explosion [days]")
ax_T.set_ylabel("Photospheric temperature [K]")
ax_T.set_ylim([3e3, 5e4])
ax_T.legend(loc="upper right", fontsize=7, frameon=False)

fig.tight_layout()