import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from uvex_transients.models.supernovae import TypeIcBLSED as SEDClass
from uvex_transients.utils.lightcurve_archive import LightcurveArchive

rng = np.random.default_rng(20260924)
n_samples = 300

params = SEDClass().sample_parameters(size=n_samples, rng=rng)
params_grid = {name: value[:, None] for name, value in params.items()}
archive = LightcurveArchive()

# Top panel: simulated light curves aligned on their own peak. Lyman+16 times are already relative to
# maximum light; the Srinivasaragavan+24 curves are shifted to their own peaks.
t_lin = np.linspace(0.05, 150, 3000) * u.day
L_lin = SEDClass.eval_bolometric(t_lin, **params_grid).to_value(u.erg / u.s)
t_peak = t_lin.to_value(u.day)[np.argmax(L_lin, axis=1)]

fig, (ax_P, ax_T) = plt.subplots(2, 1, figsize=(6.4, 7.2))

for row in range(n_samples):
    ax_P.plot(t_lin.to_value(u.day) - t_peak[row], L_lin[row], color="C0", lw=0.4, alpha=0.15)

srini_events = [name for name in archive.events("supernovae/Ic-BL") if name.endswith("srinivasaragavan2024")]
for i, name in enumerate(srini_events):
    lbol_obs = archive.table("supernovae/Ic-BL", name, "L_bol")
    t_obs = lbol_obs["time"].to_value(u.day)
    L_obs = lbol_obs["L_bol"].to_value(u.erg / u.s)
    ax_P.plot(
        t_obs - t_obs[np.argmax(L_obs)], L_obs, color="0.5", lw=0.8, alpha=0.5,
        label="Srinivasaragavan+24, own peak (n=%d)" % len(srini_events) if i == 0 else None,
    )
lyman_events = [name for name in archive.events("supernovae/Ic-BL") if name.endswith("lyman2016")]
for i, name in enumerate(lyman_events):
    lbol_obs = archive.table("supernovae/Ic-BL", name, "L_bol")
    ax_P.plot(
        lbol_obs["time"].to_value(u.day), lbol_obs["L_bol"].to_value(u.erg / u.s),
        color="k", lw=0.8, marker="o", ms=2.5, alpha=0.7,
        label="Lyman+16 (n=%d)" % len(lyman_events) if i == 0 else None,
    )
ax_P.set_yscale("log")
ax_P.set_xlim([-25, 80])
ax_P.set_ylim([1e40, 10**44.2])
ax_P.set_xlabel("Time since peak [days]")
ax_P.set_ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
ax_P.set_title(f"Type Ic-BL: simulated bolometric light curves (n={n_samples})")
ax_P.legend(loc="lower right", fontsize=8, frameon=False)

# Bottom panel: photospheric temperature against time since explosion.
t = np.geomspace(0.1, 1000, 400) * u.day
T = SEDClass.temperature(t, **params_grid)
for row in range(n_samples):
    ax_T.plot(t.to_value(u.day), T[row].to_value(u.K), color="C3", lw=0.4, alpha=0.15)

ax_T.set_xscale("log")
ax_T.set_yscale("log")
ax_T.set_xlim(1e-1, 1e3)
ax_T.set_xlabel("Time since explosion [days]")
ax_T.set_ylabel("Photospheric temperature [K]")
ax_T.set_title(f"Type Ic-BL: simulated photospheric temperatures (n={n_samples})")

fig.tight_layout()