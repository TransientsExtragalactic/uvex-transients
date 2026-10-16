import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from uvex_transients.models.supernovae import TypeIaSED as SEDClass
from uvex_transients.utils.lightcurve_archive import LightcurveArchive

rng = np.random.default_rng(20260923)
n_samples = 300

params = SEDClass().sample_parameters(size=n_samples, rng=rng)
params_grid = {name: value[:, None] for name, value in params.items()}
archive = LightcurveArchive()

t = np.geomspace(0.5, 365, 400) * u.day
L_bol = SEDClass.eval_bolometric(t, **params_grid)
T = SEDClass.temperature(t, **params_grid)

fig, (ax_L, ax_T) = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True)

for row in range(n_samples):
    ax_L.plot(t.to_value(u.day), L_bol[row].to_value(u.erg / u.s), color="C0", lw=0.4, alpha=0.15)
    ax_T.plot(t.to_value(u.day), T[row].to_value(u.K), color="C3", lw=0.4, alpha=0.15)

sharon_events = archive.events("supernovae/Ia")
for i, name in enumerate(sharon_events):
    lbol_obs = archive.table("supernovae/Ia", name, "L_bol")
    ax_L.plot(
        lbol_obs["time"].to_value(u.day), lbol_obs["L_bol"].to_value(u.erg / u.s),
        color="k", lw=0.8, marker="o", ms=2.5, alpha=0.7,
        label="Sharon+25 (n=%d)" % len(sharon_events) if i == 0 else None,
    )

ax_L.set_xscale("log")
ax_L.set_yscale("log")
ax_L.set_ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
ax_L.set_title(f"Type Ia: simulated bolometric light curves (n={n_samples})")
ax_L.legend(loc="lower left", fontsize=8, frameon=False)

ax_T.set_xscale("log")
ax_T.set_yscale("log")
ax_T.set_xlabel("Time since explosion [days]")
ax_T.set_ylabel("Photospheric temperature [K]")
ax_T.set_title(f"Type Ia: simulated photospheric temperatures (n={n_samples})")

fig.tight_layout()