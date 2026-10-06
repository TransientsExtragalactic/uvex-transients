import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from uvex_transients.utils.lightcurve_archive import LightcurveArchive

from uvex_transients.models.supernovae import ArnettMagnetarSpindownSED as SEDClass

rng = np.random.default_rng(20260921)
n_samples = 300

params = SEDClass().sample_parameters(size=n_samples, rng=rng)
params_grid = {name: value[:, None] for name, value in params.items()}

t = np.geomspace(1, 1500, 400) * u.day
L_bol = SEDClass.eval_bolometric(t, **params_grid)
T = SEDClass.temperature(t, **params_grid)

fig, (ax_L, ax_T) = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True)

for row in range(n_samples):
    ax_L.plot(t.to_value(u.day), L_bol[row].to_value(u.erg / u.s), color="C0", lw=0.4, alpha=0.15)
    ax_T.plot(t.to_value(u.day), T[row].to_value(u.K), color="C3", lw=0.4, alpha=0.15)

# Add the archival sources.
archive = LightcurveArchive()
events = archive.events("supernovae/SLSN-I")

n_temp_events = sum(1 for name in events if "T_phot" in archive.fields("supernovae/SLSN-I", name))
temp_plotted = 0

for i, name in enumerate(events):
    lbol = archive.table("supernovae/SLSN-I", name, "L_bol")
    ax_L.plot(
        lbol["time"].to_value(u.day), lbol["L_bol"].to_value(u.erg / u.s),
        color="k", lw=0.5, alpha=0.25,
        label=f"Gomez+2024 (n={len(events)})" if i == 0 else None,
    )

    if "T_phot" in archive.fields("supernovae/SLSN-I", name):
        t_phot = archive.table("supernovae/SLSN-I", name, "T_phot")
        ax_T.plot(
            t_phot["time"].to_value(u.day), t_phot["T_phot"].to_value(u.K),
            color="k", lw=0.5, alpha=0.25,
            label=f"Gomez+2024 (n={n_temp_events})" if temp_plotted == 0 else None,
        )
        temp_plotted += 1

ax_L.set_xscale("log")
ax_L.set_yscale("log")
ax_L.set_ylim([1e41, 1e46])
ax_L.set_ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
ax_L.set_title(f"SLSN-I: simulated bolometric light curves (n={n_samples})")
ax_L.legend(loc="upper right", fontsize=8, frameon=False)

ax_T.set_xscale("log")
ax_T.set_yscale("log")
ax_T.set_xlabel("Time since explosion [days]")
ax_T.set_ylabel("Photospheric temperature [K]")
ax_T.set_title(f"SLSN-I: simulated photospheric temperatures (n={n_samples})")
ax_T.legend(loc="upper right", fontsize=8, frameon=False)

fig.tight_layout()