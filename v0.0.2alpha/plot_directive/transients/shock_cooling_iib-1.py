import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from uvex_transients.models.supernovae import MoragShockCoolingSED as SEDClass

rng = np.random.default_rng(20260910)
n_samples = 500

params = SEDClass().sample_parameters(size=n_samples, rng=rng)
params_grid = {name: value[:, None] for name, value in params.items()}

t = np.geomspace(0.005, 15, 250) * u.day
L_bol = SEDClass.eval_bolometric(t, **params_grid)
T = SEDClass.temperature(t, **params_grid)

fig, (ax_L, ax_T) = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True)

for row in range(n_samples):
    ax_L.plot(t.to_value(u.day), L_bol[row].to_value(u.erg / u.s), color="C0", lw=0.4, alpha=0.08)
    ax_T.plot(t.to_value(u.day), T[row].to_value(u.K), color="C3", lw=0.4, alpha=0.08)

ax_L.set_xscale("log")
ax_L.set_yscale("log")
ax_L.set_ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
ax_L.set_title("Shock-cooling IIb: simulated bolometric light curves (n=500)")

ax_T.set_yscale("log")
ax_T.set_xlabel("Time since explosion [days]")
ax_T.set_ylabel("Photospheric temperature [K]")

fig.tight_layout()