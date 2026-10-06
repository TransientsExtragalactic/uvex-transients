import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from uvex_transients.models.kilonovae import KilonovaCoolingBlackbodySED as SEDClass

rng = np.random.default_rng(20260910)
n_samples = 200

params = SEDClass().sample_parameters(size=n_samples, rng=rng)
params_grid = {name: value[:, None] for name, value in params.items()}

t = np.geomspace(0.02, 30, 200) * u.day
L_bol = SEDClass.eval_bolometric(t, **params_grid)

plt.plot(t.to_value(u.day), L_bol.to_value(u.erg / u.s).T, color="C0", lw=0.5, alpha=0.15)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Time since explosion [days]")
plt.ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
plt.title(f"{n_samples} simulated kilonova light curves")