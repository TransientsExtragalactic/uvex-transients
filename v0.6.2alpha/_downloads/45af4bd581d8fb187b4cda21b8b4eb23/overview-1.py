import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from uvex_transients.models.kilonovae import KilonovaCoolingBlackbodySED

sed = KilonovaCoolingBlackbodySED()
params = sed.sample_parameters(rng=0)  # one realization of every parameter

t = np.geomspace(0.02, 30, 200) * u.day
L_bol = sed.eval_bolometric(t, **params)

plt.plot(t.to_value(u.day), L_bol.to_value(u.erg / u.s))
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Time since explosion [days]")
plt.ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
plt.title("One simulated kilonova bolometric light curve")