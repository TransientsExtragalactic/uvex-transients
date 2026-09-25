import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from uvex_transients.models.kilonovae import KilonovaCoolingBlackbodySED

sed = KilonovaCoolingBlackbodySED()
nu = (2000 * u.AA).to(u.Hz, equivalencies=u.spectral())

L = sed.simulate(nu, 0.6 * u.day, size=2000, rng=3)

plt.hist(np.log10(L.to_value(u.erg / u.s / u.Hz)), bins=40, color="C0")
plt.xlabel(r"$\log_{10} L_\nu$ [erg s$^{-1}$ Hz$^{-1}$] at t = 0.6 d")
plt.ylabel("Realizations")
plt.title("Spread in peak-time UV luminosity across 2000 draws")