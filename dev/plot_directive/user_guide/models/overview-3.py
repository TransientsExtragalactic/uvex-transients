import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from uvex_transients.models.kilonovae import KilonovaCoolingBlackbodySED

sed = KilonovaCoolingBlackbodySED()
params = sed.sample_parameters(rng=0)

t = np.geomspace(0.1, 20, 100) * u.day
nu = (2000 * u.AA).to(u.Hz, equivalencies=u.spectral())

for z in [0.005, 0.02, 0.05]:
    F_bol = sed.flux_bolometric(t, redshift=z, **params)
    plt.plot(t.to_value(u.day), F_bol.to_value(u.erg / u.s / u.cm**2), label=f"z = {z}")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Time since explosion [days]")
plt.ylabel(r"$F_\mathrm{bol}$ [erg s$^{-1}$ cm$^{-2}$]")
plt.title("Observed bolometric flux at three redshifts")
plt.legend()