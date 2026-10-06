import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from uvex_transients.models.kilonovae import KilonovaCoolingBlackbodySED

sed = KilonovaCoolingBlackbodySED()
params = sed.sample_parameters(rng=0)

# A toy ~2250 A bandpass, standing in for a real UVEX filter.
lam0, fwhm = 2250 * u.AA, 400 * u.AA
wave = np.linspace(lam0 - 3 * fwhm, lam0 + 3 * fwhm, 200)
throughput = np.exp(-0.5 * ((wave - lam0) / (fwhm / 2.3548)) ** 2).value
nu_grid = wave.to(u.Hz, equivalencies=u.spectral())

t = np.geomspace(0.1, 20, 100) * u.day
mag = sed.mag_band(nu_grid, throughput, t, redshift=0.01, **params)

plt.plot(t.to_value(u.day), mag.value)
plt.gca().invert_yaxis()
plt.xscale("log")
plt.xlabel("Time since explosion [days]")
plt.ylabel("AB magnitude (band-averaged)")
plt.title("A toy UV bandpass light curve, z = 0.01")