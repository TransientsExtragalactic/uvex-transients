import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from uvex_transients.models.kilonovae import KilonovaCoolingBlackbodySED

sed = KilonovaCoolingBlackbodySED()
params = sed.sample_parameters(rng=0)

wave = np.linspace(1000, 10000, 300) * u.AA
nu = wave.to(u.Hz, equivalencies=u.spectral())

fig, (ax_L, ax_S) = plt.subplots(1, 2, figsize=(9, 3.5))

for t in [0.5, 2, 8] * u.day:
    L_nu = sed.eval(nu, t, **params)
    S = sed.eval_spectrum(nu, t, **params)
    ax_L.plot(wave.to_value(u.AA), L_nu.to_value(u.erg / u.s / u.Hz), label=f"t = {t}")
    ax_S.plot(wave.to_value(u.AA), S.to_value(1 / u.Hz), label=f"t = {t}")

ax_L.set_yscale("log")
ax_L.set_xlabel(r"Wavelength [$\AA$]")
ax_L.set_ylabel(r"$L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]")
ax_L.set_title("Spectral luminosity")
ax_L.legend(fontsize=8)

ax_S.set_xlabel(r"Wavelength [$\AA$]")
ax_S.set_ylabel(r"$S(\nu, t)$ [Hz$^{-1}$]")
ax_S.set_title("Normalized shape (integrates to 1 over $\\nu$)")

fig.tight_layout()