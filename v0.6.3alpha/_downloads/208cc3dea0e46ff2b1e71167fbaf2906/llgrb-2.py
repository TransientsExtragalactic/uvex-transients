import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from uvex_transients.models.supernovae import LLGRBSED
from uvex_transients.utils.cosmology import get_cosmology
from uvex_transients.utils.filter_utils import get_bandpass
from uvex_transients.utils.swift_sne_archive import SwiftSNeArchive

rng = np.random.default_rng(20261021)
n_samples = 100

archive = SwiftSNeArchive()
redshift = float(archive.meta("SN2006aj")["redshift"])
d_L = get_cosmology().luminosity_distance(redshift)

params = LLGRBSED().sample_parameters(size=n_samples, rng=rng)
grid = {name: value[:, None] for name, value in params.items()}
t = np.geomspace(0.05, 40, 150) * u.day

fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
for ax, band, color in zip(axes, ["UVW2", "UVM2", "UVW1"], ["C3", "C1", "C0"]):
    nu_eff = (archive.meta("SN2006aj", band)["lambda_eff_AA"] * u.AA).to(u.Hz, equivalencies=u.spectral())
    mag = LLGRBSED.mag_bandpass(get_bandpass(f"Swift/UVOT.{band}"), t, redshift=redshift, **grid)
    nuLnu = (4 * np.pi * d_L**2 * nu_eff * mag.to(u.erg / u.s / u.cm**2 / u.Hz)).to_value(u.erg / u.s)
    ax.plot(t.to_value(u.day), nuLnu.T, color="0.4", lw=0.4, alpha=0.12)

    lc = archive.lightcurve("SN2006aj", band)
    ax.errorbar(lc["t"].value, lc["L"].value, lc["L_err"].value, fmt="o", ms=3, color=color, label="SN 2006aj")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Time since GRB trigger [days]")
    ax.set_title(band)
    ax.legend(loc="upper right", fontsize=8, frameon=False)

axes[0].set_ylim(1e40, 1e44)
axes[0].set_ylabel(r"$\nu L_\nu$ [erg s$^{-1}$]")
fig.tight_layout()