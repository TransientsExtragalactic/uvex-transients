import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from uvex_transients.transients.TDEs import TidalDisruptionEvent
from uvex_transients.utils.lightcurve_archive import LightcurveArchive

rng = np.random.default_rng(20260912)

TDEs = TidalDisruptionEvent()
params = TDEs.sed.sample_parameters(size=10000, rng=rng)
log_T_draws = np.log10(params["temperature"].to_value(u.K))

archive = LightcurveArchive()
log_T_yao = np.array([
    np.log10(archive.table("tdes", name, "T_phot")["T_phot"].to_value(u.K)[0])
    for name in archive.events("tdes") if name.endswith("_yao2023")
])

bins = np.linspace(3.9, 4.7, 13)

fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.hist(log_T_draws, bins=bins, density=True, color="C0", alpha=0.55, label="Prior draws (n=10000)")
ax.hist(
    log_T_yao, bins=bins, density=True, histtype="step", color="k", lw=1.6,
    label=f"Yao+2023 (n={len(log_T_yao)})",
)

ax.set_xlabel(r"$\log_{10}(T/\mathrm{K})$")
ax.set_ylabel("Probability density")
ax.set_title("Tidal disruption events: photospheric temperature")
ax.legend(loc="upper right", fontsize=8, frameon=False)

fig.tight_layout()
plt.show()