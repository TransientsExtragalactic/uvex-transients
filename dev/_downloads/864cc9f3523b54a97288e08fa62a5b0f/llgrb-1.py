import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from uvex_transients.models.arnett import ArnettDecaySED
from uvex_transients.models.lightcurves.generic import BazinLightcurve
from uvex_transients.models.supernovae import LLGRBSED

rng = np.random.default_rng(20261020)
n_samples = 500

params = LLGRBSED().sample_parameters(size=n_samples, rng=rng)
grid = {name: value[:, None] for name, value in params.items()}

t = np.geomspace(0.03, 40, 200) * u.day
L_total = LLGRBSED.eval_bolometric(t, **grid)
L_sn = ArnettDecaySED.eval_bolometric(t, **{name: grid[name] for name in ArnettDecaySED._DEFAULT_PARAMETERS})
L_bump = BazinLightcurve.eval(
    t, amplitude=grid["A_bump"], t0=grid["t0_bump"], rise=grid["tau_rise"], fall=grid["tau_fall"]
)
T_sn, T_bump = LLGRBSED.temperature(t, **grid)

fig, (ax_L, ax_T) = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True)
days = t.to_value(u.day)
for row in range(n_samples):
    ax_L.plot(days, L_total[row].to_value(u.erg / u.s), color="k", lw=0.4, alpha=0.05)
    ax_L.plot(days, L_bump[row].to_value(u.erg / u.s), color="C3", lw=0.4, alpha=0.05)
    ax_L.plot(days, L_sn[row].to_value(u.erg / u.s), color="C0", lw=0.4, alpha=0.05)
    ax_T.plot(days, T_bump[row].to_value(u.K), color="C3", lw=0.4, alpha=0.05)
    ax_T.plot(days, T_sn[row].to_value(u.K), color="C0", lw=0.4, alpha=0.05)

for color, label in [("k", "total"), ("C3", "early excess"), ("C0", "supernova")]:
    ax_L.plot([], [], color=color, label=label)
ax_L.set_xscale("log")
ax_L.set_yscale("log")
ax_L.set_ylim(1e40, 1e46)
ax_L.set_ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
ax_L.set_title("llGRBs: simulated bolometric light curves (n=500)")
ax_L.legend(loc="upper right", fontsize=8, frameon=False)

ax_T.set_yscale("log")
ax_T.set_xlabel("Time since explosion [days]")
ax_T.set_ylabel("Photospheric temperature [K]")
fig.tight_layout()