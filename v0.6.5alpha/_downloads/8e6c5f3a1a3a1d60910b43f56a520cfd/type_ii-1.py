import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from uvex_transients.models.supernovae import TypeIIPSED
from uvex_transients.utils.lightcurve_archive import LightcurveArchive

n_samples = 1000
colors = {"plain": "#2166ac", "csm": "#d6604d"}
t = np.geomspace(0.3, 200, 300) * u.day
t_days = t.to_value(u.day)

sims = {}
for key, csm, seed in [("plain", 0.0, 1), ("csm", 1.0, 2)]:
    params = TypeIIPSED(csm=csm).sample_parameters(size=n_samples, rng=np.random.default_rng(seed))
    params_grid = {name: value[:, None] for name, value in params.items()}
    sims[key] = {
        "L": TypeIIPSED.eval_bolometric(t, **params_grid).to_value(u.erg / u.s),
        "T": TypeIIPSED.temperature(t, **params_grid).to_value(u.K),
    }

fig, (ax_L, ax_T) = plt.subplots(2, 1, figsize=(6.4, 7.6), sharex=True)
for key, label in [("plain", "No CSM"), ("csm", "CSM")]:
    for ax, quantity in [(ax_L, "L"), (ax_T, "T")]:
        lo95, lo68, med, hi68, hi95 = np.percentile(sims[key][quantity], [2.5, 16, 50, 84, 97.5], axis=0)
        ax.fill_between(t_days, lo95, hi95, color=colors[key], alpha=0.15, lw=0)
        ax.fill_between(t_days, lo68, hi68, color=colors[key], alpha=0.35, lw=0)
        ax.plot(
            t_days, med, color=colors[key], lw=2,
            label=f"{label} model: median, 68%, 95% (n={n_samples})" if quantity == "L" else None,
        )

archive = LightcurveArchive()
observed = {
    "plain": ("supernovae/IIP", "o", 14, "Literature Type IIP"),
    "csm": ("supernovae/II", "s", 22, "SN 2023ixf, SN 2024ggi (CSM)"),
}
for key, (group, marker, size, label) in observed.items():
    first = True
    for event in archive.events(group):
        for ax, field in [(ax_L, "L_bol"), (ax_T, "T_phot")]:
            try:
                table = archive.table(group, event, field)
            except Exception:
                continue
            good = np.isfinite(table[field].value) & (table[field].value > 0)
            ax.scatter(
                table["time"].to_value(u.day)[good], table[field].value[good],
                marker=marker, s=size, color=colors[key], edgecolor="white", linewidth=0.3, zorder=5,
                label=label if (first and field == "L_bol") else None,
            )
        first = False

ax_L.set_xscale("log")
ax_L.set_yscale("log")
ax_L.set_ylabel(r"$L_\mathrm{bol}$ [erg s$^{-1}$]")
ax_L.set_title("Type IIP SNe with and without CSM: simulated light curves")
ax_L.legend(loc="lower left", fontsize=8, frameon=False)
ax_L.set_ylim(1e40, 3e44)
ax_L.set_xlim(0.3, 200)
ax_T.set_yscale("log")
ax_T.set_xlabel("Time since explosion [days]")
ax_T.set_ylabel("Photospheric temperature [K]")
ax_T.set_ylim(3e3, 8e4)

fig.tight_layout()