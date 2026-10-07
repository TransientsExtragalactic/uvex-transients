"""
Cut: Transient Type
========================

:meth:`~uvex_transients.simulation.core.SurveySimulator.filter_by_transient_type` keeps
only rows whose ``transient_type`` is one of a given list, dropping every other type. It
is useful once a `~uvex_transients.simulation.core.SurveySimulator` has more than one
type registered (e.g. to compare their yields, or to run a further cut against only
one), since every other cut screens whatever types are present without regard to type.
"""

from matplotlib import pyplot as plt

from uvex_transients.missions import uvex_fast as uvex
from uvex_transients.simulation.core import SurveySimulator
from uvex_transients.surveys import get_schedule
from uvex_transients.transients.supernovae import MagnetarSLSNe, TypeIaSNe
from uvex_transients.utils.plotting import resolve_fig_axes, set_plot_style

set_plot_style()

schedule = get_schedule()
slsn = MagnetarSLSNe()
snia = TypeIaSNe()
simulator = SurveySimulator(schedule, transients={"slsn": slsn, "type_ia": snia}, simulation_seed=42)

TIME_BINS = 10
NSIDE = 32
DOWNSAMPLE = 200

catalog = simulator.generate_events(time_bins=TIME_BINS, nside=NSIDE, downsample=DOWNSAMPLE)
for name in sorted(simulator.transient_collection):
    n = int((catalog.table["transient_type"] == name).sum())
    print(f"{name}: {n * DOWNSAMPLE} sampled.")

# %%
# Keeping only the SLSNe
# ---------------------------
#
# A later cut, or an action like
# :meth:`~uvex_transients.simulation.core.SurveySimulator.run_photometry_action`, may
# only make sense for one type at a time.

slsn_only = simulator.filter_by_transient_type(catalog, uvex, types=["slsn"])
kept_types = sorted({str(t) for t in slsn_only.table["transient_type"]})
print(f"After the cut: {len(slsn_only) * DOWNSAMPLE} events, all {kept_types}.")

# %%
# Sampled counts per type
# -----------------------------

fig, ax = resolve_fig_axes(fig_size=(5, 4))
counts_before = {
    name: int((catalog.table["transient_type"] == name).sum()) * DOWNSAMPLE
    for name in sorted(simulator.transient_collection)
}
counts_after = {"slsn": len(slsn_only) * DOWNSAMPLE, "type_ia": 0}
x = range(len(counts_before))
ax.bar(x, counts_before.values(), width=0.4, label="Before cut", color="#888888")
ax.bar([i + 0.4 for i in x], counts_after.values(), width=0.4, label="After cut", color="#4C72B0")
ax.set_xticks([i + 0.2 for i in x], counts_before.keys())
ax.set_ylabel("Events (simulated)")
ax.set_title("Effect of the transient-type cut")
ax.legend()
fig.tight_layout()
plt.show()
