"""
Cut: Arbitrary Query
=========================

:meth:`~uvex_transients.simulation.core.SurveySimulator.filter_by_query` cuts an
`~uvex_transients.simulation.event_catalog.EventCatalog` by an arbitrary boolean
expression over its own columns, each bound to its own native type
(`~astropy.units.Quantity`, `~astropy.coordinates.SkyCoord`, `~astropy.time.Time`, or a
plain array), so comparisons stay unit- and frame-aware. It is the escape hatch for a
cut none of the named ones cover.
"""

from m4opt.missions import uvex
from matplotlib import pyplot as plt

from uvex_transients.simulation.core import SurveySimulator
from uvex_transients.surveys import get_schedule
from uvex_transients.transients.supernovae import MagnetarSLSNe
from uvex_transients.utils.plotting import resolve_fig_axes, set_plot_style

set_plot_style()

schedule = get_schedule()
slsn = MagnetarSLSNe()
simulator = SurveySimulator(schedule, transients={"slsn": slsn}, simulation_seed=42)

TIME_BINS = 10
NSIDE = 32
DOWNSAMPLE = 50

catalog = simulator.generate_events(time_bins=TIME_BINS, nside=NSIDE, downsample=DOWNSAMPLE)
print(f"{len(catalog) * DOWNSAMPLE} events sampled.")

# %%
# Combining two columns in one expression
# ----------------------------------------------
#
# Combine conditions with ``&``/``|``/``~`` (elementwise), not Python's
# ``and``/``or``/``not``. Here: nearby *and* lightly reddened by Milky Way dust.

low_z_low_ebv = simulator.filter_by_query(catalog, uvex, expr="(redshift < 0.3) & (ebv < 0.05)")
print(f"z < 0.3 and E(B-V) < 0.05: {len(low_z_low_ebv) * DOWNSAMPLE} events.")

# %%
# Compared against the named `filter_by_redshift`/`filter_by_query` alone
# ------------------------------------------------------------------------------

low_z_only = simulator.filter_by_query(catalog, uvex, expr="redshift < 0.3")
low_ebv_only = simulator.filter_by_query(catalog, uvex, expr="ebv < 0.05")

fig, ax = resolve_fig_axes(fig_size=(6, 4))
labels = ["z < 0.3", "E(B-V) < 0.05", "Both"]
counts = [len(low_z_only) * DOWNSAMPLE, len(low_ebv_only) * DOWNSAMPLE, len(low_z_low_ebv) * DOWNSAMPLE]
ax.bar(labels, counts, color=["#888888", "#888888", "#4C72B0"])
ax.set_ylabel("Events surviving")
ax.set_title("Effect of the query cut")
fig.tight_layout()
plt.show()
