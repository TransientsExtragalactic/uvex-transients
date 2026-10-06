"""
Loading the Release Data
=========================

Every release attaches four data products from the full run. :func:`~uvex_transients.utils.get_results`
downloads the ones you ask for into the package cache and returns their local paths, and each
file loads with its own ``from_disk``:

- ``events``: the :class:`~uvex_transients.simulation.event_catalog.EventCatalog` of events that
  survived the magnitude and SNR cuts,
- ``exposure``: the :class:`~uvex_transients.simulation.exposure_catalog.ExposureCatalog`, the
  theoretical event counts for the survey's footprint,
- ``summary``: the per-event table from
  :meth:`~uvex_transients.simulation.core.SurveySimulator.run_event_summary_action`,
- ``photometry``: the synthetic photometry of the surviving events. It is over 100 MB, so this
  example leaves it out.
"""

import numpy as np
from astropy.table import QTable
from matplotlib import pyplot as plt

from uvex_transients.simulation import EventCatalog, ExposureCatalog, estimate_yield
from uvex_transients.utils import get_results
from uvex_transients.utils.plotting import resolve_fig_axes, set_plot_style

set_plot_style()

paths = get_results(["events", "exposure"])

events = EventCatalog.from_disk(paths["events"])
exposure = ExposureCatalog.from_disk(paths["exposure"])

# Releases cut before the event summary table was introduced do not carry one, so it is optional here.
try:
    summary = QTable.read(get_results(["summary"])["summary"])
except LookupError as err:
    print(f"No event summary in the latest release, skipping the yield analysis: {err}")
    summary = None

print(f"{len(events)} events survive the cuts, out of {sum(events.pre_cut_counts.values())} generated.")
print("Intrinsic UVEX events by type:", {name: f"{mu0:,.0f}" for name, mu0 in exposure.total_expected_events.items()})

# %%
# The summary table
# ------------------
#
# One row per surviving event. Its ``meta`` records how many events of each type were generated, the
# intrinsic expected count of each, and the rate uncertainty, which is everything
# :func:`~uvex_transients.simulation.rates.estimate_yield` needs to turn a selection of rows into
# an expected number of real events.

if summary is not None:
    print(summary.colnames)
    print(dict(summary.meta["n_pre_cut"]))

# %%
# Expected detections per type
# -----------------------------
#
# With no mask, every row counts, so this is the yield of the cuts the release was built with. Pass a
# boolean ``mask`` to restrict it, for example to events detected in at least two epochs.

if summary is not None:
    everything = estimate_yield(summary)
    repeated = estimate_yield(summary, mask=np.asarray(summary["n_det"]) >= 2)
    print(everything["transient_type", "n_selected", "fraction", "expected_events"])

    fig, ax = resolve_fig_axes(fig_size=(8, 5))
    x = np.arange(len(everything))
    for table, label, offset in ((everything, "any detection", -0.2), (repeated, r"$\geq 2$ detected epochs", 0.2)):
        expected = np.asarray(table["expected_events"])
        lower = np.asarray(table["expected_events_binom_lower"])
        upper = np.asarray(table["expected_events_binom_upper"])
        ax.errorbar(
            x + offset,
            np.where(expected > 0, expected, np.nan),
            yerr=[np.clip(expected - lower, 0, None), upper - expected],
            fmt="o",
            capsize=3,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xticks(x, everything["transient_type"], rotation=45, ha="right")
    ax.set_ylabel("Expected UVEX events")
    ax.legend()
    fig.tight_layout()
    plt.show()
