"""
Expected Detections
===================

How many events of each transient type does UVEX detect, and how repeatable are those detections?

An event is **detected** if the survey observes its position after it explodes and, in at least
one observation, its signal-to-noise ratio exceeds 5 in at least one band. An event whose only such
observation is the first-ever visit to that field is not counted, since there is no reference image to
detect it against. No further selection is made here: this is the upper bound on what can be studied,
and the :ref:`characterization pages <report_full_run_characterization>` apply the cuts that bring it down.

Everything on this page is read from the event summary table of the latest release, through
:class:`~uvex_transients.utils.summary_report.SummaryReport`.
"""

import numpy as np

from uvex_transients.simulation.rates import estimate_yield
from uvex_transients.utils.plotting import set_plot_style
from uvex_transients.utils.summary_report import POPULATION_GROUPS, STANDARD_POPULATIONS, SummaryReport
import matplotlib.pyplot as plt

set_plot_style()

try:
    report = SummaryReport.from_release(STANDARD_POPULATIONS)
except (ConnectionError, LookupError, OSError, ValueError) as err:
    report = None
    print(f"Skipping this page's analysis, the release data is not available: {err}")
else:
    print(f"{len(report.summary):,} detected events in the release's summary table.")

# %%
# Expected detections per type
# -----------------------------
#
# The summary table has one row per Monte Carlo event that UVEX detects. One such event stands for
# :math:`\mu_0 / n` real ones, with :math:`n` the number generated and :math:`\mu_0` the number that
# exist in UVEX's footprint and observing window, so the expected number of detections is
# :math:`\mu_0 k / n` for :math:`k` detected events
# (:func:`~uvex_transients.simulation.rates.estimate_yield`).
#
# The black error bars are the Monte Carlo uncertainty on :math:`k / n`, a Clopper-Pearson interval
# from the finite number of events generated. The pale band is the uncertainty on the event rate, which
# scales every bar by the same factor and does not shrink as more events are generated. A type with
# no detected events has only an upper limit. Types are never summed, since that would mix independent
# Monte Carlo errors with a shared rate uncertainty.

if report is not None:
    report.plot_yields(mask=report.detected)

plt.show()
# %%
# The numbers behind the bars, with the fraction of the generated events that is detected:

if report is not None:
    detected = estimate_yield(report.summary, transient_types=list(report.populations))
    detected = detected[[list(detected["transient_type"]).index(key) for key in report.populations]]  # plot order
    detected["transient_type"] = [report.populations[str(key)].label for key in detected["transient_type"]]
    detected["fraction"].info.format = ".2e"
    detected["expected_events"].info.format = ",.0f"
    detected["transient_type", "n_pre_cut", "n_selected", "fraction", "expected_events"].pprint_all()

# %%
# How often is each event detected?
# ----------------------------------
#
# A single detection is a snapshot. An event detected many times has a light curve, which is what
# classifying it takes. Each panel shows, for the types of one group, the percentage of the detected events
# that are detected at least :math:`N` times. Every curve starts at 100% at :math:`N = 1`, and the
# faster it falls the more of the type's detections are one-offs. The number in the legend is the number
# of Monte Carlo events behind the curve, so a short curve from a few events is noisy.

if report is not None:
    report.plot_detection_counts(POPULATION_GROUPS, mask=report.detected)

plt.show()

# %%
# Median number of detections per type:

if report is not None:
    for key, spec in report.populations.items():
        n_det = report.n_det[report.detected & (report.types == key)]
        if n_det.size:
            print(
                f"{spec.label:<20s} median {np.median(n_det):>5.0f}   90th percentile {np.percentile(n_det, 90):>6.0f}"
            )
