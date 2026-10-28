"""
Overlap of Characterized Events
===============================

Which of the characterized events can also be seen by other surveys, and where in the UVEX surveys are they?

This repeats the :ref:`detection analysis <sphx_glr_report_full_run_event_detection_plot_footprint_overlap.py>`
for the events that pass the standard cuts of the
:ref:`previous page <sphx_glr_report_full_run_transient_characterization_plot_characterization.py>`. These are
the events with enough repeat detections, a bright enough peak and a tightly bracketed explosion to be
classified, so overlap with ZTF and Rubin here is overlap that can be used for classification and follow-up.

.. note::

   Overlap is on the sky only. It does not say whether ZTF or Rubin was observing the position while the
   event was bright. Testing against the Rubin footprint needs the optional ``rubin_scheduler`` package
   (``pip install "uvex-transients[rubin]"``), and without it that column reads "n/a".
"""

from matplotlib import pyplot as plt

from uvex_transients.utils.plotting import set_plot_style
from uvex_transients.utils.summary_report import STANDARD_POPULATIONS, SummaryReport

set_plot_style()

# The standard cuts, as on the previous page. Change them here to see a different selection.
MIN_DETECTIONS = 2
PEAK_MAG_LIMIT = 23.0

try:
    report = SummaryReport.from_release(
        STANDARD_POPULATIONS, min_detections=MIN_DETECTIONS, peak_mag_limit=PEAK_MAG_LIMIT
    )
except (ConnectionError, LookupError, OSError, ValueError) as err:
    report = None
    print(f"Skipping this page's analysis, the release data is not available: {err}")
else:
    print(f"{report.selected.sum():,} of {len(report.summary):,} detected events are characterized.")

# %%
# Overlap by type
# ----------------
#
# Each cell is the percentage of a type's characterized events whose position is inside the footprint of a
# survey, and the number in brackets after the type is its expected number of characterized events. A type's
# percentages need not add up to 100%, since the UVEX regions overlap each other and an event can be inside the
# footprints of several surveys. A type with no characterized events reads "n/a".

if report is not None:
    report.plot_footprint_overlap()

plt.show()

# %%
# The same numbers as a table:

if report is not None:
    overlap = report.footprint_overlap()
    overlap["expected_events"].info.format = ",.0f"
    for column in overlap.colnames[2:]:
        overlap[column].info.format = ".1f"
    overlap.pprint_all()

# %%
# Where the events are
# ---------------------
#
# The characterized events on the sky, each type in its own color, over the three UVEX regions. The supernovae
# are so numerous that they mostly trace the survey itself, so they are drawn apart from the rare types, which
# are the ones whose sky distribution is worth reading. The positions are Monte Carlo events, so the number
# of points is not the expected number of events.

if report is not None:
    rare = ["slsne", "tde", "llgrb", "lfbot", "kilonova"]
    supernovae = SummaryReport(
        report.summary,
        {k: v for k, v in report.populations.items() if k not in rare},
        min_detections=MIN_DETECTIONS,
        peak_mag_limit=PEAK_MAG_LIMIT,
    )
    supernovae.plot_sky(mask=supernovae.selected)

plt.show()

# %%
# The rare types, SLSNe, TDEs, llGRBs, LFBOTs and kilonovae:

# sphinx_gallery_thumbnail_number = 3
if report is not None:
    exotic = SummaryReport(
        report.summary,
        {k: v for k, v in report.populations.items() if k in rare},
        min_detections=MIN_DETECTIONS,
        peak_mag_limit=PEAK_MAG_LIMIT,
    )
    exotic.plot_sky(mask=exotic.selected)

plt.show()
