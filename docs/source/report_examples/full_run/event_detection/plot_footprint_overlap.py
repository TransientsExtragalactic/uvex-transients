"""
Overlap With Other Surveys
==========================

Which of the events that UVEX detects can also be seen by other surveys, and where in the UVEX surveys
do the different transient types turn up?

Overlap with ZTF and Rubin is what makes follow-up and pre-discovery data possible, and the UVEX regions
(the wide and deep parts of the Low-Medium-Latitude Zone survey and the Magellanic Clouds) decide how often
an event is observed. No selection is made here beyond detection. The
:ref:`characterization pages <report_full_run_characterization>` repeat this analysis for the events that
pass the standard cuts.

.. note::

   Overlap is on the sky only. It does not say whether ZTF or Rubin was observing the position while the
   event was bright. Testing against the Rubin footprint needs the optional ``rubin_scheduler`` package
   (``pip install "uvex-transients[rubin]"``), and without it that column reads "n/a".
"""

from uvex_transients.utils.plotting import set_plot_style
from uvex_transients.utils.summary_report import STANDARD_POPULATIONS, SummaryReport
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
# Overlap by type
# ----------------
#
# Each cell is the percentage of a type's detected events whose position is inside the footprint of a
# survey, and the number in brackets after the type is its expected number of detections. A type's
# percentages need not add up to 100%, since the UVEX regions overlap each other and an event can be inside
# the footprints of several surveys.

if report is not None:
    report.plot_footprint_overlap(mask=report.detected)
plt.show()

# %%
# The same numbers as a table:

if report is not None:
    overlap = report.footprint_overlap(mask=report.detected)
    overlap["expected_events"].info.format = ",.0f"
    for column in overlap.colnames[2:]:
        overlap[column].info.format = ".1f"
    overlap.pprint_all()

# %%
# Where the events are
# ---------------------
#
# The detected events on the sky, each type in its own color, over the three UVEX regions. The supernovae
# are so numerous that they mostly trace the survey itself, so they are drawn apart from the rare types, which
# are the ones whose sky distribution is worth reading. The positions are Monte Carlo events, so the number
# of points is not the expected number of detections.

if report is not None:
    rare = ["slsne", "tde", "llgrb", "lfbot", "kilonova"]
    supernovae = SummaryReport(report.summary, {k: v for k, v in report.populations.items() if k not in rare})
    supernovae.plot_sky(mask=supernovae.detected)

plt.show()

# %%
# The rare types, SLSNe, TDEs, llGRBs, LFBOTs and kilonovae:

# sphinx_gallery_thumbnail_number = 3
if report is not None:
    exotic = SummaryReport(report.summary, {k: v for k, v in report.populations.items() if k in rare})
    exotic.plot_sky(mask=exotic.detected)
plt.show()
