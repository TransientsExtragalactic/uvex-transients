"""
Rise Time Brackets
==================

How tightly does UVEX bracket the rise of an infant supernova? The event summary table records, for each
event, ``t_rise_<N>mag``: the first detection at least N magnitudes brighter than the detection limit at the
event's last non-detection. The gap ``t_rise_<N>mag - t_last_nondet`` is therefore an upper limit on how long
the event took to rise by N magnitudes, as an observer would see it.

This example downloads the infant supernova run's event summary with
:func:`~uvex_transients.utils.get_results` and counts, for each transient type, the events whose 3 magnitude
rise is bracketed within 6, 12, 24 and 48 hours. If the data product cannot be obtained, for instance
without network access or from a release cut before the rise columns existed, the analysis is skipped.
"""

import numpy as np
from astropy.table import QTable, Table

from uvex_transients.utils import get_results

#: The rise to bracket, in magnitudes. The summary carries 1, 2 and 3 by default.
N_MAG = 3

#: The bracket limits to count within, in hours.
LIMITS_HR = (6, 12, 24, 48)

try:
    summary = QTable.read(get_results(["summary"], simulation="infant_sne")["summary"])
except (ConnectionError, LookupError, OSError, ValueError) as err:
    summary = None
    print(f"Skipping this page's analysis, the release data is not available: {err}")
else:
    print(f"{len(summary):,} events in the infant supernova summary table.")

# %%
# The brackets
# -------------
#
# An event has a bracket only if some detection gets ``N_MAG`` magnitudes above the limit at its last
# non-detection, and only if it has a non-detection at all. Every other event is left out of the counts
# below, so each column is a number of events, not a fraction.

if summary is not None and f"t_rise_{N_MAG}mag" not in summary.colnames:
    print(f"The summary has no t_rise_{N_MAG}mag column, it predates the rise time columns.")
    summary = None

if summary is not None:
    rise = summary[f"t_rise_{N_MAG}mag"]
    last_nondet = summary["t_last_nondet"]
    has_bracket = ~np.asarray(rise.mask) & ~np.asarray(last_nondet.mask)

    bracket_hr = np.full(len(summary), np.nan)
    bracket_hr[has_bracket] = (rise[has_bracket] - last_nondet[has_bracket]).to_value("hr")
    types = np.asarray(summary["transient_type"]).astype(str)

    rows = []
    for name in [*sorted(set(types)), "total"]:
        in_type = np.ones(len(summary), dtype=bool) if name == "total" else types == name
        rows.append(
            (
                name,
                int(in_type.sum()),
                int((in_type & has_bracket).sum()),
                *(int((in_type & has_bracket & (bracket_hr <= limit)).sum()) for limit in LIMITS_HR),
            )
        )

    table = Table(
        rows=rows,
        names=["transient_type", "n_events", f"n_{N_MAG}mag_bracket", *[f"within_{limit}h" for limit in LIMITS_HR]],
    )
    print(f"Events whose {N_MAG} mag rise is bracketed within each limit:")
    table.pprint_all()
