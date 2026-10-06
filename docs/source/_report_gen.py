"""
Generates the "Report" docs page from the latest published GitHub Release.

The page shows one pre-computed table, the expected yield of every transient type, computed from
the release's ``event_summary.ecsv`` with `~uvex_transients.simulation.rates.estimate_yield`,
followed by the gallery of examples in ``report_examples/``. Those examples download the same
release data through `~uvex_transients.utils.get_results` and do whatever analysis they like with
it. ``configs/full_run.yaml`` and ``.github/workflows/build_and_release.yml`` produce and attach
the assets.

Hooked into the Sphinx build from ``conf.py``'s ``setup()``, on the ``builder-inited`` event, so
the generated ``report/index.rst`` exists before Sphinx reads any source files, both in CI
(``build_documentation.yml``) and in a local ``make html``. Never fails the build: a missing
release, an unreachable GitHub API, or a release without the expected asset all fall back to a
placeholder page instead of raising.
"""

from pathlib import Path

import numpy as np
from astropy.table import QTable

from uvex_transients.simulation.rates import estimate_yield
from uvex_transients.utils.results import RELEASES_URL, find_release, get_results

_DISPLAY_NAMES = {
    "kilonova": "Kilonova",
    "tde": "Tidal Disruption Event",
    "lfbot": "LFBOT",
    "ia_sne": "Type Ia SNe",
    "iip_sne": "Type IIP SNe",
    "iip_excess_sne": "Type IIP SNe (Excess)",
    "iib_sne": "Type IIb SNe",
    "ib_sne": "Type Ib SNe",
    "ic_sne": "Type Ic SNe",
    "icbl_sne": "Type Ic-BL SNe",
    "slsne": "SLSNe (Magnetar)",
}


def _display_name(key: str) -> str:
    return _DISPLAY_NAMES.get(key, key.replace("_", " ").title())


# ----------------------------------------- #
# Table Rendering                            #
# ----------------------------------------- #
def _fmt(value: float, precision: int = 3) -> str:
    """Format to `precision` significant figures as fixed-point (never `1.2e+03`, which MathJax renders literally)."""
    if np.isnan(value):
        return "--"
    if value == 0:
        return "0"
    magnitude = int(np.floor(np.log10(abs(value))))
    decimals = max(0, precision - 1 - magnitude)
    return f"{value:,.{decimals}f}"


def _asym_math(value: float, lower: float, upper: float, precision: int = 3) -> str:
    """Render ``value`` with one asymmetric interval as an RST ``:math:`` role."""
    if np.isnan(value):
        return "--"
    return (
        f":math:`{_fmt(value, precision)}^{{+{_fmt(upper - value, precision)}}}_{{-{_fmt(value - lower, precision)}}}`"
    )


def _double_asym_math(
    value: float,
    binom_lower: float,
    binom_upper: float,
    rate_lower: float,
    rate_upper: float,
    precision: int = 3,
) -> str:
    """Render ``value`` with both stacked asymmetric intervals (Monte Carlo, then rate) as an RST ``:math:`` role."""
    if np.isnan(value):
        return "--"
    return (
        f":math:`{_fmt(value, precision)}"
        f"^{{+{_fmt(binom_upper - value, precision)}}}_{{-{_fmt(value - binom_lower, precision)}}}"
        f"\\,{{}}^{{+{_fmt(rate_upper - value, precision)}}}_{{-{_fmt(value - rate_lower, precision)}}}`"
    )


def _summary_table_rst(yields: QTable) -> str:
    """Render `estimate_yield`'s per-type table as an RST ``list-table`` (detection probability, expected events)."""
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 26 37 37",
        "",
        "   * - Transient Type",
        "     - Detection Probability",
        "     - Expected Detections",
    ]
    for row in yields:
        lines.append(f"   * - {_display_name(str(row['transient_type']))}")
        lines.append(
            "     - " + _asym_math(float(row["fraction"]), float(row["fraction_lower"]), float(row["fraction_upper"]))
        )
        lines.append(
            "     - "
            + _double_asym_math(
                float(row["expected_events"]),
                float(row["expected_events_binom_lower"]),
                float(row["expected_events_binom_upper"]),
                float(row["expected_events_rate_lower"]),
                float(row["expected_events_rate_upper"]),
            )
        )
    return "\n".join(lines)


# ----------------------------------------- #
# Page Assembly                              #
# ----------------------------------------- #
_EXAMPLES_RST = """\
Examples
========

The examples below download this release's data products with
:func:`~uvex_transients.utils.get_results` and analyze them: the event catalog that survived the
cuts, the exposure catalog, the per-event summary table, and the synthetic photometry. Each page
is rebuilt with the docs, so they follow the latest release.

.. toctree::
   :maxdepth: 1

   auto_examples/index
"""

_PLACEHOLDER_RST = f"""\
Transient Yield Report
=======================

.. note::

   No published release report is available yet (or it could not be fetched during this
   documentation build, see the build log for details). Once a ``vX.Y.Z`` release with
   ``event_summary.ecsv`` attached is published, this page will be generated automatically on
   the next documentation build.

{_EXAMPLES_RST}
"""


def _write_placeholder(report_dir: Path) -> None:
    (report_dir / "index.rst").write_text(_PLACEHOLDER_RST)


def _write_report(report_dir: Path, tag: str, release_url: str, yields: QTable) -> None:
    confidence = int(round(100 * yields.meta["confidence"]))
    rst = f"""\
Transient Yield Report
=======================

.. note::

   This page is generated automatically from the latest published release,
   `{tag} <{release_url}>`__, of UVEX Transients, specifically its ``event_summary.ecsv`` data
   product (see :doc:`../user_guide`). It is regenerated on every documentation build, so it may
   lag between releases.

.. seealso::

   Every data product from this run (the event catalog, photometry, exposure, and per-event
   summary) is attached to the `{tag} release page <{release_url}>`__ on GitHub, and
   :func:`~uvex_transients.utils.get_results` downloads them into a local cache. Earlier
   releases are listed on the `releases page <{RELEASES_URL}>`__.

Summary
=======

For each transient type, the fraction of simulated events that UVEX detects and the expected
number of real events that corresponds to. The first uncertainty on each value is the
{confidence}% Clopper-Pearson interval from the finite Monte Carlo sample, and the second, on the
expected detections only, is the uncertainty on the event rate.

{_summary_table_rst(yields)}

{_EXAMPLES_RST}
"""
    (report_dir / "index.rst").write_text(rst)


def generate_report(app) -> None:
    """Sphinx ``builder-inited`` callback: (re)generate ``report/index.rst`` from the latest release."""
    report_dir = Path(app.srcdir) / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    try:
        release = find_release()
        tag = release["tag_name"]
        path = get_results(["summary"], tag=tag)["summary"]
        events = QTable.read(path)
        yields = estimate_yield(events)
    except (ConnectionError, LookupError, OSError, ValueError, KeyError) as exc:
        print(f"[report] could not build the report from the latest release ({exc}); using a placeholder.")
        _write_placeholder(report_dir)
        return

    _write_report(report_dir, tag, release.get("html_url") or f"{RELEASES_URL}/tag/{tag}", yields)
    print(f"[report] generated report/index.rst from release {tag}.")
