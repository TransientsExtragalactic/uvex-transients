"""
Generates the "Report" docs page from the latest published GitHub Release.

The page shows one pre-computed table, the expected yield of every transient type, computed from
the primary run's ``event_summary.ecsv`` with `~uvex_transients.simulation.rates.estimate_yield`,
followed by one card per simulation run (see ``_report_runs.py``), each linking to a gallery of that
run's analyses built from ``report_examples/<run>/``. Those examples download the run's release data
through `~uvex_transients.utils.get_results` and do whatever analysis they like with it.
``configs/*.yaml`` and ``.github/workflows/build_and_release.yml`` produce and attach the assets.

Hooked into the Sphinx build from ``conf.py``'s ``setup()``, on the ``builder-inited`` event, so
the generated ``report/index.rst`` exists before Sphinx reads any source files, both in CI
(``build_documentation.yml``) and in a local ``make html``. Never fails the build: a missing
release, an unreachable GitHub API, or a release without the expected asset all fall back to a
placeholder page instead of raising.
"""

from pathlib import Path

import numpy as np
from _report_runs import RunSpec, discover_runs
from astropy.table import QTable

from uvex_transients.simulation.rates import estimate_yield
from uvex_transients.utils.results import GITHUB_REPO, RELEASES_URL, find_release, get_results

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
# Run Cards                                  #
# ----------------------------------------- #
def _indent(text: str, spaces: int) -> str:
    return "\n".join(" " * spaces + line if line else line for line in text.splitlines())


def _run_card_rst(run: RunSpec) -> str:
    """Render one run as a sphinx-design ``grid-item-card`` linking to its gallery's index."""
    badges = " ".join(f":bdg-secondary-line:`{tag}`" for tag in run.tags)
    footer = ":bdg-primary:`Primary run` · " if run.is_primary else ""
    count = run.n_analyses
    footer += "Analyses coming soon" if count == 0 else f"{count} {'analysis' if count == 1 else 'analyses'}"
    if run.config:
        footer += f" · `{run.config} <https://github.com/{GITHUB_REPO}/blob/main/{run.config}>`__"
    lines = [
        f".. grid-item-card:: :octicon:`{run.icon};1.3em;sd-text-primary` {run.title}",
        f"   :link: {run.key}/index",
        "   :link-type: doc",
        "   :class-card: sd-shadow-sm sd-border-1 run-card",
        "",
        f"   {run.summary}",
    ]
    if badges:
        lines += ["", f"   {badges}"]
    lines += ["", "   +++", f"   {footer}"]
    return "\n".join(lines)


def _runs_rst(runs: list[RunSpec]) -> str:
    """Render the "Simulation Runs" section: a grid of run cards, and a hidden toctree for the sidebar."""
    if not runs:
        return ""
    cards = "\n\n".join(_run_card_rst(run) for run in runs)
    toctree = "\n".join(f"   {run.key}/index" for run in runs)
    return f"""\
Simulation Runs
===============

Each run is its own simulation, with its own population, selection and data products, and has a
gallery of the analyses done on it. Every example downloads its run's data from the latest release
with :func:`~uvex_transients.utils.get_results` (cached after the first download), and the pages are
rebuilt with the docs, so they follow the latest release.

.. grid:: 1 1 2 2
   :gutter: 3

{_indent(cards, 3)}

.. toctree::
   :hidden:
   :maxdepth: 1

{toctree}
"""


# ----------------------------------------- #
# Page Assembly                              #
# ----------------------------------------- #
def _placeholder_rst(runs: list[RunSpec]) -> str:
    return f"""\
.. _report_page:

Transient Yield Report
=======================

.. note::

   No published release report is available yet (or it could not be fetched during this
   documentation build, see the build log for details). Once a ``vX.Y.Z`` release with
   ``event_summary.ecsv`` attached is published, this page will be generated automatically on
   the next documentation build.

{_runs_rst(runs)}
"""


def _report_rst(tag: str, release_url: str, yields: QTable, runs: list[RunSpec]) -> str:
    confidence = int(round(100 * yields.meta["confidence"]))
    return f"""\
.. _report_page:

Transient Yield Report
=======================

.. note::

   This page is generated automatically from the latest published release,
   `{tag} <{release_url}>`__, of UVEX Transients, specifically its ``event_summary.ecsv`` data
   product (see :doc:`../user_guide`). It is regenerated on every documentation build, so it may
   lag between releases.

.. seealso::

   Every data product from these runs (the event catalog, photometry, exposure, and per-event
   summary) is attached to the `{tag} release page <{release_url}>`__ on GitHub, and
   :func:`~uvex_transients.utils.get_results` downloads them into a local cache. Earlier
   releases are listed on the `releases page <{RELEASES_URL}>`__.

Summary
=======

For each transient type of the primary run, the fraction of simulated events that UVEX detects and
the expected number of real events that corresponds to. The first uncertainty on each value is the
{confidence}% Clopper-Pearson interval from the finite Monte Carlo sample, and the second, on the
expected detections only, is the uncertainty on the event rate.

{_summary_table_rst(yields)}

{_runs_rst(runs)}
"""


def generate_report(app) -> None:
    """Sphinx ``builder-inited`` callback: (re)generate ``report/index.rst`` from the latest release."""
    report_dir = Path(app.srcdir) / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(Path(app.srcdir) / "report_examples")

    try:
        release = find_release()
        tag = release["tag_name"]
        path = get_results(["summary"], tag=tag)["summary"]
        events = QTable.read(path)
        yields = estimate_yield(events)
    except (ConnectionError, LookupError, OSError, ValueError, KeyError) as exc:
        print(f"[report] could not build the report from the latest release ({exc}); using a placeholder.")
        (report_dir / "index.rst").write_text(_placeholder_rst(runs))
        return

    release_url = release.get("html_url") or f"{RELEASES_URL}/tag/{tag}"
    (report_dir / "index.rst").write_text(_report_rst(tag, release_url, yields, runs))
    print(f"[report] generated report/index.rst from release {tag}.")
