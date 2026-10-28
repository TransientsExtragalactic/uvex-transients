"""
Generates the "Report" docs page from the latest published GitHub Release.

The page opens with an introduction under its title (what the report is, and a table of its simulations), then the
detection report: a pre-computed table of the expected detections of every transient type, with
its overlap with ZTF and Rubin, computed from the primary run's ``event_summary.ecsv`` with
`~uvex_transients.simulation.rates.estimate_yield`. It ends with one card per simulation run (see
``_report_runs.py``), each linking to a gallery of that run's analyses built from
``report_examples/<run>/``. Those examples download the run's release data through
`~uvex_transients.utils.get_results` and do whatever analysis they like with it.
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
    "llgrb": "Low-Luminosity GRB",
    "ia_sne": "Type Ia SNe",
    "iip_sne": "Type IIP SNe",
    "iip_excess_sne": "Type IIP SNe (Excess)",
    "iib_sne": "Type IIb SNe",
    "ib_sne": "Type Ib SNe",
    "ic_sne": "Type Ic SNe",
    "icbl_sne": "Type Ic-BL SNe",
    "slsne": "SLSNe (Magnetar)",
}


# The docs page under ``transients/`` that documents each type, for the links in the detection table.
_TRANSIENT_PAGES = {
    "kilonova": "kilonovae",
    "tde": "tdes",
    "lfbot": "lfbots",
    "llgrb": "llgrb",
    "ia_sne": "type_i",
    "ib_sne": "type_i",
    "ic_sne": "type_i",
    "icbl_sne": "type_i",
    "iip_sne": "type_ii",
    "iip_excess_sne": "type_ii",
    "iib_sne": "type_ii",
    "slsne": "slsne",
}

# The event summary's footprint-membership columns that the overlap columns are read from.
_ZTF_COLUMN = "in_ztf_main"
_RUBIN_COLUMN = "in_lsst_combined"


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


def _overlap(events: QTable, transient_type: str, column: str) -> str:
    """Percent of `transient_type`'s detected events inside the footprint `column` flags, ``--`` if not recorded."""
    if column not in events.colnames:
        return "--"
    inside = np.asarray(events[column], dtype=bool)[np.asarray(events["transient_type"]).astype(str) == transient_type]
    return f"{100 * inside.mean():.1f}%" if inside.size else "--"


def _summary_table_rst(yields: QTable, events: QTable) -> str:
    """Render `estimate_yield`'s per-type table as an RST ``list-table`` with ZTF and Rubin overlap."""
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 34 38 14 14",
        "",
        "   * - Transient Type",
        "     - Expected Detections",
        "     - % ZTF Overlap",
        "     - % Rubin Overlap",
    ]
    for row in yields:
        key = str(row["transient_type"])
        page = _TRANSIENT_PAGES.get(key)
        lines.append(f"   * - {f':doc:`{_display_name(key)} <../transients/{page}>`' if page else _display_name(key)}")
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
        lines.append(f"     - {_overlap(events, key, _ZTF_COLUMN)}")
        lines.append(f"     - {_overlap(events, key, _RUBIN_COLUMN)}")
    return "\n".join(lines)


# ----------------------------------------- #
# Run Table and Cards                        #
# ----------------------------------------- #
def _indent(text: str, spaces: int) -> str:
    return "\n".join(" " * spaces + line if line else line for line in text.splitlines())


def _runs_table_rst(runs: list[RunSpec]) -> str:
    """Render the intro's table of simulations, with the default (primary) run in bold."""
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 22 58 20",
        "",
        "   * - Simulation",
        "     - Description",
        "     - Analyses",
    ]
    for run in runs:
        count = run.n_analyses
        lines.append(f"   * - {f'**{run.title}**' if run.is_primary else run.title}")
        lines.append(f"     - {run.summary}")
        lines.append(f"     - :doc:`{count} {'analysis' if count == 1 else 'analyses'} <{run.key}/index>`")
    return "\n".join(lines)


def _run_card_rst(run: RunSpec) -> str:
    """Render one run as a sphinx-design ``grid-item-card`` linking to its gallery's index."""
    badges = " ".join(f":bdg-secondary-line:`{tag}`" for tag in run.tags)
    footer = ":bdg-primary:`Default run` · " if run.is_primary else ""
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
    """Render the "Simulations" section: a grid of run cards, and a hidden toctree for the sidebar."""
    if not runs:
        return ""
    cards = "\n\n".join(_run_card_rst(run) for run in runs)
    toctree = "\n".join(f"   {run.key}/index" for run in runs)
    return f"""\
Simulations
-----------

Each **simulation** is its own run of the pipeline, built for one science case: its own transient classes,
parameter priors, sampling and selection cuts, and therefore its own data products. Each card below opens a
gallery of the analyses done on that simulation. The examples download its data from the latest release with
:func:`~uvex_transients.utils.results.get_results` (cached after the first download), and the galleries are
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
def _overview_rst(runs: list[RunSpec], release_note: str) -> str:
    """Render the page header (logo, title), the introduction under it, the contents, and the table of simulations."""
    runs_section = (
        f"""\
Available Simulations
---------------------

The TYR contains several **simulations**, each with its own transient populations, selection and
data products. The default is the **Standard Run**, which covers every registered transient class at
full scale; the others each target one science case.

{_runs_table_rst(runs)}

"""
        if runs
        else ""
    )
    return f"""\
.. raw:: html

    <p align="center">
      <img src="../_static/uvex_logo_dark.png" width="400" alt="uvex_logo">
    </p>

.. _report_page:

.. rst-class:: report-title

Transient Yield Report
=======================

The **UVEX Transient Yield Report** (TYR) is the expected transient science of the UVEX survey: how many
events of each type UVEX should detect, given the *latest published release* of the survey schedule, the
transient SED models and their volumetric rates. Every number on these pages comes from the data products
of that release, and the pages are regenerated when the documentation is rebuilt.

.. important::

   The TYR is a **first look** at the simulated data, not a final analysis. The numbers will change as the
   survey schedule, the transient models and the rates are refined. Detailed analyses, such as follow-up
   planning, classification or science-case-specific selections, are up to the user and the UVEX team,
   starting from the released data products.

{release_note}

.. seealso::

   - :ref:`user_guide_simulation` for how a simulation is built, from sampling events to the cuts and
     actions applied to them, and :ref:`user_guide_cli` for the config files that define one.
   - :ref:`yield-statistics` for how the expected detections and their uncertainties are computed.
   - :ref:`transients` for the SED model, rate and priors of every transient type.
   - :func:`~uvex_transients.utils.results.get_results` for downloading a release's data products.

.. contents:: On this page
   :local:
   :depth: 1

{runs_section}"""


def _placeholder_rst(runs: list[RunSpec]) -> str:
    note = """\
.. note::

   No published release report is available yet (or it could not be fetched during this
   documentation build, see the build log for details). Once a ``vX.Y.Z`` release with
   ``event_summary.ecsv`` attached is published, this page will be generated automatically on
   the next documentation build.
"""
    return _overview_rst(runs, note) + _runs_rst(runs)


def _report_rst(tag: str, release_url: str, yields: QTable, events: QTable, runs: list[RunSpec]) -> str:
    confidence = int(round(100 * yields.meta["confidence"]))
    note = f"""\
.. note::

   This page is generated from the latest published release, `{tag} <{release_url}>`__, specifically
   its ``event_summary.ecsv`` data product. Every data product (the event catalog, photometry,
   exposure, and per-event summary) is attached to that release on GitHub, and earlier releases are
   listed on the `releases page <{RELEASES_URL}>`__.
"""
    return (
        _overview_rst(runs, note)
        + f"""\
Transient Detection Report
--------------------------

The **0th-order** result of the TYR is the number of transients UVEX *detects* over its 2-year prime
mission. Here, an event is **detected** if the survey observes its position after it explodes and, in at
least one observation and in at least one band (FUV or NUV), its signal-to-noise ratio exceeds 5. An
event whose only such observation is the first-ever visit to that field is not counted, since there is no
reference image to detect it against. A detection says only that UVEX *sees* the event: it does not mean
the event can be characterized or classified, which needs more observations and is a subject for
downstream analysis.

The table gives, for each transient type of the Standard Run, the expected number of real events that
UVEX detects. The first uncertainty is the {confidence}% Clopper-Pearson interval from the finite
Monte Carlo sample, and the second is the uncertainty on the event rate. The overlap columns are the
percentage of those detected events whose sky position lies inside the footprint of ZTF
(Palomar-visible sky) or Rubin (the LSST main survey and deep drilling fields). They are sky overlap
only, and do not account for whether those surveys observed the event at the right time.

{_summary_table_rst(yields, events)}

"""
        + _runs_rst(runs)
    )


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
    (report_dir / "index.rst").write_text(_report_rst(tag, release_url, yields, events, runs))
    print(f"[report] generated report/index.rst from release {tag}.")
