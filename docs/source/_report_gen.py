"""
Generates the "Report" docs page from the latest published GitHub Release's
``yield_summary.ecsv``/``detection_counts.ecsv`` assets (see
``configs/full_run.yaml`` and ``.github/workflows/build_and_release.yml``, which
produce and attach them).

Hooked into the Sphinx build from ``conf.py``'s ``setup()``, on the
``builder-inited`` event -- fires before Sphinx reads any source files, so the
generated ``report/index.rst`` and its figures exist in time to be picked up by
the normal build, both in CI (``build_documentation.yml``) and in a local
``make html``. Never fails the build: a missing release, an unreachable GitHub
API, or a release without the expected assets all fall back to a placeholder
page instead of raising (mirroring ``check_switcher = False`` in ``conf.py`` for
the version switcher, which has the same "don't have network/a match" problem).
"""

import json
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable

GITHUB_REPO = "TransientsExtragalactic/uvex-transients"
ASSET_NAMES = ("yield_summary.ecsv", "detection_counts.ecsv")

# Single-hue series color + muted gridline/text color, from the validated
# reference palette (categorical slot 1 / a mid-gray) -- one series per figure,
# so no categorical assignment is needed.
_SERIES_COLOR = "#2a78d6"
_MUTED = "#8a8a86"

_DISPLAY_NAMES = {
    "kilonova": "Kilonova",
    "tde": "Tidal Disruption Event",
    "lfbot": "LFBOT",
    "iip_sne": "Type IIP SNe",
    "iip_excess_sne": "Type IIP SNe (Excess)",
    "iib_sne": "Type IIb SNe",
    "ib_sne": "Type Ib SNe",
    "ic_sne": "Type Ic SNe",
    "slsne": "SLSNe (Magnetar)",
}


def _display_name(key: str) -> str:
    return _DISPLAY_NAMES.get(key, key.replace("_", " ").title())


# ----------------------------------------- #
# GitHub Release Fetching                    #
# ----------------------------------------- #
def _headers() -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _api_get(url: str) -> dict:
    request = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def _download(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(request, timeout=30) as response:
        dest.write_bytes(response.read())


def _fetch_latest_release(cache_dir: Path) -> tuple[str, dict[str, Path]] | None:
    """Download the newest release's report assets into `cache_dir`, or `None` on any failure.

    Lists releases (`GET /releases`, newest first) rather than using the `/releases/latest`
    endpoint, since that endpoint explicitly excludes prereleases -- and every UVEX Transients
    release so far (e.g. ``v0.0.2alpha``) is one, per `setuptools-scm`'s ``vX.Y.Zalpha`` tagging.
    """
    try:
        releases = _api_get(f"https://api.github.com/repos/{GITHUB_REPO}/releases")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        print(f"[report] could not reach the GitHub releases API ({exc}); skipping report generation.")
        return None

    release = next((r for r in releases if not r.get("draft", False)), None)
    if release is None:
        print("[report] no published releases found; skipping report generation.")
        return None

    tag = release.get("tag_name", "unknown")
    assets = {asset["name"]: asset["browser_download_url"] for asset in release.get("assets", [])}
    missing = [name for name in ASSET_NAMES if name not in assets]
    if missing:
        print(f"[report] release {tag} is missing asset(s) {missing}; skipping report generation.")
        return None

    paths = {}
    for name in ASSET_NAMES:
        dest = cache_dir / name
        try:
            _download(assets[name], dest)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"[report] failed to download {name} ({exc}); skipping report generation.")
            return None
        paths[name] = dest
    return tag, paths


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
    """Render ``value`` with an asymmetric interval as an RST ``:math:`` role."""
    if np.isnan(value):
        return "--"
    return (
        f":math:`{_fmt(value, precision)}^{{+{_fmt(upper - value, precision)}}}_{{-{_fmt(value - lower, precision)}}}`"
    )


def _summary_table_rst(yield_table: QTable) -> str:
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 24 16 22 22",
        "",
        "   * - Transient Type",
        "     - Detected Events",
        "     - Detection Probability",
        "     - Expected Detections",
    ]
    for row in yield_table:
        lines.append(f"   * - {_display_name(str(row['transient_type']))}")
        lines.append(f"     - {int(row['detected_events'])}")
        lines.append(
            "     - "
            + _asym_math(
                float(row["detection_probability"]),
                float(row["detection_probability_binom_lower"]),
                float(row["detection_probability_binom_upper"]),
            )
        )
        lines.append(
            "     - "
            + _asym_math(
                float(row["expected_detections"]),
                float(row["expected_detections_binom_lower"]),
                float(row["expected_detections_binom_upper"]),
            )
        )
    return "\n".join(lines)


# ----------------------------------------- #
# Plotting                                   #
# ----------------------------------------- #
def _plot_detection_curve(sub_table: QTable, out_path: Path, title: str) -> bool:
    """Plot expected events with >= k detected epochs vs. k; returns whether anything was drawn."""
    k = np.asarray(sub_table["n_detections"])
    mask = k >= 1
    if not mask.any():
        return False

    k = k[mask]
    expected = np.asarray(sub_table["expected_events"], dtype=float)[mask]
    lower = np.asarray(sub_table["expected_events_binom_lower"], dtype=float)[mask]
    upper = np.asarray(sub_table["expected_events_binom_upper"], dtype=float)[mask]
    yerr = np.vstack([expected - lower, upper - expected])

    fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=150)
    ax.errorbar(
        k,
        expected,
        yerr=yerr,
        fmt="o-",
        color=_SERIES_COLOR,
        ecolor=_SERIES_COLOR,
        linewidth=2,
        markersize=7,
        capsize=3,
        elinewidth=1.5,
    )
    ax.set_xlabel(r"Detected epochs $N_\mathrm{det} \geq k$")
    ax.set_ylabel("Expected UVEX events")
    ax.set_title(title, fontsize=11, color="#3a3a38")
    ax.set_xticks(k.astype(int))
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_MUTED)
    ax.spines["bottom"].set_color(_MUTED)
    ax.tick_params(colors=_MUTED)
    ax.grid(axis="y", color=_MUTED, alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, transparent=True)
    plt.close(fig)
    return True


# ----------------------------------------- #
# Page Assembly                              #
# ----------------------------------------- #
_PLACEHOLDER_RST = """\
Transient Yield Report
=======================

.. note::

   No published release report is available yet (or it could not be fetched
   during this documentation build -- see the build log for details). Once a
   ``vX.Y.Z`` release with ``yield_summary.ecsv``/``detection_counts.ecsv``
   attached is published, this page will be generated automatically on the
   next documentation build.
"""


def _write_placeholder(report_dir: Path) -> None:
    (report_dir / "index.rst").write_text(_PLACEHOLDER_RST)


def _write_report(report_dir: Path, images_dir: Path, tag: str, yield_table: QTable, detection_table: QTable) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)

    names = sorted(np.unique(np.asarray(detection_table["transient_type"]).astype(str)))
    tab_items = []
    for name in names:
        sub = detection_table[np.asarray(detection_table["transient_type"]).astype(str) == name]
        image_path = images_dir / f"{name}.png"
        if not _plot_detection_curve(sub, image_path, _display_name(name)):
            continue
        tab_items.append(
            f"   .. tab-item:: {_display_name(name)}\n\n      .. image:: _images/{name}.png\n         :width: 100%\n         :align: center\n"
        )

    tabs_rst = ".. tab-set::\n\n" + "\n".join(tab_items) if tab_items else "*No detection-count data available.*"

    rst = f"""\
Transient Yield Report
=======================

.. note::

   This page is generated automatically from the latest published release,
   **{tag}**, of UVEX Transients -- specifically its ``yield_summary.ecsv`` and
   ``detection_counts.ecsv`` data products (see :doc:`../user_guide`). It is
   regenerated on every documentation build, so it may lag between releases.

Summary
=======

{_summary_table_rst(yield_table)}

Detections vs. Threshold
=========================

For each transient type, the expected number of UVEX events detected in at
least :math:`k` SNR-qualifying epochs, as :math:`k` increases. Error bars are
90% Clopper-Pearson simulation-only uncertainty.

{tabs_rst}
"""
    (report_dir / "index.rst").write_text(rst)


def generate_report(app) -> None:
    """Sphinx ``builder-inited`` callback: (re)generate ``report/index.rst`` and its figures."""
    report_dir = Path(app.srcdir) / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        result = _fetch_latest_release(Path(tmp))
        if result is None:
            _write_placeholder(report_dir)
            return

        tag, paths = result
        try:
            yield_table = QTable.read(paths["yield_summary.ecsv"])
            detection_table = QTable.read(paths["detection_counts.ecsv"])
        except Exception as exc:  # pragma: no cover - defensive against a malformed release asset
            print(f"[report] could not parse release {tag}'s report assets ({exc}); skipping report generation.")
            _write_placeholder(report_dir)
            return

        _write_report(report_dir, report_dir / "_images", tag, yield_table, detection_table)
        print(f"[report] generated report/index.rst from release {tag}.")
