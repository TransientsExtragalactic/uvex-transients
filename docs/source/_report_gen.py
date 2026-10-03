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
import matplotlib.ticker
import numpy as np
from astropy.table import QTable
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from uvex_transients.utils.plotting import get_categorical_colors, resolve_fig_axes, set_plot_style

GITHUB_REPO = "TransientsExtragalactic/uvex-transients"
RELEASES_URL = f"https://github.com/{GITHUB_REPO}/releases"
ASSET_NAMES = ("yield_summary.ecsv", "detection_counts.ecsv")

# Muted gridline/text color (a mid-gray legible on both the light and dark docs themes). The series
# color is the first of ``config["plotting.categorical_colors"]`` (one series per figure, so no
# further categorical assignment is needed).
_MUTED = "#8a8a86"

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


def _fetch_latest_release(cache_dir: Path) -> tuple[str, str, dict[str, Path]] | None:
    """Download the newest release's report assets into `cache_dir`, or `None` on any failure.

    On success returns ``(tag, release_url, paths)``, where `release_url` is the release's GitHub page.

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
    release_url = release.get("html_url") or f"{RELEASES_URL}/tag/{tag}"
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
    return tag, release_url, paths


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


def _double_asym_math(
    value: float,
    binom_lower: float,
    binom_upper: float,
    rate_lower: float,
    rate_upper: float,
    precision: int = 3,
) -> str:
    """Render ``value`` with both stacked asymmetric intervals (binomial, then rate) as an RST ``:math:`` role."""
    if np.isnan(value):
        return "--"
    return (
        f":math:`{_fmt(value, precision)}"
        f"^{{+{_fmt(binom_upper - value, precision)}}}_{{-{_fmt(value - binom_lower, precision)}}}"
        f"\\,{{}}^{{+{_fmt(rate_upper - value, precision)}}}_{{-{_fmt(value - rate_lower, precision)}}}`"
    )


def _summary_table_rst(yield_table: QTable) -> str:
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 26 37 37",
        "",
        "   * - Transient Type",
        "     - Detection Probability",
        "     - Expected Detections",
    ]
    for row in yield_table:
        lines.append(f"   * - {_display_name(str(row['transient_type']))}")
        lines.append(
            "     - "
            + _double_asym_math(
                float(row["detection_probability"]),
                float(row["detection_probability_binom_lower"]),
                float(row["detection_probability_binom_upper"]),
                float(row["detection_probability_rate_lower"]),
                float(row["detection_probability_rate_upper"]),
            )
        )
        lines.append(
            "     - "
            + _double_asym_math(
                float(row["expected_detections"]),
                float(row["expected_detections_binom_lower"]),
                float(row["expected_detections_binom_upper"]),
                float(row["expected_detections_rate_lower"]),
                float(row["expected_detections_rate_upper"]),
            )
        )
    return "\n".join(lines)


# ----------------------------------------- #
# Plotting                                   #
# ----------------------------------------- #
def _plot_detection_curve(sub_table: QTable, out_path: Path, title: str) -> bool:
    """Plot expected events with >= k detected epochs vs. k, log-log, as a line with two uncertainty bands.

    Returns whether anything was drawn.
    """
    k = np.asarray(sub_table["n_detections"])
    mask = k >= 1
    if not mask.any():
        return False

    k = k[mask]
    expected = np.asarray(sub_table["expected_events"], dtype=float)[mask]
    binom_lower = np.asarray(sub_table["expected_events_binom_lower"], dtype=float)[mask]
    binom_upper = np.asarray(sub_table["expected_events_binom_upper"], dtype=float)[mask]
    rate_lower = np.asarray(sub_table["expected_events_rate_lower"], dtype=float)[mask]
    rate_upper = np.asarray(sub_table["expected_events_rate_upper"], dtype=float)[mask]

    # A rate lower bound can be non-positive (e.g. a published error larger than the rate itself),
    # which a log axis cannot show. Pin the axis floor just under the smallest positive value in
    # view and clip such bounds below it, so the band simply runs off the bottom of the axes.
    lows = np.concatenate([expected, binom_lower, rate_lower])
    y_floor = lows[lows > 0].min() / 1.5
    binom_lower = np.where(binom_lower > 0, binom_lower, y_floor / 10)
    rate_lower = np.where(rate_lower > 0, rate_lower, y_floor / 10)

    set_plot_style()
    series_color = get_categorical_colors(1)[0]
    fig, ax = resolve_fig_axes(fig_size=(5.5, 3.8), dpi=150)

    # Wider, paler rate-uncertainty band behind the binomial (Clopper-Pearson) band, with a thin
    # outline so its edges stay readable where the two bands overlap.
    ax.fill_between(k, rate_lower, rate_upper, facecolor=to_rgba(series_color, 0.18), edgecolor="none", zorder=1)
    for bound in (rate_lower, rate_upper):
        ax.plot(k, bound, color=series_color, alpha=0.55, linewidth=0.8, zorder=1)
    ax.fill_between(k, binom_lower, binom_upper, facecolor=to_rgba(series_color, 0.45), edgecolor="none", zorder=2)
    ax.plot(k, expected, color=series_color, linewidth=2.0, zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")

    # Decade majors labeled as plain integers (1, 10, 100), with unlabeled 2-9 minors: ticking
    # every integer k piles the labels on top of each other once k spans more than a decade or so.
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=10))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(k.min(), k.max())
    ax.set_ylim(bottom=y_floor, top=max(rate_upper.max(), binom_upper.max()) * 1.5)

    ax.set_xlabel(r"Detected epochs $N_\mathrm{det} \geq k$")
    ax.set_ylabel("Expected UVEX events")
    ax.set_title(title, fontsize=11, color="#3a3a38")
    for spine in ax.spines.values():
        spine.set_color(_MUTED)
    ax.tick_params(which="both", colors=_MUTED, top=True, right=True)
    ax.grid(which="major", color=_MUTED, alpha=0.3, linewidth=0.8)
    ax.grid(which="minor", axis="x", color=_MUTED, alpha=0.12, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            Line2D([], [], color=series_color, linewidth=2.0, label="Expected"),
            Patch(facecolor=to_rgba(series_color, 0.45), label="MC (statistical) uncertainty"),
            Patch(
                facecolor=to_rgba(series_color, 0.18), edgecolor=to_rgba(series_color, 0.55), label="Rate uncertainty"
            ),
        ],
        loc="upper right",
        fontsize=8,
        frameon=False,
        labelcolor=_MUTED,
    )
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


def _write_report(
    report_dir: Path, images_dir: Path, tag: str, release_url: str, yield_table: QTable, detection_table: QTable
) -> None:
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
   `{tag} <{release_url}>`__, of UVEX Transients -- specifically its ``yield_summary.ecsv`` and
   ``detection_counts.ecsv`` data products (see :doc:`../user_guide`). It is
   regenerated on every documentation build, so it may lag between releases.

.. seealso::

   Every data product from this run (the full event catalog, photometry, exposure, and the
   tables behind this page) is attached to the `{tag} release page <{release_url}>`__ on
   GitHub, where you can download whichever you need. Earlier releases are listed on the
   `releases page <{RELEASES_URL}>`__.

Summary
=======

{_summary_table_rst(yield_table)}

Detections vs. Threshold
=========================

For each transient type, the expected number of UVEX events detected in at
least :math:`k` SNR-qualifying epochs, as :math:`k` increases. The darker
band is the 90% Clopper-Pearson simulation-only uncertainty; the paler,
outlined band is the event-rate uncertainty.

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

        tag, release_url, paths = result
        try:
            yield_table = QTable.read(paths["yield_summary.ecsv"])
            detection_table = QTable.read(paths["detection_counts.ecsv"])
        except Exception as exc:  # pragma: no cover - defensive against a malformed release asset
            print(f"[report] could not parse release {tag}'s report assets ({exc}); skipping report generation.")
            _write_placeholder(report_dir)
            return

        _write_report(report_dir, report_dir / "_images", tag, release_url, yield_table, detection_table)
        print(f"[report] generated report/index.rst from release {tag}.")
