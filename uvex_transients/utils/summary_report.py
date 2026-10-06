r"""
Standard per-population report on an event summary table, the one a release's ``event_summary.ecsv`` holds.

A table from `~uvex_transients.simulation.core.SurveySimulator.run_event_summary_action` already
carries everything needed to say how many events UVEX should find and what they look like: one row
per event that cleared the detection screens, plus, in its ``meta``, the generated count, intrinsic
expected count and rate uncertainty of every population. `SummaryReport` applies a science
selection to those rows (repeat detections, a bright peak, and a tightly bracketed explosion time)
and produces the figures and tables that go with it:

- the **detection funnel**, `SummaryReport.plot_funnel`,
- the **expected yield** per population, `SummaryReport.yield_table`,
- the **distributions** of the properties behind each cut, `SummaryReport.plot_distributions`,
- the **sky positions** of the selected events, `SummaryReport.plot_sky`,
- one reconstructed **example light curve**, `SummaryReport.plot_example_event`.

Everything but the example light curve reads only the summary table, so `SummaryReport.from_release`
builds a full report from nothing but the published release.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
from astropy.table import QTable, Table
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ..simulation.rates import estimate_yield
from .plotting import (
    add_funnel_legend,
    compute_funnel_bounds,
    get_band_color,
    get_categorical_colors,
    plot_band_light_curve,
    plot_detection_funnel,
)
from .results import get_results

if TYPE_CHECKING:
    from m4opt.missions import Mission

    from ..simulation.event_catalog import EventCatalog
    from ..surveys.base import SurveySchedule
    from ..transients.base import TransientBase

__all__ = ["PopulationSpec", "SummaryReport"]

REQUIRED_COLUMNS = (
    "event_id",
    "transient_type",
    "redshift",
    "coord",
    "weight",
    "n_det",
    "t_explosion",
    "t_first_det",
    "t_last_constraining_nondet",
    "peak_mag",
)
"""tuple of str: The columns of the summary table that `SummaryReport` reads."""

REQUIRED_META = ("n_pre_cut", "expected_events", "rate_ci")
"""tuple of str: The ``meta`` entries of the summary table that `SummaryReport` reads."""


@dataclass(frozen=True)
class PopulationSpec:
    """How `SummaryReport` presents and selects one transient population."""

    label: str
    """str: Display name, used in legends and tables."""

    max_gap: float
    """float: Longest allowed gap, in days, between the last constraining non-detection and the first detection.

    The selection keeps an event only if its explosion time is bracketed more tightly than this,
    roughly the time the event takes to rise.
    """

    color: str | None = None
    """str or None: Matplotlib color for the population.

    The default takes the next color of ``config["plotting.categorical_colors"]``, in the order the
    populations were given.
    """


def _filled(column) -> NDArray[np.float64]:
    """Convert a masked column to a plain float array, with masked entries as NaN so that they fail every comparison."""
    return np.ma.filled(np.ma.asarray(column, dtype=float), np.nan)


def _fmt(x: float) -> str:
    """Format a count with thousands separators, or 3 significant figures when small."""
    return f"{x:,.0f}" if x >= 100 else f"{x:.3g}"


def _make_bins(values: NDArray[np.float64], log: bool = False, integer: bool = False, n: int = 25) -> NDArray:
    """Histogram bin edges spanning `values`: log-spaced for `log`, whole numbers for `integer`."""
    lo, hi = values.min(), values.max()
    if integer:
        return np.unique(np.round(np.geomspace(1, hi + 1, n)))
    if log:
        lo = max(lo, 1e-3)
        return np.geomspace(lo, max(hi, 2 * lo), n)
    return np.linspace(lo, hi, n) if hi > lo else np.linspace(lo - 0.5, hi + 0.5, n)


class SummaryReport:
    r"""
    The science selection, yields and standard figures for the populations in an event summary table.

    The selection is three cumulative cuts on the table's rows:

    1. **Repeat detections**: ``n_det`` is at least `min_detections`. A lone detection is hard to
       tell apart from noise or an artifact.
    2. **Bright peak**: ``peak_mag`` is below `peak_mag_limit`.
    3. **Bracketed explosion**: ``t_first_det - t_last_constraining_nondet`` is at most the
       population's `PopulationSpec.max_gap`, so that the explosion time is pinned down. An event
       with no constraining non-detection has no bracket and is dropped.

    Every count is converted to **expected UVEX events** with the table's own bookkeeping: one
    Monte Carlo event stands for :math:`\mu_0 / n` real ones, with :math:`n` the number generated
    and :math:`\mu_0` the intrinsic expected number (see
    `~uvex_transients.simulation.rates.estimate_yield`).

    Parameters
    ----------
    summary : ~astropy.table.QTable
        A table from
        `~uvex_transients.simulation.core.SurveySimulator.run_event_summary_action`, such as the
        one a release publishes. It needs the columns in `REQUIRED_COLUMNS` and the ``meta``
        entries in `REQUIRED_META`.
    populations : Mapping of str to PopulationSpec
        The populations to report, keyed by the ``transient_type`` they have in `summary`. Any
        that `summary` did not generate are left out, and every other type in it is ignored.
    min_detections : int, optional
        Fewest detected epochs an event may have. The default is 2.
    peak_mag_limit : float, optional
        The peak AB magnitude must be below this. The default is 23.
    snr_threshold : float, optional
        The detection threshold of `summary`, used only to label figures. The default is the
        ``snr_threshold`` in its ``meta``, or 5 if it has none.

    Attributes
    ----------
    summary : ~astropy.table.QTable
        The table, as given.
    populations : dict of str to PopulationSpec
        The populations being reported, each with its color filled in.
    repeated, bright, selected : numpy.ndarray of bool
        The cumulative cuts, one entry per row of `summary`: the first alone, the first two, and
        all three.

    Raises
    ------
    ValueError
        If `summary` lacks a required column or ``meta`` entry.
    LookupError
        If `summary` has none of the requested populations.
    """

    STAGE_NAMES = (
        "Generated",
        "Detected",
        "Repeat detections",
        "Bright peak",
        "Explosion bracketed",
    )
    """tuple of str: The stages of the detection funnel, in order. Each is a subset of the one before."""

    def __init__(
        self,
        summary: QTable,
        populations: Mapping[str, PopulationSpec],
        min_detections: int = 2,
        peak_mag_limit: float = 23.0,
        snr_threshold: float | None = None,
    ):
        missing = [name for name in REQUIRED_COLUMNS if name not in summary.colnames]
        missing += [f"meta[{name!r}]" for name in REQUIRED_META if name not in summary.meta]
        if missing:
            raise ValueError(f"The summary table lacks {missing}; it must come from run_event_summary_action.")

        present = [key for key in populations if key in summary.meta["n_pre_cut"]]
        if not present:
            raise LookupError(
                f"The summary table has none of the populations {list(populations)}; "
                f"it has {list(summary.meta['n_pre_cut'])}."
            )
        colors = iter(get_categorical_colors(len(present)))
        self.populations = {
            key: PopulationSpec(
                populations[key].label, populations[key].max_gap, populations[key].color or next(colors)
            )
            for key in present
        }

        self.summary = summary
        self.min_detections = int(min_detections)
        self.peak_mag_limit = float(peak_mag_limit)
        self.snr_threshold = float(
            snr_threshold if snr_threshold is not None else summary.meta.get("snr_threshold", 5.0)
        )

        self.types = np.asarray(summary["transient_type"]).astype(str)
        self.n_det = np.asarray(summary["n_det"])
        self.peak_mag = _filled(summary["peak_mag"])
        self.redshift = _filled(summary["redshift"])
        self.weight = _filled(summary["weight"])
        # Days from the last constraining non-detection to the first detection, NaN where there is none.
        self.gap = np.ma.filled(
            (summary["t_first_det"] - summary["t_last_constraining_nondet"]).to_value(u.day), np.nan
        )
        # Days from the explosion to the first detection (observer frame).
        self.t_first = np.ma.filled((summary["t_first_det"] - summary["t_explosion"]).to_value(u.day), np.nan)
        max_gap = np.array([self.populations[k].max_gap if k in self.populations else np.nan for k in self.types])

        with np.errstate(invalid="ignore"):
            self.repeated = self.n_det >= self.min_detections
            self.bright = self.repeated & (self.peak_mag < self.peak_mag_limit)
            self.selected = self.bright & (self.gap <= max_gap)

    @classmethod
    def from_release(
        cls, populations: Mapping[str, PopulationSpec], tag: str | None = None, **kwargs
    ) -> "SummaryReport":
        """
        Build a report from the event summary table of a published release.

        The table is downloaded into the package cache with
        `~uvex_transients.utils.results.get_results`, so only the first call pays for the download.

        Parameters
        ----------
        populations : Mapping of str to PopulationSpec
            As for the constructor.
        tag : str, optional
            The release tag. The default is the newest release.
        **kwargs
            Forwarded to the constructor.

        Returns
        -------
        SummaryReport
            The report on the release's summary table.

        Raises
        ------
        ConnectionError
            If the GitHub releases API cannot be reached.
        LookupError
            If there is no such release, it has no event summary table (releases cut before the
            table was introduced do not), or the table has none of the requested populations.
        ValueError
            If the table lacks something `SummaryReport` needs, as an older release's might.
        """
        path = get_results(["summary"], tag=tag)["summary"]
        return cls(QTable.read(path), populations, **kwargs)

    def _in(self, key: str) -> NDArray[np.bool_]:
        """Mask of the rows of population `key`."""
        return self.types == key

    @property
    def stage_masks(self) -> list[NDArray[np.bool_]]:
        """list[numpy.ndarray]: The row mask of each funnel stage after the first, as `STAGE_NAMES` lists them."""
        return [np.ones(len(self.summary), dtype=bool), self.repeated, self.bright, self.selected]

    def stage_counts(self) -> dict[str, list[int]]:
        """
        Count the Monte Carlo events of each population that reach every stage of the funnel.

        Returns
        -------
        dict of str to list of int
            For each population, the counts at the stages of `STAGE_NAMES`. The first is the number
            generated, before any cut, and the rest are rows of the summary table.
        """
        return {
            key: [int(self.summary.meta["n_pre_cut"][key])]
            + [int(np.sum(self._in(key) & mask)) for mask in self.stage_masks]
            for key in self.populations
        }

    def yields(self) -> QTable:
        """
        Estimate the expected number of selected events per population.

        Returns
        -------
        ~astropy.table.QTable
            `~uvex_transients.simulation.rates.estimate_yield` of the selection, one row per
            population (sorted by name, as that function does).
        """
        return estimate_yield(self.summary, mask=self.selected, transient_types=list(self.populations))

    def yield_table(self) -> Table:
        """
        Summarize the expected yield of every population as a table of display strings.

        Returns
        -------
        ~astropy.table.Table
            One row per population, in the order they were given: the intrinsic number of UVEX
            events, how many Monte Carlo events were selected out of those generated, the selected
            fraction with its confidence interval, and the expected number of selected events with
            its Monte Carlo and rate uncertainties. A population with nothing selected has only an upper limit.
        """
        yields = self.yields()
        ci = f"{yields.meta['confidence']:.0%}"
        by_type = {str(row["transient_type"]): row for row in yields}
        rows = []
        for key in self.populations:  # in the order the populations were given, not estimate_yield's
            row = by_type[key]
            expected = row["expected_events"]
            fraction = (
                f"{row['fraction']:.2%} ({row['fraction_lower']:.2%} to {row['fraction_upper']:.2%})"
                if row["n_pre_cut"]
                else "n/a"
            )
            if row["n_selected"] == 0:
                result = f"< {_fmt(row['expected_events_binom_upper'])} ({ci} upper limit)"
            else:
                up_mc = row["expected_events_binom_upper"] - expected
                lo_mc = expected - row["expected_events_binom_lower"]
                up_rate = row["expected_events_rate_upper"] - expected
                lo_rate = expected - row["expected_events_rate_lower"]
                result = (
                    f"{_fmt(expected)} (+{_fmt(up_mc)} / -{_fmt(lo_mc)} MC, +{_fmt(up_rate)} / -{_fmt(lo_rate)} rate)"
                )
            rows.append(
                (
                    self.populations[str(row["transient_type"])].label,
                    _fmt(row["intrinsic_events"]),
                    f"{row['n_selected']} / {row['n_pre_cut']}",
                    fraction,
                    result,
                )
            )
        return Table(
            rows=rows,
            names=(
                "Population",
                "Intrinsic UVEX events",
                "Selected / generated",
                f"Fraction ({ci} CI)",
                "Expected selected events",
            ),
        )

    def plot_funnel(self, fig_size: tuple[float, float] = (10, 4.5)) -> Figure:
        r"""
        Plot how many events of each population survive each stage of the selection.

        The bars are expected UVEX events: the Monte Carlo count at a stage times :math:`\mu_0 / n`
        for the population, so the first bar is the intrinsic number :math:`\mu_0`. Black error bars
        are the Monte Carlo (Clopper-Pearson) uncertainty on each stage, and the pale band is the
        rate uncertainty, which scales every stage by the same factor.

        Parameters
        ----------
        fig_size : tuple of float, optional
            Figure size in inches. The default is ``(10, 4.5)``.

        Returns
        -------
        matplotlib.figure.Figure
            The sky map.
        """
        labels = [
            "Generated",
            f"Detected\n(SNR > {self.snr_threshold:g})",
            f"At least {self.min_detections}\ndetections",
            f"Peak brighter\nthan {self.peak_mag_limit:g} mag",
            "Explosion\nbracketed",
        ]
        meta = self.summary.meta
        stage_counts = self.stage_counts()
        fig, ax = plt.subplots(figsize=fig_size)
        bar_width = 0.8 / len(self.populations)
        positive: list[float] = []
        for i, (key, spec) in enumerate(self.populations.items()):
            raw = stage_counts[key]
            per_event = meta["expected_events"][key] / max(meta["n_pre_cut"][key], 1)
            mc_lower, mc_upper, rate_lower, rate_upper = compute_funnel_bounds(raw, rate_ci=meta["rate_ci"][key])
            counts = np.asarray(raw, dtype=float) * per_event
            x = np.arange(len(labels)) + (i - (len(self.populations) - 1) / 2) * bar_width
            plot_detection_funnel(
                ax,
                x=x,
                counts=counts,
                mc_lower=mc_lower * per_event,
                mc_upper=mc_upper * per_event,
                rate_lower=rate_lower * per_event,
                rate_upper=rate_upper * per_event,
                color=spec.color,
                width=bar_width,
                label=spec.label,
            )
            tops = np.maximum(rate_upper, mc_upper) * per_event
            positive += [v for v in np.concatenate([counts, tops]) if v > 0]
            for xi, count, top in zip(x, counts, tops):
                ax.text(xi, max(top, min(positive) / 3), _fmt(count), ha="center", va="bottom", fontsize=7)
        ax.set_xticks(np.arange(len(labels)), labels)
        ax.set_yscale("log")
        if positive:
            ax.set_ylim(min(positive) / 3, max(positive) * 30)  # headroom for the legend above the bars
        ax.set_ylabel("Expected UVEX events")
        ax.set_title("Detection funnel")
        add_funnel_legend(ax)
        fig.tight_layout()
        return fig

    def _panels(self) -> dict[str, tuple[str, NDArray[np.float64], bool, Callable[[str], float] | None]]:
        """Each distribution's axis label, values, whether it is log-scaled, and its cut threshold (per population)."""
        return {
            "redshift": ("Redshift", self.redshift, False, None),
            "peak_mag": ("Peak magnitude (AB)", self.peak_mag, False, lambda key: self.peak_mag_limit),
            "n_det": (
                f"Number of detections (SNR > {self.snr_threshold:g})",
                self.n_det.astype(float),
                True,
                lambda key: self.min_detections,
            ),
            "t_first": ("Days from explosion to first detection", self.t_first, False, None),
            "gap": (
                "Days from last constraining non-detection to first detection",
                self.gap,
                True,
                lambda key: self.populations[key].max_gap,
            ),
        }

    def _draw_panel(self, ax, quantity: str, key: str) -> None:
        """Weighted histogram of one quantity for one population, all detected events behind the selected ones."""
        xlabel, values, log_x, cut = self._panels()[quantity]
        usable = self._in(key) & np.isfinite(values)
        chosen = usable & self.selected
        ax.set_xlabel(xlabel)
        if not usable.any():
            ax.text(0.5, 0.5, "no events", transform=ax.transAxes, ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            return
        bins = _make_bins(values[usable], log=log_x, integer=quantity == "n_det")
        ax.hist(
            values[usable],
            bins=bins,
            weights=self.weight[usable],
            color="0.8",
            label=f"Detected ({_fmt(self.weight[usable].sum())})",
        )
        ax.hist(
            values[chosen],
            bins=bins,
            weights=self.weight[chosen],
            color=self.populations[key].color,
            label=f"Selected ({_fmt(self.weight[chosen].sum())})",
        )
        if cut is not None:
            ax.axvline(cut(key), color="k", ls="--", lw=1, label="Cut")
        ax.set_yscale("log")
        if log_x:
            ax.set_xscale("log")
        ax.legend(fontsize=7)

    def plot_distributions(self) -> list[Figure]:
        """
        Plot the distributions of the properties behind each cut, plus the redshift.

        Each event is weighted by its ``weight``, so a bin's height is an expected number of UVEX
        events and the legend entries are totals: the gray histogram is the expected number of events
        detected at all, and the colored one the expected yield. Dashed lines mark the cuts. Each
        panel shows a single cut, so some events on the surviving side of a line are still removed
        by one of the others.

        Returns
        -------
        list of matplotlib.figure.Figure
            One figure with a panel per quantity for a single population, otherwise one figure per
            quantity with a panel per population.
        """
        panels = self._panels()
        if len(self.populations) == 1:
            ((key, spec),) = self.populations.items()
            fig, axes = plt.subplots(2, 3, figsize=(13, 7))
            for ax, quantity in zip(axes.flat, panels):
                self._draw_panel(ax, quantity, key)
            for ax in axes[:, 0]:
                ax.set_ylabel("Expected UVEX events per bin")
            axes.flat[-1].axis("off")
            fig.suptitle(f"{spec.label}: distributions")
            fig.tight_layout()
            return [fig]

        figs = []
        for quantity in panels:
            fig, axes = plt.subplots(
                1, len(self.populations), figsize=(4.5 * len(self.populations), 3.6), sharey=True, squeeze=False
            )
            for ax, (key, spec) in zip(axes[0], self.populations.items()):
                self._draw_panel(ax, quantity, key)
                ax.set_title(spec.label)
            axes[0, 0].set_ylabel("Expected UVEX events per bin")
            fig.tight_layout()
            figs.append(fig)
        return figs

    def plot_sky(self) -> Figure:
        """
        Plot where the detected and the selected events fall on the sky, in an Aitoff projection.

        Returns
        -------
        matplotlib.figure.Figure
            The sky map.
        """
        coord = self.summary["coord"]
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111, projection="aitoff")
        ax.grid(True)
        detected = np.isin(self.types, list(self.populations))
        ax.scatter(
            coord.ra[detected].wrap_at(180 * u.deg).radian,
            coord.dec[detected].radian,
            s=2,
            alpha=0.2,
            color="#888888",
            label=f"Detected ({int(detected.sum()):,})",
        )
        for key, spec in self.populations.items():
            chosen = self._in(key) & self.selected
            ax.scatter(
                coord.ra[chosen].wrap_at(180 * u.deg).radian,
                coord.dec[chosen].radian,
                s=8,
                color=spec.color,
                label=f"{spec.label} selected ({int(chosen.sum()):,})",
            )
        ax.legend(loc="lower right", markerscale=2, fontsize=7)
        ax.set_title("Sky distribution (Monte Carlo events)")
        fig.tight_layout()
        return fig

    def plot_example_event(
        self,
        key: str,
        events: "EventCatalog",
        transients: Mapping[str, "TransientBase"],
        schedule: "SurveySchedule",
        mission: "Mission",
        seed: int = 1,
    ) -> Figure:
        """
        Plot the synthetic light curve of one randomly chosen selected event of a population.

        The event is rebuilt from `events` (see `~uvex_transients.simulation.event_catalog.EventCatalog.get_events`)
        and its photometry simulated against every observation `schedule` made of it. Filled
        squares are detections, open triangles upper limits, and the lines are the noiseless model.

        Parameters
        ----------
        key : str
            The population, a key of `populations`.
        events : ~uvex_transients.simulation.event_catalog.EventCatalog
            The event catalog the summary was built from, as a release publishes it. It must hold
            the event.
        transients : Mapping of str to TransientBase
            The transient instances, keyed as the ``transient_type`` column is.
        schedule : ~uvex_transients.surveys.base.SurveySchedule
            The survey schedule, ideally the one the release was run against.
        mission : m4opt.missions.Mission
            The mission to simulate the photometry for.
        seed : int, optional
            Seed of the draw of the event. The default is 1.

        Returns
        -------
        matplotlib.figure.Figure
            The light curve.

        Raises
        ------
        ValueError
            If no event of the population survived the selection.
        """
        ids = np.asarray(self.summary["event_id"])[self._in(key) & self.selected]
        if len(ids) == 0:
            raise ValueError(f"No {self.populations[key].label} events survived the selection.")
        event = events.get_events(int(np.random.default_rng(seed).choice(ids)), dict(transients), schedule)

        phot = event.simulate_photometry(mission)
        t_since_explosion = (phot["obs_time"] - event.t_explosion).to(u.day)
        t_theory = np.linspace(0, transients[key].duration_limit.to_value(u.day), 300) * u.day

        fig, ax = plt.subplots(figsize=(7, 4))
        for band in np.unique(np.asarray(phot["band"])):
            plot_band_light_curve(
                ax,
                str(band),
                t_since_explosion,
                phot,
                t_theory=t_theory,
                theory_mag=event.mag(t_theory, mission, band=str(band)),
                snr_threshold=self.snr_threshold,
                color=get_band_color(str(band)),
                err_scale=5.0,
                label=str(band),
            )
        ax.invert_yaxis()
        ax.set_xlabel("Days since explosion")
        ax.set_ylabel("AB magnitude")
        label = self.populations[key].label
        ax.set_title(f"{label} event {event.event_id} (z={event.redshift:.3f}, {event.n_observations} observations)")
        ax.legend()
        fig.tight_layout()
        return fig
