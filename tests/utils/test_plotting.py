"""Tests for `uvex_transients.utils.plotting`."""

import os
import subprocess
import sys

import matplotlib as mpl

mpl.use("Agg")

import astropy.units as u
import numpy as np
import pytest
from astropy.table import QTable
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

from uvex_transients.utils.config import config
from uvex_transients.utils.plotting import (
    add_funnel_legend,
    compute_funnel_bounds,
    get_band_color,
    get_cmap,
    get_default_cmap,
    plot_band_light_curve,
    plot_detection_funnel,
    plot_healpix_map,
    plot_histogram,
    plot_rate_bars,
    resolve_fig_axes,
    set_plot_style,
)


class TestResolveFigAxes:
    """Tests for `resolve_fig_axes`."""

    def test_creates_new_figure_and_axes(self):
        """With nothing given, a new figure/axes pair is created."""
        fig, ax = resolve_fig_axes()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        plt.close(fig)

    def test_uses_configured_default_figsize(self):
        """A newly created figure should use `config["plotting.default_figsize"]` by default."""
        fig, _ = resolve_fig_axes()
        assert tuple(fig.get_size_inches()) == tuple(config["plotting.default_figsize"])
        plt.close(fig)

    def test_explicit_fig_size_overrides_default(self):
        """An explicit `fig_size` should override the configured default."""
        fig, _ = resolve_fig_axes(fig_size=(3, 2))
        assert tuple(fig.get_size_inches()) == (3.0, 2.0)
        plt.close(fig)

    def test_reuses_existing_axes(self):
        """Passing existing axes should return them (and their parent figure) unchanged."""
        fig, ax = plt.subplots()
        out_fig, out_ax = resolve_fig_axes(axes=ax)
        assert out_ax is ax
        assert out_fig is fig
        plt.close(fig)

    def test_reuses_existing_figure_without_axes(self):
        """Passing only a figure should attach new axes to it via `Figure.gca`."""
        fig = plt.figure()
        out_fig, out_ax = resolve_fig_axes(fig=fig)
        assert out_fig is fig
        assert isinstance(out_ax, Axes)
        plt.close(fig)

    def test_subplot_kw_forwarded_for_new_axes(self):
        """`subplot_kw` should be forwarded to `~matplotlib.pyplot.subplots` for a new figure."""
        fig, ax = resolve_fig_axes(subplot_kw={"projection": "aitoff"})
        assert ax.name == "aitoff"
        plt.close(fig)


class TestSetPlotStyle:
    """Tests for `set_plot_style`."""

    def test_applies_configured_dpi_and_tick_direction(self):
        """`rcParams` should reflect the configured dpi and the shared inward tick convention."""
        set_plot_style()
        assert plt.rcParams["figure.dpi"] == config["plotting.dpi"]
        assert plt.rcParams["xtick.direction"] == "in"
        assert plt.rcParams["ytick.direction"] == "in"


class TestGetDefaultCmap:
    """Tests for `get_default_cmap`."""

    def test_returns_configured_colormap(self):
        """The returned colormap should be the one named by `config["plotting.default_cmap"]`."""
        assert get_default_cmap().name == config["plotting.default_cmap"]


class TestGetCmap:
    """Tests for `get_cmap`."""

    def test_resolves_string_name(self):
        """A colormap name should resolve to a `Colormap` instance of that name."""
        cmap = get_cmap("plasma")
        assert isinstance(cmap, Colormap)
        assert cmap.name == "plasma"

    def test_returns_colormap_instance_unchanged(self):
        """An already-resolved `Colormap` should be returned unchanged."""
        original = get_default_cmap()
        assert get_cmap(original) is original

    def test_unknown_name_raises_value_error(self):
        """An unregistered colormap name should raise `ValueError`."""
        with pytest.raises(ValueError, match="Unknown colormap"):
            get_cmap("not-a-real-colormap")

    def test_wrong_type_raises_type_error(self):
        """A non-str, non-Colormap argument should raise `TypeError`."""
        with pytest.raises(TypeError):
            get_cmap(1234)


class TestGetBandColor:
    """Tests for `get_band_color`."""

    def test_known_band_matches_config(self):
        """A band present in `config["plotting.band_colors"]` should return that exact color."""
        assert get_band_color("FUV") == config["plotting.band_colors"]["FUV"]

    def test_unknown_band_is_deterministic(self):
        """An unconfigured band name should still map to the same color on repeated calls."""
        assert get_band_color("made-up-band") == get_band_color("made-up-band")

    def test_unknown_band_is_deterministic_across_processes(self):
        """The fallback color must not depend on `hash()`'s per-process salt (PYTHONHASHSEED)."""
        script = "from uvex_transients.utils.plotting import get_band_color; print(get_band_color('made-up-band'))"
        outputs = {
            subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                check=True,
                env={**os.environ, "PYTHONHASHSEED": seed},
            ).stdout
            for seed in ("0", "1", "2")
        }
        assert len(outputs) == 1

    def test_different_unknown_bands_can_differ(self):
        """Different unconfigured band names are not guaranteed, but commonly do, map differently.

        Any single hardcoded pair could in principle collide in the finite-resolution colormap, so
        this checks that a modest set of arbitrary names doesn't *all* collapse onto one color,
        rather than asserting inequality for one specific pair.
        """
        colors = {get_band_color(f"unknown-band-{i}") for i in range(10)}
        assert len(colors) > 1


class TestPlotHealpixMap:
    """Tests for `plot_healpix_map`."""

    def test_runs_without_error(self):
        """A small full-sky map should plot without raising, onto an aitoff-projected axes."""
        rng = np.random.default_rng(0)
        nside = 4
        values = rng.lognormal(size=12 * nside**2)
        fig, ax = plot_healpix_map(values, nside=nside, title="t", cbar_label="c")
        assert ax.name == "aitoff"
        assert ax.get_title() == "t"
        plt.close(fig)

    def test_non_positive_values_are_dropped(self):
        """Zero/negative pixels should not break the log-scaled color normalization."""
        nside = 4
        values = np.zeros(12 * nside**2)
        values[0] = 1.0
        values[1] = 10.0
        fig, ax = plot_healpix_map(values, nside=nside)
        plt.close(fig)


class TestPlotHistogram:
    """Tests for `plot_histogram`."""

    def test_runs_without_error(self):
        """A strictly positive, log-spanning sample should histogram without raising."""
        rng = np.random.default_rng(0)
        values = rng.lognormal(size=200)
        fig, ax = plot_histogram(values, title="t", xlabel="x")
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"
        plt.close(fig)

    def test_drops_non_positive_and_non_finite_values(self):
        """Zero, negative, and non-finite entries should be dropped rather than raise."""
        values = np.array([1.0, 2.0, -1.0, 0.0, np.nan, np.inf, 3.0])
        fig, ax = plot_histogram(values)
        plt.close(fig)


class TestPlotBandLightCurve:
    """Tests for `plot_band_light_curve`."""

    def _phot_table(self):
        return QTable(
            {
                "band": ["FUV", "FUV", "FUV", "NUV"],
                "ab_mag": [22.0, 23.0, 25.0, 21.0],
                "snr": [10.0, 8.0, 2.0, 15.0],
                "mag_err": [0.05, 0.08, 0.2, 0.03],
                "mag_lower": [21.8, 22.7, 24.5, 20.8],
                "mag_upper": [22.2, 23.3, 25.8, 21.2],
            }
        )

    def test_draws_theory_curve_detections_and_upper_limits(self):
        """Detections should be errorbar points; sub-threshold visits, upper limits; onto one Axes."""
        phot = self._phot_table()
        t_obs = np.array([0.0, 1.0, 2.0, 0.0]) * u.day
        t_theory = np.linspace(0, 2, 10) * u.day
        theory_mag = np.linspace(22, 24, 10)

        fig, ax = plt.subplots()
        plot_band_light_curve(
            ax,
            "FUV",
            t_obs,
            phot,
            t_theory=t_theory,
            theory_mag=theory_mag,
            snr_threshold=5.0,
            label="FUV",
        )
        lines = ax.get_lines()
        # One theory curve; errorbar containers hold their own markers/lines separately.
        assert len(lines) >= 1
        assert len(ax.containers) == 2  # one for detections, one for upper limits
        plt.close(fig)

    def test_no_upper_limits_without_bounds_columns(self):
        """Without `mag_lower`/`mag_upper` columns, upper limits should be silently skipped."""
        phot = self._phot_table()
        phot.remove_columns(["mag_lower", "mag_upper"])
        t_obs = np.array([0.0, 1.0, 2.0, 0.0]) * u.day

        fig, ax = plt.subplots()
        plot_band_light_curve(ax, "FUV", t_obs, phot, snr_threshold=5.0)
        assert len(ax.containers) == 1  # detections only
        plt.close(fig)

    def test_defaults_to_band_color(self):
        """Without an explicit `color`, the detection points should use `get_band_color(band)`."""
        phot = self._phot_table()
        t_obs = np.array([0.0, 1.0, 2.0, 0.0]) * u.day

        fig, ax = plt.subplots()
        plot_band_light_curve(ax, "FUV", t_obs, phot, snr_threshold=5.0)
        from matplotlib.colors import to_rgba

        assert to_rgba(ax.containers[0][0].get_markerfacecolor()) == to_rgba(get_band_color("FUV"))
        plt.close(fig)


class TestComputeFunnelBounds:
    """Tests for `compute_funnel_bounds`."""

    def test_first_stage_upper_is_exact(self):
        """The first stage has k == n, so its Clopper-Pearson upper bound is the exact k=n case (1.0)."""
        mc_lower, mc_upper, rate_lower, rate_upper = compute_funnel_bounds([100, 40, 10])
        assert mc_upper[0] == 100

    def test_bounds_bracket_point_estimate(self):
        """The MC bounds must bracket the point estimate they're centered on."""
        stage_counts = [100, 40, 10]
        mc_lower, mc_upper, rate_lower, rate_upper = compute_funnel_bounds(stage_counts)
        counts = np.asarray(stage_counts, dtype=float)
        assert np.all(mc_lower <= counts)
        assert np.all(mc_upper >= counts)

    def test_no_rate_ci_collapses_to_point_estimate(self):
        """`rate_ci=None` should leave the rate bounds equal to the point estimate."""
        stage_counts = [100, 40, 10]
        *_, rate_lower, rate_upper = compute_funnel_bounds(stage_counts)
        assert np.allclose(rate_lower, stage_counts)
        assert np.allclose(rate_upper, stage_counts)

    def test_rate_ci_scales_every_stage_uniformly(self):
        """`rate_ci` should scale every stage's point estimate by the same multiplicative factor."""
        stage_counts = [100, 40, 10]
        *_, rate_lower, rate_upper = compute_funnel_bounds(stage_counts, rate_ci=(0.5, 2.0))
        assert np.allclose(rate_lower, 0.5 * np.asarray(stage_counts, dtype=float))
        assert np.allclose(rate_upper, 2.0 * np.asarray(stage_counts, dtype=float))

    def test_monotonic_confidence_widening(self):
        """A higher confidence level must widen (never narrow) the MC bounds."""
        stage_counts = [100, 40, 10]
        mc_lower_narrow, mc_upper_narrow, *_ = compute_funnel_bounds(stage_counts, confidence=0.5)
        mc_lower_wide, mc_upper_wide, *_ = compute_funnel_bounds(stage_counts, confidence=0.99)
        assert np.all(mc_lower_wide <= mc_lower_narrow)
        assert np.all(mc_upper_wide >= mc_upper_narrow)


class TestPlotDetectionFunnel:
    """Tests for `plot_detection_funnel`."""

    def test_runs_without_error(self):
        """A single shared bar color should draw one bar per stage without raising."""
        fig, ax = plt.subplots()
        stage_counts = [100, 40, 10]
        mc_lower, mc_upper, rate_lower, rate_upper = compute_funnel_bounds(stage_counts, rate_ci=(0.8, 1.3))
        plot_detection_funnel(
            ax,
            x=np.arange(3),
            counts=np.asarray(stage_counts, dtype=float),
            mc_lower=mc_lower,
            mc_upper=mc_upper,
            rate_lower=rate_lower,
            rate_upper=rate_upper,
            color="#4C72B0",
            label="test",
        )
        # 3 count bars + 3 rate-band rectangles (one outlined box per stage).
        assert len(ax.patches) == 6
        plt.close(fig)

    def test_rate_band_rectangles_span_the_rate_bounds(self):
        """Rate-band rectangles must span `rate_lower` to `rate_upper` with a visible, opaque edge."""
        fig, ax = plt.subplots()
        stage_counts = [100, 40, 10]
        mc_lower, mc_upper, rate_lower, rate_upper = compute_funnel_bounds(stage_counts, rate_ci=(0.8, 1.3))
        plot_detection_funnel(
            ax,
            x=np.arange(3),
            counts=np.asarray(stage_counts, dtype=float),
            mc_lower=mc_lower,
            mc_upper=mc_upper,
            rate_lower=rate_lower,
            rate_upper=rate_upper,
            color="#4C72B0",
        )
        # The rate-band rectangles are the last 3 patches added (after the 3 count bars); each must
        # have a fully opaque edge (visible outline) and a translucent fill (shaded envelope).
        band_patches = ax.patches[3:]
        for patch, lo, hi in zip(band_patches, rate_lower, rate_upper):
            assert patch.get_y() == lo
            assert patch.get_y() + patch.get_height() == hi
            assert patch.get_edgecolor()[3] > patch.get_facecolor()[3]
        plt.close(fig)

    def test_accepts_per_bar_colors(self):
        """A per-bar color list should draw one bar per stage without raising."""
        fig, ax = plt.subplots()
        stage_counts = [100, 40, 10]
        mc_lower, mc_upper, rate_lower, rate_upper = compute_funnel_bounds(stage_counts)
        plot_detection_funnel(
            ax,
            x=np.arange(3),
            counts=np.asarray(stage_counts, dtype=float),
            mc_lower=mc_lower,
            mc_upper=mc_upper,
            rate_lower=rate_lower,
            rate_upper=rate_upper,
            color=["#888888", "#4C72B0", "#55A868"],
        )
        assert len(ax.patches) == 6
        plt.close(fig)


class TestAddFunnelLegend:
    """Tests for `add_funnel_legend`."""

    def test_adds_two_entries_to_bare_axes(self):
        """Both proxy legend entries should appear even with no prior legend content."""
        fig, ax = plt.subplots()
        add_funnel_legend(ax)
        legend = ax.get_legend()
        labels = [t.get_text() for t in legend.get_texts()]
        assert "MC (statistical) uncertainty" in labels
        assert "Rate (systematic) uncertainty" in labels
        plt.close(fig)

    def test_preserves_existing_bar_labels(self):
        """Existing labeled handles (e.g. per-transient-type bars) should stay in the legend."""
        fig, ax = plt.subplots()
        ax.bar([0], [1], label="kne")
        add_funnel_legend(ax)
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ["kne", "MC (statistical) uncertainty", "Rate (systematic) uncertainty"]
        plt.close(fig)


class TestPlotRateBars:
    """Tests for `plot_rate_bars`."""

    def test_bar_heights_match_visible_fraction_times_rate(self):
        """Each bar height should be ``visible_counts[i] / n_samples * all_sky_rate``."""
        fig, ax = plt.subplots()
        rates = plot_rate_bars(
            ax,
            ["FUV", "NUV"],
            visible_counts=[300, 900],
            n_samples=3000,
            all_sky_rate=1000 / u.yr,
        )
        assert np.allclose(rates, [100.0, 300.0])
        plt.close(fig)

    def test_draws_one_bar_per_category(self):
        """One count bar (plus, with `rate_ci`, one rate-band rectangle) per category."""
        fig, ax = plt.subplots()
        plot_rate_bars(
            ax,
            ["FUV", "NUV"],
            visible_counts=[300, 900],
            n_samples=3000,
            all_sky_rate=1000 / u.yr,
            rate_ci=(0.8, 1.3),
        )
        assert len(ax.patches) == 4
        plt.close(fig)

    def test_no_rate_ci_omits_rate_band(self):
        """Without `rate_ci`, the rate band should collapse to zero height (invisible, but present)."""
        fig, ax = plt.subplots()
        plot_rate_bars(
            ax,
            ["FUV", "NUV"],
            visible_counts=[300, 900],
            n_samples=3000,
            all_sky_rate=1000 / u.yr,
        )
        band_patches = ax.patches[2:]
        for patch in band_patches:
            assert patch.get_height() == 0
        plt.close(fig)

    def test_sets_category_xticks(self):
        """X-tick labels should be the category names, in order."""
        fig, ax = plt.subplots()
        plot_rate_bars(
            ax,
            ["FUV", "NUV"],
            visible_counts=[300, 900],
            n_samples=3000,
            all_sky_rate=1000 / u.yr,
        )
        assert [t.get_text() for t in ax.get_xticklabels()] == ["FUV", "NUV"]
        plt.close(fig)

    def test_respects_rate_unit(self):
        """A non-default `rate_unit` should rescale the returned bar heights accordingly."""
        fig, ax = plt.subplots()
        rates_per_year = plot_rate_bars(ax, ["FUV"], visible_counts=[300], n_samples=3000, all_sky_rate=1000 / u.yr)
        plt.close(fig)

        fig, ax = plt.subplots()
        rates_per_day = plot_rate_bars(
            ax,
            ["FUV"],
            visible_counts=[300],
            n_samples=3000,
            all_sky_rate=1000 / u.yr,
            rate_unit=1 / u.day,
        )
        plt.close(fig)

        assert np.allclose(rates_per_day, (rates_per_year / u.yr).to_value(1 / u.day))
