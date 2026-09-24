"""Tests for `uvex_transients.utils.plotting`."""

import matplotlib as mpl

mpl.use("Agg")

import numpy as np
from matplotlib import pyplot as plt

from uvex_transients.utils.plotting import add_funnel_legend, compute_funnel_bounds, plot_detection_funnel


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
