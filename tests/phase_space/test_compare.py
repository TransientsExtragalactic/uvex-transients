"""Tests for differencing two recovery maps, on synthetic map tables (no simulation, schedule or scikit-learn)."""

import numpy as np
import pytest
from astropy.table import Table
from scipy.special import expit

from uvex_transients.phase_space.compare import difference_map

NAMES = ("log_luminosity", "log_half_time")
N = 6
META = {
    "band": "FUV",
    "luminosity_unit": "erg/s",
    "spectral_index": -2.0,
    "rise_ratio": 0.1,
    "fade_mag": 6.0,
    "absolute_magnitude_zero_point": 10.0,
}


def make_map(logit, sigma=0.2, volume_time=1e6, **meta):
    """A map of an N x N grid whose recovery is `expit(logit)`, with a logit interval of `sigma`."""
    log_l, log_t = np.meshgrid(np.linspace(41.0, 44.0, N), np.linspace(0.0, 2.0, N), indexing="ij")
    logit = np.broadcast_to(logit, log_l.shape).ravel()
    table = Table()
    table[NAMES[0]], table[NAMES[1]] = log_l.ravel(), log_t.ravel()
    table["volume_time"] = np.full(log_l.size, float(volume_time))
    for suffix, offset in (("", 0.0), ("_lo", -sigma), ("_hi", sigma)):
        table["f" + suffix] = expit(logit + offset)
        table["recovered_volume_time" + suffix] = table["f" + suffix] * table["volume_time"]
    table.meta.update({**META, **meta})
    return table


def test_the_difference_is_the_log_ratio_of_the_recovered_volume_times():
    # Same recovery, a volume-time ten times larger: one dex, positive for the comparison.
    baseline, comparison = make_map(0.0), make_map(0.0, volume_time=1e7)
    diff = difference_map(baseline, comparison)
    assert np.allclose(diff["delta_dex"], 1.0)
    assert np.all(diff["limit"] == 0)

    # The recovery enters as its own ratio, and swapping the maps changes the sign.
    diff = difference_map(make_map(-1.0), make_map(1.0))
    assert np.allclose(diff["delta_dex"], np.log10(expit(1.0) / expit(-1.0)))
    assert np.allclose(difference_map(make_map(1.0), make_map(-1.0))["delta_dex"], -diff["delta_dex"])


def test_the_uncertainties_of_the_maps_combine_in_quadrature():
    a, b = make_map(0.0, sigma=0.3), make_map(0.0, sigma=0.4)
    diff = difference_map(a, b)
    half_width = lambda t: 0.5 * np.log10(t["f_hi"] / t["f_lo"])  # noqa: E731
    expected = np.hypot(half_width(a), half_width(b))
    assert np.allclose(diff["sigma_dex"], expected)
    assert np.allclose(diff["delta_dex_lo"], diff["delta_dex"] - expected)
    assert np.allclose(diff["delta_dex_hi"], diff["delta_dex"] + expected)
    assert np.allclose(diff["significance"], diff["delta_dex"] / expected)


def test_a_map_below_the_floor_makes_the_difference_a_bound_and_two_make_it_undefined():
    # Row 0: no recovery in the baseline. Row 1: none in either map. Row 2: none in the comparison.
    base_logit = np.full((N, N), 1.0)
    comp_logit = np.full((N, N), 1.0)
    base_logit[:2] = -20.0
    comp_logit[1:3] = -20.0
    diff = difference_map(make_map(base_logit), make_map(comp_logit), floor=1e-3)
    delta, limit = (np.asarray(diff[name]).reshape(N, N) for name in ("delta_dex", "limit"))

    # Baseline below the floor, comparison above it: a lower bound, with the baseline held at the floor.
    assert np.all(limit[0] == 1) and np.all(np.isfinite(delta[0]))
    assert np.allclose(delta[0], np.log10(expit(1.0) / 1e-3))
    # Both below the floor: undefined, and not a bound.
    assert np.all(np.isnan(delta[1])) and np.all(limit[1] == 0)
    # Comparison below the floor only: an upper bound.
    assert np.all(limit[2] == -1)
    assert np.allclose(delta[2], np.log10(1e-3 / expit(1.0)))
    assert np.all(limit[3:] == 0)

    # The floor is a parameter, and is recorded.
    diff = difference_map(make_map(base_logit), make_map(comp_logit), floor=1e-12)
    assert diff.meta["floor"] == 1e-12 and np.all(diff["limit"] == 0)


def test_the_floor_is_the_resolution_of_the_maps_unless_given():
    # f = 3e-5 in the baseline and 3e-4 in the comparison, everywhere
    low, high = np.log(3e-5 / (1 - 3e-5)), np.log(3e-4 / (1 - 3e-4))

    # Maps that do not record n_max resolve 1e-4, so the baseline is a bound and the comparison is measured.
    diff = difference_map(make_map(low), make_map(high))
    assert diff.meta["floor"] == 1e-4 and np.all(diff["limit"] == 1)

    # One recorded n_max of 1e4 (resolution 1e-4) and one of 1e5: the coarser resolution is the floor.
    diff = difference_map(make_map(low, n_max=1e5), make_map(high, n_max=1e4))
    assert diff.meta["floor"] == 1e-4 and np.all(diff["limit"] == 1)

    # Resolved by both: nothing is a bound, and a rare event is a measurement whatever its volume-time.
    diff = difference_map(make_map(low, n_max=1e6), make_map(high, n_max=1e6))
    assert diff.meta["floor"] == pytest.approx(1e-6) and np.all(diff["limit"] == 0)
    assert np.allclose(diff["delta_dex"], np.log10(3e-4 / 3e-5))

    # An explicit floor wins
    assert difference_map(make_map(low), make_map(high), floor=1e-2).meta["floor"] == 1e-2


def test_maps_on_different_grids_are_refused():
    baseline = make_map(0.0)
    with pytest.raises(ValueError, match="same grid"):
        difference_map(baseline, make_map(0.0)[:-1])
    shifted = make_map(0.0)
    shifted[NAMES[0]] += 0.5
    with pytest.raises(ValueError, match="same grid"):
        difference_map(baseline, shifted)
    renamed = make_map(0.0)
    renamed.rename_column(NAMES[0], "other")
    with pytest.raises(ValueError, match="different coordinates"):
        difference_map(baseline, renamed)
    incomplete = make_map(0.0)
    incomplete.remove_column("f_hi")
    with pytest.raises(ValueError, match="f_hi"):
        difference_map(baseline, incomplete)


@pytest.mark.parametrize("meta", [{"band": "NUV"}, {"spectral_index": -1.0}, {"rise_ratio": 0.2}, {"fade_mag": 8.0}])
def test_maps_of_different_axes_are_refused_unless_asked_not_to_check(meta):
    baseline, comparison = make_map(0.0), make_map(0.0, **meta)
    with pytest.raises(ValueError, match=next(iter(meta))):
        difference_map(baseline, comparison)
    assert np.allclose(difference_map(baseline, comparison, check_meta=False)["delta_dex"], 0.0)


def test_a_map_that_predates_a_recorded_quantity_is_not_held_to_it():
    old = make_map(0.0)
    for key in ("rise_ratio", "fade_mag"):
        del old.meta[key]
    assert np.allclose(difference_map(old, make_map(0.0, rise_ratio=0.2))["delta_dex"], 0.0)


def test_the_result_carries_the_axes_and_the_provenance_of_both_maps():
    diff = difference_map(make_map(0.0), make_map(0.0, spectral_index=-2.0, band="FUV"))
    assert diff.colnames[:2] == list(NAMES)
    assert diff.meta["band"] == "FUV" and diff.meta["absolute_magnitude_zero_point"] == 10.0
    assert diff.meta["baseline_band"] == diff.meta["comparison_band"] == "FUV"
    assert diff.meta["baseline_fade_mag"] == 6.0


def test_the_figure_marks_what_is_not_a_measurement():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from uvex_transients.phase_space.plotting import plot_difference

    # Left to right: a clear gain, no difference, a bound and nothing at all.
    base_logit = np.full((N, N), 1.0)
    comp_logit = np.full((N, N), 1.0)
    comp_logit[:2] = 3.0
    base_logit[4], comp_logit[4] = -20.0, 1.0
    base_logit[5], comp_logit[5] = -20.0, -20.0
    diff = difference_map(make_map(base_logit, volume_time=1e6), make_map(comp_logit, volume_time=1e7), floor=1e-3)

    overlay = Table({"log_luminosity": np.linspace(42, 43, 20), "log_half_time": np.linspace(0.5, 1.5, 20)})
    overlay["measured"] = True
    overlay.meta["class"] = "Test"
    figure = plot_difference(diff, title="test", overlays=[overlay])
    try:
        labels = " ".join(ax.get_ylabel() + ax.get_xlabel() for ax in figure.axes)
        assert "absolute AB magnitude" in labels and "comparison / baseline" in labels
        entries = [text.get_text() for text in figure.legends[0].get_texts()]
        assert any("lower bound" in entry for entry in entries) and "Test" in entries
        # The colour scale is symmetric about no difference.
        norm = figure.axes[0].collections[0].norm
        assert norm.vmin == -norm.vmax
    finally:
        matplotlib.pyplot.close(figure)


def test_a_difference_with_nothing_defined_still_draws():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from uvex_transients.phase_space.plotting import plot_difference

    nothing = make_map(-20.0)
    figure = plot_difference(difference_map(nothing, nothing))
    matplotlib.pyplot.close(figure)
