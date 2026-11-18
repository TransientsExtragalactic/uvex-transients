"""Tests for the canonical map of an `Experiment` that has been saved and loaded: its tables, its rate and its figure."""

import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table

pytest.importorskip("sklearn")

from uvex_transients.phase_space.experiment import CampaignSettings, Experiment  # noqa: E402
from uvex_transients.phase_space.sampling import CanonicalSampler, StoppingRule  # noqa: E402

BOUNDS = [(41.5, 44.0), (0.0, 2.0)]
NAMES = ("log_luminosity", "log_half_time")
RATE = 10.0 / u.Gpc**3 / u.yr


@pytest.fixture(scope="module")
def schedule(make_schedule):
    return make_schedule(n_sched=20)


@pytest.fixture(scope="module")
def run(schedule, tmp_path_factory):
    """A small finished run: the experiment and the file it was saved to."""
    sampler = CanonicalSampler(schedule, "uvex_fast", nside=32, order="nested", time_bins=3, n_visits=1)
    experiment = Experiment.create(
        sampler,
        bounds=BOUNDS,
        rule=StoppingRule(chunk_size=100, k_target=3, n_max=200),
        campaign=CampaignSettings(n_initial=8, batch_size=4, max_points=8, tolerance=0.0, seed=4),
        seed=1,
        n_candidates=64,
        n_restarts=0,
    )
    experiment.run()
    path = tmp_path_factory.mktemp("run") / "points.ecsv"
    experiment.save(path)
    return experiment, path


@pytest.fixture(scope="module")
def loaded(run, schedule):
    # The synthetic schedule is not in the registry, so the file cannot name it.
    return Experiment.load(run[1], schedule=schedule)


@pytest.fixture(scope="module")
def points(loaded):
    return loaded.phase_map.table


def test_a_loaded_run_reproduces_the_map(run, loaded):
    original, again = run[0].grid_table(4), loaded.grid_table(4)
    assert original.colnames == again.colnames
    for name in original.colnames:
        np.testing.assert_allclose(again[name], original[name], rtol=1e-6, err_msg=name)


def test_the_maps_are_built_from_the_recovery_and_the_scales(loaded):
    table = loaded.grid_table(4)
    f, coverage, volume_time = (np.asarray(table[name]) for name in ("f", "coverage", "volume_time"))
    np.testing.assert_allclose(table["efficiency"], f * coverage)
    np.testing.assert_allclose(table["recovered_volume_time"], f * volume_time)
    np.testing.assert_allclose(table["all_sky_volume_time"], volume_time / coverage, rtol=1e-10)
    assert np.all((0 < coverage) & (coverage <= 1))
    assert np.all(table["efficiency"] <= coverage)


def test_the_expected_recoveries_scale_with_the_rate_and_accept_any_units(loaded):
    base = loaded.grid_table(4, rate=RATE)
    np.testing.assert_allclose(
        loaded.grid_table(4, rate=3 * RATE)["expected_recoveries"], 3 * np.asarray(base["expected_recoveries"])
    )
    # 10 per Gpc^3 per yr is 1e-8 per Mpc^3 per yr, whichever way it is given
    np.testing.assert_allclose(
        loaded.grid_table(4, rate=1e-8 / u.Mpc**3 / u.yr)["expected_recoveries"],
        base["expected_recoveries"],
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        base["expected_recoveries"], 1e-8 * np.asarray(base["recovered_volume_time"]), rtol=1e-10
    )


def test_a_map_can_be_evaluated_beyond_the_bounds_it_was_run_on(loaded):
    table = loaded.grid_table(4, bounds=[(40.5, 45.5), (-1.0, 2.5)])
    assert np.asarray(table[NAMES[0]]).min() == 40.5
    assert np.all(np.isfinite(table["efficiency"]))


def test_the_plot_has_a_panel_for_each_quantity(loaded, points):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from uvex_transients.phase_space.plotting import plot_map

    figure = plot_map(loaded.grid_table(6, rate=RATE), points, title="test")
    try:
        # Four panels, and a twin magnitude axis on each of the two on the right.
        assert len(figure.axes) >= 4
        labels = " ".join(ax.get_ylabel() + ax.get_xlabel() for ax in figure.axes)
        assert "absolute AB magnitude" in labels
        # The third panel is the rate for one recovery, in units of the local core-collapse supernova rate.
        assert "rate for one recovery / local CCSN rate" in labels
    finally:
        matplotlib.pyplot.close(figure)


def test_overlays_are_shaded_and_keyed_by_a_legend_and_do_not_stretch_the_axes(loaded, points):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from uvex_transients.phase_space.plotting import plot_map

    rng = np.random.default_rng(0)
    overlay = Table({"log_luminosity": rng.normal(43.0, 0.3, 80), "log_half_time": rng.normal(1.0, 0.2, 80)})
    overlay["measured"] = True
    overlay.meta["class"] = "Test"
    # Too small for a density estimate, so it is drawn as a box instead, and far outside the map.
    far = Table({"log_luminosity": [30.0, 31.0], "log_half_time": [5.0, 6.0], "measured": [True, True]})
    far.meta["class"] = "Far"

    grid = loaded.grid_table(6)
    plain = plot_map(grid, points)
    figure = plot_map(grid, points, overlays=[overlay, far])
    try:
        top, bottom = figure.axes[0], figure.axes[2]
        # The classes are named by one legend, in the colours of their shading, and not by text on the panels.
        assert [text.get_text() for text in figure.legends[0].get_texts()] == ["Test", "Far"]
        names = {text.get_text() for panel in (top, bottom) for text in panel.texts}
        assert names.isdisjoint({"Test", "Far"})
        # The shading and its outline are added to every panel, and the box of the small population is a patch.
        assert len(bottom.collections) > len(plain.axes[2].collections)
        assert len(bottom.patches) == len(plain.axes[2].patches) + 1
        assert top.get_xlim() == plain.axes[0].get_xlim() and top.get_ylim() == plain.axes[0].get_ylim()
    finally:
        matplotlib.pyplot.close(figure)
        matplotlib.pyplot.close(plain)


def test_the_simulated_points_are_only_marked_on_request(loaded, points):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from uvex_transients.phase_space.plotting import plot_map

    grid = loaded.grid_table(6)
    hidden, shown = plot_map(grid, points), plot_map(grid, points, show_points=True)
    try:
        assert len(shown.axes[0].collections) == len(hidden.axes[0].collections) + 1
    finally:
        matplotlib.pyplot.close(hidden)
        matplotlib.pyplot.close(shown)
