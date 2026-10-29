"""Tests for reloading a run, and evaluating its maps, through `PhaseSpaceSampler`."""

import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table

pytest.importorskip("sklearn")

from uvex_transients.phase_space.evaluator import RecoveryEvaluator  # noqa: E402
from uvex_transients.phase_space.sampler import PhaseSpaceSampler  # noqa: E402

BOUNDS = [(41.5, 44.0), (0.0, 2.0)]
NAMES = ("log_luminosity", "log_half_time")
RATE = 10.0 / u.Gpc**3 / u.yr


def make_evaluator(schedule, **kwargs):
    defaults = {
        "mission": "uvex_fast",
        "nside": 32,
        "order": "nested",
        "time_bins": 3,
        "chunk_size": 100,
        "n_max": 200,
        "k_target": 3,
        "n_visits": 1,
        "seed": 4,
    }
    defaults.update(kwargs)
    return RecoveryEvaluator(schedule, **defaults)


@pytest.fixture(scope="module")
def schedule(make_schedule):
    return make_schedule(n_sched=20)


@pytest.fixture(scope="module")
def run(schedule, tmp_path_factory):
    """A small finished run: the sampler, its evaluator and the file it was saved to."""
    evaluator = make_evaluator(schedule)
    sampler = PhaseSpaceSampler(
        evaluator, BOUNDS, names=NAMES, n_initial=8, max_points=8, n_candidates=64, n_restarts=0, seed=1
    ).run()
    path = tmp_path_factory.mktemp("run") / "points.ecsv"
    sampler.save(path)
    return sampler, evaluator, path


@pytest.fixture(scope="module")
def loaded(run, schedule):
    _, _, path = run
    # The synthetic schedule is not in the registry, so the file cannot name it.
    return PhaseSpaceSampler.load(path, evaluator_kwargs={"schedule": schedule}, n_restarts=0, seed=1)


def test_the_evaluator_describes_itself_in_plain_types(run):
    _, evaluator, _ = run
    settings = evaluator.settings
    assert settings["mission"] == "uvex_fast" and settings["band"] == "FUV"
    assert settings["time_bins"] == 3 and settings["nside"] == 32 and settings["n_visits"] == 1
    assert settings["schedule_name"] is None
    assert all(value is None or isinstance(value, (str, int, float, bool, list)) for value in settings.values())


def test_the_file_names_the_evaluator_and_load_rebuilds_it(run, loaded):
    sampler, evaluator, path = run
    spec = Table.read(path, format="ascii.ecsv").meta["evaluator"]
    assert spec["class"] == "uvex_transients.phase_space.evaluator.RecoveryEvaluator"
    assert spec["settings"] == evaluator.settings

    assert isinstance(loaded._evaluate, RecoveryEvaluator)
    assert loaded._evaluate.settings == evaluator.settings
    assert len(loaded) == len(sampler) == 8


def test_a_schedule_that_is_not_in_the_registry_must_be_given(run):
    _, _, path = run
    with pytest.raises(ValueError, match="no schedule"):
        PhaseSpaceSampler.load(path)


def test_a_file_cannot_name_an_evaluator_outside_the_package(run, schedule, tmp_path):
    _, _, path = run
    table = Table.read(path, format="ascii.ecsv")
    table.meta["evaluator"]["class"] = "os.path.join"
    bad = tmp_path / "bad.ecsv"
    table.write(bad, format="ascii.ecsv")
    with pytest.raises(ValueError, match="Refusing"):
        PhaseSpaceSampler.load(bad, evaluator_kwargs={"schedule": schedule})


def test_a_loaded_run_reproduces_every_map(run, loaded):
    sampler, _, _ = run
    log_l, log_dt = np.meshgrid(np.linspace(41.8, 43.8, 4), np.linspace(0.2, 1.8, 3), indexing="ij")
    for name in (
        "recovery",
        "efficiency",
        "recovered_volume_time",
        "uncertainty",
        "coverage",
        "volume_time",
    ):
        original = getattr(sampler, f"evaluate_{name}_map")(log_l, log_dt)
        again = getattr(loaded, f"evaluate_{name}_map")(log_l, log_dt)
        np.testing.assert_allclose(again, original, rtol=1e-6, err_msg=name)
    np.testing.assert_allclose(loaded.evaluate_z_limit_map(log_l), sampler.evaluate_z_limit_map(log_l))
    np.testing.assert_allclose(
        loaded.evaluate_all_sky_volume_time_map(log_l), sampler.evaluate_all_sky_volume_time_map(log_l)
    )


def test_the_maps_are_built_from_the_recovery_and_the_scales(loaded):
    log_l, log_dt = np.array([42.3, 43.1, 43.9]), np.array([0.5, 1.0, 1.5])
    f = loaded.evaluate_recovery_map(log_l, log_dt)
    coverage = loaded.evaluate_coverage_map(log_l, log_dt)
    volume_time = loaded.evaluate_volume_time_map(log_l, log_dt)

    np.testing.assert_allclose(loaded.evaluate_efficiency_map(log_l, log_dt), f * coverage)
    np.testing.assert_allclose(loaded.evaluate_recovered_volume_time_map(log_l, log_dt), f * volume_time)
    np.testing.assert_allclose(loaded.evaluate_all_sky_volume_time_map(log_l), volume_time / coverage, rtol=1e-10)
    assert np.all((0 < coverage) & (coverage <= 1))
    assert np.all(loaded.evaluate_efficiency_map(log_l, log_dt) <= coverage)


def test_the_expected_recoveries_scale_with_the_rate_and_accept_any_units(loaded):
    log_l, log_dt = np.array([42.5, 43.5]), np.array([1.0, 1.0])
    base = loaded.evaluate_expected_recoveries_map(log_l, log_dt, RATE)
    np.testing.assert_allclose(loaded.evaluate_expected_recoveries_map(log_l, log_dt, 3 * RATE), 3 * base)
    np.testing.assert_allclose(
        loaded.evaluate_expected_recoveries_map(log_l, log_dt, 1e-8 / u.Mpc**3 / u.yr), base, rtol=1e-10
    )
    np.testing.assert_allclose(base, 1e-8 * loaded.evaluate_recovered_volume_time_map(log_l, log_dt), rtol=1e-10)


def test_the_grid_table_agrees_with_the_point_maps(loaded):
    table = loaded.map_table(5, rate=RATE)
    assert len(table) == 25
    assert table.colnames[:2] == list(NAMES)
    for column in (
        "f",
        "f_lo",
        "f_hi",
        "efficiency",
        "recovered_volume_time",
        "expected_recoveries_hi",
        "coverage",
        "volume_time",
        "all_sky_volume_time",
        "z_limit",
        "sigma_log",
    ):
        assert column in table.colnames
    log_l, log_dt = np.asarray(table[NAMES[0]]), np.asarray(table[NAMES[1]])
    np.testing.assert_allclose(table["efficiency"], loaded.evaluate_efficiency_map(log_l, log_dt))
    np.testing.assert_allclose(
        table["expected_recoveries"], loaded.evaluate_expected_recoveries_map(log_l, log_dt, RATE)
    )
    assert np.all(table["f_lo"] <= table["f"]) and np.all(table["f"] <= table["f_hi"])
    # `ij` order: the luminosity varies slowest, and the grid spans the bounds.
    assert np.all(np.diff(np.asarray(table[NAMES[0]]).reshape(5, 5), axis=1) == 0)
    assert log_l.min() == BOUNDS[0][0] and log_l.max() == BOUNDS[0][1]
    assert log_dt.min() == BOUNDS[1][0] and log_dt.max() == BOUNDS[1][1]
    assert "expected_recoveries" not in loaded.map_table(3).colnames


def test_a_map_can_be_evaluated_beyond_the_bounds_it_was_run_on(loaded):
    table = loaded.map_table(4, bounds=[(40.5, 45.5), (-1.0, 2.5)])
    assert np.asarray(table[NAMES[0]]).min() == 40.5
    assert np.all(np.isfinite(table["efficiency"]))


def test_the_plot_has_a_panel_for_each_quantity(loaded):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from uvex_transients.phase_space.plotting import plot_map

    figure = plot_map(loaded.map_table(6, rate=RATE), loaded.table, title="test")
    try:
        # Four panels, and a twin magnitude axis on each of the two on the right.
        assert len(figure.axes) >= 4
        labels = " ".join(ax.get_ylabel() + ax.get_xlabel() for ax in figure.axes)
        assert "absolute AB magnitude" in labels
        # The third panel is the rate for one recovery, in units of the local core-collapse supernova rate.
        assert "rate for one recovery / local CCSN rate" in labels
    finally:
        matplotlib.pyplot.close(figure)


def test_overlays_are_shaded_and_keyed_by_a_legend_and_do_not_stretch_the_axes(loaded):
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

    grid = loaded.map_table(6)
    plain = plot_map(grid, loaded.table)
    figure = plot_map(grid, loaded.table, overlays=[overlay, far])
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


def test_the_simulated_points_are_only_marked_on_request(loaded):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from uvex_transients.phase_space.plotting import plot_map

    grid = loaded.map_table(6)
    hidden, shown = plot_map(grid, loaded.table), plot_map(grid, loaded.table, show_points=True)
    try:
        assert len(shown.axes[0].collections) == len(hidden.axes[0].collections) + 1
    finally:
        matplotlib.pyplot.close(hidden)
        matplotlib.pyplot.close(shown)
