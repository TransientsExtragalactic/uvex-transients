"""Tests for `run_experiment`, `plot_grid` and `Experiment.resume_from`."""

import matplotlib as mpl
import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table

mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from uvex_transients.phase_space.experiment import CampaignSettings, Experiment  # noqa: E402
from uvex_transients.phase_space.plotting import plot_grid  # noqa: E402
from uvex_transients.phase_space.run import run_experiment  # noqa: E402
from uvex_transients.phase_space.sampling import LogisticSampler, StoppingRule, Weighted  # noqa: E402
from uvex_transients.phase_space.space import Axis, Space  # noqa: E402

pytest.importorskip("sklearn")

RULE = StoppingRule(chunk_size=100, k_target=15, n_max=1000)
CAMPAIGN = CampaignSettings(n_initial=10, batch_size=4, max_points=14, tolerance=0.0, seed=2)


class _Yielding(LogisticSampler, kind="logistic_with_a_yield"):
    """A logistic sampler that also offers a yield, a product with an interval."""

    @property
    def products(self):
        return (*super().products, Weighted("yield", lambda x: 10.0 ** x[..., 0], label="toy yield", unit="events"))


@pytest.fixture
def space():
    return Space(
        [
            Axis("log_L0", 0.0, 3.0, label=r"peak luminosity $L_0$", unit="erg/s"),
            Axis("log_t0", -1.0, 2.0, label=r"time scale $t_0$", unit="s"),
        ]
    )


@pytest.fixture
def experiment(space):
    sampler = _Yielding(space, center=[1.5, 0.5], slope=[3.0, 2.0])
    return Experiment.create(sampler, rule=RULE, campaign=CAMPAIGN, seed=2, n_restarts=0, n_candidates=128)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------- #
# run_experiment                                                               #
# ---------------------------------------------------------------------------- #
def test_a_run_writes_the_points_the_map_and_the_figure(experiment, tmp_path):
    written = run_experiment(experiment, tmp_path / "out", grid_size=8)
    assert set(written) == {"points", "map", "figure"}
    assert all(path.exists() and path.stat().st_size > 0 for path in written.values())

    points = Table.read(written["points"], format="ascii.ecsv")
    assert len(points) == 14 and points.meta["schema_version"] == 2
    grid = Table.read(written["map"], format="ascii.ecsv")
    assert len(grid) == 64 and {"f", "yield", "yield_lo", "yield_hi", "true_probability"} <= set(grid.colnames)


def test_the_files_of_a_run_rebuild_the_experiment(experiment, tmp_path):
    written = run_experiment(experiment, tmp_path, grid_size=5, plot=False)
    assert "figure" not in written and not (tmp_path / "map.png").exists()
    loaded = Experiment.load(written["points"])
    assert loaded.sampler.kind == "logistic_with_a_yield"
    np.testing.assert_array_equal(loaded.phase_map.points, experiment.phase_map.points)


def test_a_run_will_not_overwrite_a_map_unless_told_to(experiment, tmp_path):
    run_experiment(experiment, tmp_path, grid_size=5, plot=False)
    with pytest.raises(FileExistsError, match="overwrite"):
        run_experiment(experiment, tmp_path, grid_size=5, plot=False)
    run_experiment(experiment, tmp_path, grid_size=5, plot=False, overwrite=True)


def test_a_callback_runs_after_the_points_are_saved(experiment, tmp_path):
    seen = []

    def callback(phase_map):
        seen.append((len(phase_map), len(Table.read(tmp_path / "points.ecsv", format="ascii.ecsv"))))

    run_experiment(experiment, tmp_path, grid_size=5, plot=False, callback=callback)
    assert seen == [(10, 10), (14, 14)]


def test_a_run_resumes_from_the_saved_points(experiment, space, tmp_path):
    run_experiment(experiment, tmp_path, grid_size=5, plot=False)
    first = experiment.phase_map.points.copy()

    sampler = _Yielding.from_settings(experiment.sampler.settings)
    longer = Experiment.create(
        sampler,
        rule=RULE,
        campaign=CampaignSettings(n_initial=10, batch_size=4, max_points=18, tolerance=0.0, seed=2),
        seed=2,
        n_restarts=0,
        n_candidates=128,
    )
    run_experiment(longer, tmp_path, grid_size=5, plot=False, resume=True)
    assert len(longer.phase_map) == 18
    np.testing.assert_array_equal(longer.phase_map.points[:14], first)


def test_resuming_needs_a_file_over_the_same_axes(experiment, tmp_path):
    run_experiment(experiment, tmp_path, grid_size=5, plot=False)
    other = Experiment.create(
        _Yielding(Space([Axis("a", 0.0, 1.0), Axis("b", 0.0, 1.0)]), [0.5, 0.5], [1.0, 1.0]),
        rule=RULE,
        campaign=CAMPAIGN,
    )
    with pytest.raises(ValueError, match="over"):
        other.resume_from(tmp_path / "points.ecsv")


def test_a_rate_adds_the_expected_recoveries_when_the_sampler_can_make_them(experiment, tmp_path):
    """The logistic sampler has no volume-time, so a rate adds nothing, rather than failing."""
    written = run_experiment(experiment, tmp_path, grid_size=5, plot=False, rate=10 / u.Gpc**3 / u.yr)
    assert "expected_recoveries" not in Table.read(written["map"], format="ascii.ecsv").colnames


# ---------------------------------------------------------------------------- #
# plot_grid                                                                    #
# ---------------------------------------------------------------------------- #
@pytest.fixture
def grid(experiment):
    experiment.run()
    return experiment.grid_table(7)


def test_a_figure_has_a_panel_for_the_fraction_every_yield_and_the_uncertainty(grid):
    figure = plot_grid(grid)
    visible = [ax for ax in figure.axes if ax.get_visible() and ax.get_label() != "<colorbar>"]
    assert len(visible) == 3


def test_the_axes_are_labelled_from_the_space_and_are_logarithmic(grid):
    figure = plot_grid(grid)
    panels = [ax for ax in figure.axes if ax.get_label() != "<colorbar>"]
    panel = panels[2]  # the bottom left, which carries both labels
    assert panel.get_xlabel() == r"time scale $t_0$ [s]"
    assert panel.get_ylabel() == r"peak luminosity $L_0$ [erg/s]"
    assert panel.get_xscale() == "log" and panel.get_yscale() == "log"


def test_a_linear_axis_is_drawn_linearly(space):
    linear = Space([Axis("a", 0.0, 1.0, scale="linear"), Axis("b", 0.0, 1.0, label="b", unit="m", scale="linear")])
    sampler = LogisticSampler(linear, center=[0.5, 0.5], slope=[4.0, 4.0])
    experiment = Experiment.create(sampler, rule=RULE, campaign=CAMPAIGN, n_restarts=0, n_candidates=64).run()
    figure = plot_grid(experiment.grid_table(6))
    panel = [ax for ax in figure.axes if ax.get_label() != "<colorbar>"][0]
    assert panel.get_xscale() == "linear" and panel.get_xlabel() == "b [m]"
    assert panel.get_ylabel() == "a"


def test_the_panels_can_be_chosen(grid):
    figure = plot_grid(grid, fields=["f", "true_probability"])
    assert len([ax for ax in figure.axes if ax.get_label() != "<colorbar>" and ax.get_visible()]) == 2
    with pytest.raises(ValueError, match="no column 'nothing'"):
        plot_grid(grid, fields=["nothing"])


def test_an_annotation_is_drawn_on_every_panel(grid):
    calls = []
    plot_grid(grid, annotate=calls.append)
    assert len(calls) == 3


def test_points_can_be_marked(grid, experiment):
    figure = plot_grid(grid, experiment.phase_map.table, show_points=True)
    first = [ax for ax in figure.axes if ax.get_label() != "<colorbar>"][0]
    scatters = [c for c in first.collections if type(c).__name__ == "PathCollection"]
    assert len(scatters) == 1 and len(scatters[0].get_offsets()) == 14


def test_a_figure_is_saved_when_given_a_path(grid, tmp_path):
    plot_grid(grid, path=tmp_path / "fig.png", title="a title")
    assert (tmp_path / "fig.png").stat().st_size > 0


def test_a_figure_needs_a_table_that_records_its_space_and_two_axes(grid):
    bare = Table(grid)
    del bare.meta["space"]
    with pytest.raises(ValueError, match="does not record its space"):
        plot_grid(bare)

    three = Space([Axis(name, 0.0, 1.0) for name in "abc"])
    sampler = LogisticSampler(three, [0.5] * 3, [1.0] * 3)
    table = Experiment.create(sampler, rule=RULE, campaign=CAMPAIGN, n_restarts=0, n_candidates=64).run().grid_table(3)
    with pytest.raises(ValueError, match="two axes"):
        plot_grid(table)
