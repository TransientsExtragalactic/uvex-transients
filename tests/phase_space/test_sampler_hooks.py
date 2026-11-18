"""Tests for the optional hooks of `Sampler`, which nothing that consumes a sampler asks its kind about."""

import matplotlib as mpl
import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table

mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from uvex_transients.phase_space.experiment import CampaignSettings, Experiment  # noqa: E402
from uvex_transients.phase_space.sampling import (  # noqa: E402
    BracketedBreakoutSampler,
    BreakoutSampler,
    CanonicalSampler,
    LogisticSampler,
    Scale,
    StoppingRule,
    SurveySampler,
    Weighted,
    get_sampler,
)
from uvex_transients.phase_space.space import Axis, Space  # noqa: E402
from uvex_transients.phase_space.stages import map_title, plot_files, run_experiment, write_map  # noqa: E402

pytest.importorskip("sklearn")


@pytest.fixture
def space():
    return Space([Axis("a", 0.0, 1.0, label="a"), Axis("b", -1.0, 1.0, label="b")])


@pytest.fixture
def sampler(space):
    return LogisticSampler(space, center=[0.5, 0.0], slope=[4.0, 4.0])


# ---------------------------------------------------------------------------- #
# The defaults                                                                 #
# ---------------------------------------------------------------------------- #
def test_the_defaults_say_that_a_sampler_has_nothing_extra(sampler):
    assert sampler.rate_products(1.0 / u.Gpc**3 / u.yr) == ()
    assert sampler.table_meta == {}
    assert LogisticSampler.HAS_OVERLAY is False
    assert LogisticSampler.describe({}) == "logistic"


def test_a_sampler_without_an_overlay_cannot_measure_one(sampler):
    with pytest.raises(NotImplementedError, match="logistic"):
        sampler.measure_overlay()


def test_only_the_canonical_sampler_has_an_overlay():
    assert {name: get_sampler(name).HAS_OVERLAY for name in ("canonical", "breakout", "breakout_bracketed")} == {
        "canonical": True,
        "breakout": False,
        "breakout_bracketed": False,
    }


def test_a_sampler_is_built_from_a_run_config_by_its_own_hook(space):
    built = LogisticSampler.from_run_config(None, space=space, center=[0.5, 0.0], slope=[1.0, 1.0])
    assert isinstance(built, LogisticSampler) and built.space is space


def test_a_setting_a_sampler_does_not_take_is_named_before_it_is_built(space):
    with pytest.raises(ValueError, match=r"'logistic' sampler does not take.*bogus"):
        LogisticSampler.from_run_config(None, space=space, center=[0, 0], slope=[1, 1], bogus=1)


def test_a_constructor_that_forwards_its_keywords_is_checked_against_its_parents():
    # `BracketedBreakoutSampler` takes ``**kwargs`` for `BreakoutSampler`, which would accept any name if not checked
    for kind, options in ((BracketedBreakoutSampler, {"gap_factor": 2.0, "z_max": 5.0}), (BreakoutSampler, {})):
        kind._check_options(None, None, band="FUV", **options)
    with pytest.raises(ValueError, match="bogus"):
        BracketedBreakoutSampler._check_options(None, None, bogus=1)
    with pytest.raises(ValueError, match="gap_factor"):
        BreakoutSampler._check_options(None, None, gap_factor=2.0)


def test_a_survey_sampler_names_what_it_is_from_its_settings_alone():
    settings = {"band": "FUV", "snr_threshold": 5.0, "n_visits": 2}
    assert CanonicalSampler.describe(settings) == "FUV: SNR >= 5 on 2 visit(s)"
    assert BreakoutSampler.describe(settings) == "FUV: SNR >= 5 on 2 visit(s)"
    assert BracketedBreakoutSampler.describe({**settings, "gap_factor": 2.0}) == (
        "FUV: SNR >= 5, explosion bracketed within 2 x its window"
    )


def test_a_title_is_what_the_sampler_says_the_map_is():
    assert map_title("logistic", {}, 10, 2000) == "logistic, 10 points, 2,000 tries"
    title = map_title("canonical", {"band": "NUV", "snr_threshold": 5.0, "n_visits": 2}, 4, 80, minutes=1.5)
    assert title == "NUV: SNR >= 5 on 2 visit(s), 4 points, 80 tries, 1.5 min"


def test_a_survey_sampler_that_never_configures_itself_says_so():
    class Unconfigured(SurveySampler):
        make_transient = redshift_limit = _place = _log_window = lambda *args, **kwargs: None  # noqa: E731

        def __init__(self):  # no super().__init__, no _configure
            self._space = None

    with pytest.raises(RuntimeError, match="_configure"):
        Unconfigured().space


# ---------------------------------------------------------------------------- #
# What consumes them                                                           #
# ---------------------------------------------------------------------------- #
class _WithRate(LogisticSampler, kind="logistic_with_a_rate"):
    """A sampler that has a volume to apply a rate to, and a plane to record."""

    def rate_products(self, rate):
        per = u.Quantity(rate).to_value(u.Gpc**-3 * u.yr**-1)
        return (Weighted("expected", lambda x: per * np.ones(x.shape[:-1]), label="expected"),)

    @property
    def table_meta(self):
        return {"band": "toy"}


def _experiment(sampler):
    rule = StoppingRule(chunk_size=50, k_target=10, n_max=200)
    campaign = CampaignSettings(n_initial=8, batch_size=4, max_points=8, tolerance=0.0, seed=1)
    return Experiment.create(sampler, rule=rule, campaign=campaign, seed=1)


def test_the_map_carries_the_rate_products_and_meta_of_its_sampler(space):
    experiment = _experiment(_WithRate(space, center=[0.5, 0.0], slope=[4.0, 4.0])).run()
    grid = experiment.grid_table(4, rate=10.0 / u.Gpc**3 / u.yr)
    assert "expected" in grid.colnames and grid.meta["band"] == "toy" and grid.meta["sampler"] == "logistic_with_a_rate"

    # A sampler that has no rate products leaves the rate with nothing to add, and says nothing extra of its plane
    plain = _experiment(LogisticSampler(space, center=[0.5, 0.0], slope=[4.0, 4.0])).run()
    grid = plain.grid_table(4, rate=10.0 / u.Gpc**3 / u.yr)
    assert "expected" not in grid.colnames and "band" not in grid.meta


def test_the_figure_of_a_map_is_drawn_by_its_sampler(sampler, tmp_path):
    experiment = _experiment(sampler).run()
    experiment.save(tmp_path / "points.ecsv")
    write_map(experiment, tmp_path, grid_size=4)
    path = plot_files(tmp_path)
    assert path.exists() and path.name == "map.png"
    plt.close("all")


def test_an_overlay_is_refused_for_a_sampler_that_has_none(sampler, tmp_path):
    experiment = _experiment(sampler).run()
    with pytest.raises(ValueError, match="no overlay"):
        write_map(experiment, tmp_path, grid_size=4, overlay=True)


# ---------------------------------------------------------------------------- #
# The figure                                                                   #
# ---------------------------------------------------------------------------- #
class _WithDepth(LogisticSampler, kind="logistic_with_a_depth"):
    """A sampler with a quantity of the point alone, which carries no interval, and an overlay of its own."""

    HAS_OVERLAY = True

    @property
    def products(self):
        return (Scale("depth", lambda x: 1.0 + x[..., 0], label="depth", unit="mag"),)

    def measure_overlay(self, n_draws=None, seed=0):
        return Table({"a": [0.2, 0.4], "b": [0.0, 0.1]})

    @classmethod
    def annotate_map(cls, ax, overlay):
        ax.plot(overlay["a"], overlay["b"], "o", color="red")
        return [Patch(facecolor="red", label="the overlay")]


def _written(sampler, directory, **kwargs):
    experiment = _experiment(sampler).run()
    experiment.save(directory / "points.ecsv")
    write_map(experiment, directory, grid_size=4, **kwargs)
    return experiment


def test_a_map_over_more_than_two_axes_is_made_but_not_drawn(tmp_path, caplog):
    space = Space([Axis("a", 0.0, 1.0), Axis("b", 0.0, 1.0), Axis("c", 0.0, 1.0)])
    experiment = _experiment(LogisticSampler(space, center=[0.5] * 3, slope=[3.0] * 3))
    written = run_experiment(experiment, tmp_path, grid_size=3)
    assert sorted(written) == ["map", "points"] and not (tmp_path / "map.png").exists()
    assert "none is drawn" in caplog.text
    assert plot_files(tmp_path) is None


def test_the_columns_to_draw_can_be_chosen_including_one_without_an_interval(space, tmp_path):
    _written(_WithDepth(space, center=[0.5, 0.0], slope=[4.0, 4.0]), tmp_path)
    grid = Table.read(tmp_path / "map.ecsv", format="ascii.ecsv")
    points = Table.read(tmp_path / "points.ecsv", format="ascii.ecsv")

    def drawn(figure):
        return [ax.get_ylabel() for ax in figure.axes]

    # The colour bars carry the name of what each panel draws
    assert not any("depth" in label for label in drawn(_WithDepth.draw_map(grid, points, None, "t")))
    chosen = _WithDepth.draw_map(grid, points, None, "t", fields=["f", "depth"])
    assert any("depth [mag]" in label for label in drawn(chosen))
    plt.close("all")


def test_a_sampler_decorates_its_figure_and_names_what_it_drew_in_the_legend(space, tmp_path):
    _written(_WithDepth(space, center=[0.5, 0.0], slope=[4.0, 4.0]), tmp_path)
    grid = Table.read(tmp_path / "map.ecsv", format="ascii.ecsv")
    points = Table.read(tmp_path / "points.ecsv", format="ascii.ecsv")
    overlay = _WithDepth(space, center=[0.5, 0.0], slope=[4.0, 4.0]).measure_overlay()

    figure = _WithDepth.draw_map(grid, points, None, "t", overlay=overlay)
    panels = [ax for ax in figure.axes if ax.get_visible() and ax.lines]
    assert panels and all(ax.lines[-1].get_color() == "red" for ax in panels)
    assert [t.get_text() for t in figure.legends[0].get_texts()] == ["the overlay"]

    # A sampler that does not decorate its figure leaves the overlay unused
    bare = LogisticSampler.draw_map(grid, points, None, "t", overlay=overlay)
    assert not bare.legends and not any(ax.lines for ax in bare.axes)
    plt.close("all")


# ---------------------------------------------------------------------------- #
# The yield a map is compared on, and what a figure draws by default           #
# ---------------------------------------------------------------------------- #
class _Volume(LogisticSampler, kind="logistic_with_a_volume"):
    """A sampler with a volume of its own, which is what two of its maps are compared on."""

    yield_weight = "volume"

    @property
    def products(self):
        return (Scale("volume", lambda x: 10.0 ** (1.0 + x[..., 0]), label="volume"),)


def _grid(sampler, seed=1):
    rule = StoppingRule(chunk_size=200, k_target=30, n_max=2000)
    campaign = CampaignSettings(n_initial=10, batch_size=4, max_points=10, tolerance=0.0, seed=seed)
    return Experiment.create(sampler, rule=rule, campaign=campaign, seed=seed).run().grid_table(5)


def test_a_sampler_declares_the_quantity_its_maps_are_compared_on(space):
    assert LogisticSampler.yield_weight is None and SurveySampler.yield_weight == "volume_time"
    assert all(get_sampler(kind).yield_weight == "volume_time" for kind in ("canonical", "breakout"))

    plain = _grid(LogisticSampler(space, center=[0.5, 0.0], slope=[4.0, 4.0]))
    weighted = _grid(_Volume(space, center=[0.5, 0.0], slope=[4.0, 4.0]))
    assert "weight" not in plain.meta and weighted.meta["weight"] == "volume"


def test_maps_of_a_sampler_with_no_weight_are_compared_on_the_success_fraction(space):
    from uvex_transients.phase_space.analysis.compare import difference_map

    baseline = _grid(LogisticSampler(space, center=[0.5, 0.0], slope=[4.0, 4.0]))
    comparison = _grid(LogisticSampler(space, center=[0.3, 0.0], slope=[4.0, 4.0]))
    difference = difference_map(baseline, comparison)
    assert difference.meta["weight"] == ""

    # It is the ratio of the fractions, with nothing multiplied in
    f_b, f_c = np.asarray(baseline["f"]), np.asarray(comparison["f"])
    ok = np.isfinite(difference["delta_dex"])
    floor = difference.meta["floor"]
    expected = np.log10(np.maximum(f_c, floor) / np.maximum(f_b, floor))
    np.testing.assert_allclose(difference["delta_dex"][ok], expected[ok])


def test_maps_weighted_by_different_quantities_are_not_compared(space):
    from uvex_transients.phase_space.analysis.compare import difference_map

    plain = _grid(LogisticSampler(space, center=[0.5, 0.0], slope=[4.0, 4.0]))
    weighted = _grid(_Volume(space, center=[0.5, 0.0], slope=[4.0, 4.0]))
    with pytest.raises(ValueError, match="different quantities"):
        difference_map(plain, weighted)


def test_a_map_from_before_the_weight_was_recorded_is_weighted_by_its_volume_time(space):
    from uvex_transients.phase_space.analysis.compare import difference_map

    legacy = _grid(_Volume(space, center=[0.5, 0.0], slope=[4.0, 4.0]))
    del legacy.meta["weight"]
    legacy.rename_column("volume", "volume_time")
    difference = difference_map(legacy, legacy)
    assert difference.meta["weight"] == "volume_time"


def test_a_product_says_whether_the_default_figure_draws_it(space):
    assert Scale("a", lambda x: x[..., 0]).draw is False and Weighted("b", lambda x: x[..., 0]).draw is True
    assert Scale("a", lambda x: x[..., 0], draw=True).draw and not Weighted("b", lambda x: x[..., 0], draw=False).draw

    class Drawn(LogisticSampler, kind="logistic_with_a_drawn_scale"):
        @property
        def products(self):
            return (Scale("depth", lambda x: 1.0 + x[..., 0], label="depth", unit="mag", draw=True),)

    grid = _grid(Drawn(space, center=[0.5, 0.0], slope=[4.0, 4.0]))
    assert grid.meta["products"][0]["draw"] is True
    figure = Drawn.draw_map(grid, Table({"a": [0.1], "b": [0.0], "n": [10]}), None, "t")
    assert any("depth [mag]" in ax.get_ylabel() for ax in figure.axes)
    plt.close("all")
