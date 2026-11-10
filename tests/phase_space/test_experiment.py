"""Tests for `CampaignSettings` and `Experiment`: running, the points file, and reading the earlier format."""

import numpy as np
import pytest
from astropy.table import Table

from uvex_transients.phase_space.evaluator import RecoveryEvaluator
from uvex_transients.phase_space.experiment import SCHEMA_VERSION, CampaignSettings, Experiment
from uvex_transients.phase_space.sampler import PhaseSpaceSampler
from uvex_transients.phase_space.samplers import CanonicalSampler, LogisticSampler, Sampler, StoppingRule
from uvex_transients.phase_space.space import Axis, Space
from uvex_transients.utils import config

pytest.importorskip("sklearn")

RULE = StoppingRule(chunk_size=100, k_target=15, n_max=1000)
CAMPAIGN = CampaignSettings(n_initial=10, batch_size=4, max_points=18, tolerance=0.0, seed=3)


@pytest.fixture
def sampler():
    space = Space([Axis("a", 0.0, 1.0, label=r"$a$", unit="m"), Axis("b", -1.0, 1.0, label=r"$b$", unit="s")])
    return LogisticSampler(space, center=[0.5, 0.0], slope=[8.0, 4.0])


@pytest.fixture
def experiment(sampler):
    return Experiment.create(sampler, rule=RULE, campaign=CAMPAIGN, seed=3, n_restarts=0, n_candidates=128)


# ---------------------------------------------------------------------------- #
# Campaign settings                                                            #
# ---------------------------------------------------------------------------- #
def test_campaign_settings_are_converted_and_immutable():
    settings = CampaignSettings(n_initial=10.0, batch_size=4, max_points=20, tolerance=0, seed=2.0)
    assert (settings.n_initial, settings.tolerance, settings.seed) == (10, 0.0, 2)
    assert type(settings.n_initial) is int and type(settings.tolerance) is float
    with pytest.raises(AttributeError):
        settings.max_points = 5


@pytest.mark.parametrize("name", ["n_initial", "batch_size", "max_points"])
@pytest.mark.parametrize("value", [0, -1, 2.5])
def test_campaign_counts_have_to_be_positive_integers(name, value):
    arguments = {"n_initial": 5, "batch_size": 2, "max_points": 10, "tolerance": 0.1, name: value}
    with pytest.raises(ValueError, match="positive integer"):
        CampaignSettings(**arguments)


def test_campaign_tolerance_may_not_be_negative():
    with pytest.raises(ValueError, match="tolerance"):
        CampaignSettings(n_initial=5, batch_size=2, max_points=10, tolerance=-0.1)


def test_campaign_settings_take_what_is_not_given_from_the_configuration():
    settings = CampaignSettings.from_config(max_points=33, seed=4)
    assert settings.n_initial == config["phase_space.sampler.n_initial"]
    assert settings.tolerance == config["phase_space.sampler.tolerance"]
    assert (settings.max_points, settings.seed) == (33, 4)


# ---------------------------------------------------------------------------- #
# Creating and running                                                         #
# ---------------------------------------------------------------------------- #
def test_an_experiment_maps_the_space_of_its_sampler_by_default(sampler):
    experiment = Experiment.create(sampler, rule=RULE, campaign=CAMPAIGN)
    assert experiment.phase_map.space == sampler.space
    assert experiment.rule is RULE and experiment.campaign is CAMPAIGN


def test_an_experiment_can_map_a_smaller_rectangle_and_keeps_the_labels(sampler):
    experiment = Experiment.create(sampler, bounds=[[0.2, 0.8], [-0.5, 0.5]], rule=RULE, campaign=CAMPAIGN)
    space = experiment.phase_map.space
    np.testing.assert_array_equal(space.bounds, [[0.2, 0.8], [-0.5, 0.5]])
    assert [axis.label for axis in space] == [r"$a$", r"$b$"]


def test_an_experiment_takes_the_rule_and_the_campaign_from_the_configuration_if_not_given(sampler):
    experiment = Experiment.create(sampler, seed=7)
    assert experiment.rule.k_target == config["phase_space.recovery.k_target"]
    assert experiment.campaign.seed == 7


def test_a_map_over_other_axes_than_the_sampler_is_refused(sampler):
    from uvex_transients.phase_space.map import PhaseSpaceMap

    other = PhaseSpaceMap(Space([Axis("a", 0.0, 1.0), Axis("c", 0.0, 1.0)]))
    with pytest.raises(ValueError, match="over"):
        Experiment(sampler, other, RULE, CAMPAIGN)


def test_running_adds_points_up_to_the_budget_and_returns_the_experiment(experiment):
    assert experiment.run() is experiment
    assert len(experiment.phase_map) == 18
    assert experiment.phase_map.table.colnames == ["a", "b", "k", "n", "n_chunks"]


def test_a_run_is_reproducible_from_its_seeds(sampler):
    def run():
        return Experiment.create(sampler, rule=RULE, campaign=CAMPAIGN, seed=3, n_restarts=0, n_candidates=128).run()

    one, two = run(), run()
    np.testing.assert_array_equal(one.phase_map.points, two.phase_map.points)
    assert list(one.phase_map.table["k"]) == list(two.phase_map.table["k"])


def test_a_callback_sees_the_map_grow(experiment):
    sizes = []
    experiment.run(callback=lambda phase_map: sizes.append(len(phase_map)))
    assert sizes == [10, 14, 18]


def test_the_grid_table_has_the_fraction_and_the_products_of_the_sampler(experiment):
    experiment.run()
    table = experiment.grid_table(6)
    assert len(table) == 36
    assert {"a", "b", "f", "f_lo", "f_hi", "sigma_log", "true_probability"} <= set(table.colnames)
    np.testing.assert_allclose(
        table["true_probability"], experiment.sampler.probability(np.column_stack([table["a"], table["b"]]))
    )


def test_the_grid_size_defaults_to_the_configuration(experiment):
    experiment.run()
    n = config["phase_space.map.grid_size"]
    assert len(experiment.grid_table()) == n * n


# ---------------------------------------------------------------------------- #
# The points file                                                              #
# ---------------------------------------------------------------------------- #
def test_the_header_records_everything_that_rebuilds_the_experiment(experiment):
    header = experiment.header
    assert header["schema_version"] == SCHEMA_VERSION == 2
    assert header["sampler"]["kind"] == "logistic"
    assert header["rule"] == {"chunk_size": 100, "k_target": 15, "n_max": 1000}
    assert header["campaign"]["max_points"] == 18 and header["campaign"]["seed"] == 3
    assert set(header["map"]) == {"acquisition", "n_candidates", "floor", "n_restarts"}
    assert Space.from_dict(header["space"]) == experiment.phase_map.space


def test_a_file_stores_no_import_path(experiment, tmp_path):
    experiment.run()
    experiment.save(tmp_path / "points.ecsv")
    text = (tmp_path / "points.ecsv").read_text()
    assert "uvex_transients." not in text
    assert "kind: logistic" in text


def test_an_experiment_round_trips_through_its_file(experiment, tmp_path):
    experiment.run()
    path = tmp_path / "points.ecsv"
    experiment.save(path)
    loaded = Experiment.load(path)

    assert isinstance(loaded.sampler, LogisticSampler)
    assert loaded.sampler.settings == experiment.sampler.settings
    assert loaded.rule == experiment.rule and loaded.campaign == experiment.campaign
    assert loaded.phase_map.space == experiment.phase_map.space
    assert loaded.phase_map.settings == experiment.phase_map.settings
    np.testing.assert_array_equal(loaded.phase_map.points, experiment.phase_map.points)
    assert list(loaded.phase_map.table["k"]) == list(experiment.phase_map.table["k"])

    # The fitted map is the same, so a table made from the file is the table made from the run
    a, b = experiment.grid_table(5), loaded.grid_table(5)
    np.testing.assert_allclose(a["f"], b["f"], rtol=1e-6)


def test_a_loaded_experiment_can_continue_running(experiment, tmp_path):
    experiment.run()
    experiment.save(tmp_path / "points.ecsv")
    loaded = Experiment.load(tmp_path / "points.ecsv")
    loaded.campaign = CampaignSettings(**{**experiment.campaign.__dict__, "max_points": 22})
    loaded.run()
    assert len(loaded.phase_map) == 22
    np.testing.assert_array_equal(loaded.phase_map.points[:18], experiment.phase_map.points)


def test_a_sampler_can_be_given_instead_of_rebuilt(experiment, tmp_path):
    experiment.run()
    experiment.save(tmp_path / "points.ecsv")
    other = LogisticSampler(experiment.sampler.space, center=[0.1, 0.1], slope=[1.0, 1.0])
    loaded = Experiment.load(tmp_path / "points.ecsv", sampler=other)
    assert loaded.sampler is other


def test_a_sampler_with_no_kind_cannot_be_saved(sampler):
    class Anonymous(LogisticSampler):
        pass

    assert Anonymous.kind is None
    experiment = Experiment.create(Anonymous(sampler.space, [0.5, 0.0], [1.0, 1.0]), rule=RULE, campaign=CAMPAIGN)
    with pytest.raises(ValueError, match="no registered kind"):
        experiment.table()


def test_a_file_that_names_an_unknown_kind_lists_the_known_ones(experiment, tmp_path):
    experiment.run()
    table = experiment.table()
    table.meta["sampler"] = {"kind": "no_such_kind", "settings": {}}
    table.write(tmp_path / "bad.ecsv", format="ascii.ecsv")
    with pytest.raises(KeyError, match="logistic"):
        Experiment.load(tmp_path / "bad.ecsv")


# ---------------------------------------------------------------------------- #
# The earlier format                                                           #
# ---------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def schedule(make_schedule):
    return make_schedule(n_sched=20)


@pytest.fixture(scope="module")
def legacy_file(schedule, tmp_path_factory):
    """Return a points file of the earlier format, written by the evaluator and the sampler that wrote them."""
    evaluator = RecoveryEvaluator(
        schedule,
        mission="uvex_fast",
        nside=32,
        order="nested",
        time_bins=3,
        chunk_size=200,
        n_max=600,
        k_target=5,
        n_visits=1,
        seed=3,
    )
    sampler = PhaseSpaceSampler(
        evaluator,
        [(40.0, 44.5), (-1.0, 2.0)],
        names=("log_luminosity", "log_half_time"),
        n_initial=4,
        batch_size=2,
        max_points=6,
        n_restarts=0,
        n_candidates=64,
        tolerance=0.0,
        seed=1,
    )
    sampler.run()
    path = tmp_path_factory.mktemp("legacy") / "points.ecsv"
    sampler.save(path)
    return path, sampler, evaluator


def test_a_file_of_the_earlier_format_is_read_as_the_canonical_experiment(schedule, legacy_file):
    path, legacy, evaluator = legacy_file
    assert "schema_version" not in Table.read(path, format="ascii.ecsv").meta

    experiment = Experiment.load(path, schedule=schedule)
    assert isinstance(experiment.sampler, CanonicalSampler)
    assert experiment.rule == StoppingRule(chunk_size=200, k_target=5, n_max=600)
    assert experiment.campaign.max_points == 6 and experiment.campaign.n_initial == 4
    assert experiment.campaign.seed == 3

    # The points are the ones the legacy sampler measured, and the model settings are carried over
    np.testing.assert_array_equal(experiment.phase_map.points, legacy.points)
    assert list(experiment.phase_map.table["k"]) == list(legacy.table["k"])
    assert experiment.phase_map.settings["acquisition"] == legacy.acquisition


def test_the_axes_of_an_old_file_get_their_labels_from_the_sampler(schedule, legacy_file):
    experiment = Experiment.load(legacy_file[0], schedule=schedule)
    space = experiment.phase_map.space
    assert space.names == ("log_luminosity", "log_half_time")
    np.testing.assert_array_equal(space.bounds, [[40.0, 44.5], [-1.0, 2.0]])
    assert [axis.unit for axis in space] == ["erg/s", "day"]
    assert "FUV" in space["log_luminosity"].label


def test_an_old_file_gives_the_same_map_through_the_new_loader_as_through_the_old_sampler(schedule, legacy_file):
    path, legacy, evaluator = legacy_file
    experiment = Experiment.load(path, schedule=schedule)
    grid = experiment.grid_table(5)
    old = legacy.map_table(5)
    np.testing.assert_allclose(grid["f"], old["f"], rtol=0.05, atol=1e-4)
    np.testing.assert_allclose(grid["volume_time"], old["volume_time"], rtol=1e-10)
    np.testing.assert_allclose(grid["recovered_volume_time"], grid["f"] * grid["volume_time"], rtol=1e-12)


def test_a_file_that_recorded_another_evaluator_is_refused(legacy_file, tmp_path):
    table = Table.read(legacy_file[0], format="ascii.ecsv")
    table.meta["evaluator"] = {**table.meta["evaluator"], "class": "uvex_transients.phase_space.evaluator.Other"}
    table.write(tmp_path / "other.ecsv", format="ascii.ecsv")
    with pytest.raises(ValueError, match="Cannot read"):
        Experiment.load(tmp_path / "other.ecsv")


def test_loading_a_sampler_needs_a_kind_the_experiment_can_rebuild():
    assert issubclass(CanonicalSampler, Sampler) and CanonicalSampler.kind == "canonical"
