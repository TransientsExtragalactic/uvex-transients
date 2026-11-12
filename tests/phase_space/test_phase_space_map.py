"""Tests for `PhaseSpaceMap`: the data, the Gaussian process model, the derived maps and the acquisition hook."""

import numpy as np
import pytest
from astropy.table import Table

from uvex_transients.phase_space.map import PhaseSpaceMap
from uvex_transients.phase_space.sampling import Estimate, LogisticSampler, Scale, StoppingRule, Weighted
from uvex_transients.phase_space.space import Axis, Space

pytest.importorskip("sklearn")

RULE = StoppingRule(chunk_size=200, k_target=20, n_max=2000)


@pytest.fixture
def space():
    return Space([Axis("a", 0.0, 1.0, label="a"), Axis("b", -1.0, 1.0, label="b")])


def measured_map(space, n_points=30, seed=0, **kwargs):
    """Return a map holding `n_points` of a Sobol design measured on a logistic sampler, and that sampler."""
    sampler = LogisticSampler(space, center=[0.5, 0.0], slope=[8.0, 4.0])
    new = PhaseSpaceMap(space, seed=seed, n_restarts=0, **kwargs)
    root = np.random.SeedSequence(seed + 100)
    for x in new.initial_design(n_points):
        outcome = RULE.run(sampler, x, root.spawn(1)[0])
        new.add(x, outcome.k, outcome.n, n_chunks=outcome.extras["n_chunks"])
    return new, sampler


# ---------------------------------------------------------------------------- #
# Construction                                                                 #
# ---------------------------------------------------------------------------- #
def test_a_map_takes_its_model_settings_from_the_configuration_and_records_them(space):
    new = PhaseSpaceMap(space)
    settings = new.settings
    assert set(settings) == {"acquisition", "n_candidates", "floor", "n_restarts"}
    assert new.space is space and len(new) == 0

    explicit = PhaseSpaceMap(space, acquisition="variance", n_candidates=64, floor=0.01, n_restarts=1)
    assert explicit.settings == {"acquisition": "variance", "n_candidates": 64, "floor": 0.01, "n_restarts": 1}


def test_a_map_refuses_an_unknown_acquisition_or_a_floor_outside_the_unit_interval(space):
    with pytest.raises(ValueError, match="acquisition"):
        PhaseSpaceMap(space, acquisition="random")
    for floor in (0.0, 1.0, -0.1, 2.0):
        with pytest.raises(ValueError, match="floor"):
            PhaseSpaceMap(space, floor=floor)


# ---------------------------------------------------------------------------- #
# Data                                                                         #
# ---------------------------------------------------------------------------- #
def test_an_empty_map_has_the_columns_of_its_space_and_the_counts(space):
    table = PhaseSpaceMap(space).table
    assert table.colnames == ["a", "b", "k", "n"] and len(table) == 0
    assert PhaseSpaceMap(space).points.shape == (0, 2)


def test_points_are_added_with_their_counts_and_extras(space):
    new = PhaseSpaceMap(space)
    new.add([0.25, -0.5], 3, 10, z_limit=0.4)
    new.add(np.array([0.75, 0.5]), 0, 10, z_limit=0.2)
    assert len(new) == 2
    table = new.table
    assert table.colnames == ["a", "b", "k", "n", "z_limit"]
    np.testing.assert_array_equal(new.points, [[0.25, -0.5], [0.75, 0.5]])
    assert list(table["k"]) == [3, 0] and list(table["n"]) == [10, 10]


def test_a_point_is_checked_when_it_is_added(space):
    new = PhaseSpaceMap(space)
    with pytest.raises(ValueError, match="shape"):
        new.add([0.5], 1, 2)
    for k, n in ((-1, 5), (6, 5), (0, 0)):
        with pytest.raises(ValueError, match="0 <= k <= n"):
            new.add([0.5, 0.0], k, n)
    assert len(new) == 0


def test_a_map_resumes_from_a_table_and_refuses_one_missing_a_column(space):
    first, _ = measured_map(space, n_points=8)
    resumed = PhaseSpaceMap(space, table=first.table)
    assert len(resumed) == 8
    np.testing.assert_array_equal(resumed.points, first.points)

    with pytest.raises(ValueError, match="no 'b' column"):
        PhaseSpaceMap(space, table=Table({"a": [0.1], "k": [1], "n": [2]}))


# ---------------------------------------------------------------------------- #
# The model                                                                    #
# ---------------------------------------------------------------------------- #
def test_the_model_needs_two_points(space):
    new = PhaseSpaceMap(space)
    new.add([0.5, 0.0], 1, 2)
    with pytest.raises(ValueError, match="at least two"):
        new.fit()


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_the_model_recovers_a_known_success_fraction_in_any_dimension(ndim):
    """The fitted map follows the logistic truth, in one, two and three dimensions."""
    space = Space([Axis(f"x{i}", 0.0, 1.0) for i in range(ndim)])
    sampler = LogisticSampler(space, center=[0.5] * ndim, slope=[6.0] * ndim)
    new = PhaseSpaceMap(space, seed=1, n_restarts=0)
    root = np.random.SeedSequence(2)
    for x in new.initial_design(24 * ndim):
        outcome = RULE.run(sampler, x, root.spawn(1)[0])
        new.add(x, outcome.k, outcome.n)

    grid = space.grid(5 if ndim == 3 else 9)
    estimate = new.estimate(grid)
    truth = sampler.probability(grid)
    assert np.mean(np.abs(estimate.p - truth)) < 0.06
    assert np.max(np.abs(estimate.p - truth)) < 0.25
    assert np.all((estimate.p_lo <= estimate.p) & (estimate.p <= estimate.p_hi))


def test_the_estimate_has_the_shape_of_the_points_without_their_last_axis(space):
    new, _ = measured_map(space)
    x = np.random.default_rng(0).random((3, 4, 2)) * [1.0, 2.0] + [0.0, -1.0]
    estimate = new.estimate(x)
    assert isinstance(estimate, Estimate) and estimate.shape == (3, 4)
    flat = new.estimate(x.reshape(-1, 2))
    np.testing.assert_array_equal(estimate.p.reshape(-1), flat.p)
    with pytest.raises(ValueError, match="2 coordinates"):
        new.estimate(np.zeros((4, 3)))


def test_a_wider_interval_with_more_sigma(space):
    new, _ = measured_map(space)
    x = np.array([[0.3, 0.1], [0.7, -0.4]])
    narrow, wide = new.estimate(x, n_sigma=1.0), new.estimate(x, n_sigma=3.0)
    assert np.all(wide.p_hi - wide.p_lo > narrow.p_hi - narrow.p_lo)
    np.testing.assert_allclose(wide.p, narrow.p)


def test_the_posterior_is_wider_far_from_the_data(space):
    sampler = LogisticSampler(space, center=[0.5, 0.0], slope=[8.0, 4.0])
    new = PhaseSpaceMap(space, seed=0, n_restarts=0)
    root = np.random.SeedSequence(3)
    # Measure only the lower half of the first axis
    for x in new.initial_design(40):
        x = x * [0.5, 1.0] + [0.0, 0.0]
        outcome = RULE.run(sampler, x, root.spawn(1)[0])
        new.add(x, outcome.k, outcome.n)
    near = new.recovery([[0.25, 0.0]])["sigma_logit"][0]
    far = new.recovery([[0.95, 0.0]])["sigma_logit"][0]
    assert far > near


# ---------------------------------------------------------------------------- #
# Derived maps                                                                 #
# ---------------------------------------------------------------------------- #
def test_evaluate_gives_the_fraction_the_uncertainty_and_every_product_column(space):
    new, sampler = measured_map(space)
    products = [Scale("double_a", lambda x: 2.0 * x[..., 0]), Weighted("weighted", lambda x: 1.0 + x[..., 1] ** 2)]
    x = space.grid(4)
    columns = new.evaluate(x, products)
    assert list(columns) == ["f", "f_lo", "f_hi", "sigma_log", "double_a", "weighted", "weighted_lo", "weighted_hi"]
    np.testing.assert_allclose(columns["double_a"], 2.0 * x[:, 0])
    np.testing.assert_allclose(columns["weighted"], columns["f"] * (1.0 + x[:, 1] ** 2))
    assert all(values.shape == (16,) for values in columns.values())


def test_a_grid_table_has_the_coordinates_the_columns_and_the_space(space):
    new, sampler = measured_map(space)
    table = new.grid_table(5, sampler.products)
    assert table.colnames[:2] == ["a", "b"] and "f" in table.colnames and "true_probability" in table.colnames
    assert len(table) == 25
    np.testing.assert_allclose(
        table["true_probability"], sampler.probability(np.column_stack([table["a"], table["b"]]))
    )
    assert Space.from_dict(table.meta["space"]) == space
    assert table.meta["products"] == [{"name": "true_probability", "label": r"true $p$", "unit": ""}]


def test_a_grid_table_can_cover_a_smaller_rectangle(space):
    new, _ = measured_map(space)
    table = new.grid_table([3, 4], bounds=[[0.2, 0.4], [-0.5, 0.5]])
    assert len(table) == 12
    assert table["a"].min() == pytest.approx(0.2) and table["b"].max() == pytest.approx(0.5)
    assert Space.from_dict(table.meta["space"]).bounds.tolist() == [[0.2, 0.4], [-0.5, 0.5]]


# ---------------------------------------------------------------------------- #
# The acquisition hook                                                         #
# ---------------------------------------------------------------------------- #
def test_the_initial_design_covers_the_space(space):
    new = PhaseSpaceMap(space, seed=4)
    design = new.initial_design(16)
    assert design.shape == (16, 2) and np.all(space.contains(design))


@pytest.mark.parametrize("acquisition", ["variance", "imse"])
def test_propose_returns_distinct_points_of_the_space_and_an_uncertainty(space, acquisition):
    new, _ = measured_map(space, acquisition=acquisition, n_candidates=128)
    points, uncertainty = new.propose(5, RULE.expected_n)
    assert points.shape == (5, 2) and np.all(space.contains(points))
    assert len({tuple(point) for point in points}) == 5
    assert uncertainty > 0.0 and np.isfinite(uncertainty)


def test_propose_is_reproducible_from_the_seed_of_the_map(space):
    one, _ = measured_map(space, n_candidates=128)
    two, _ = measured_map(space, n_candidates=128)
    np.testing.assert_array_equal(one.propose(4, RULE.expected_n)[0], two.propose(4, RULE.expected_n)[0])


def test_more_points_lower_the_uncertainty_that_a_campaign_stops_on(space):
    few, _ = measured_map(space, n_points=12, n_candidates=256)
    many, _ = measured_map(space, n_points=60, n_candidates=256)
    assert many.propose(4, RULE.expected_n)[1] < few.propose(4, RULE.expected_n)[1]


def test_propose_works_in_three_dimensions():
    space = Space([Axis(name, 0.0, 1.0) for name in "abc"])
    sampler = LogisticSampler(space, center=[0.5] * 3, slope=[5.0] * 3)
    new = PhaseSpaceMap(space, seed=0, n_restarts=0, n_candidates=128)
    root = np.random.SeedSequence(0)
    for x in new.initial_design(30):
        outcome = RULE.run(sampler, x, root.spawn(1)[0])
        new.add(x, outcome.k, outcome.n)
    points, _ = new.propose(4, RULE.expected_n)
    assert points.shape == (4, 3) and np.all(space.contains(points))
