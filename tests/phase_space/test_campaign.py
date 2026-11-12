"""Tests for `run_campaign` and `point_seed`."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from uvex_transients.phase_space.campaign import point_seed, run_campaign
from uvex_transients.phase_space.map import PhaseSpaceMap
from uvex_transients.phase_space.sampling import LogisticSampler, StoppingRule
from uvex_transients.phase_space.space import Axis, Space

pytest.importorskip("sklearn")

RULE = StoppingRule(chunk_size=100, k_target=15, n_max=1000)
SETTINGS = {"n_initial": 12, "batch_size": 4, "max_points": 24, "tolerance": 0.0}


@pytest.fixture
def space():
    return Space([Axis("a", 0.0, 1.0), Axis("b", -1.0, 1.0)])


@pytest.fixture
def sampler(space):
    return LogisticSampler(space, center=[0.5, 0.0], slope=[8.0, 4.0])


def new_map(space, seed=0):
    return PhaseSpaceMap(space, seed=seed, n_restarts=0, n_candidates=128)


def campaign(space, sampler, seed=5, **kwargs):
    return run_campaign(new_map(space), sampler, RULE, seed=seed, **{**SETTINGS, **kwargs})


# ---------------------------------------------------------------------------- #
# Seeds                                                                        #
# ---------------------------------------------------------------------------- #
def test_a_point_seed_is_a_pure_function_of_the_root_and_the_index():
    root = np.random.SeedSequence(7)
    first = point_seed(root, 3)
    assert root.n_children_spawned == 0
    assert point_seed(root, 3).generate_state(2).tolist() == first.generate_state(2).tolist()
    assert point_seed(root, 4).generate_state(2).tolist() != first.generate_state(2).tolist()
    assert point_seed(np.random.SeedSequence(8), 3).generate_state(2).tolist() != first.generate_state(2).tolist()


def test_a_point_seed_does_not_depend_on_how_much_of_the_root_was_used():
    root = np.random.SeedSequence(7)
    before = point_seed(root, 2).generate_state(2).tolist()
    root.spawn(5)
    assert point_seed(root, 2).generate_state(2).tolist() == before


# ---------------------------------------------------------------------------- #
# The loop                                                                     #
# ---------------------------------------------------------------------------- #
def test_a_campaign_measures_an_initial_design_then_batches_up_to_the_budget(space, sampler):
    result = campaign(space, sampler)
    assert len(result) == 24
    table = result.table
    assert table.colnames == ["a", "b", "k", "n", "n_chunks"]
    assert np.all(space.contains(result.points))
    assert np.all((table["k"] >= 0) & (table["k"] <= table["n"]))
    assert np.all(table["n"] % RULE.chunk_size == 0)


def test_a_campaign_returns_the_map_it_was_given_fitted(space, sampler):
    phase_map = new_map(space)
    assert run_campaign(phase_map, sampler, RULE, seed=1, **SETTINGS) is phase_map
    assert phase_map._gp is not None


def test_a_campaign_stops_when_the_map_is_good_enough(space, sampler):
    result = campaign(space, sampler, tolerance=10.0)
    assert len(result) == SETTINGS["n_initial"]


def test_the_budget_is_not_overrun_by_a_batch_that_does_not_fit(space, sampler):
    assert len(campaign(space, sampler, max_points=14)) == 14


def test_a_callback_is_called_after_the_design_and_every_batch(space, sampler):
    sizes = []
    campaign(space, sampler, callback=lambda m: sizes.append(len(m)))
    assert sizes == [12, 16, 20, 24]


def test_a_campaign_takes_what_is_not_given_from_the_configuration(space, sampler):
    result = run_campaign(new_map(space), sampler, RULE, seed=1, n_initial=5, max_points=5)
    assert len(result) == 5


@pytest.mark.parametrize("name", ["n_initial", "batch_size", "max_points"])
@pytest.mark.parametrize("value", [0, -2, 1.5])
def test_the_counts_have_to_be_positive_integers(space, sampler, name, value):
    with pytest.raises(ValueError, match="positive integer"):
        campaign(space, sampler, **{name: value})


def test_the_sampler_has_to_be_over_the_space_of_the_map(space, sampler):
    other = Space([Axis("a", 0.0, 1.0), Axis("c", -1.0, 1.0)])
    with pytest.raises(ValueError, match="over"):
        run_campaign(new_map(other), sampler, RULE, **SETTINGS)


# ---------------------------------------------------------------------------- #
# Reproducibility                                                              #
# ---------------------------------------------------------------------------- #
def test_the_same_seeds_give_the_same_map(space, sampler):
    one, two = campaign(space, sampler), campaign(space, sampler)
    np.testing.assert_array_equal(one.points, two.points)
    assert list(one.table["k"]) == list(two.table["k"])


def test_another_seed_measures_the_same_points_differently(space, sampler):
    one, two = campaign(space, sampler, seed=5), campaign(space, sampler, seed=6)
    np.testing.assert_array_equal(one.points[:12], two.points[:12])
    assert list(one.table["k"][:12]) != list(two.table["k"][:12])


def test_the_order_a_batch_is_measured_in_does_not_change_the_result(space, sampler):
    """Each point carries the seed of its index, so a reversed evaluation gives the same table, which a parallel run needs."""

    def reversed_map(function, tasks):
        return list(map(function, reversed(list(tasks))))[::-1]

    forward = campaign(space, sampler)
    backward = campaign(space, sampler, map_fn=reversed_map)
    np.testing.assert_array_equal(forward.points, backward.points)
    assert list(forward.table["k"]) == list(backward.table["k"])
    assert list(forward.table["n"]) == list(backward.table["n"])


def test_a_thread_pool_gives_the_same_map_as_a_loop(space, sampler):
    with ThreadPoolExecutor(max_workers=4) as pool:
        threaded = campaign(space, sampler, map_fn=pool.map)
    serial = campaign(space, sampler)
    np.testing.assert_array_equal(serial.points, threaded.points)
    assert list(serial.table["k"]) == list(threaded.table["k"])


def test_resuming_continues_the_sequence_of_seeds(space, sampler):
    """The points measured after a resume use the seeds of their own indices, as if the campaign had not stopped."""
    first = campaign(space, sampler, max_points=16)
    resumed = run_campaign(
        PhaseSpaceMap(space, seed=0, n_restarts=0, n_candidates=128, table=first.table),
        sampler,
        RULE,
        seed=5,
        **{**SETTINGS, "max_points": 20},
    )
    assert len(resumed) == 20
    np.testing.assert_array_equal(resumed.points[:16], first.points)

    # The 17th point is measured with the seed of index 16, whatever the map did before
    x = resumed.points[16]
    expected = RULE.run(sampler, x, point_seed(np.random.SeedSequence(5), 16))
    assert (resumed.table["k"][16], resumed.table["n"][16]) == (expected.k, expected.n)


# ---------------------------------------------------------------------------- #
# What it finds                                                                #
# ---------------------------------------------------------------------------- #
def test_a_campaign_recovers_the_known_success_fraction(space, sampler):
    """The end to end check: a map built by a campaign on a sampler with a known answer follows that answer."""
    result = run_campaign(
        new_map(space), sampler, RULE, seed=2, n_initial=24, batch_size=6, max_points=48, tolerance=0.0
    )
    grid = space.grid(9)
    estimate = result.estimate(grid)
    truth = sampler.probability(grid)
    assert np.mean(np.abs(estimate.p - truth)) < 0.06
    inside = (truth > 0.05) & (truth < 0.95)
    assert (
        np.mean((truth[inside] >= estimate.p_lo[inside] - 0.05) & (truth[inside] <= estimate.p_hi[inside] + 0.05)) > 0.7
    )


def test_a_campaign_spends_nothing_on_the_saturated_corner_and_most_where_the_fraction_is_informative(space, sampler):
    """The acquisition chases the relative uncertainty of the fraction, so it leaves the corner where it is one alone."""
    result = run_campaign(
        new_map(space), sampler, RULE, seed=3, n_initial=16, batch_size=6, max_points=46, tolerance=0.0
    )
    p = sampler.probability(result.points[16:])
    uniform = sampler.probability(space.grid(41))

    # A uniform grid puts 14% of its points where the fraction is above 0.98, and the campaign puts none
    assert (uniform > 0.98).mean() > 0.1
    assert (p > 0.98).mean() < 0.05

    # And a good share of what it adds is on the transition rather than in the low tail
    assert ((p > 0.02) & (p < 0.98)).mean() > 0.3


# ---------------------------------------------------------------------------- #
# Worker processes                                                             #
# ---------------------------------------------------------------------------- #
class _Unpicklable(LogisticSampler):
    """A sampler that cannot be pickled, as a survey-backed one cannot (its schedule holds a satellite orbit)."""

    def __init__(self, space, center, slope):
        super().__init__(space, center, slope)
        import threading

        self._lock = threading.Lock()


def test_a_sampler_that_cannot_be_pickled_cannot_go_to_an_ordinary_process_pool(space):
    import pickle

    with pytest.raises(TypeError):
        pickle.dumps(_Unpicklable(space, [0.5, 0.0], [8.0, 4.0]))


def test_forked_workers_measure_what_the_loop_does_with_a_sampler_that_cannot_be_pickled(space):
    from uvex_transients.phase_space.campaign import fork_map

    sampler = _Unpicklable(space, [0.5, 0.0], [8.0, 4.0])
    serial = campaign(space, sampler)
    with fork_map(sampler, RULE, processes=2) as map_fn:
        forked = campaign(space, sampler, map_fn=map_fn)
    np.testing.assert_array_equal(serial.points, forked.points)
    assert list(serial.table["k"]) == list(forked.table["k"])
    assert list(serial.table["n"]) == list(forked.table["n"])


def test_the_workers_share_what_the_sampler_had_computed_when_they_were_forked(space):
    """A table built before the pool is made is inherited, not rebuilt in every worker."""
    from uvex_transients.phase_space.campaign import _FORKED, fork_map

    sampler = _Unpicklable(space, [0.5, 0.0], [8.0, 4.0])
    with fork_map(sampler, RULE, processes=1) as map_fn:
        assert _FORKED["sampler"] is sampler and _FORKED["rule"] is RULE
        assert map_fn(None, []) == []
    assert _FORKED == {}
