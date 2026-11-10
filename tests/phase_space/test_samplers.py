"""Tests for `Sampler`, `Outcome`, `Product`, `StoppingRule` and the sampler registry."""

import numpy as np
import pytest

from uvex_transients.phase_space.samplers import (
    LogisticSampler,
    Outcome,
    Product,
    Sampler,
    StoppingRule,
    available_samplers,
    get_sampler,
)
from uvex_transients.phase_space.samplers import base as base_module
from uvex_transients.phase_space.space import Axis, Space
from uvex_transients.utils import config


@pytest.fixture
def space():
    return Space([Axis("a", 0.0, 1.0), Axis("b", -1.0, 1.0)])


@pytest.fixture
def registry():
    """Let a test register samplers, and leave the registry as it was."""
    before = dict(base_module._REGISTRY)
    yield base_module._REGISTRY
    base_module._REGISTRY.clear()
    base_module._REGISTRY.update(before)


class _Scripted(Sampler):
    """A sampler that returns the successes it is told to, and records the tries it is asked for."""

    def __init__(self, successes, space):
        self._successes = list(successes)
        self._space = space
        self.calls = []

    @property
    def space(self):
        return self._space

    def draw(self, x, n, seed):
        self.calls.append((np.asarray(x), n, seed))
        return self._successes[len(self.calls) - 1] if len(self.calls) <= len(self._successes) else 0

    @property
    def settings(self):
        return {}

    @classmethod
    def from_settings(cls, settings, **context):
        raise NotImplementedError


# ---------------------------------------------------------------------------- #
# Outcome and Product                                                          #
# ---------------------------------------------------------------------------- #
def test_an_outcome_holds_its_counts_and_extras():
    outcome = Outcome(3, 10, {"n_chunks": 2})
    assert (outcome.k, outcome.n, outcome.extras) == (3, 10, {"n_chunks": 2})
    assert outcome.fraction == pytest.approx(0.3)
    assert Outcome(0, 5).extras == {}


def test_an_outcome_converts_integers_and_copies_its_extras():
    extras = {"x": 1.0}
    outcome = Outcome(np.int64(2), np.int64(4), extras)
    assert type(outcome.k) is int and type(outcome.n) is int
    extras["y"] = 2.0
    assert outcome.extras == {"x": 1.0}


@pytest.mark.parametrize("k, n", [(-1, 5), (6, 5), (0, 0), (1, 0), (0, -2)])
def test_an_outcome_needs_k_between_zero_and_n(k, n):
    with pytest.raises(ValueError, match="0 <= k <= n"):
        Outcome(k, n)


def test_an_outcome_and_a_product_are_immutable():
    with pytest.raises(AttributeError):
        Outcome(1, 2).k = 2
    with pytest.raises(AttributeError):
        Product("x", "scale").name = "y"
    assert Product("rvt", "volume_time", label="label", unit="Mpc3 yr").unit == "Mpc3 yr"


# ---------------------------------------------------------------------------- #
# StoppingRule                                                                 #
# ---------------------------------------------------------------------------- #
@pytest.mark.parametrize("chunk, k, n_max", [(0, 5, 100), (10, 0, 100), (10, 5, 9)])
def test_a_rule_needs_a_chunk_a_target_and_room_for_a_chunk(chunk, k, n_max):
    with pytest.raises(ValueError, match="Need chunk_size >= 1"):
        StoppingRule(chunk, k, n_max)


def test_a_rule_converts_to_integers_and_is_immutable():
    rule = StoppingRule(10.0, 5.0, 100.0)
    assert (rule.chunk_size, rule.k_target, rule.n_max) == (10, 5, 100)
    assert all(type(v) is int for v in (rule.chunk_size, rule.k_target, rule.n_max))
    with pytest.raises(AttributeError):
        rule.n_max = 1


def test_a_rule_stops_at_the_target_and_counts_chunks(space):
    sampler = _Scripted([3, 4, 5, 6], space)
    outcome = StoppingRule(chunk_size=20, k_target=10, n_max=1000).run(sampler, [0.5, 0.0], np.random.SeedSequence(1))
    assert (outcome.k, outcome.n, outcome.extras["n_chunks"]) == (12, 60, 3)
    assert [call[1] for call in sampler.calls] == [20, 20, 20]


def test_a_rule_stops_at_the_budget_when_the_target_is_out_of_reach(space):
    sampler = _Scripted([], space)
    outcome = StoppingRule(chunk_size=20, k_target=10, n_max=100).run(sampler, [0.5, 0.0], np.random.SeedSequence(1))
    assert (outcome.k, outcome.n, outcome.extras["n_chunks"]) == (0, 100, 5)


def test_the_last_chunk_may_take_a_point_past_the_budget(space):
    outcome = StoppingRule(chunk_size=30, k_target=10, n_max=100).run(
        _Scripted([], space), [0.5, 0.0], np.random.SeedSequence(1)
    )
    assert outcome.n == 120


def test_a_point_that_reaches_the_target_in_one_chunk_takes_one(space):
    outcome = StoppingRule(chunk_size=20, k_target=5, n_max=1000).run(
        _Scripted([9], space), [0.5, 0.0], np.random.SeedSequence(1)
    )
    assert (outcome.k, outcome.n, outcome.extras["n_chunks"]) == (9, 20, 1)


def test_every_chunk_draws_from_its_own_child_of_the_seed(space):
    """The chunks are seeded with successive children of the point's seed, which is advanced by them."""
    sampler = _Scripted([], space)
    seed = np.random.SeedSequence(7)
    StoppingRule(chunk_size=10, k_target=1, n_max=30).run(sampler, [0.5, 0.0], seed)
    reference = np.random.SeedSequence(7)
    for _, _, child in sampler.calls:
        expected = reference.spawn(1)[0]
        assert child.spawn_key == expected.spawn_key and child.entropy == expected.entropy
    assert seed.n_children_spawned == 3


def test_a_rule_passes_the_point_as_a_float_array(space):
    sampler = _Scripted([1], space)
    StoppingRule(10, 1, 10).run(sampler, [1, 0], np.random.SeedSequence(0))
    assert sampler.calls[0][0].dtype == np.float64
    np.testing.assert_array_equal(sampler.calls[0][0], [1.0, 0.0])


def test_the_expected_tries_are_enough_for_the_target_within_the_limits():
    rule = StoppingRule(chunk_size=100, k_target=10, n_max=5000)
    np.testing.assert_allclose(rule.expected_n([1.0, 0.5, 0.1, 1e-3, 1e-9]), [100.0, 100.0, 100.0, 5000.0, 5000.0])
    np.testing.assert_allclose(rule.expected_n([0.05, 0.02]), [200.0, 500.0])
    assert rule.expected_n(0.5).shape == ()


def test_a_rule_takes_what_is_not_given_from_the_configuration():
    rule = StoppingRule.from_config()
    assert rule.chunk_size == config["phase_space.recovery.chunk_size"]
    assert rule.k_target == config["phase_space.recovery.k_target"]
    assert rule.n_max == config["phase_space.recovery.n_max"]

    explicit = StoppingRule.from_config(k_target=7)
    assert explicit.k_target == 7 and explicit.chunk_size == rule.chunk_size


# ---------------------------------------------------------------------------- #
# The Sampler base class                                                       #
# ---------------------------------------------------------------------------- #
def test_the_base_class_cannot_be_instantiated_and_has_four_abstract_pieces():
    assert Sampler.__abstractmethods__ == {"space", "draw", "settings", "from_settings"}
    with pytest.raises(TypeError, match="abstract"):
        Sampler()


def test_a_sampler_has_no_scales_or_products_unless_it_says_so(space):
    sampler = _Scripted([], space)
    assert sampler.scales(np.zeros((4, 2))) == {}
    assert Sampler.products == ()


def test_a_class_that_names_a_kind_is_registered(registry, space):
    class Mine(_Scripted, kind="mine"):
        pass

    assert Mine.kind == "mine"
    assert get_sampler("mine") is Mine
    assert "mine" in available_samplers()


def test_a_class_that_names_no_kind_is_not_registered_and_does_not_inherit_one(registry):
    class Registered(_Scripted, kind="parent"):
        pass

    class Child(Registered):
        pass

    assert Child.kind is None
    assert get_sampler("parent") is Registered
    assert _Scripted.kind is None
    assert "Child" not in str(available_samplers())


def test_a_kind_can_be_registered_only_once(registry):
    class First(_Scripted, kind="taken"):
        pass

    with pytest.raises(TypeError, match="already registered"):

        class Second(_Scripted, kind="taken"):
            pass


@pytest.mark.parametrize("kind", ["", 3])
def test_a_kind_is_a_non_empty_string(registry, kind):
    with pytest.raises(TypeError, match="non-empty string"):

        class Bad(_Scripted, kind=kind):
            pass


def test_an_unknown_kind_lists_the_registered_ones():
    with pytest.raises(KeyError, match="logistic"):
        get_sampler("nothing_of_the_sort")


# ---------------------------------------------------------------------------- #
# LogisticSampler                                                              #
# ---------------------------------------------------------------------------- #
@pytest.fixture
def logistic(space):
    return LogisticSampler(space, center=[0.5, 0.0], slope=[4.0, 2.0])


def test_the_logistic_sampler_is_registered_and_a_sampler(logistic):
    assert LogisticSampler.kind == "logistic"
    assert get_sampler("logistic") is LogisticSampler
    assert isinstance(logistic, Sampler)
    assert logistic.space.names == ("a", "b")


def test_the_probability_is_half_on_the_boundary_and_rises_along_the_slope(logistic):
    assert logistic.probability([0.5, 0.0]) == pytest.approx(0.5)
    assert logistic.probability([0.5 + 1.0, 0.0]) == pytest.approx(1.0 / (1.0 + np.exp(-4.0)))
    assert logistic.probability([0.0, 0.0]) < 0.5 < logistic.probability([1.0, 0.0])
    p = logistic.probability(np.random.default_rng(0).random((5, 3, 2)))
    assert p.shape == (5, 3) and np.all((p > 0) & (p < 1))


def test_the_probability_checks_the_dimension(logistic):
    with pytest.raises(ValueError, match="2 coordinates"):
        logistic.probability([0.5, 0.0, 1.0])


def test_the_draws_follow_the_probability(logistic):
    x = [0.75, 0.25]
    p = logistic.probability(x)
    n = 40000
    k = logistic.draw(x, n, np.random.SeedSequence(3))
    assert abs(k - p * n) < 5.0 * np.sqrt(n * p * (1.0 - p))


def test_the_draws_are_reproducible_from_the_seed(logistic):
    a = logistic.draw([0.5, 0.0], 1000, np.random.SeedSequence(5))
    assert a == logistic.draw([0.5, 0.0], 1000, np.random.SeedSequence(5))
    assert a != logistic.draw([0.5, 0.0], 1000, np.random.SeedSequence(6))


def test_a_rule_run_on_a_logistic_sampler_recovers_its_probability(logistic):
    x = [0.9, 0.5]
    outcome = StoppingRule(chunk_size=500, k_target=200, n_max=20000).run(logistic, x, np.random.SeedSequence(11))
    p = logistic.probability(x)
    assert outcome.k >= 200
    assert abs(outcome.fraction - p) < 5.0 * np.sqrt(p * (1.0 - p) / outcome.n)


def test_the_logistic_sampler_round_trips_through_its_settings_and_the_registry(logistic):
    rebuilt = get_sampler("logistic").from_settings(logistic.settings)
    assert rebuilt.space == logistic.space
    x = np.random.default_rng(1).random((6, 2))
    np.testing.assert_array_equal(rebuilt.probability(x), logistic.probability(x))
    assert rebuilt.settings == logistic.settings


def test_the_settings_are_plain_types(logistic):
    def plain(value):
        if isinstance(value, dict):
            return all(isinstance(k, str) and plain(v) for k, v in value.items())
        if isinstance(value, list):
            return all(plain(v) for v in value)
        return isinstance(value, (str, int, float, bool))

    assert plain(logistic.settings)


def test_the_logistic_sampler_needs_a_center_and_slope_for_every_axis(space):
    with pytest.raises(ValueError, match="one entry for each"):
        LogisticSampler(space, center=[0.5], slope=[1.0, 1.0])
    with pytest.raises(ValueError, match="one entry for each"):
        LogisticSampler(space, center=[0.5, 0.0], slope=[1.0, 1.0, 1.0])


def test_the_logistic_sampler_works_in_any_dimension():
    four = Space([Axis(name, 0.0, 1.0) for name in "abcd"])
    sampler = LogisticSampler(four, center=[0.5] * 4, slope=[1.0] * 4)
    assert sampler.probability([0.5] * 4) == pytest.approx(0.5)
    assert 0 <= sampler.draw([0.2, 0.4, 0.6, 0.8], 50, np.random.SeedSequence(0)) <= 50
