"""Tests for the `Prior.registry` auto-registration mechanism and `MixturePrior`."""

from dataclasses import dataclass

import numpy as np
import pytest

from uvex_transients.models.core.priors import (
    DiscretePrior,
    MixturePrior,
    NormalPrior,
    Prior,
    UniformPrior,
)


def test_registry_contains_every_builtin_prior():
    """Every concrete `Prior` subclass in the package is keyed by its own `DISTRIBUTION_NAME`."""
    registry = Prior.registry()
    assert registry["uniform"] is UniformPrior
    assert registry["discrete"] is DiscretePrior
    assert registry["mixture"] is MixturePrior
    assert len(registry) == 9


def test_registry_is_a_copy():
    """`registry()` returns a snapshot, not a live view -- mutating it can't corrupt the real registry."""
    registry = Prior.registry()
    registry["not_real"] = object()
    assert "not_real" not in Prior.registry()


def test_new_subclass_registers_itself():
    """Defining a new concrete `Prior` subclass adds it to `Prior.registry()` automatically."""

    @dataclass(frozen=True)
    class _ThrowawayPrior(Prior):
        DISTRIBUTION_NAME = "throwaway"
        value: float

        def _validate(self) -> None:
            pass

        @property
        def support(self) -> tuple[float, float]:
            return (self.value, self.value)

        def _logpdf(self, x):
            return np.zeros_like(x)

    assert Prior.registry()["throwaway"] is _ThrowawayPrior


class TestMixturePrior:
    """`MixturePrior` combines component priors with weights that need not sum to 1."""

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            MixturePrior(components=(NormalPrior(mean=0.0, sigma=1.0),), weights=np.array([0.5, 0.5]))

    def test_rejects_all_zero_weights(self):
        with pytest.raises(ValueError, match="must be positive"):
            MixturePrior(
                components=(NormalPrior(mean=0.0, sigma=1.0), NormalPrior(mean=1.0, sigma=1.0)),
                weights=np.array([0.0, 0.0]),
            )

    def test_support_is_union_of_components(self):
        prior = MixturePrior(
            components=(UniformPrior(lower=0.0, upper=1.0), UniformPrior(lower=2.0, upper=5.0)),
            weights=np.array([1.0, 1.0]),
        )
        assert prior.support == (0.0, 5.0)

    def test_pdf_integrates_to_one(self):
        prior = MixturePrior(
            components=(NormalPrior(mean=-3.0, sigma=0.5), NormalPrior(mean=3.0, sigma=0.5)),
            weights=np.array([0.25, 0.75]),
        )
        x = np.linspace(-10, 10, 20001)
        assert np.trapezoid(prior.pdf(x), x) == pytest.approx(1.0, abs=1e-3)

    def test_sample_recovers_mixture_weights_and_means(self):
        prior = MixturePrior(
            components=(NormalPrior(mean=-5.0, sigma=0.1), NormalPrior(mean=5.0, sigma=0.1)),
            weights=np.array([0.2, 0.8]),
        )
        samples = prior.sample(size=20000, rng=np.random.default_rng(0))
        assert samples.size == 20000
        # Components are well-separated (10 sigma apart), so a hard split recovers each mode cleanly.
        assert np.mean(samples < 0) == pytest.approx(0.2, abs=0.01)
        assert samples[samples < 0].mean() == pytest.approx(-5.0, abs=0.05)
        assert samples[samples > 0].mean() == pytest.approx(5.0, abs=0.05)

    def test_unnormalized_weights_behave_like_normalized(self):
        rng = np.random.default_rng(1)
        prior_a = MixturePrior(
            components=(NormalPrior(mean=0.0, sigma=1.0), NormalPrior(mean=10.0, sigma=1.0)),
            weights=np.array([1.0, 1.0]),
        )
        prior_b = MixturePrior(
            components=(NormalPrior(mean=0.0, sigma=1.0), NormalPrior(mean=10.0, sigma=1.0)),
            weights=np.array([3.0, 3.0]),
        )
        x = np.linspace(-5, 15, 101)
        np.testing.assert_allclose(prior_a.pdf(x), prior_b.pdf(x))
        assert prior_a.sample(size=10, rng=rng).size == 10
