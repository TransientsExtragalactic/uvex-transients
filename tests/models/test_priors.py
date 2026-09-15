"""Tests for the `Prior.registry` auto-registration mechanism."""

from dataclasses import dataclass

import numpy as np

from uvex_transients.models.core.priors import DiscretePrior, Prior, UniformPrior


def test_registry_contains_every_builtin_prior():
    """Every concrete `Prior` subclass in the package is keyed by its own `DISTRIBUTION_NAME`."""
    registry = Prior.registry()
    assert registry["uniform"] is UniformPrior
    assert registry["discrete"] is DiscretePrior
    assert len(registry) == 8


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
