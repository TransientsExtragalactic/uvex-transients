r"""
The sampler: how many detections in n tries at a point of the space.

A phase-space map asks one question of the world, again and again: at this point `x`, of `n` independent tries, how
many succeed? A `Sampler` answers it, and nothing else. It does not decide how many tries to make (that is the
`~uvex_transients.phase_space.samplers.stopping.StoppingRule`) and it does not know about the Gaussian process that
models the answers (that is the map). What it does know is the physics of one kind of experiment:

* `Sampler.space` is the `~uvex_transients.phase_space.space.Space` the experiment is over, with its axes and their
  default ranges;
* `Sampler.draw` makes `n` tries at a point and returns the number of successes;
* `Sampler.scales` gives the noise-free quantities of a point (a redshift limit, an exposure, a volume-time) that turn
  the fitted success fraction into a physical yield, and `Sampler.products` says how.

Every sampler has a `kind`, a name under which it is registered, so that a file can record which sampler made it and
`get_sampler` can find the class again without importing it by path. A subclass registers itself by naming its kind in
the class statement,

.. code-block:: python

    class MySampler(Sampler, kind="my_sampler"): ...

and an abstract intermediate class leaves it out.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from uvex_transients.phase_space.space import Space

__all__ = ["Outcome", "Product", "Sampler", "available_samplers", "get_sampler"]


# ============================================ #
# Outcome                                      #
# ============================================ #
@dataclass(frozen=True)
class Outcome:
    """
    The result of sampling one point: `k` successes of `n` tries, and whatever else was worth recording.

    Raises
    ------
    ValueError
        If the counts do not satisfy ``0 <= k <= n`` with ``n >= 1``.
    """

    k: int
    """int: The number of successes."""

    n: int
    """int: The number of tries, at least 1."""

    extras: Mapping[str, Any] = field(default_factory=dict)
    """Mapping: Other entries of the point, by name, which become columns of the table of points."""

    def __post_init__(self):
        k, n = int(self.k), int(self.n)
        if n < 1 or not 0 <= k <= n:
            raise ValueError(f"An outcome needs 0 <= k <= n and n >= 1, got k={self.k!r} and n={self.n!r}.")
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "extras", dict(self.extras))

    @property
    def fraction(self) -> float:
        """float: The observed success fraction, ``k / n``."""
        return self.k / self.n


# ============================================ #
# Product                                      #
# ============================================ #
@dataclass(frozen=True)
class Product:
    """
    A physical quantity of the map: the success fraction times one of the sampler's noise-free scales.

    The fitted map is of the success fraction :math:`f`. A yield, such as the expected number of recoveries per unit
    rate, is :math:`f` times a scale of the sampler at the same point (a volume-time, say), and a `Product` records
    which one, and what to call the result.
    """

    name: str
    """str: The name of the product, which is its column name in the table of the map."""

    scale: str
    """str: The name of the entry of `Sampler.scales` that multiplies the success fraction."""

    label: str = ""
    """str: What a figure calls the product, in mathtext if it has symbols."""

    unit: str = ""
    """str: The unit of the product, which is the unit of the scale."""


# ============================================ #
# Registry                                     #
# ============================================ #
_REGISTRY: dict[str, type["Sampler"]] = {}


def available_samplers() -> tuple[str, ...]:
    """
    Return the kinds of every registered sampler.

    Returns
    -------
    tuple of str
        The kinds, sorted.
    """
    return tuple(sorted(_REGISTRY))


def get_sampler(kind: str) -> type["Sampler"]:
    """
    Return the registered sampler class of a kind.

    Parameters
    ----------
    kind : str
        The kind the class was registered under.

    Returns
    -------
    type of Sampler
        The class.

    Raises
    ------
    KeyError
        If no sampler is registered under `kind`.
    """
    try:
        return _REGISTRY[kind]
    except KeyError:
        raise KeyError(f"No sampler of kind {kind!r}; the registered kinds are {list(available_samplers())}.") from None


# ============================================ #
# Sampler                                      #
# ============================================ #
class Sampler(ABC):
    r"""
    Answer "how many detections in n tries at a point?" for one kind of experiment.

    A subclass supplies `space`, `draw`, `settings` and `from_settings`, and `scales` and `products` if its points
    have a physical scale. Name a `kind` in the class statement to register it; see the module documentation.

    Raises
    ------
    TypeError
        If a subclass names a kind that is already registered, or one that is not a non-empty string.
    """

    kind: ClassVar[str | None] = None
    """str or None: The name the class is registered under, which a file records. `None` for an unregistered class."""

    products: ClassVar[tuple[Product, ...]] = ()
    """tuple of Product: The physical quantities of a map of this sampler, each a success fraction times a scale."""

    def __init_subclass__(cls, kind: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)

        # A subclass that does not name a kind is not registered, and does not claim the kind of its parent
        cls.kind = None
        if kind is None:
            return
        if not isinstance(kind, str) or not kind:
            raise TypeError(f"The kind of {cls.__name__} must be a non-empty string, got {kind!r}.")
        if kind in _REGISTRY:
            raise TypeError(f"A sampler of kind {kind!r} is already registered: {_REGISTRY[kind].__name__}.")
        cls.kind = kind
        _REGISTRY[kind] = cls

    # -------------------------------------- #
    # What a subclass supplies               #
    # -------------------------------------- #
    @property
    @abstractmethod
    def space(self) -> Space:
        """Space: The axes the experiment is over, with their default ranges."""

    @abstractmethod
    def draw(self, x: ArrayLike, n: int, seed: np.random.SeedSequence) -> int:
        """
        Make `n` independent tries at a point and return how many succeed.

        Parameters
        ----------
        x : array-like
            The point, in the coordinates of `space`, shape ``(ndim,)``.
        n : int
            The number of tries.
        seed : numpy.random.SeedSequence
            The source of all the randomness of the tries. The same point, count and seed give the same number of
            successes.

        Returns
        -------
        int
            The number of successes, between 0 and `n`.
        """

    @property
    @abstractmethod
    def settings(self) -> dict:
        """dict: What `from_settings` needs to rebuild this sampler, as plain types that an ECSV header can hold."""

    @classmethod
    @abstractmethod
    def from_settings(cls, settings: Mapping, **context) -> "Sampler":
        """
        Rebuild a sampler from the `settings` of another one.

        Parameters
        ----------
        settings : Mapping
            The `settings` of the sampler to rebuild.
        **context
            What a file cannot record, such as a survey schedule that is not in the registry, for the samplers that
            need it.

        Returns
        -------
        Sampler
            A sampler that makes the same experiment.
        """

    # -------------------------------------- #
    # What a subclass may supply             #
    # -------------------------------------- #
    def scales(self, x: ArrayLike) -> dict[str, NDArray[np.float64]]:
        """
        Return the noise-free scales of points of the space.

        These are quantities that depend on the point but involve no Monte Carlo, such as the redshift limit or the
        volume-time of the survey for an event there. The fitted success fraction times one of them is a physical
        yield (see `products`).

        Parameters
        ----------
        x : array-like
            Points in the coordinates of `space`, shape ``(..., ndim)``.

        Returns
        -------
        dict of str to numpy.ndarray
            Each scale by name, of shape ``(...)``. Empty by default.
        """
        return {}
