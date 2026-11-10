r"""
The stopping rule: how many tries a point gets.

A sampler makes however many tries it is asked for. How many to ask for is a question of statistics, not of physics: a
point where the success fraction is high needs few tries to pin it down, and one where it is small needs many, so a
point is sampled in chunks until it has enough successes or has used its budget. `StoppingRule` is that rule, and it is
a separate object because two parties need it: the campaign applies it, and the map's acquisition predicts from it the
noise of a point it has not measured yet.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from uvex_transients.phase_space.samplers.base import Outcome, Sampler
from uvex_transients.utils import config

__all__ = ["StoppingRule"]


@dataclass(frozen=True)
class StoppingRule:
    """
    Sample a point in chunks of tries until it has `k_target` successes or has used `n_max` tries.

    Raises
    ------
    ValueError
        If `chunk_size` or `k_target` is below 1, or `n_max` is below `chunk_size`.
    """

    chunk_size: int
    """int: The tries made at a time, and so the fewest a point gets."""

    k_target: int
    """int: Stop adding chunks to a point once it has this many successes."""

    n_max: int
    """int: Never make more than this many tries at a point (the last chunk may take it past)."""

    def __post_init__(self):
        chunk_size, k_target, n_max = int(self.chunk_size), int(self.k_target), int(self.n_max)
        if chunk_size < 1 or k_target < 1 or n_max < chunk_size:
            raise ValueError(
                f"Need chunk_size >= 1, k_target >= 1 and n_max >= chunk_size, got {chunk_size}, {k_target}, {n_max}."
            )
        object.__setattr__(self, "chunk_size", chunk_size)
        object.__setattr__(self, "k_target", k_target)
        object.__setattr__(self, "n_max", n_max)

    @classmethod
    def from_config(
        cls,
        *,
        chunk_size: int | None = None,
        k_target: int | None = None,
        n_max: int | None = None,
    ) -> "StoppingRule":
        """
        Build the rule, taking what is not given from ``config["phase_space.recovery"]``.

        Parameters
        ----------
        chunk_size, k_target, n_max : int, optional
            As for the class. Any that is `None` (the default) is read from the configuration.

        Returns
        -------
        StoppingRule
            The rule.
        """
        given = {"chunk_size": chunk_size, "k_target": k_target, "n_max": n_max}
        return cls(
            **{
                name: config[f"phase_space.recovery.{name}"] if value is None else value
                for name, value in given.items()
            }
        )

    def run(self, sampler: Sampler, x: ArrayLike, seed: np.random.SeedSequence) -> Outcome:
        """
        Sample a point until the rule says to stop.

        Each chunk draws from its own child of `seed`, so a chunk does not depend on the others, and `seed` is
        advanced by one child per chunk. Give every point a seed of its own.

        Parameters
        ----------
        sampler : Sampler
            What makes the tries.
        x : array-like
            The point, in the coordinates of the sampler's space.
        seed : numpy.random.SeedSequence
            The root seed of the point, whose children seed the chunks.

        Returns
        -------
        Outcome
            The successes and tries, with the number of chunks as ``extras["n_chunks"]``.
        """
        x = np.asarray(x, dtype=np.float64)
        k = n = chunks = 0

        # Top the point up until it has enough successes, or has used all the tries it is allowed
        while n < self.n_max and k < self.k_target:
            k += sampler.draw(x, self.chunk_size, seed.spawn(1)[0])
            n += self.chunk_size
            chunks += 1
        return Outcome(k, n, {"n_chunks": chunks})

    def expected_n(self, p: ArrayLike) -> NDArray[np.float64]:
        """
        Predict how many tries a point of success fraction `p` will get.

        Enough to expect `k_target` successes, bounded by one chunk below and `n_max` above. This is what the map's
        acquisition uses to predict the noise of a point before it is measured.

        Parameters
        ----------
        p : array-like
            The success fraction, which must be positive.

        Returns
        -------
        numpy.ndarray
            The expected number of tries, of the shape of `p`.
        """
        return np.clip(self.k_target / np.asarray(p, dtype=np.float64), self.chunk_size, self.n_max)
