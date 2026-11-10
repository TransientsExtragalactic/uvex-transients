r"""
A sampler with a known answer, for testing the machinery that sits on it.

`LogisticSampler` succeeds with a probability that is a logistic function of the point, :math:`p(x)=\mathrm{expit}
[\sum_i s_i (x_i - c_i)]`, so the success fraction of a map of it is known exactly. That is what lets the map, the
acquisition and the campaign be tested against the truth without running a survey simulation.
"""

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit

from uvex_transients.phase_space.samplers.base import Sampler
from uvex_transients.phase_space.space import Space

__all__ = ["LogisticSampler"]


class LogisticSampler(Sampler, kind="logistic"):
    r"""
    Succeed with probability :math:`p(x)=\mathrm{expit}[\sum_i s_i (x_i - c_i)]`.

    The boundary :math:`p=1/2` is the plane through `center` that is perpendicular to `slope`, and `slope` sets how
    sharp it is.

    Parameters
    ----------
    space : Space
        The axes of the experiment.
    center : sequence of float
        A point on the boundary, one coordinate per axis.
    slope : sequence of float
        How fast the logit of the probability rises along each axis.

    Raises
    ------
    ValueError
        If `center` or `slope` does not have one entry per axis.
    """

    def __init__(self, space: Space, center: Sequence[float], slope: Sequence[float]):
        center = np.asarray(center, dtype=np.float64)
        slope = np.asarray(slope, dtype=np.float64)
        if center.shape != (space.ndim,) or slope.shape != (space.ndim,):
            raise ValueError(
                f"`center` and `slope` need one entry for each of the {space.ndim} axes, "
                f"got shapes {center.shape} and {slope.shape}."
            )
        self._space = space
        self._center = center
        self._slope = slope

    @property
    def space(self) -> Space:
        """Space: The axes of the experiment."""
        return self._space

    def probability(self, x: ArrayLike) -> NDArray[np.float64]:
        """
        Return the probability of success at points of the space.

        Parameters
        ----------
        x : array-like
            Points in the coordinates of the space, shape ``(..., ndim)``.

        Returns
        -------
        numpy.ndarray
            The probability, of shape ``(...)``.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 0 or x.shape[-1] != self._space.ndim:
            raise ValueError(f"Points of this space have {self._space.ndim} coordinates, got shape {x.shape}.")
        return expit(np.sum(self._slope * (x - self._center), axis=-1))

    def draw(self, x: ArrayLike, n: int, seed: np.random.SeedSequence) -> int:
        """
        Make `n` tries at a point.

        Parameters
        ----------
        x : array-like
            The point, shape ``(ndim,)``.
        n : int
            The number of tries.
        seed : numpy.random.SeedSequence
            The source of the randomness.

        Returns
        -------
        int
            The number of successes.
        """
        return int(np.random.default_rng(seed).binomial(int(n), float(self.probability(x))))

    @property
    def settings(self) -> dict:
        """dict: The space, `center` and `slope`, as plain types."""
        return {"space": self._space.to_dict(), "center": self._center.tolist(), "slope": self._slope.tolist()}

    @classmethod
    def from_settings(cls, settings: Mapping, **context) -> "LogisticSampler":
        """
        Rebuild a sampler from `settings`.

        Parameters
        ----------
        settings : Mapping
            The `settings` of the sampler to rebuild.
        **context
            Ignored: this sampler needs nothing a file cannot record.

        Returns
        -------
        LogisticSampler
            A sampler with the same space, `center` and `slope`.
        """
        return cls(Space.from_dict(settings["space"]), settings["center"], settings["slope"])
