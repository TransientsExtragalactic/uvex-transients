r"""
Products: the quantities of a map, each of which owns how it is made.

A fitted map is of one thing, the success fraction :math:`f` over the space. Everything else a figure or a table shows
is made from it and from the point: the fraction times a volume-time is an expected yield, the fraction times a rate on
top of that is an expected count, and a redshift limit does not involve the fraction at all. A `Product` is one of those
quantities. It is a callable that takes the points and the map's `Estimate` of the success fraction there and returns
the columns it creates, so it is free to combine them however the quantity needs. Nothing else has to know what a
volume-time or a rate is.

A sampler supplies its own products (`~uvex_transients.phase_space.samplers.base.Sampler.products`), which are built
around whatever machinery of the sampler they need. `Scale` and `Weighted` cover the two common cases, a quantity of
the point alone and the success fraction times one.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["Estimate", "Product", "Scale", "Weighted"]

#: Suffixes of the columns of an interval, by convention: a product that carries one names them ``<name>_lo`` and
#: ``<name>_hi``.
LOWER = "_lo"
UPPER = "_hi"


# ============================================ #
# Estimate                                     #
# ============================================ #
@dataclass(frozen=True)
class Estimate:
    """
    The map's estimate of the success fraction at points: a value with an interval.

    Raises
    ------
    ValueError
        If the three arrays do not have the same shape.
    """

    p: NDArray[np.float64]
    """numpy.ndarray: The estimated success fraction, of shape ``(...)``."""

    p_lo: NDArray[np.float64]
    """numpy.ndarray: The lower end of its interval, of the shape of `p`."""

    p_hi: NDArray[np.float64]
    """numpy.ndarray: The upper end of its interval, of the shape of `p`."""

    def __post_init__(self):
        p, p_lo, p_hi = (np.asarray(value, dtype=np.float64) for value in (self.p, self.p_lo, self.p_hi))
        if not p.shape == p_lo.shape == p_hi.shape:
            raise ValueError(
                f"The estimate and its interval need one shape, got {p.shape}, {p_lo.shape}, {p_hi.shape}."
            )
        object.__setattr__(self, "p", p)
        object.__setattr__(self, "p_lo", p_lo)
        object.__setattr__(self, "p_hi", p_hi)

    @property
    def shape(self) -> tuple[int, ...]:
        """Tuple of int: The shape of the estimate, which is the shape of the points without their last axis."""
        return self.p.shape


# ============================================ #
# Product                                      #
# ============================================ #
class Product(ABC):
    r"""
    A quantity of the map, and how it is made from the points and the estimate of the success fraction.

    Parameters
    ----------
    name : str
        The name of the product, which is the name of its main column. A valid Python identifier, so that it can be
        used in an expression.
    label : str, optional
        What a figure calls the quantity, in mathtext if it has symbols.
    unit : str, optional
        The unit of the quantity.

    Raises
    ------
    ValueError
        If `name` is not an identifier.
    """

    def __init__(self, name: str, label: str = "", unit: str = ""):
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"The name of a product must be a valid identifier, got {name!r}.")
        self.name = name
        """str: The name of the product, and of its main column."""
        self.label = label
        """str: What a figure calls the quantity."""
        self.unit = unit
        """str: The unit of the quantity."""

    @abstractmethod
    def __call__(self, x: ArrayLike, estimate: Estimate) -> dict[str, NDArray[np.float64]]:
        """
        Make the columns of the product at points of the space.

        Parameters
        ----------
        x : array-like
            Points in the coordinates of the sampler's space, shape ``(..., ndim)``.
        estimate : Estimate
            The map's estimate of the success fraction at those points, of shape ``(...)``.

        Returns
        -------
        dict of str to numpy.ndarray
            The columns it creates, each of shape ``(...)``. The one called `name` is the main one; an interval is
            carried by convention in ``<name>_lo`` and ``<name>_hi``.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


def _points(x: ArrayLike, estimate: Estimate) -> NDArray[np.float64]:
    """Return `x` as floats, after checking that it has the points that `estimate` is of."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0 or x.shape[:-1] != estimate.shape:
        raise ValueError(f"The points {x.shape} and the estimate {estimate.shape} do not describe the same points.")
    return x


class Scale(Product):
    r"""
    A quantity of the point alone, which does not involve the success fraction.

    A redshift limit, a coverage or a volume-time is a `Scale`: it follows from the physics at the point, with no Monte
    Carlo, and so has no interval.

    Parameters
    ----------
    name : str
        The name of the product, and of its one column.
    function : callable
        ``function(x)`` takes points of shape ``(..., ndim)`` and returns the quantity, of shape ``(...)``.
    label, unit : str, optional
        As for `Product`.
    """

    def __init__(self, name: str, function: Callable[[NDArray], ArrayLike], label: str = "", unit: str = ""):
        super().__init__(name, label, unit)
        self._function = function

    def __call__(self, x: ArrayLike, estimate: Estimate) -> dict[str, NDArray[np.float64]]:
        """
        Make the column of the quantity at points of the space.

        Parameters
        ----------
        x : array-like
            Points, shape ``(..., ndim)``.
        estimate : Estimate
            The estimate at those points; it only supplies their shape.

        Returns
        -------
        dict of str to numpy.ndarray
            ``{name: values}``, of shape ``(...)``.
        """
        x = _points(x, estimate)
        return {self.name: np.broadcast_to(np.asarray(self._function(x), dtype=np.float64), estimate.shape).copy()}


class Weighted(Product):
    r"""
    The success fraction times a quantity of the point, with its interval.

    A recovered volume-time is the success fraction times the volume-time of the survey for an event there, and an
    efficiency is the fraction times the coverage. The interval of the fraction carries through, since the weight is
    positive.

    Parameters
    ----------
    name : str
        The name of the product. Its columns are `name`, ``<name>_lo`` and ``<name>_hi``.
    weight : callable
        ``weight(x)`` takes points of shape ``(..., ndim)`` and returns the weight, of shape ``(...)``, which must
        not be negative.
    label, unit : str, optional
        As for `Product`.
    """

    def __init__(self, name: str, weight: Callable[[NDArray], ArrayLike], label: str = "", unit: str = ""):
        super().__init__(name, label, unit)
        self._weight = weight

    def __call__(self, x: ArrayLike, estimate: Estimate) -> dict[str, NDArray[np.float64]]:
        """
        Make the columns of the weighted estimate at points of the space.

        Parameters
        ----------
        x : array-like
            Points, shape ``(..., ndim)``.
        estimate : Estimate
            The estimate of the success fraction at those points.

        Returns
        -------
        dict of str to numpy.ndarray
            ``name``, ``<name>_lo`` and ``<name>_hi``, each of shape ``(...)``.

        Raises
        ------
        ValueError
            If a weight is negative, which would swap the ends of the interval.
        """
        x = _points(x, estimate)
        weight = np.broadcast_to(np.asarray(self._weight(x), dtype=np.float64), estimate.shape)
        if np.any(weight < 0.0):
            raise ValueError(f"The weight of {self.name!r} must not be negative.")
        return {
            self.name: estimate.p * weight,
            self.name + LOWER: estimate.p_lo * weight,
            self.name + UPPER: estimate.p_hi * weight,
        }
