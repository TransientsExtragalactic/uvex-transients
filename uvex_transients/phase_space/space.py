r"""
The space a phase-space map is over: its axes, and the moves between physical points and the unit cube.

A map is over any number of axes. An `Axis` names one coordinate and says what it is (a label and a unit for a figure,
and whether the coordinate is the logarithm of the quantity), and a `Space` is the ordered set of them. Everything that
needs the shape of the map takes a `Space`, so none of it hard-codes two dimensions or the names of the axes:

* the Gaussian process works on the unit cube, which `Space.to_unit` and `Space.from_unit` move to and from;
* `Space.sobol` draws a scrambled Sobol design over it, for the initial points and the candidates;
* `Space.grid` lays out the regular grid that a map is evaluated on;
* a figure labels its axes from `Axis.label` and `Axis.unit`, and a file records the space with `Space.to_dict`.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import qmc

__all__ = ["Axis", "Space"]

#: What an axis coordinate can be: the base-10 logarithm of a quantity, or the quantity itself.
_SCALES = ("log10", "linear")


def _unspent(seed: int | np.random.SeedSequence | None) -> int | np.random.SeedSequence | None:
    """
    Return `seed`, or for a `~numpy.random.SeedSequence` an identical copy, so that using it does not advance it.

    Drawing a scrambled Sobol design spawns from the seed sequence of its generator, which counts a child on the
    sequence it was given, so the same sequence would give a different design the second time it was used.
    """
    if isinstance(seed, np.random.SeedSequence):
        return np.random.SeedSequence(
            entropy=seed.entropy,
            spawn_key=seed.spawn_key,
            pool_size=seed.pool_size,
            n_children_spawned=seed.n_children_spawned,
        )
    return seed


# ============================================ #
# Axis                                         #
# ============================================ #
@dataclass(frozen=True)
class Axis:
    r"""
    One axis of a phase-space map.

    Raises
    ------
    ValueError
        If `name` is not an identifier, the range is not finite and increasing, or `scale` is not one of the two.
    """

    name: str
    """str: The name of the coordinate, which is its column name in every table. A valid Python identifier, so that it
    can be used in an expression."""

    lo: float
    """float: The lower end of the range, in the units of the coordinate (so in dex for a ``"log10"`` axis)."""

    hi: float
    """float: The upper end of the range, in the units of the coordinate. Finite, and greater than `lo`."""

    label: str = ""
    r"""str: What a figure calls the axis, in mathtext if it has symbols, for example ``r"$L_\mathrm{FUV}$"``."""

    unit: str = ""
    """str: The unit of the underlying quantity, for example ``"erg/s"`` or ``"day"``, which a figure writes next to the
    label. It is the unit of the quantity, not of the coordinate, so it is the same for a ``"log10"`` and a
    ``"linear"`` axis."""

    scale: str = "log10"
    """str: Whether the coordinate is the base-10 logarithm of the quantity (``"log10"``, the default) or the quantity
    itself (``"linear"``)."""

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError(f"An axis name must be a valid identifier, got {self.name!r}.")
        lo, hi = float(self.lo), float(self.hi)
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError(f"Axis {self.name!r} needs a finite range with lo < hi, got ({self.lo!r}, {self.hi!r}).")
        if self.scale not in _SCALES:
            raise ValueError(f"Axis {self.name!r} has scale {self.scale!r}; it must be one of {_SCALES}.")

        # Frozen, so the converted floats are set through the base class
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)

    @property
    def bounds(self) -> tuple[float, float]:
        """Tuple of float: The range ``(lo, hi)``."""
        return (self.lo, self.hi)

    def to_dict(self) -> dict:
        """
        Return the axis as plain types, which an ECSV header can hold.

        Returns
        -------
        dict
            The fields of the axis, which `from_dict` turns back into it.
        """
        return {
            "name": self.name,
            "lo": self.lo,
            "hi": self.hi,
            "label": self.label,
            "unit": self.unit,
            "scale": self.scale,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "Axis":
        """
        Rebuild an axis from `to_dict`.

        Parameters
        ----------
        data : Mapping
            The fields of the axis.

        Returns
        -------
        Axis
            The axis.
        """
        return cls(**dict(data))


# ============================================ #
# Space                                        #
# ============================================ #
class Space:
    r"""
    The ordered axes of a phase-space map, and the moves between physical points and the unit cube.

    A point of the space is a length-``ndim`` array in the coordinates of the axes. Functions take any number of
    leading axes, so a single point, a batch of them and a grid all work, as long as the last axis has length
    ``ndim``.

    Parameters
    ----------
    axes : sequence of Axis
        The axes, in order. At least one, with distinct names.

    Raises
    ------
    ValueError
        If there are no axes or two have the same name.
    """

    def __init__(self, axes: Sequence[Axis]):
        axes = tuple(axes)
        if not axes:
            raise ValueError("A space needs at least one axis.")
        if not all(isinstance(axis, Axis) for axis in axes):
            raise TypeError("Every axis of a space must be an Axis.")
        names = [axis.name for axis in axes]
        if len(set(names)) != len(names):
            raise ValueError(f"The axes of a space need distinct names, got {names}.")

        self._axes = axes
        self._lo = np.array([axis.lo for axis in axes], dtype=np.float64)
        self._hi = np.array([axis.hi for axis in axes], dtype=np.float64)

    # -------------------------------------- #
    # Description                            #
    # -------------------------------------- #
    @property
    def axes(self) -> tuple[Axis, ...]:
        """Tuple of Axis: The axes, in order."""
        return self._axes

    @property
    def names(self) -> tuple[str, ...]:
        """Tuple of str: The names of the axes, in order."""
        return tuple(axis.name for axis in self._axes)

    @property
    def ndim(self) -> int:
        """int: The number of axes."""
        return len(self._axes)

    @property
    def bounds(self) -> NDArray[np.float64]:
        """numpy.ndarray: The ranges, shape ``(ndim, 2)`` with rows ``(lo, hi)``."""
        return np.column_stack([self._lo, self._hi])

    def index(self, name: str) -> int:
        """
        Return the position of an axis.

        Parameters
        ----------
        name : str
            The name of the axis.

        Returns
        -------
        int
            Its index in `axes`.

        Raises
        ------
        KeyError
            If there is no axis of that name.
        """
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"No axis named {name!r}; the axes are {list(self.names)}.") from None

    def __len__(self) -> int:
        return self.ndim

    def __getitem__(self, key: int | str) -> Axis:
        return self._axes[self.index(key) if isinstance(key, str) else key]

    def __iter__(self):
        return iter(self._axes)

    def __eq__(self, other) -> bool:
        return isinstance(other, Space) and self._axes == other._axes

    def __hash__(self) -> int:
        return hash(self._axes)

    def __repr__(self) -> str:
        axes = ", ".join(f"{axis.name}=[{axis.lo:g}, {axis.hi:g}]" for axis in self._axes)
        return f"Space({axes})"

    # -------------------------------------- #
    # Points                                 #
    # -------------------------------------- #
    def _check(self, x: ArrayLike) -> NDArray[np.float64]:
        """Return `x` as floats, after checking that its last axis runs over the axes of the space."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 0 or x.shape[-1] != self.ndim:
            raise ValueError(f"Points of this space have {self.ndim} coordinates, got an array of shape {x.shape}.")
        return x

    def to_unit(self, x: ArrayLike) -> NDArray[np.float64]:
        """
        Rescale points from the ranges of the axes to the unit cube.

        Parameters
        ----------
        x : array-like
            Points in the coordinates of the axes, shape ``(..., ndim)``.

        Returns
        -------
        numpy.ndarray
            The points in the unit cube, of the same shape.
        """
        return (self._check(x) - self._lo) / (self._hi - self._lo)

    def from_unit(self, u: ArrayLike) -> NDArray[np.float64]:
        """
        Rescale points from the unit cube to the ranges of the axes.

        Parameters
        ----------
        u : array-like
            Points in the unit cube, shape ``(..., ndim)``.

        Returns
        -------
        numpy.ndarray
            The points in the coordinates of the axes, of the same shape.
        """
        return self._lo + self._check(u) * (self._hi - self._lo)

    def contains(self, x: ArrayLike) -> NDArray[np.bool_]:
        """
        Return whether points lie inside the ranges of the axes, edges included.

        Parameters
        ----------
        x : array-like
            Points in the coordinates of the axes, shape ``(..., ndim)``.

        Returns
        -------
        numpy.ndarray
            Boolean, of shape ``(...)``.
        """
        x = self._check(x)
        return np.all((x >= self._lo) & (x <= self._hi), axis=-1)

    # -------------------------------------- #
    # Designs                                #
    # -------------------------------------- #
    def sobol(self, n: int, seed: int | np.random.SeedSequence | None = None) -> NDArray[np.float64]:
        """
        Draw a scrambled Sobol design over the unit cube.

        A Sobol sequence is balanced in blocks of a power of two, so this draws the next power of two points and keeps
        the first `n`.

        Parameters
        ----------
        n : int
            The number of points, at least 1.
        seed : int or numpy.random.SeedSequence, optional
            Seed of the scrambling. The same seed gives the same design, and a `~numpy.random.SeedSequence` is not
            advanced by being used, so it can be passed again.

        Returns
        -------
        numpy.ndarray
            Points in the unit cube, shape ``(n, ndim)``. Use `from_unit` for points in the coordinates of the axes.

        Raises
        ------
        ValueError
            If `n` is not a positive integer.
        """
        if int(n) != n or n < 1:
            raise ValueError(f"`n` must be a positive integer, got {n!r}.")
        sobol = qmc.Sobol(d=self.ndim, scramble=True, seed=np.random.default_rng(_unspent(seed)))
        return sobol.random_base2(int(np.ceil(np.log2(n))))[: int(n)]

    def grid(self, size: int | Sequence[int], bounds: ArrayLike | None = None) -> NDArray[np.float64]:
        """
        Lay out a regular grid of points, with the first axis varying slowest.

        Parameters
        ----------
        size : int or sequence of int
            Points per axis, one number for every axis or a sequence with one for each. At least 2.
        bounds : array-like, optional
            The rectangle to cover, shape ``(ndim, 2)``, which can differ from the ranges of the axes. By default
            the ranges of the axes.

        Returns
        -------
        numpy.ndarray
            The grid points, shape ``(prod(size), ndim)``, in the order of ``numpy.meshgrid(..., indexing="ij")``
            flattened, so that reshaping a column to `size` gives that axis laid out over the grid.

        Raises
        ------
        ValueError
            If `size` or `bounds` does not match the space.
        """
        sizes = np.broadcast_to(np.asarray(size), (self.ndim,)) if np.ndim(size) == 0 else np.asarray(size)
        if sizes.shape != (self.ndim,) or np.any(sizes < 2) or np.any(sizes != np.round(sizes)):
            raise ValueError(f"`size` needs an integer of at least 2 for each of the {self.ndim} axes, got {size!r}.")
        limits = self.bounds if bounds is None else np.asarray(bounds, dtype=np.float64)
        if limits.shape != (self.ndim, 2):
            raise ValueError(f"`bounds` must have shape ({self.ndim}, 2), got {limits.shape}.")

        lines = [np.linspace(lo, hi, int(count)) for (lo, hi), count in zip(limits, sizes)]
        return np.column_stack([mesh.ravel() for mesh in np.meshgrid(*lines, indexing="ij")])

    # -------------------------------------- #
    # Changing and recording                 #
    # -------------------------------------- #
    def with_bounds(self, bounds: ArrayLike) -> "Space":
        """
        Return a space of the same axes over other ranges.

        Parameters
        ----------
        bounds : array-like
            The new ranges, shape ``(ndim, 2)`` with rows ``(lo, hi)``.

        Returns
        -------
        Space
            A new space; this one is unchanged.

        Raises
        ------
        ValueError
            If `bounds` does not have one row per axis, or a row is not a finite, increasing range.
        """
        bounds = np.asarray(bounds, dtype=np.float64)
        if bounds.shape != (self.ndim, 2):
            raise ValueError(f"`bounds` must have shape ({self.ndim}, 2), got {bounds.shape}.")
        return Space([replace(axis, lo=lo, hi=hi) for axis, (lo, hi) in zip(self._axes, bounds)])

    def to_dict(self) -> dict:
        """
        Return the space as plain types, which an ECSV header can hold.

        Returns
        -------
        dict
            ``{"axes": [...]}``, which `from_dict` turns back into the space.
        """
        return {"axes": [axis.to_dict() for axis in self._axes]}

    @classmethod
    def from_dict(cls, data: Mapping) -> "Space":
        """
        Rebuild a space from `to_dict`.

        Parameters
        ----------
        data : Mapping
            What `to_dict` returned.

        Returns
        -------
        Space
            The space.
        """
        return cls([Axis.from_dict(axis) for axis in data["axes"]])
