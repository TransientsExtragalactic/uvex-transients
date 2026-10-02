"""Named survey sky footprints: lazy generation, disk caching, a name registry, and point queries."""

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from mocpy import MOC

from uvex_transients.utils import cache_dir

__all__ = ["SurveyFootprint", "FootprintRegistry", "default_registry", "combine_MOC", "combine_footprints"]

_Generator = Callable[..., MOC]


# ====================================== #
# Footprint Type                         #
# ====================================== #
@dataclass
class SurveyFootprint:
    """
    A named survey sky footprint, built lazily and optionally cached to disk.

    The `~mocpy.MOC` is only generated when first needed (see :attr:`moc`). If
    :attr:`persist` is set, it is then saved under a key covering everything that
    can change it (see :attr:`cache_key`) and reloaded from there on later runs.
    Constructing a footprint registers it in `default_registry`.
    """

    name: str
    """str: Colon-separated name, ordered observatory, survey, region (e.g. ``uvex:lmlz:wide``).

    `FootprintRegistry` looks footprints up by this full string.
    """

    generator: _Generator
    """Callable[..., MOC]: Builds the `~mocpy.MOC`, called as ``generator(max_order=..., **params)``."""

    version: str = "v1"
    """str: Part of :attr:`cache_key`; bump it when the output changes without the inputs changing
    (e.g. an upstream data fix)."""

    description: str | None = None
    """str or None: Short description of what the footprint covers."""

    params: dict = field(default_factory=dict)
    """dict: Keyword arguments passed to `generator`."""

    persist: bool = False
    """bool: Whether to cache the generated MOC at :attr:`cache_path` and reload it from there."""

    MOC_max_order: int = 10
    """int: Maximum HEALPix order of the MOC (order 10 is ~3.4 arcmin cells)."""

    _moc: MOC | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Register this footprint in `default_registry`."""
        self.register()

    # ------------------- #
    # Caching             #
    # ------------------- #
    @property
    def cache_key(self) -> str:
        """
        str: Hash of the name, version, generator, params, and max order.

        Changing any of them invalidates the cached file.
        """
        ident = {
            "name": self.name,
            "version": self.version,
            "generator": f"{self.generator.__module__}.{self.generator.__qualname__}",
            "params": self.params,
            "max_order": self.MOC_max_order,
        }
        blob = json.dumps(ident, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    @property
    def cache_path(self) -> Path:
        """
        Path: Location of this footprint's cached MOC file.

        Each ``:``-separated part of :attr:`name` becomes a directory (``:`` is not portable in paths).
        """
        return cache_dir / "footprints" / Path(*self.name.split(":")) / f"{self.cache_key}.fits"

    @property
    def is_cached(self) -> bool:
        """bool: Whether a cached MOC matching the current :attr:`cache_key` exists."""
        return self.cache_path.is_file()

    def clear_from_cache(self) -> None:
        """Delete the cached MOC file, if any, and drop the in-memory copy."""
        self.cache_path.unlink(missing_ok=True)
        self._moc = None

    def _write_cache(self, moc: MOC) -> None:
        # Write to a temp file, then rename, so a killed job never leaves a truncated cache file.
        path = self.cache_path
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".fits.tmp")
        os.close(fd)
        try:
            moc.save(tmp, format="fits", overwrite=True)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    # ------------------- #
    # Generator Action    #
    # ------------------- #
    def generate_MOC(self) -> MOC:
        """
        Run `generator`, degrade to :attr:`MOC_max_order` if needed, and cache if :attr:`persist`.

        Returns
        -------
        mocpy.MOC
            The generated MOC, also kept in memory for :attr:`moc`.

        Raises
        ------
        TypeError
            If `generator` does not return a `~mocpy.MOC`.
        """
        moc = self.generator(max_order=self.MOC_max_order, **self.params)
        if not isinstance(moc, MOC):
            raise TypeError(f"Generator for {self.name!r} returned {type(moc).__name__}, not a MOC.")
        if moc.max_order > self.MOC_max_order:
            moc = moc.degrade_to_order(self.MOC_max_order)
        if self.persist:
            self._write_cache(moc)
        self._moc = moc
        return moc

    # ------------------- #
    # Loading             #
    # ------------------- #
    @property
    def moc(self) -> MOC:
        """MOC: The footprint, from memory, else the disk cache, else `generator`."""
        if self._moc is None:
            if self.persist and self.is_cached:
                self._moc = MOC.from_fits(str(self.cache_path))
            else:
                self.generate_MOC()
        return self._moc

    # ------------------- #
    # Querying            #
    # ------------------- #
    def contains(self, ra, dec) -> np.ndarray:
        """
        Test which sky positions fall inside the footprint.

        Parameters
        ----------
        ra, dec : float or array-like
            Positions in degrees (ICRS), broadcast against each other.

        Returns
        -------
        numpy.ndarray
            Boolean mask, one entry per position.
        """
        ra, dec = np.broadcast_arrays(np.asarray(ra, dtype=float), np.asarray(dec, dtype=float))
        coords = SkyCoord(ra=np.atleast_1d(ra) * u.deg, dec=np.atleast_1d(dec) * u.deg, frame="icrs")
        return self.moc.contains_skycoords(coords)

    def contains_skycoord(self, coord: SkyCoord) -> np.ndarray:
        """
        Test which positions of a `~astropy.coordinates.SkyCoord` fall inside the footprint.

        A vectorized MOC lookup, so large catalogs are cheap. Any frame is accepted.

        Parameters
        ----------
        coord : ~astropy.coordinates.SkyCoord
            Scalar or array of positions.

        Returns
        -------
        numpy.ndarray
            Boolean mask with the shape of `coord` (``(1,)`` for a scalar).
        """
        return np.asarray(self.moc.contains_skycoords(coord.reshape(-1) if coord.isscalar else coord))

    # ------------------- #
    # Registering         #
    # ------------------- #
    def register(self, registry: "FootprintRegistry | None" = None, overwrite: bool = False) -> "SurveyFootprint":
        """
        Add this footprint to a registry.

        Construction already registers in `default_registry`; call this to add it
        to another registry or to replace an existing entry.

        Parameters
        ----------
        registry : FootprintRegistry, optional
            Target registry. Default `default_registry`.
        overwrite : bool, optional
            Replace an existing entry of the same name instead of raising. Default `False`.

        Returns
        -------
        SurveyFootprint
            This footprint, for chaining.

        Raises
        ------
        KeyError
            If the name is already registered and `overwrite` is `False`.
        """
        (registry if registry is not None else default_registry).register(self, overwrite=overwrite)
        return self


# ====================================== #
# Footprint Registry                     #
# ====================================== #
class FootprintRegistry:
    """Case-insensitive lookup of `SurveyFootprint` objects by :attr:`~SurveyFootprint.name`."""

    def __init__(self):
        self._footprints: dict[str, SurveyFootprint] = {}

    @staticmethod
    def _key(name: str) -> str:
        return name.strip().lower()

    def register(self, footprint: SurveyFootprint, overwrite: bool = False) -> None:
        """
        Add `footprint`, keyed on its name.

        Parameters
        ----------
        footprint : SurveyFootprint
            The footprint to register.
        overwrite : bool, optional
            Replace an existing entry of the same name instead of raising. Default `False`.

        Raises
        ------
        KeyError
            If the name is already registered and `overwrite` is `False`.
        """
        key = self._key(footprint.name)
        if key in self._footprints and not overwrite:
            raise KeyError(f"Footprint {footprint.name!r} is already registered.")
        self._footprints[key] = footprint

    def get(self, name: str) -> SurveyFootprint:
        """
        Look up a footprint by its full name, e.g. ``"uvex:lmlz:wide"``.

        Parameters
        ----------
        name : str
            Full name, matched case-insensitively.

        Returns
        -------
        SurveyFootprint
            The registered footprint.

        Raises
        ------
        KeyError
            If no footprint has that name.
        """
        try:
            return self._footprints[self._key(name)]
        except KeyError:
            raise KeyError(f"Unknown footprint {name!r}. Known: {', '.join(self.names())}") from None

    def resolve(self, footprint: "SurveyFootprint | str") -> SurveyFootprint:
        """
        Return `footprint` itself if it is a `SurveyFootprint`, else look it up by name.

        Parameters
        ----------
        footprint : SurveyFootprint or str
            A footprint, or its full name.

        Returns
        -------
        SurveyFootprint
            The footprint.

        Raises
        ------
        KeyError
            If a name is not registered.
        """
        return footprint if isinstance(footprint, SurveyFootprint) else self.get(footprint)

    def __getitem__(self, name: str) -> SurveyFootprint:
        """Alias for :meth:`get`."""
        return self.get(name)

    def __contains__(self, name: str) -> bool:
        """Whether `name` is registered (case-insensitive)."""
        return self._key(name) in self._footprints

    def names(self, prefix: str = "") -> list[str]:
        """
        List registered names, optionally only those starting with `prefix`.

        Parameters
        ----------
        prefix : str, optional
            Name prefix to filter on, e.g. ``"uvex:lmlz"``. Default: every name.

        Returns
        -------
        list of str
            Matching names, sorted.
        """
        return sorted(fp.name for fp in self._footprints.values() if fp.name.startswith(prefix))

    def match(self, ra, dec, names: Iterable[str]) -> dict[str, np.ndarray]:
        """
        Test sky positions against several footprints.

        Parameters
        ----------
        ra, dec : float or array-like
            Positions in degrees (ICRS); see `SurveyFootprint.contains`.
        names : Iterable of str
            Full names of the footprints to test.

        Returns
        -------
        dict of str to numpy.ndarray
            Each name mapped to its `SurveyFootprint.contains` mask.

        Raises
        ------
        KeyError
            If a name is not registered.
        """
        return {n: self.get(n).contains(ra, dec) for n in names}


default_registry = FootprintRegistry()
"""FootprintRegistry: The package-wide default footprint registry."""


# ====================================== #
# Footprint Combinations                 #
# ====================================== #
_OPERATIONS: dict[str, Callable[[MOC, MOC], MOC]] = {
    "union": MOC.union,
    "intersection": MOC.intersection,
    "difference": MOC.difference,
    "xor": MOC.symmetric_difference,
}
_OPERATION_ALIASES = {
    "or": "union",
    "and": "intersection",
    "minus": "difference",
    "subtract": "difference",
    "symmetric_difference": "xor",
}


def _resolve_operation(operation: str) -> str:
    key = operation.strip().lower()
    key = _OPERATION_ALIASES.get(key, key)
    if key not in _OPERATIONS:
        raise ValueError(f"Unknown operation {operation!r}. Known: {', '.join(_OPERATIONS)}.")
    return key


def combine_MOC(*, max_order: int, operation: str, operands: Sequence[str], operand_keys: Sequence[str] = ()) -> MOC:
    """
    Registry-contract generator that combines the MOCs of registered footprints.

    Parameters
    ----------
    max_order : int
        Maximum order of the result. Operands finer than this are degraded first.
    operation : {"union", "intersection", "difference", "xor"}
        How to fold the operands together, left to right. For ``"difference"`` that
        is the first operand minus each later one. Aliases ``"or"``, ``"and"``,
        ``"minus"`` and ``"symmetric_difference"`` are accepted.
    operands : sequence of str
        Full names of footprints in `default_registry`.
    operand_keys : sequence of str, optional
        Unused; the operands' cache keys, carried in the params so the combined cache key
        changes whenever an operand does.

    Returns
    -------
    mocpy.MOC
        The combined MOC.

    Raises
    ------
    ValueError
        If `operation` is unknown or `operands` is empty.
    KeyError
        If an operand is not registered.
    """
    func = _OPERATIONS[_resolve_operation(operation)]
    if not operands:
        raise ValueError("At least one operand footprint is required.")
    mocs = []
    for name in operands:
        moc = default_registry.get(name).moc
        mocs.append(moc.degrade_to_order(max_order) if moc.max_order > max_order else moc)
    return reduce(func, mocs)


def combine_footprints(
    name: str,
    operation: str,
    footprints: Sequence["SurveyFootprint | str"],
    description: str | None = None,
    **kwargs,
) -> SurveyFootprint:
    """
    Build and register a new footprint from set operations on existing ones.

    The result is an ordinary `SurveyFootprint` whose generator is `combine_MOC`. It is
    lazy and, by default, cached. Its :attr:`~SurveyFootprint.cache_key` includes the
    operands' keys, so changing an operand invalidates the combined cache too.

    Parameters
    ----------
    name : str
        Name of the new footprint (``observatory:survey:region``).
    operation : {"union", "intersection", "difference", "xor"}
        See `combine_MOC`.
    footprints : sequence of SurveyFootprint or str
        Operands, or their names. They must be registered in `default_registry`.
    description : str, optional
        Description. Default: generated from the operation and operand names.
    **kwargs
        Passed to `SurveyFootprint` (e.g. ``persist``, ``version``, ``MOC_max_order``).
        ``persist`` defaults to `True`.

    Returns
    -------
    SurveyFootprint
        The new, registered footprint.

    Raises
    ------
    ValueError
        If `operation` is unknown or `footprints` is empty.
    KeyError
        If an operand is not registered, or `name` already is.
    """
    op = _resolve_operation(operation)
    if not footprints:
        raise ValueError("At least one operand footprint is required.")
    operands = [fp.name if isinstance(fp, SurveyFootprint) else fp for fp in footprints]
    keys = [default_registry.get(n).cache_key for n in operands]  # also validates registration
    kwargs.setdefault("persist", True)
    return SurveyFootprint(
        name=name,
        generator=combine_MOC,
        description=description or f"{op} of {', '.join(operands)}",
        params={"operation": op, "operands": operands, "operand_keys": keys},
        **kwargs,
    )
