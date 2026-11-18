"""
The files of a recovery map: where they are, which format a points file is in, and what its header records.

A map is a directory of files (see :data:`~uvex_transients.utils.results.PHASE_SPACE_FILES`). This module knows that
layout and nothing else about a map: it reads and writes no experiment, so it imports neither the samplers nor
scikit-learn, and a command that only inspects files (`describe`, `find_maps`) stays cheap.

The points file comes in two formats, the current one (`SCHEMA_VERSION`, which records a registered sampler kind and its
settings) and the earlier canonical one, which recorded an evaluator. `flavor` tells them apart and
`upgrade_legacy_header` turns the header of the earlier one into the current one, so that the files of a published
release can still be read.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from astropy.table import Table

from uvex_transients.phase_space.space import Space
from uvex_transients.utils.results import PHASE_SPACE_FILES

__all__ = [
    "CURRENT",
    "LEGACY",
    "SCHEMA_VERSION",
    "MapInfo",
    "describe",
    "find_maps",
    "flavor",
    "map_paths",
    "read_table",
    "upgrade_legacy_header",
]

#: The version of the points file that `Experiment.save` writes.
SCHEMA_VERSION = 2

#: What the earlier format recorded as its evaluator, which is the canonical sampler with a stopping rule.
_LEGACY_EVALUATOR = "uvex_transients.phase_space.evaluator.RecoveryEvaluator"

#: What of the settings of the earlier format's evaluator belongs to the stopping rule and the seed, not the sampler.
_LEGACY_RULE_KEYS = ("chunk_size", "k_target", "n_max")

#: The two formats of a points file.
LEGACY, CURRENT = "legacy", "experiment"


@dataclass(frozen=True)
class MapInfo:
    """What the points file of a map records, and which of its files are there."""

    directory: Path
    """pathlib.Path: The directory of the map."""

    flavor: str
    """str: ``"experiment"`` for the current format of the points file, ``"legacy"`` for the earlier canonical one."""

    kind: str
    """str: The registered kind of the sampler, ``"canonical"`` for the earlier format."""

    space: Space
    """Space: The axes the map was made over."""

    n_points: int
    """int: The number of points measured."""

    n_tries: int
    """int: The total number of tries (events) over all points."""

    header: dict
    """dict: The header in the current format (``space``, ``map``, ``rule``, ``campaign``, ``sampler``)."""

    files: tuple[str, ...]
    """tuple of str: The keys of :data:`~uvex_transients.utils.results.PHASE_SPACE_FILES` whose file is present."""

    @property
    def settings(self) -> dict:
        """dict: The settings the sampler records."""
        return dict(self.header["sampler"]["settings"])

    @property
    def band(self) -> str | None:
        """Str or None: The band of the map, if the sampler records one."""
        return self.settings.get("band")


def map_paths(directory: str | Path) -> dict[str, Path]:
    """
    Return the file paths of a map directory.

    Parameters
    ----------
    directory : str or pathlib.Path
        The directory of the map.

    Returns
    -------
    dict of str to pathlib.Path
        The path of each file, by its key in :data:`~uvex_transients.utils.results.PHASE_SPACE_FILES`, whether or not it
        exists.
    """
    directory = Path(directory)
    return {key: directory / name for key, name in PHASE_SPACE_FILES.items()}


def read_table(path: str | Path, what: str) -> Table:
    """
    Read an ECSV file of a map.

    Parameters
    ----------
    path : str or pathlib.Path
        The file.
    what : str
        What the table is needed for, which the error says if the file is missing.

    Returns
    -------
    ~astropy.table.Table
        The table, with its header in ``meta``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; it is needed for {what}.")
    return Table.read(path, format="ascii.ecsv")


def flavor(meta: Mapping) -> str:
    """
    Return the format of a points file from its header.

    Parameters
    ----------
    meta : Mapping
        The header of the points file.

    Returns
    -------
    str
        `CURRENT` or `LEGACY`.

    Raises
    ------
    ValueError
        If the header records neither a sampler nor an evaluator.
    """
    if "schema_version" in meta:
        return CURRENT
    if "evaluator" in meta:
        return LEGACY
    raise ValueError("The file is not a points file of a phase-space map: its header records no sampler or evaluator.")


def find_maps(paths: Sequence[str | Path]) -> list[Path]:
    """
    Resolve the paths a command was given to the directories of maps.

    A directory that holds a ``points.ecsv`` is a map. One that does not is taken to be the output of a run, and its
    subdirectories that are maps are used, in order of name. A ``points.ecsv`` file stands for its directory.

    Parameters
    ----------
    paths : sequence of str or pathlib.Path
        Directories (or points files).

    Returns
    -------
    list of pathlib.Path
        The directories of the maps, in the order given, without repeats.

    Raises
    ------
    FileNotFoundError
        If a path does not exist, or is neither a map nor holds any.
    """
    points = PHASE_SPACE_FILES["points"]
    found: dict[Path, None] = {}
    for path in map(Path, paths):
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist.")
        if path.is_file():
            if path.name != points:
                raise FileNotFoundError(f"{path} is a file, but not a {points}.")
            path = path.parent
        if (path / points).exists():
            found[path] = None
            continue
        children = sorted(child for child in path.iterdir() if (child / points).exists()) if path.is_dir() else []
        if not children:
            raise FileNotFoundError(f"{path} is not a map and holds none: there is no {points} in it or below it.")
        found.update(dict.fromkeys(children))
    return list(found)


def describe(directory: str | Path) -> MapInfo:
    """
    Describe a map from the header of its points file.

    Parameters
    ----------
    directory : str or pathlib.Path
        The directory of the map.

    Returns
    -------
    MapInfo
        The kind, the axes, the number of points and tries, and the files present.

    Raises
    ------
    FileNotFoundError
        If the directory has no ``points.ecsv``.
    ValueError
        If the points file does not record a phase-space map.
    """
    directory = Path(directory)
    paths = map_paths(directory)
    table = read_table(paths["points"], "describing a map")
    meta = dict(table.meta)
    fmt = flavor(meta)
    header = upgrade_legacy_header(meta) if fmt == LEGACY else meta
    return MapInfo(
        directory=directory,
        flavor=fmt,
        kind=header["sampler"]["kind"],
        space=Space.from_dict(header["space"]),
        n_points=len(table),
        n_tries=int(table["n"].sum()) if len(table) else 0,
        header=header,
        files=tuple(key for key, path in paths.items() if path.exists()),
    )


def upgrade_legacy_header(meta: Mapping) -> dict:
    """
    Turn the header of the earlier points format into that of the current one.

    The earlier format recorded the bounds and names, the settings of the map and of its loop, and an evaluator: the
    canonical sampler together with the stopping rule and a seed.

    Parameters
    ----------
    meta : Mapping
        The header of a file of the earlier format.

    Returns
    -------
    dict
        The header in the current format.

    Raises
    ------
    ValueError
        If the file recorded an evaluator that is not the canonical one.
    """
    evaluator = meta["evaluator"]
    if evaluator["class"] != _LEGACY_EVALUATOR:
        raise ValueError(f"Cannot read a file that records the evaluator {evaluator['class']!r}.")
    settings = dict(evaluator["settings"])
    legacy_map = dict(meta.get("sampler", {}))

    # The stopping rule and the seed were settings of the evaluator, and are no longer settings of the sampler
    rule = {key: int(settings.pop(key)) for key in _LEGACY_RULE_KEYS}
    seed = settings.pop("seed", None)

    return {
        "schema_version": SCHEMA_VERSION,
        "space": {
            "axes": [
                {"name": name, "lo": lo, "hi": hi, "label": "", "unit": "", "scale": "log10"}
                for name, (lo, hi) in zip(meta["names"], meta["bounds"])
            ]
        },
        "map": {
            key: legacy_map[key] for key in ("acquisition", "n_candidates", "floor", "n_restarts") if key in legacy_map
        },
        "rule": rule,
        "campaign": {
            **{
                key: legacy_map[key]
                for key in ("n_initial", "batch_size", "max_points", "tolerance")
                if key in legacy_map
            },
            "seed": int(seed) if seed is not None else 0,
        },
        "sampler": {"kind": "canonical", "settings": settings},
    }
