"""
The stages of a recovery map as operations on its directory.

A map is a directory of files (see `~uvex_transients.utils.results.PHASE_SPACE_FILES`), and each stage reads some of
them and writes others, so that a stage can be run, repeated or changed without the ones before it:

1. ``points.ecsv`` is the simulation, made by `~uvex_transients.phase_space.run.run_map` or
   `~uvex_transients.phase_space.run.run_experiment` (with ``products=False`` to stop there). It is the only expensive
   file, and its header records everything needed to rebuild the experiment.
2. `fit_map` makes ``map.ecsv`` (and, for the canonical axes, ``overlay.ecsv``) from the points. It rebuilds the
   sampler, so it needs the survey schedule and scikit-learn.
3. `plot_files` makes ``map.png`` from the files alone, and so needs neither.
4. `compare_maps` differences the ``map.ecsv`` of two maps, again from the files alone.

`describe` reads only the header of the points. Every function accepts both formats of points file, the current one
(which records a registered sampler kind) and the earlier canonical one (which records an evaluator), and chooses the
code that made it: the earlier format is fitted and drawn exactly as `~uvex_transients.phase_space.run.run_map` does, so
the files of a published release can be reused.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from astropy.table import Table
from astropy.units import Quantity

from uvex_transients.phase_space.space import Space
from uvex_transients.utils.results import PHASE_SPACE_FILES

__all__ = ["DIFFERENCE_FILES", "MapInfo", "compare_maps", "describe", "find_maps", "fit_map", "plot_files"]

#: The files `compare_maps` writes, by key.
DIFFERENCE_FILES = {"difference": "difference.ecsv", "figure": "difference.png"}

#: The axes of the canonical map, which the overlay of the shipped classes is measured on.
_CANONICAL_NAMES = ("log_luminosity", "log_half_time")

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
    """tuple of str: The keys of `~uvex_transients.utils.results.PHASE_SPACE_FILES` whose file is present."""

    @property
    def settings(self) -> dict:
        """dict: The settings the sampler records."""
        return dict(self.header["sampler"]["settings"])

    @property
    def band(self) -> str | None:
        """Str or None: The band of the map, if the sampler records one."""
        return self.settings.get("band")


def _paths(directory: str | Path) -> dict[str, Path]:
    """Return the file paths of a map directory, by key."""
    directory = Path(directory)
    return {key: directory / name for key, name in PHASE_SPACE_FILES.items()}


def _read(path: Path, what: str) -> Table:
    """Read an ECSV file, or raise a `FileNotFoundError` that says what is missing and which stage makes it."""
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; it is needed for {what}.")
    return Table.read(path, format="ascii.ecsv")


def _flavor(meta) -> str:
    """Return the format of a points file from its header."""
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
    from uvex_transients.phase_space.experiment import _upgrade_legacy_header

    directory = Path(directory)
    paths = _paths(directory)
    table = _read(paths["points"], "describing a map")
    meta = dict(table.meta)
    flavor = _flavor(meta)
    header = _upgrade_legacy_header(meta) if flavor == LEGACY else meta
    return MapInfo(
        directory=directory,
        flavor=flavor,
        kind=header["sampler"]["kind"],
        space=Space.from_dict(header["space"]),
        n_points=len(table),
        n_tries=int(table["n"].sum()) if len(table) else 0,
        header=header,
        files=tuple(key for key, path in paths.items() if path.exists()),
    )


def fit_map(
    directory: str | Path,
    *,
    grid_size: int | None = None,
    rate: Quantity | None = None,
    overlay: bool = False,
    overlay_draws: int | None = None,
    seed: int | None = None,
    schedule=None,
    mission=None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """
    Fit the map of the points in a directory, and write ``map.ecsv`` (and ``overlay.ecsv``).

    The experiment is rebuilt from the header of ``points.ecsv`` and the model is fitted to its points. Nothing is
    simulated. The result is the file the run itself would have written.

    Parameters
    ----------
    directory : str or pathlib.Path
        The directory of the map.
    grid_size : int, optional
        Points per axis of the fitted map. By default ``config["phase_space.map.grid_size"]``.
    rate : ~astropy.units.Quantity, optional
        A volumetric rate; adds ``expected_recoveries`` to the map.
    overlay : bool, optional
        Whether to measure the shipped transient classes in the band of the map, for ``overlay.ecsv``. Only the
        canonical map has an overlay.
    overlay_draws : int, optional
        SED draws per class. By default ``config["phase_space.overlay.n_draws"]``.
    seed : int, optional
        The seed of the overlay draws. By default the seed of the campaign that the file records.
    schedule : ~uvex_transients.surveys.base.SurveySchedule, optional
        The survey schedule, if it is not one the file can fetch by its registry name.
    mission : str or m4opt.missions.Mission, optional
        The mission, if the file does not record one.
    overwrite : bool, optional
        Replace a ``map.ecsv`` (and ``overlay.ecsv``) that is already there.

    Returns
    -------
    dict of str to pathlib.Path
        The files written, by their key in `~uvex_transients.utils.results.PHASE_SPACE_FILES`.

    Raises
    ------
    FileNotFoundError
        If the directory has no ``points.ecsv``.
    FileExistsError
        If a file to write exists and `overwrite` is not set.
    ValueError
        If an overlay is asked for a map that is not over the canonical axes, or the points file is not a map's.
    ImportError
        If scikit-learn is not installed.
    """
    info = describe(directory)
    paths = _paths(directory)
    targets = ["map", *(["overlay"] if overlay else [])]
    existing = [paths[key].name for key in targets if paths[key].exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{directory} already holds {existing}; pass overwrite to replace them.")
    if overlay and (info.flavor != LEGACY and info.space.names != _CANONICAL_NAMES):
        raise ValueError(f"Only a map over {_CANONICAL_NAMES} has an overlay, and this one is over {info.space.names}.")

    context = {key: value for key, value in (("schedule", schedule), ("mission", mission)) if value is not None}
    written = {}
    overlays_from = None
    if info.flavor == LEGACY:
        # The map as `run_map` makes it, so that a file of a release can be fitted again
        from uvex_transients.phase_space.sampler import PhaseSpaceSampler

        sampler = PhaseSpaceSampler.load(paths["points"], evaluator_kwargs=context)
        grid = sampler.map_table(grid_size, rate=rate)
        overlays_from = sampler.evaluator
    else:
        from uvex_transients.phase_space.experiment import Experiment

        experiment = Experiment.load(paths["points"], **context)
        grid = experiment.grid_table(grid_size, rate=rate)

    grid.write(paths["map"], format="ascii.ecsv", overwrite=True)
    written["map"] = paths["map"]

    if overlay:
        from uvex_transients.phase_space.overlay import measure_class, shipped_classes, stack_measurements

        if overlays_from is None:
            raise ValueError("The overlay is made from the evaluator of a canonical map made by `run_map`.")
        bandpass = overlays_from.mission.detector.bandpasses[overlays_from.band]
        seed = int(info.header["campaign"].get("seed", 0)) if seed is None else int(seed)
        measured = [measure_class(cls, bandpass, n_draws=overlay_draws, rng=seed) for cls in shipped_classes().values()]
        stack_measurements(measured).write(paths["overlay"], format="ascii.ecsv", overwrite=True)
        written["overlay"] = paths["overlay"]
    return written


def plot_files(
    directory: str | Path,
    *,
    path: str | Path | None = None,
    title: str | None = None,
    show_points: bool = False,
    overlay: bool = True,
    overwrite: bool = True,
) -> Path:
    """
    Draw the map of a directory from its ``map.ecsv`` and ``points.ecsv``, and write ``map.png``.

    The earlier canonical format is drawn with `~uvex_transients.phase_space.plotting.plot_map` and the current one with
    `~uvex_transients.phase_space.plotting.plot_grid`, as the runs do. It needs matplotlib only.

    Parameters
    ----------
    directory : str or pathlib.Path
        The directory of the map.
    path : str or pathlib.Path, optional
        Where to write the figure. By default ``map.png`` in the directory.
    title : str, optional
        A suptitle. By default one that says what the map is, from the points file.
    show_points : bool, optional
        Whether to mark the measured points.
    overlay : bool, optional
        Whether to shade the shipped classes of ``overlay.ecsv``, if the directory has one.
    overwrite : bool, optional
        Replace the figure if it exists. The default is `True`, since a figure is cheap and the point of drawing it
        again is usually to change it.

    Returns
    -------
    pathlib.Path
        The figure.

    Raises
    ------
    FileNotFoundError
        If the directory lacks ``map.ecsv`` or ``points.ecsv``.
    FileExistsError
        If the figure exists and `overwrite` is not set.
    """
    import matplotlib.pyplot as plt

    from uvex_transients.phase_space.plotting import plot_grid, plot_map

    info = describe(directory)
    paths = _paths(directory)
    figure_path = Path(path) if path is not None else paths["figure"]
    if figure_path.exists() and not overwrite:
        raise FileExistsError(f"{figure_path} exists; pass overwrite to replace it.")
    grid = _read(paths["map"], "drawing the map (make it with `fit`)")
    points = _read(paths["points"], "drawing the map")

    if title is None:
        tries = f"{info.n_points} points, {info.n_tries:,} " + ("events" if info.flavor == LEGACY else "tries")
        if info.flavor == LEGACY:
            settings = info.settings
            title = (
                f"{settings['band']}: SNR >= {settings['snr_threshold']:g} on {settings['n_visits']} visit(s), {tries}"
            )
        else:
            title = f"{info.kind}: {tries}"

    if info.flavor == LEGACY:
        overlays = None
        if overlay and paths["overlay"].exists():
            from uvex_transients.phase_space.overlay import split_measurements

            overlays = split_measurements(_read(paths["overlay"], "shading the classes"))
        figure = plot_map(grid, points, figure_path, title, overlays=overlays, show_points=show_points)
    else:
        figure = plot_grid(grid, points, figure_path, title, show_points=show_points)
    plt.close(figure)
    return figure_path


def compare_maps(
    baseline: str | Path,
    comparison: str | Path,
    out_dir: str | Path,
    *,
    floor: float | None = None,
    check_meta: bool = True,
    plot: bool = True,
    overlay: bool = True,
    title: str | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """
    Difference the ``map.ecsv`` of two maps, and write ``difference.ecsv`` and ``difference.png``.

    See `~uvex_transients.phase_space.compare.difference_map` for what is compared and refused. It reads only the
    fitted maps, so it needs neither the schedule nor scikit-learn.

    Parameters
    ----------
    baseline, comparison : str or pathlib.Path
        The directories of the two maps. The difference is positive where the comparison recovers more.
    out_dir : str or pathlib.Path
        Where to write the files, created if needed.
    floor : float, optional
        The recovery fraction below which a map is taken not to recover anything; see `difference_map`.
    check_meta : bool, optional
        Whether to refuse maps that do not measure the same axes; see `difference_map`.
    plot : bool, optional
        Whether to draw the figure. The default is `True`.
    overlay : bool, optional
        Whether to shade the shipped classes of the baseline's ``overlay.ecsv``, if it has one and the axes are
        canonical.
    title : str, optional
        The title of the figure. By default one that names the two directories.
    overwrite : bool, optional
        Replace files that are already in `out_dir`.

    Returns
    -------
    dict of str to pathlib.Path
        The files written, by their key in `DIFFERENCE_FILES`.

    Raises
    ------
    FileNotFoundError
        If a map has no ``map.ecsv``.
    FileExistsError
        If a file to write exists and `overwrite` is not set.
    ValueError
        As for `difference_map`.
    """
    import matplotlib.pyplot as plt

    from uvex_transients.phase_space.compare import difference_map
    from uvex_transients.phase_space.plotting import plot_difference

    out_dir = Path(out_dir)
    targets = {key: out_dir / name for key, name in DIFFERENCE_FILES.items()}
    existing = [path.name for path in targets.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{out_dir} already holds {existing}; pass overwrite to replace them.")

    maps = [
        _read(_paths(directory)["map"], "a comparison (make it with `fit`)") for directory in (baseline, comparison)
    ]
    difference = difference_map(*maps, floor=floor, check_meta=check_meta)
    out_dir.mkdir(parents=True, exist_ok=True)
    difference.write(targets["difference"], format="ascii.ecsv", overwrite=True)
    written = {"difference": targets["difference"]}

    if plot:
        overlays = None
        overlay_path = _paths(baseline)["overlay"]
        names = Space.from_dict(difference.meta["space"]).names if "space" in difference.meta else _CANONICAL_NAMES
        if overlay and overlay_path.exists() and names == _CANONICAL_NAMES:
            from uvex_transients.phase_space.overlay import split_measurements

            overlays = split_measurements(_read(overlay_path, "shading the classes"))
        title = title or f"{Path(comparison).name} relative to {Path(baseline).name}"
        figure = plot_difference(difference, targets["figure"], title, overlays=overlays)
        plt.close(figure)
        written["figure"] = targets["figure"]
    return written
