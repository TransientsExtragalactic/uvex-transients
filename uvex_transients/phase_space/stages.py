"""
The stages of a recovery map as operations on its directory.

A map is a directory of files (see :mod:`~uvex_transients.phase_space.files` and
:data:`~uvex_transients.utils.results.PHASE_SPACE_FILES`), and each stage reads some of them and writes others, so that
a stage can be run, repeated or changed without the ones before it:

1. `run_experiment` makes ``points.ecsv`` (with ``products=False`` to stop there), and then the products below. It is
   the only expensive stage, and the header of the points records everything needed to rebuild the experiment.
2. `fit_map` makes ``map.ecsv`` (and ``overlay.ecsv``, for a sampler that has an overlay) from the points; `write_map`
   does the same for an experiment that is already in memory. They rebuild the sampler, so they need the survey
   schedule and scikit-learn.
3. `plot_files` makes ``map.png`` from the files alone, and so needs neither.
4. `compare_maps` differences the ``map.ecsv`` of two maps, again from the files alone.

Up to four files are written to the directory of a map:

``points.ecsv``
    Every measured point: its coordinates, ``k`` successes out of ``n`` tries, and what the sampler records of it. It
    is rewritten after every batch and read back by ``resume``.
``map.ecsv``
    The fitted map on a grid (see :meth:`~uvex_transients.phase_space.experiment.Experiment.grid_table`): the success
    fraction with a 1 sigma interval (``_lo``, ``_hi``), and the quantities of the sampler.
``overlay.ecsv``
    For a sampler that has an overlay (the canonical one) and with ``overlay``, the measured peak luminosity and
    half-decline time of draws from every shipped transient class, in the band of the map.
``map.png``
    The figure of the map.

Every stage accepts both formats of points file, so the files of a published release can be fitted again and drawn.
Running a map needs the optional ``phase_space`` extra.
"""

import contextlib
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from astropy.units import Quantity
from tqdm import tqdm

from uvex_transients.phase_space.files import describe, map_paths, read_table
from uvex_transients.phase_space.sampling.base import get_sampler
from uvex_transients.utils import logger
from uvex_transients.utils.results import PHASE_SPACE_FILES

if TYPE_CHECKING:
    from uvex_transients.phase_space.experiment import Experiment
    from uvex_transients.phase_space.map import PhaseSpaceMap

__all__ = [
    "DIFFERENCE_FILES",
    "compare_maps",
    "fit_map",
    "map_title",
    "plot_files",
    "run_experiment",
    "write_map",
]

#: The files `compare_maps` writes, by key.
DIFFERENCE_FILES = {"difference": "difference.ecsv", "figure": "difference.png"}


#: The progress bar constructor, kept so that `_quiet_progress` can put it back.
_TQDM_INIT = tqdm.__init__


def _silent_tqdm_init(self, *args, **kwargs) -> None:
    """
    Construct a progress bar that is disabled, whatever it asks for.

    Parameters
    ----------
    *args
        Positional arguments of `tqdm.tqdm`.
    **kwargs
        Keyword arguments of `tqdm.tqdm`; ``disable`` is overridden.
    """
    kwargs["disable"] = True
    _TQDM_INIT(self, *args, **kwargs)


@contextlib.contextmanager
def _quiet_progress(enabled: bool) -> Iterator[None]:
    """
    Silence every progress bar of the simulation for the duration of the block, unless `enabled`.

    Some bars pass ``disable`` explicitly, so a default would not silence them, and a map makes thousands of them.

    Parameters
    ----------
    enabled : bool
        Whether the bars are wanted, in which case nothing is changed.

    Yields
    ------
    None
        The block runs with the bars silenced.
    """
    if enabled:
        yield
        return

    tqdm.__init__ = _silent_tqdm_init
    try:
        yield
    finally:
        tqdm.__init__ = _TQDM_INIT


def run_experiment(
    experiment: "Experiment",
    out_dir: str | Path,
    *,
    grid_size: int | None = None,
    rate: Quantity | None = None,
    overlay: bool = False,
    overlay_draws: int | None = None,
    plot: bool = True,
    show_points: bool = False,
    resume: bool = False,
    overwrite: bool = False,
    progress: bool = False,
    callback: Callable[["PhaseSpaceMap"], None] | None = None,
    map_fn: Callable = map,
    title: str | None = None,
    products: bool = True,
) -> dict[str, Path]:
    """
    Run an experiment of any sampler and write its data products and figure.

    The points are saved after every batch, so a run that stops can be resumed, and the fitted map is written on a grid
    with the quantities of the sampler. The files are the ones the module's docstring lists: ``points.ecsv`` (see
    :meth:`~uvex_transients.phase_space.experiment.Experiment.save`), ``map.ecsv`` (see
    :meth:`~uvex_transients.phase_space.experiment.Experiment.grid_table`), ``overlay.ecsv`` and ``map.png`` (see
    :func:`~uvex_transients.phase_space.stages.plot_files`, which needs a map over two axes).

    Parameters
    ----------
    experiment : ~uvex_transients.phase_space.experiment.Experiment
        What to run: a sampler, the map, the stopping rule and the campaign.
    out_dir : str or pathlib.Path
        Directory for the files, created if needed.
    grid_size : int, optional
        Points per axis of the fitted map. If `None` (the default), uses ``config["phase_space.grid_size"]``.
    rate : ~astropy.units.Quantity, optional
        A volumetric rate; adds ``expected_recoveries`` to the map for a sampler that can make it.
    overlay : bool, optional
        Whether to measure the shipped transient classes in the band of the map and shade them on the figure. Only a
        sampler with ``HAS_OVERLAY`` (the canonical one) can.
    overlay_draws : int, optional
        SED draws per class. By default ``config["phase_space.overlay.n_draws"]``.
    plot : bool, optional
        Whether to draw the figure. The default is `True`. A map over more than two axes has no figure.
    show_points : bool, optional
        Whether to mark the measured points on the figure.
    resume : bool, optional
        Continue from the ``points.ecsv`` in `out_dir`, if there is one.
    overwrite : bool, optional
        Replace the files that are already in `out_dir`. Without it, or `resume`, existing ``points.ecsv`` or
        ``map.ecsv`` raises.
    progress : bool, optional
        Whether to show the progress bars of the simulation. The default is `False`.
    callback : callable, optional
        Called with the map after the initial design and after every batch, once the points are saved.
    map_fn : callable, optional
        ``map_fn(function, tasks)`` returning the results in order, to measure a batch in parallel.
    title : str, optional
        The title of the figure. By default one that says how many points and tries it took.
    products : bool, optional
        Whether to make the map and the figure after the sampling. If `False`, only ``points.ecsv`` is written, and the
        rest can be made from it later with :func:`~uvex_transients.phase_space.stages.fit_map` and
        :func:`~uvex_transients.phase_space.stages.plot_files`. The default is `True`.

    Returns
    -------
    dict of str to pathlib.Path
        The files written, by their key in `PHASE_SPACE_FILES`.

    Raises
    ------
    FileExistsError
        If `out_dir` already holds a map and neither `resume` nor `overwrite` is set.
    """
    out_dir = Path(out_dir)
    paths = {key: out_dir / name for key, name in PHASE_SPACE_FILES.items()}
    existing = [paths[key].name for key in ("points", "map") if paths[key].exists()]
    if existing and not (resume or overwrite):
        raise FileExistsError(f"{out_dir} already holds {existing}; pass overwrite (or resume) to continue.")
    out_dir.mkdir(parents=True, exist_ok=True)

    if resume and paths["points"].exists():
        experiment.resume_from(paths["points"])
        logger.info("Resuming from %d points in %s.", len(experiment.phase_map), paths["points"])

    def checkpoint(phase_map: "PhaseSpaceMap") -> None:
        """
        Save the points, then pass the map on to the caller's callback.

        Parameters
        ----------
        phase_map : PhaseSpaceMap
            The map, after its initial design or a batch.
        """
        experiment.save(paths["points"])
        if callback is not None:
            callback(phase_map)

    start = time.perf_counter()
    with _quiet_progress(progress):
        experiment.run(map_fn=map_fn, callback=checkpoint)
    elapsed = time.perf_counter() - start
    experiment.save(paths["points"])
    if not products:
        logger.info(
            "Sampled %s in %.1f min with %d points.", experiment.sampler.kind, elapsed / 60, len(experiment.phase_map)
        )
        return {"points": paths["points"]}

    points = experiment.phase_map.table
    written = {
        "points": paths["points"],
        **write_map(experiment, out_dir, grid_size=grid_size, rate=rate, overlay=overlay, overlay_draws=overlay_draws),
    }

    if plot:
        title = title or map_title(
            experiment.sampler.kind,
            experiment.sampler.settings,
            len(points),
            int(points["n"].sum()),
            minutes=elapsed / 60,
        )
        figure = plot_files(out_dir, title=title, show_points=show_points, overlay=overlay)
        if figure is not None:
            written["figure"] = figure

    logger.info("Ran %s in %.1f min with %d points.", experiment.sampler.kind, elapsed / 60, len(points))
    return written


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
        Points per axis of the fitted map. By default ``config["phase_space.grid_size"]``.
    rate : ~astropy.units.Quantity, optional
        A volumetric rate; adds ``expected_recoveries`` to the map.
    overlay : bool, optional
        Whether to measure the shipped transient classes in the band of the map, for ``overlay.ecsv``. Only a sampler
        with ``HAS_OVERLAY`` (the canonical one) can.
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
        The files written, by their key in :data:`~uvex_transients.utils.results.PHASE_SPACE_FILES`.

    Raises
    ------
    FileNotFoundError
        If the directory has no ``points.ecsv``.
    FileExistsError
        If a file to write exists and `overwrite` is not set.
    ValueError
        If an overlay is asked for a map whose sampler has none, or the points file is not a map's.
    ImportError
        If scikit-learn is not installed.
    """
    info = describe(directory)
    paths = map_paths(directory)
    targets = ["map", *(["overlay"] if overlay else [])]
    existing = [paths[key].name for key in targets if paths[key].exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{directory} already holds {existing}; pass overwrite to replace them.")
    if overlay and not get_sampler(info.kind).HAS_OVERLAY:
        raise ValueError(f"A map of kind {info.kind!r} has no overlay.")

    from uvex_transients.phase_space.experiment import Experiment

    context = {key: value for key, value in (("schedule", schedule), ("mission", mission)) if value is not None}
    experiment = Experiment.load(paths["points"], **context)
    return write_map(
        experiment, directory, grid_size=grid_size, rate=rate, overlay=overlay, overlay_draws=overlay_draws, seed=seed
    )


def write_map(
    experiment,
    directory: str | Path,
    *,
    grid_size: int | None = None,
    rate: Quantity | None = None,
    overlay: bool = False,
    overlay_draws: int | None = None,
    seed: int | None = None,
) -> dict[str, Path]:
    """
    Write ``map.ecsv`` (and ``overlay.ecsv``) of an experiment whose model is fitted to its points.

    This is what `fit_map` does after rebuilding the experiment, and what a run does with the experiment it has just
    measured.

    Parameters
    ----------
    experiment : ~uvex_transients.phase_space.experiment.Experiment
        The experiment, with the points measured.
    directory : str or pathlib.Path
        The directory of the map, created if needed.
    grid_size : int, optional
        Points per axis of the fitted map. By default ``config["phase_space.grid_size"]``.
    rate : ~astropy.units.Quantity, optional
        A volumetric rate; adds ``expected_recoveries`` to the map.
    overlay : bool, optional
        Whether to measure the shipped transient classes in the band of the map, for ``overlay.ecsv``. Only a sampler
        with ``HAS_OVERLAY`` (the canonical one) can.
    overlay_draws : int, optional
        SED draws per class. By default ``config["phase_space.overlay.n_draws"]``.
    seed : int, optional
        The seed of the overlay draws. By default the seed of the experiment's campaign.

    Returns
    -------
    dict of str to pathlib.Path
        The files written, by their key in :data:`~uvex_transients.utils.results.PHASE_SPACE_FILES`.

    Raises
    ------
    ValueError
        If an overlay is asked for a map whose sampler has none.
    """
    sampler = experiment.sampler
    if overlay and not sampler.HAS_OVERLAY:
        raise ValueError(f"A map of kind {sampler.kind!r} has no overlay.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = map_paths(directory)

    grid = experiment.grid_table(grid_size, rate=rate)
    grid.write(paths["map"], format="ascii.ecsv", overwrite=True)
    written = {"map": paths["map"]}

    if overlay:
        seed = experiment.campaign.seed if seed is None else int(seed)
        sampler.measure_overlay(overlay_draws, seed).write(paths["overlay"], format="ascii.ecsv", overwrite=True)
        written["overlay"] = paths["overlay"]
    return written


def map_title(kind: str, settings: dict, n_points: int, n_tries: int, minutes: float | None = None) -> str:
    """
    Return the title of the figure of a map.

    Parameters
    ----------
    kind : str
        The kind of the sampler.
    settings : dict
        The settings the sampler records, which its ``describe`` turns into what the map is.
    n_points, n_tries : int
        The number of points measured and of tries made.
    minutes : float, optional
        How long the run took, if it is known.

    Returns
    -------
    str
        The title.
    """
    took = "" if minutes is None else f", {minutes:.1f} min"
    return f"{get_sampler(kind).describe(settings)}, {n_points} points, {n_tries:,} tries{took}"


def plot_files(
    directory: str | Path,
    *,
    path: str | Path | None = None,
    title: str | None = None,
    show_points: bool = False,
    overlay: bool = True,
    overwrite: bool = True,
    fields: Sequence[str] | None = None,
) -> Path | None:
    """
    Draw the map of a directory from its ``map.ecsv`` and ``points.ecsv``, and write ``map.png``.

    The figure is the one the map's sampler draws (``draw_map``): for the canonical map
    :func:`~uvex_transients.phase_space.plotting.plot_map`, which can shade the classes of its overlay, and for any
    other
    :func:`~uvex_transients.phase_space.plotting.plot_grid`. It needs matplotlib only, and the class of the sampler but
    not the survey.

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
    fields : sequence of str, optional
        The columns of ``map.ecsv`` to draw, one panel each. By default the figure the map's sampler draws.

    Returns
    -------
    pathlib.Path or None
        The figure, or `None` if the map's sampler does not draw one (the default figure is of two axes, so a map over
        more is not drawn).

    Raises
    ------
    FileNotFoundError
        If the directory lacks ``map.ecsv`` or ``points.ecsv``.
    FileExistsError
        If the figure exists and `overwrite` is not set.
    """
    import matplotlib.pyplot as plt

    info = describe(directory)
    paths = map_paths(directory)
    figure_path = Path(path) if path is not None else paths["figure"]
    if figure_path.exists() and not overwrite:
        raise FileExistsError(f"{figure_path} exists; pass overwrite to replace it.")
    grid = read_table(paths["map"], "drawing the map (make it with `fit`)")
    points = read_table(paths["points"], "drawing the map")

    if title is None:
        title = map_title(info.kind, info.settings, info.n_points, info.n_tries)

    overlay_table = None
    if overlay and paths["overlay"].exists():
        overlay_table = read_table(paths["overlay"], "shading the classes")
    figure = get_sampler(info.kind).draw_map(
        grid, points, figure_path, title, overlay=overlay_table, show_points=show_points, fields=fields
    )
    if figure is None:
        return None
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

    See :func:`~uvex_transients.phase_space.analysis.compare.difference_map` for what is compared and refused. It reads
    only the fitted maps, so it needs neither the schedule nor scikit-learn.

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
        Whether to draw the figure. The default is `True`. Maps over more than two axes have no figure.
    overlay : bool, optional
        Whether to shade the baseline's ``overlay.ecsv`` as its sampler draws it (``annotate_map``), if it has one
        (only a map of a sampler with an overlay does).
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

    from uvex_transients.phase_space.analysis.compare import difference_map
    from uvex_transients.phase_space.plotting import plot_difference

    out_dir = Path(out_dir)
    targets = {key: out_dir / name for key, name in DIFFERENCE_FILES.items()}
    existing = [path.name for path in targets.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{out_dir} already holds {existing}; pass overwrite to replace them.")

    maps = [
        read_table(map_paths(directory)["map"], "a comparison (make it with `fit`)")
        for directory in (baseline, comparison)
    ]
    difference = difference_map(*maps, floor=floor, check_meta=check_meta)
    out_dir.mkdir(parents=True, exist_ok=True)
    difference.write(targets["difference"], format="ascii.ecsv", overwrite=True)
    written = {"difference": targets["difference"]}

    ndim = len(difference.meta["space"]["axes"]) if "space" in difference.meta else 2
    if plot and ndim != 2:
        logger.warning("The maps are over %d axes and a figure is of two, so none is drawn.", ndim)
    elif plot:
        # What is shaded on the figure is up to the sampler of the baseline, which is the one that made the overlay
        annotate = None
        overlay_path = map_paths(baseline)["overlay"]
        if overlay and overlay_path.exists():
            sampler_class = get_sampler(describe(baseline).kind)
            measurements = read_table(overlay_path, "shading the classes")
            annotate = lambda ax: sampler_class.annotate_map(ax, measurements)  # noqa: E731
        title = title or f"{Path(comparison).name} relative to {Path(baseline).name}"
        figure = plot_difference(difference, targets["figure"], title, annotate=annotate)
        plt.close(figure)
        written["figure"] = targets["figure"]
    return written
