"""
Run one recovery map: place the points, fit the map, and write the data products and the figure.

`run_experiment` is the whole of a map for one `~uvex_transients.phase_space.experiment.Experiment`, whatever its
sampler, and what the ``uvex-transients phase-space`` command calls once per map of a run-config. It needs the optional
``phase_space`` extra.

Up to four files are written to the output directory (see `~uvex_transients.utils.results.PHASE_SPACE_FILES`):

``points.ecsv``
    Every measured point: its coordinates, ``k`` successes out of ``n`` tries, and what the sampler records of it. It
    is rewritten after every batch and read back by ``resume``.
``map.ecsv``
    The fitted map on a grid (see `~uvex_transients.phase_space.experiment.Experiment.grid_table`): the success
    fraction with a 1 sigma interval (``_lo``, ``_hi``), and the quantities of the sampler.
``overlay.ecsv``
    For the canonical map only, and with ``overlay``, the measured peak luminosity and half-decline time of draws from
    every shipped transient class, in the band of the map.
``map.png``
    The figure of the map.
"""

import contextlib
import time
from collections.abc import Callable, Iterator
from pathlib import Path

from astropy.units import Quantity
from tqdm import tqdm

from uvex_transients.phase_space.experiment import Experiment
from uvex_transients.phase_space.map import PhaseSpaceMap
from uvex_transients.phase_space.stages import map_title, plot_files, write_map
from uvex_transients.utils import logger
from uvex_transients.utils.results import PHASE_SPACE_FILES

__all__ = ["MAP_FILES", "run_experiment"]

#: The files of a map, by the key `run_experiment` returns them under.
MAP_FILES = PHASE_SPACE_FILES


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
    experiment: Experiment,
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
    callback: Callable[[PhaseSpaceMap], None] | None = None,
    map_fn: Callable = map,
    title: str | None = None,
    products: bool = True,
) -> dict[str, Path]:
    """
    Run an experiment of any sampler and write its data products and figure.

    The points are saved after every batch, so a run that stops can be resumed, and the fitted map is written on a grid
    with the quantities of the sampler. The files are those in the module's docstring: ``points.ecsv`` (see
    `~uvex_transients.phase_space.experiment.Experiment.save`), ``map.ecsv`` (see
    `~uvex_transients.phase_space.experiment.Experiment.grid_table`), ``overlay.ecsv`` and ``map.png`` (see
    `~uvex_transients.phase_space.stages.plot_files`, which needs a map over two axes).

    Parameters
    ----------
    experiment : ~uvex_transients.phase_space.experiment.Experiment
        What to run: a sampler, the map, the stopping rule and the campaign.
    out_dir : str or pathlib.Path
        Directory for the files, created if needed.
    grid_size : int, optional
        Points per axis of the fitted map. If `None` (the default), uses ``config["phase_space.map.grid_size"]``.
    rate : ~astropy.units.Quantity, optional
        A volumetric rate; adds ``expected_recoveries`` to the map for a sampler that can make it.
    overlay : bool, optional
        Whether to measure the shipped transient classes in the band of the map and shade them on the figure. Only the
        canonical map has an overlay.
    overlay_draws : int, optional
        SED draws per class. By default ``config["phase_space.overlay.n_draws"]``.
    plot : bool, optional
        Whether to draw the figure. The default is `True`.
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
        rest can be made from it later with `~uvex_transients.phase_space.stages.fit_map` and
        `~uvex_transients.phase_space.stages.plot_files`. The default is `True`.

    Returns
    -------
    dict of str to pathlib.Path
        The files written, by their key in `MAP_FILES`.

    Raises
    ------
    FileExistsError
        If `out_dir` already holds a map and neither `resume` nor `overwrite` is set.
    """
    out_dir = Path(out_dir)
    paths = {key: out_dir / name for key, name in MAP_FILES.items()}
    existing = [paths[key].name for key in ("points", "map") if paths[key].exists()]
    if existing and not (resume or overwrite):
        raise FileExistsError(f"{out_dir} already holds {existing}; pass overwrite (or resume) to continue.")
    out_dir.mkdir(parents=True, exist_ok=True)

    if resume and paths["points"].exists():
        experiment.resume_from(paths["points"])
        logger.info("Resuming from %d points in %s.", len(experiment.phase_map), paths["points"])

    def checkpoint(phase_map: PhaseSpaceMap) -> None:
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
        written["figure"] = plot_files(out_dir, title=title, show_points=show_points, overlay=overlay)

    logger.info("Ran %s in %.1f min with %d points.", experiment.sampler.kind, elapsed / 60, len(points))
    return written
