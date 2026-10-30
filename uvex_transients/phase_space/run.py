"""
Run one recovery map: place the points, fit the map, and write the data products and the figure.

`run_map` is the whole of a map for one `~uvex_transients.phase_space.evaluator.RecoveryEvaluator`, and what the
``uvex-transients phase-space`` command calls once per map of a run-config. It needs the optional ``phase_space`` extra.

Four files are written to the output directory:

``points.ecsv``
    Every simulated point: its coordinates, ``k`` recoveries out of ``n`` events, the redshift limit, the footprint
    coverage and the effective volume-time. It is rewritten after every batch and read back by ``resume``.
``map.ecsv``
    The fitted map on a grid, from `~uvex_transients.phase_space.sampler.PhaseSpaceSampler.map_table`: each recovery
    quantity with a 1 sigma interval (``_lo``, ``_hi``), and the noise-free scales it is built from.
``overlay.ecsv``
    With ``overlay``, the measured peak luminosity and half-decline time of draws from every shipped transient class,
    in the band of the map.
``map.png``
    The recovery fraction, the all-sky efficiency, the rate needed to recover one event, and the remaining uncertainty.
"""

import contextlib
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

from astropy.units import Quantity
from tqdm import tqdm

from uvex_transients.phase_space.evaluator import RecoveryEvaluator
from uvex_transients.phase_space.overlay import measure_class, shipped_classes, stack_measurements
from uvex_transients.phase_space.plotting import plot_map
from uvex_transients.phase_space.sampler import PhaseSpaceSampler
from uvex_transients.utils import config, logger
from uvex_transients.utils.results import PHASE_SPACE_FILES

__all__ = ["MAP_FILES", "NAMES", "run_map"]

#: Names of the two coordinates of a map, as they appear in its files.
NAMES = ("log_luminosity", "log_half_time")

#: The files of a map, by the key `run_map` returns them under.
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


def _save(sampler: PhaseSpaceSampler, path: Path, callback: Callable[[PhaseSpaceSampler], None] | None) -> None:
    """
    Save the points of a sampler, then pass it on to the caller's callback.

    Parameters
    ----------
    sampler : PhaseSpaceSampler
        The sampler, after its initial design or a batch.
    path : pathlib.Path
        Where to write ``points.ecsv``.
    callback : callable or None
        The caller's callback, if any.
    """
    sampler.save(path)
    if callback is not None:
        callback(sampler)


def run_map(
    evaluator: RecoveryEvaluator,
    out_dir: str | Path,
    *,
    bounds: Sequence[Sequence[float]] | None = None,
    sampler_settings: Mapping | None = None,
    grid_size: int | None = None,
    rate: Quantity | None = None,
    overlay: bool = False,
    overlay_draws: int | None = None,
    show_points: bool = False,
    plot: bool = True,
    resume: bool = False,
    overwrite: bool = False,
    seed: int = 0,
    progress: bool = False,
    callback: Callable[[PhaseSpaceSampler], None] | None = None,
) -> dict[str, Path]:
    """
    Map the recovery of an evaluator over its two axes and write the data products.

    Parameters
    ----------
    evaluator : ~uvex_transients.phase_space.evaluator.RecoveryEvaluator
        What recovers the events; it fixes the band, the depth and the definition of a detection.
    out_dir : str or pathlib.Path
        Directory for the files (see `MAP_FILES`), created if needed.
    bounds : sequence of (float, float), optional
        ``[(log10 L_band / (erg/s) lo, hi), (log10 t_half / day lo, hi)]``. If `None` (the default), uses
        ``config["phase_space.map.log_luminosity"]`` and ``config["phase_space.map.log_half_time"]``.
    sampler_settings : Mapping, optional
        Arguments of `~uvex_transients.phase_space.sampler.PhaseSpaceSampler` (``n_initial``, ``batch_size``,
        ``max_points``, ``tolerance``, ``acquisition``). Any that is missing or `None` is taken from
        ``config["phase_space.sampler"]``.
    grid_size : int, optional
        Points per axis of the fitted map. If `None` (the default), uses ``config["phase_space.map.grid_size"]``.
    rate : ~astropy.units.Quantity, optional
        A volumetric rate; adds ``expected_recoveries`` to the map and its contours to the figure.
    overlay : bool, optional
        Whether to measure the shipped transient classes in the band of the map and shade them on the figure.
    overlay_draws : int, optional
        SED draws per class. If `None` (the default), uses ``config["phase_space.overlay.n_draws"]``.
    show_points : bool, optional
        Whether to mark the simulated points on the figure.
    plot : bool, optional
        Whether to draw the figure. The default is `True`.
    resume : bool, optional
        Continue from the ``points.ecsv`` in `out_dir`, if there is one.
    overwrite : bool, optional
        Replace the files that are already in `out_dir`. Without it, or `resume`, existing ``points.ecsv`` or
        ``map.ecsv`` raises.
    seed : int, optional
        Root seed of the sampler and of the overlay draws. The default is 0.
    progress : bool, optional
        Whether to show the progress bars of the simulation. The default is `False`.
    callback : callable, optional
        Called with the sampler after the initial design and after every batch, once the points are saved.

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

    # Anything not given comes from the configuration
    if bounds is None:
        bounds = [config["phase_space.map.log_luminosity"], config["phase_space.map.log_half_time"]]
    bounds = [tuple(axis) for axis in bounds]
    settings = {"names": NAMES, "seed": seed, **{k: v for k, v in (sampler_settings or {}).items() if v is not None}}

    # ---------------------------------------------- #
    # Sampling                                       #
    # ---------------------------------------------- #
    start = time.perf_counter()
    if resume and paths["points"].exists():
        sampler = PhaseSpaceSampler.load(paths["points"], evaluator, bounds=bounds, **settings)
        logger.info("Resuming from %d points in %s.", len(sampler), paths["points"])
    else:
        sampler = PhaseSpaceSampler(evaluator, bounds, **settings)

    # The points are saved after every batch, so a run that stops can be resumed
    with _quiet_progress(progress):
        sampler.run(callback=lambda state: _save(state, paths["points"], callback))
    elapsed = time.perf_counter() - start

    # ---------------------------------------------- #
    # Products                                       #
    # ---------------------------------------------- #
    points = sampler.table
    grid = sampler.map_table(grid_size, rate=rate)
    grid.write(paths["map"], format="ascii.ecsv", overwrite=True)
    written = {"points": paths["points"], "map": paths["map"]}

    # The shipped classes, measured in the band of the map
    overlays = None
    if overlay:
        bandpass = evaluator.mission.detector.bandpasses[evaluator.band]
        overlays = [measure_class(cls, bandpass, n_draws=overlay_draws, rng=seed) for cls in shipped_classes().values()]
        stack_measurements(overlays).write(paths["overlay"], format="ascii.ecsv", overwrite=True)
        written["overlay"] = paths["overlay"]

    if plot:
        title = (
            f"{evaluator.band}: SNR >= {evaluator.snr_threshold:g} on {evaluator.n_visits} visit(s), "
            f"{len(points)} points, {int(points['n'].sum()):,} events, {elapsed / 60:.1f} min"
        )
        figure = plot_map(grid, points, paths["figure"], title, overlays=overlays, show_points=show_points)
        written["figure"] = paths["figure"]

        # A run makes several maps, so do not keep the figures open
        import matplotlib.pyplot as plt

        plt.close(figure)

    logger.info("Mapped %s in %.1f min with %d points.", evaluator.band, elapsed / 60, len(points))
    return written
