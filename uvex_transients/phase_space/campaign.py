r"""
The campaign: the loop that connects a map, a sampler and a stopping rule.

A map says where to measure next, a sampler measures a point, and a stopping rule says how many tries a point gets.
`run_campaign` is the loop that joins them: measure an initial space-filling design, then repeat, asking the map for
a batch of points, measuring them and adding the counts, until the budget is spent or the map is good enough. It
is short on purpose, and everything it needs to be extended (a different sampler, a different rule, another acquisition)
is in the three objects it is given.

**Seeds.** The randomness of the measurement of a point is the seed of the point, `point_seed`: a child of the
campaign's root seed that is identified by the *index* of the point in the map, not by when it happened to be measured.
A point is measured the same way whatever the order a batch is evaluated in, and so whether it is evaluated in a loop,
on a thread pool or on a cluster, and resuming a campaign from its saved points continues the same sequence of seeds.
"""

import contextlib
import functools
import multiprocessing
from collections.abc import Callable, Iterator, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from uvex_transients.phase_space.map import PhaseSpaceMap, _configured
from uvex_transients.phase_space.sampling.base import Outcome, Sampler, StoppingRule
from uvex_transients.utils import logger

__all__ = ["fork_map", "point_seed", "run_campaign"]


def point_seed(root: np.random.SeedSequence, index: int) -> np.random.SeedSequence:
    """
    Return the seed of the point with a given index in a campaign.

    It is a pure function of the root seed and the index, so the same point always gets the same seed, and it does not
    advance `root`.

    Parameters
    ----------
    root : numpy.random.SeedSequence
        The root seed of the campaign.
    index : int
        The index of the point in the map, from 0.

    Returns
    -------
    numpy.random.SeedSequence
        The seed of that point, independent of the seeds of all the others.
    """
    return np.random.SeedSequence(entropy=root.entropy, spawn_key=(*root.spawn_key, int(index)))


def _measure(sampler: Sampler, rule: StoppingRule, task: tuple[NDArray, np.random.SeedSequence]) -> Outcome:
    """Measure one point: apply the rule to the sampler there, with the seed of the point."""
    x, seed = task
    return rule.run(sampler, x, seed)


#: What the forked workers of `fork_map` measure with, set before they are forked and so inherited by them.
_FORKED: dict = {}


def _forked_measure(task: tuple[NDArray, np.random.SeedSequence]) -> Outcome:
    """Measure one point in a forked worker, with the sampler and rule it inherited."""
    return _measure(_FORKED["sampler"], _FORKED["rule"], task)


@contextlib.contextmanager
def fork_map(sampler: Sampler, rule: StoppingRule, processes: int | None = None) -> Iterator[Callable]:
    """
    Provide a ``map_fn`` that measures the points of a batch in forked worker processes.

    A survey-backed sampler cannot be pickled (its schedule holds a satellite orbit), so an ordinary process pool cannot
    be sent one. The workers of this pool are forked instead, and inherit the sampler as it is when the pool is made,
    so anything it has computed by then (a table of redshift limits, say) is shared and not repeated. Each point
    carries its own seed, so a result does not depend on which worker measured it, or on how many there are.

    .. code-block:: python

        with fork_map(
            experiment.sampler,
            experiment.rule,
            processes=8,
        ) as map_fn:
            experiment.run(map_fn=map_fn)

    Parameters
    ----------
    sampler : ~uvex_transients.phase_space.sampling.base.Sampler
        What measures a point. It is not changed by the workers.
    rule : ~uvex_transients.phase_space.sampling.base.StoppingRule
        How many tries a point gets.
    processes : int, optional
        The number of worker processes. By default the number of CPUs.

    Yields
    ------
    callable
        ``map_fn(function, tasks)``, for the ``map_fn`` of `run_campaign`, `Experiment.run` or `run_experiment`. It
        measures the tasks and ignores `function`, which it could not send to the workers.

    Raises
    ------
    ValueError
        If the platform cannot fork.
    """
    if "fork" not in multiprocessing.get_all_start_methods():
        raise ValueError("fork_map needs a platform that can fork processes.")
    _FORKED.update(sampler=sampler, rule=rule)
    try:
        with multiprocessing.get_context("fork").Pool(processes) as pool:
            yield lambda function, tasks: pool.map(_forked_measure, list(tasks), chunksize=1)
    finally:
        _FORKED.clear()


def _measure_points(
    phase_map: PhaseSpaceMap,
    sampler: Sampler,
    rule: StoppingRule,
    points: ArrayLike,
    root: np.random.SeedSequence,
    map_fn: Callable,
) -> None:
    """Measure points and add their counts to the map, in order, each with the seed of its index."""
    points = np.asarray(points, dtype=np.float64)
    first = len(phase_map)
    tasks = [(point, point_seed(root, first + offset)) for offset, point in enumerate(points)]
    outcomes = list(map_fn(functools.partial(_measure, sampler, rule), tasks))
    for point, outcome in zip(points, outcomes):
        phase_map.add(point, outcome.k, outcome.n, **outcome.extras)


def run_campaign(
    phase_map: PhaseSpaceMap,
    sampler: Sampler,
    rule: StoppingRule,
    *,
    n_initial: int | None = None,
    batch_size: int | None = None,
    max_points: int | None = None,
    tolerance: float | None = None,
    seed: int | np.random.SeedSequence | None = None,
    map_fn: Callable = map,
    callback: Callable[[PhaseSpaceMap], None] | None = None,
) -> PhaseSpaceMap:
    r"""
    Measure points of a map until its budget is spent or it is converged.

    The initial design is measured first if the map has fewer than `n_initial` points. Each round then fits the model,
    asks the map for the next batch, and stops without spending the batch if the uncertainty is already below
    `tolerance`. Every argument that is `None` (the default) takes ``config["phase_space.sampler.<name>"]``.

    Parameters
    ----------
    phase_map : ~uvex_transients.phase_space.map.PhaseSpaceMap
        The map, which is added to. It can already hold points, in which case the campaign continues from them.
    sampler : ~uvex_transients.phase_space.sampling.base.Sampler
        What measures a point. Its space has to be the map's.
    rule : ~uvex_transients.phase_space.sampling.base.StoppingRule
        How many tries a point gets. The map uses it to predict the noise of a point it has not measured.
    n_initial : int, optional
        Size of the scrambled Sobol design measured before the first fit.
    batch_size : int, optional
        Points chosen per round.
    max_points : int, optional
        Stop once the map has this many points.
    tolerance : float, optional
        Stop once the largest posterior standard deviation of :math:`\ln(p+\epsilon)` over the candidates falls below
        this, a relative uncertainty on the success fraction.
    seed : int or numpy.random.SeedSequence, optional
        The root seed of the measurements; see `point_seed`. This is separate from the seed of the map, which is that of
        its designs and its model.
    map_fn : callable, optional
        ``map_fn(function, tasks)`` returning the results in order; replace it to measure a batch in parallel. The
        default is the builtin `map`. Because each point carries its own seed, the order does not matter.
    callback : callable, optional
        Called with the map after the initial design and after every batch, for example to save a checkpoint.

    Returns
    -------
    ~uvex_transients.phase_space.map.PhaseSpaceMap
        The map that was passed in, now holding the new points and fitted.

    Raises
    ------
    ValueError
        If the sampler's space is not the map's, or a count is not a positive integer.
    """
    n_initial = _configured(n_initial, "sampler.n_initial")
    batch_size = _configured(batch_size, "sampler.batch_size")
    max_points = _configured(max_points, "sampler.max_points")
    tolerance = float(_configured(tolerance, "sampler.tolerance"))
    for label, value in (("n_initial", n_initial), ("batch_size", batch_size), ("max_points", max_points)):
        if int(value) != value or value < 1:
            raise ValueError(f"`{label}` must be a positive integer, got {value!r}.")
    if sampler.space.names != phase_map.space.names:
        raise ValueError(
            f"The sampler is over {list(sampler.space.names)} but the map is over {list(phase_map.space.names)}."
        )
    n_initial, batch_size, max_points = int(n_initial), int(batch_size), int(max_points)
    root = np.random.SeedSequence(seed) if not isinstance(seed, np.random.SeedSequence) else seed

    def measure(points: Sequence) -> None:
        """
        Measure a batch of points and add them to the map, then call the callback.

        Parameters
        ----------
        points : sequence
            The points, in the coordinates of the map's space.
        """
        _measure_points(phase_map, sampler, rule, points, root, map_fn)
        if callback is not None:
            callback(phase_map)

    if len(phase_map) < n_initial:
        logger.info("Campaign: measuring an initial design of %d points.", n_initial - len(phase_map))
        measure(phase_map.initial_design(n_initial - len(phase_map)))

    # Rounds of propose, measure and refit, until the budget is spent or the map is good enough
    while len(phase_map) < max_points:
        n_new = min(batch_size, max_points - len(phase_map))
        points, uncertainty = phase_map.propose(n_new, rule.expected_n)
        if uncertainty < tolerance:
            logger.info("Campaign: converged (%.3f < %.3f).", uncertainty, tolerance)
            break
        logger.info(
            "Campaign: %d points, uncertainty %.3f (tolerance %.3f); measuring %d more.",
            len(phase_map),
            uncertainty,
            tolerance,
            n_new,
        )
        measure(points)

    phase_map.fit()
    return phase_map
