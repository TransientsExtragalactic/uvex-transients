r"""
The experiment: a sampler, a map, a stopping rule and a campaign, as one object that can be run and saved.

A phase-space scenario is made of four things. A :class:`~uvex_transients.phase_space.sampling.base.Sampler` answers
"how many detections in n tries at a point", a :class:`~uvex_transients.phase_space.sampling.base.StoppingRule` says how
many tries a point gets, a :class:`~uvex_transients.phase_space.map.PhaseSpaceMap` holds the points and models the
success fraction, and :class:`~uvex_transients.phase_space.experiment.CampaignSettings` says how long to run.
:class:`~uvex_transients.phase_space.experiment.Experiment` holds the four, so a new scenario is a new sampler and
nothing else:

.. code-block:: python

    experiment = Experiment.create(
        MySampler(...),
        bounds=[(41.0, 45.0), (0.0, 3.0)],
    )
    experiment.run()
    experiment.save("points.ecsv")
    table = experiment.grid_table(60)

**Files.** :meth:`Experiment.save <uvex_transients.phase_space.experiment.Experiment.save>` writes the points as an ECSV
table whose header records everything needed to rebuild the experiment: the space, the settings of the model, the
stopping rule, the campaign, and the sampler as its registered kind and settings (see
:func:`~uvex_transients.phase_space.sampling.base.get_sampler`), so that no Python import path is stored.
:meth:`Experiment.load <uvex_transients.phase_space.experiment.Experiment.load>` reads that, and also the files of the
earlier format, which recorded a ``RecoveryEvaluator`` (a class that no longer exists); it rebuilds those as the
canonical sampler.

**The campaign.** :func:`~uvex_transients.phase_space.experiment.run_campaign` is the loop that joins a map, a sampler
and a stopping rule. The map says where to measure next, the sampler measures a point, and the rule says how many tries
a point gets, so the loop measures an initial space-filling design, then repeats: ask the map for a batch of points,
measure them and add the counts, until the budget is spent or the map is good enough. It is short on purpose, and
everything it needs to be extended (a different sampler, a different rule, another acquisition) is in the three objects
it is given.

**Seeds.** The randomness of the measurement of a point is the seed of the point,
:func:`~uvex_transients.phase_space.experiment.point_seed`: a child of the campaign's root seed that is identified by
the *index* of the point in the map, not by when it happened to be measured. A point is measured the same way whatever
the order a batch is evaluated in, and so whether it is evaluated in a loop, on a thread pool or on a cluster
(:func:`~uvex_transients.phase_space.experiment.fork_map`), and resuming a campaign from its saved points continues the
same sequence of seeds.
"""

import contextlib
import functools
import multiprocessing
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from astropy.table import Table
from astropy.units import Quantity
from numpy.typing import ArrayLike, NDArray

from uvex_transients.phase_space._config import configured, default
from uvex_transients.phase_space.files import LEGACY, SCHEMA_VERSION, flavor, upgrade_legacy_header
from uvex_transients.phase_space.map import PhaseSpaceMap
from uvex_transients.phase_space.sampling.base import Outcome, Product, Sampler, StoppingRule, get_sampler
from uvex_transients.phase_space.space import Space
from uvex_transients.utils import logger

__all__ = ["CampaignSettings", "Experiment", "fork_map", "point_seed", "run_campaign"]


# ============================================ #
# The campaign loop                            #
# ============================================ #
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
        ``map_fn(function, tasks)``, for the ``map_fn`` of :func:`~uvex_transients.phase_space.experiment.run_campaign`,
        :meth:`Experiment.run <uvex_transients.phase_space.experiment.Experiment.run>` or
        :func:`~uvex_transients.phase_space.stages.run_experiment`. It measures the tasks and ignores `function`, which
        it could not send to the workers.

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
    campaign: "CampaignSettings | None" = None,
    *,
    map_fn: Callable = map,
    callback: Callable[[PhaseSpaceMap], None] | None = None,
) -> PhaseSpaceMap:
    r"""
    Measure points of a map until its budget is spent or it is converged.

    The initial design is measured first if the map has fewer than ``campaign.n_initial`` points. Each round then fits
    the model, asks the map for the next batch, and stops without spending the batch if the uncertainty is already
    below ``campaign.tolerance``.

    Parameters
    ----------
    phase_map : ~uvex_transients.phase_space.map.PhaseSpaceMap
        The map, which is added to. It can already hold points, in which case the campaign continues from them.
    sampler : ~uvex_transients.phase_space.sampling.base.Sampler
        What measures a point. Its space has to be the map's.
    rule : ~uvex_transients.phase_space.sampling.base.StoppingRule
        How many tries a point gets. The map uses it to predict the noise of a point it has not measured.
    campaign : CampaignSettings, optional
        How long to run, how many points to place per round, and the root seed of the measurements (see
        :func:`~uvex_transients.phase_space.experiment.point_seed`; this is separate from the seed of the map, which is
        that of its designs and its model). By default
        :meth:`CampaignSettings.from_config <uvex_transients.phase_space.experiment.CampaignSettings.from_config>` for
        the kinds of the sampler.
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
        If the sampler's space is not the map's.
    """
    if campaign is None:
        campaign = CampaignSettings.from_config(kinds=sampler.kinds())
    if sampler.space.names != phase_map.space.names:
        raise ValueError(
            f"The sampler is over {list(sampler.space.names)} but the map is over {list(phase_map.space.names)}."
        )
    n_initial, batch_size, max_points, tolerance = (
        campaign.n_initial,
        campaign.batch_size,
        campaign.max_points,
        campaign.tolerance,
    )
    root = np.random.SeedSequence(campaign.seed)

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


# ============================================ #
# Campaign settings                            #
# ============================================ #
@dataclass(frozen=True)
class CampaignSettings:
    r"""
    How long a campaign runs and how it spends its points.

    Raises
    ------
    ValueError
        If a count is not a positive integer, or the tolerance is negative.
    """

    n_initial: int
    """int: Size of the space-filling design measured before the first fit."""

    batch_size: int
    """int: Points chosen per round."""

    max_points: int
    """int: Stop once the map has this many points."""

    tolerance: float
    r"""float: Stop once the largest posterior standard deviation of :math:`\ln(p+\epsilon)` falls below this."""

    seed: int = 0
    """int: The root seed of the measurements of the points; see
    :func:`~uvex_transients.phase_space.experiment.point_seed`."""

    def __post_init__(self):
        for name in ("n_initial", "batch_size", "max_points"):
            value = getattr(self, name)
            if int(value) != value or value < 1:
                raise ValueError(f"`{name}` must be a positive integer, got {value!r}.")
            object.__setattr__(self, name, int(value))
        if self.tolerance < 0:
            raise ValueError(f"`tolerance` must not be negative, got {self.tolerance!r}.")
        object.__setattr__(self, "tolerance", float(self.tolerance))
        object.__setattr__(self, "seed", int(self.seed))

    @classmethod
    def from_config(
        cls,
        *,
        n_initial: int | None = None,
        batch_size: int | None = None,
        max_points: int | None = None,
        tolerance: float | None = None,
        seed: int = 0,
        kinds: Sequence[str] = (),
    ) -> "CampaignSettings":
        """
        Build the settings, taking what is not given from ``config["phase_space.campaign"]``.

        Parameters
        ----------
        n_initial, batch_size, max_points, tolerance : optional
            As for the class. Any that is `None` (the default) is read from the configuration.
        seed : int, optional
            The root seed of the measurements.
        kinds : sequence of str, optional
            The kinds of the sampler the campaign is for, whose own sections of the configuration take precedence (see
            :func:`~uvex_transients.phase_space._config.default`).

        Returns
        -------
        CampaignSettings
            The settings.
        """
        given = {"n_initial": n_initial, "batch_size": batch_size, "max_points": max_points, "tolerance": tolerance}
        return cls(
            **{name: default("campaign", name, kinds) if value is None else value for name, value in given.items()},
            seed=seed,
        )


# ============================================ #
# Experiment                                   #
# ============================================ #
class Experiment:
    """
    A sampler, the map of its success fraction, the stopping rule for its points, and a campaign.

    Parameters
    ----------
    sampler : ~uvex_transients.phase_space.sampling.base.Sampler
        What measures a point.
    phase_map : ~uvex_transients.phase_space.map.PhaseSpaceMap
        The map, over the space of the sampler (with the ranges to be mapped).
    rule : ~uvex_transients.phase_space.sampling.base.StoppingRule
        How many tries a point gets.
    campaign : CampaignSettings
        How long to run.

    Raises
    ------
    ValueError
        If the map is not over the axes of the sampler.
    """

    def __init__(self, sampler: Sampler, phase_map: PhaseSpaceMap, rule: StoppingRule, campaign: CampaignSettings):
        if phase_map.space.names != sampler.space.names:
            raise ValueError(
                f"The map is over {list(phase_map.space.names)} but the sampler is over {list(sampler.space.names)}."
            )
        self.sampler = sampler
        self.phase_map = phase_map
        self.rule = rule
        self.campaign = campaign

    @classmethod
    def create(
        cls,
        sampler: Sampler,
        *,
        bounds: ArrayLike | None = None,
        rule: StoppingRule | None = None,
        campaign: CampaignSettings | None = None,
        seed: int = 0,
        **map_settings,
    ) -> "Experiment":
        """
        Create an experiment of a sampler, with an empty map.

        Parameters
        ----------
        sampler : ~uvex_transients.phase_space.sampling.base.Sampler
            What measures a point.
        bounds : array-like, optional
            The ranges to map, shape ``(ndim, 2)``. By default the ranges of the sampler's space.
        rule : ~uvex_transients.phase_space.sampling.base.StoppingRule, optional
            By default :meth:`~uvex_transients.phase_space.sampling.base.StoppingRule.from_config` for the sampler.
        campaign : CampaignSettings, optional
            By default
            :meth:`CampaignSettings.from_config <uvex_transients.phase_space.experiment.CampaignSettings.from_config>`
            with `seed`.
        seed : int, optional
            The root seed of the measurements, and the seed of the map.
        **map_settings
            Settings of the map (``acquisition``, ``n_candidates``, ``floor``, ``n_restarts``, ``kernel``).

        Returns
        -------
        Experiment
            The experiment, not yet run.
        """
        space = sampler.space if bounds is None else sampler.space.with_bounds(bounds)
        return cls(
            sampler,
            PhaseSpaceMap(space, seed=seed, **map_settings),
            StoppingRule.from_config(kinds=sampler.kinds()) if rule is None else rule,
            CampaignSettings.from_config(seed=seed, kinds=sampler.kinds()) if campaign is None else campaign,
        )

    # -------------------------------------- #
    # Running                                #
    # -------------------------------------- #
    def run(self, *, map_fn: Callable = map, callback: Callable[[PhaseSpaceMap], None] | None = None) -> "Experiment":
        """
        Run the campaign, adding points to the map until its budget is spent or it is converged.

        Parameters
        ----------
        map_fn : callable, optional
            ``map_fn(function, tasks)`` returning the results in order, to measure a batch in parallel.
        callback : callable, optional
            Called with the map after the initial design and after every batch, for example to save a checkpoint.

        Returns
        -------
        Experiment
            This experiment, so that ``experiment.run().grid_table()`` works.
        """
        run_campaign(self.phase_map, self.sampler, self.rule, self.campaign, map_fn=map_fn, callback=callback)
        return self

    def resume_from(self, path: str | Path) -> "Experiment":
        """
        Take the points of a file as the points of this experiment's map, to continue from them.

        The sampler, rule and campaign of this experiment are kept; only the measured points come from the file, which
        can be of either format.

        Parameters
        ----------
        path : str or pathlib.Path
            An ECSV file of points over the same axes as this experiment's map.

        Returns
        -------
        Experiment
            This experiment.

        Raises
        ------
        ValueError
            If the file is over other axes.
        """
        stored = Experiment.load(path, sampler=self.sampler).phase_map
        if stored.space.names != self.phase_map.space.names:
            raise ValueError(
                f"{path} is over {list(stored.space.names)} but this experiment is over "
                f"{list(self.phase_map.space.names)}."
            )
        self.phase_map = PhaseSpaceMap(
            self.phase_map.space, table=stored.table, seed=self.campaign.seed, **self.phase_map.settings
        )
        return self

    def grid_table(
        self,
        size: int | Sequence[int] | None = None,
        *,
        bounds: ArrayLike | None = None,
        products: Sequence[Product] | None = None,
        rate: Quantity | None = None,
        n_sigma: float = 1.0,
    ) -> Table:
        """
        Evaluate the map on a regular grid, with the quantities of the sampler, as one table.

        Parameters
        ----------
        size : int or sequence of int, optional
            Points per axis. By default ``config["phase_space.grid_size"]``.
        bounds : array-like, optional
            The rectangle to evaluate, which can differ from the ranges of the map.
        products : sequence of Product, optional
            What to make besides the success fraction. By default the ``products`` of the sampler, and with a `rate`
            its ``rate_products``.
        rate : ~astropy.units.Quantity, optional
            A volumetric rate, which adds the sampler's ``rate_products`` (the expected number of recoveries, for a
            survey sampler).
        n_sigma : float, optional
            Half-width of the interval, in posterior standard deviations of the logit.

        Returns
        -------
        astropy.table.Table
            The table of :meth:`~uvex_transients.phase_space.map.PhaseSpaceMap.grid_table`, whose ``meta`` also holds
            the ``sampler`` kind, whatever the sampler records in its ``table_meta`` (for a figure's labels and for
            :func:`~uvex_transients.phase_space.analysis.compare.difference_map` to check) and the ``n_max`` of the
            stopping rule.
        """
        size = configured(size, "grid_size")
        if products is None:
            products = list(self.sampler.products)
            if rate is not None:
                products += list(self.sampler.rate_products(rate))
        table = self.phase_map.grid_table(size, products, bounds=bounds, n_sigma=n_sigma)

        # What `difference_map` checks before it compares two maps, and the resolution it takes its floor from
        table.meta["sampler"] = self.sampler.kind
        table.meta.update(self.sampler.table_meta)
        if self.sampler.yield_weight in table.colnames:
            table.meta["weight"] = self.sampler.yield_weight
        table.meta["n_max"] = self.rule.n_max
        return table

    # -------------------------------------- #
    # Files                                  #
    # -------------------------------------- #
    @property
    def header(self) -> dict:
        """dict: What a points file records to rebuild this experiment, as plain types an ECSV header can hold."""
        return {
            "schema_version": SCHEMA_VERSION,
            "space": self.phase_map.space.to_dict(),
            "map": self.phase_map.settings,
            "rule": asdict(self.rule),
            "campaign": asdict(self.campaign),
            "sampler": {"kind": self.sampler.kind, "settings": self.sampler.settings},
        }

    def table(self) -> Table:
        """
        Return the points as a table whose header records the experiment.

        Returns
        -------
        astropy.table.Table
            One row per measured point, with the header of
            :attr:`~uvex_transients.phase_space.experiment.Experiment.header` as its ``meta``.

        Raises
        ------
        ValueError
            If the sampler has no registered kind, so a file could not rebuild it.
        """
        if self.sampler.kind is None:
            raise ValueError(
                f"{type(self.sampler).__name__} has no registered kind, so a file could not rebuild it; "
                "name one in the class statement, `class X(Sampler, kind='x')`."
            )
        table = self.phase_map.table
        table.meta.update(self.header)
        return table

    def save(self, path: str | Path) -> None:
        """
        Write the points, with the header that rebuilds the experiment, as an ECSV file, replacing any existing one.

        Parameters
        ----------
        path : str or pathlib.Path
            The destination.
        """
        self.table().write(Path(path), format="ascii.ecsv", overwrite=True)

    @classmethod
    def load(cls, path: str | Path, *, sampler: Sampler | None = None, **context) -> "Experiment":
        """
        Rebuild an experiment from a points file of either format.

        The file is one written by :meth:`~uvex_transients.phase_space.experiment.Experiment.save`, or by the earlier
        format.

        Parameters
        ----------
        path : str or pathlib.Path
            An ECSV file of points.
        sampler : ~uvex_transients.phase_space.sampling.base.Sampler, optional
            A sampler to use instead of rebuilding the one the file records, for example one whose survey schedule is
            not in the registry.
        **context
            What the file could not record, passed to the sampler's ``from_settings``, such as ``schedule``.

        Returns
        -------
        Experiment
            The experiment, holding the stored points. The model is fitted on first use.

        Raises
        ------
        ValueError
            If the file records no sampler that can be rebuilt and none is given.
        """
        table = Table.read(Path(path), format="ascii.ecsv")
        meta = dict(table.meta)
        if flavor(meta) == LEGACY:
            meta = upgrade_legacy_header(meta)

        if sampler is None:
            record = meta.get("sampler", {})
            sampler = get_sampler(record["kind"]).from_settings(record["settings"], **context)
        space = Space.from_dict(meta["space"])

        # The earlier format recorded only the names and ranges, so the labels and units come from the sampler
        axes = []
        for axis in space:
            known = sampler.space[axis.name] if axis.name in sampler.space.names else None
            if known is not None and not axis.label and not axis.unit:
                axis = replace(axis, label=known.label, unit=known.unit)
            axes.append(axis)
        space = Space(axes)

        campaign = CampaignSettings(**meta["campaign"])
        phase_map = PhaseSpaceMap(space, table=table, seed=campaign.seed, **meta.get("map", {}))
        return cls(sampler, phase_map, StoppingRule(**meta["rule"]), campaign)
