r"""
The experiment: a sampler, a map, a stopping rule and a campaign, as one object that can be run and saved.

A phase-space scenario is made of four things. A `~uvex_transients.phase_space.sampling.base.Sampler` answers "how many
detections in n tries at a point", a `~uvex_transients.phase_space.sampling.base.StoppingRule` says how many tries a
point gets, a `~uvex_transients.phase_space.map.PhaseSpaceMap` holds the points and models the success fraction, and
`CampaignSettings` says how long to run. `Experiment` holds the four, so a new scenario is a new sampler and nothing
else:

.. code-block:: python

    experiment = Experiment.create(
        MySampler(...),
        bounds=[(41.0, 45.0), (0.0, 3.0)],
    )
    experiment.run()
    experiment.save("points.ecsv")
    table = experiment.grid_table(60)

**Files.** `Experiment.save` writes the points as an ECSV table whose header records everything needed to rebuild the
experiment: the space, the settings of the model, the stopping rule, the campaign, and the sampler as its registered
kind and settings (see `~uvex_transients.phase_space.sampling.base.get_sampler`), so that no Python import path is
stored.
`Experiment.load` reads that, and also the files of the earlier format, which recorded a ``RecoveryEvaluator`` (a class
that no longer exists); it rebuilds those as the canonical sampler.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from astropy.table import Table
from astropy.units import Quantity
from numpy.typing import ArrayLike

from uvex_transients.phase_space.campaign import run_campaign
from uvex_transients.phase_space.map import PhaseSpaceMap, _configured
from uvex_transients.phase_space.sampling.base import Product, Sampler, StoppingRule, get_sampler
from uvex_transients.phase_space.space import Space

__all__ = ["CampaignSettings", "Experiment", "SCHEMA_VERSION"]

#: The version of the points file that `Experiment.save` writes.
SCHEMA_VERSION = 2

#: What the earlier format recorded as its evaluator, which is the canonical sampler with a stopping rule.
_LEGACY_EVALUATOR = "uvex_transients.phase_space.evaluator.RecoveryEvaluator"

#: What of the settings of the earlier format's evaluator belongs to the stopping rule and the seed, not the sampler.
_LEGACY_RULE_KEYS = ("chunk_size", "k_target", "n_max")


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
    """int: The root seed of the measurements of the points; see `~uvex_transients.phase_space.campaign.point_seed`."""

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
    ) -> "CampaignSettings":
        """
        Build the settings, taking what is not given from ``config["phase_space.sampler"]``.

        Parameters
        ----------
        n_initial, batch_size, max_points, tolerance : optional
            As for the class. Any that is `None` (the default) is read from the configuration.
        seed : int, optional
            The root seed of the measurements.

        Returns
        -------
        CampaignSettings
            The settings.
        """
        return cls(
            n_initial=_configured(n_initial, "sampler.n_initial"),
            batch_size=_configured(batch_size, "sampler.batch_size"),
            max_points=_configured(max_points, "sampler.max_points"),
            tolerance=_configured(tolerance, "sampler.tolerance"),
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
            By default `~uvex_transients.phase_space.sampling.base.StoppingRule.from_config`.
        campaign : CampaignSettings, optional
            By default `CampaignSettings.from_config` with `seed`.
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
            StoppingRule.from_config() if rule is None else rule,
            CampaignSettings.from_config(seed=seed) if campaign is None else campaign,
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
        run_campaign(
            self.phase_map,
            self.sampler,
            self.rule,
            n_initial=self.campaign.n_initial,
            batch_size=self.campaign.batch_size,
            max_points=self.campaign.max_points,
            tolerance=self.campaign.tolerance,
            seed=self.campaign.seed,
            map_fn=map_fn,
            callback=callback,
        )
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
            Points per axis. By default ``config["phase_space.map.grid_size"]``.
        bounds : array-like, optional
            The rectangle to evaluate, which can differ from the ranges of the map.
        products : sequence of Product, optional
            What to make besides the success fraction. By default the ``products`` of the sampler, and with a `rate`
            its ``rate_products``, if it has them.
        rate : ~astropy.units.Quantity, optional
            A volumetric rate, which adds the expected number of recoveries if the sampler can make it.
        n_sigma : float, optional
            Half-width of the interval, in posterior standard deviations of the logit.

        Returns
        -------
        astropy.table.Table
            The table of `~uvex_transients.phase_space.map.PhaseSpaceMap.grid_table`, whose ``meta`` also holds the
            ``sampler`` kind, whatever the sampler records in its ``table_meta`` (for a figure's labels and for
            `~uvex_transients.phase_space.compare.difference_map` to check) and the ``n_max`` of the stopping rule.
        """
        size = _configured(size, "map.grid_size")
        if products is None:
            products = list(self.sampler.products)
            if rate is not None and hasattr(self.sampler, "rate_products"):
                products += list(self.sampler.rate_products(rate))
        table = self.phase_map.grid_table(size, products, bounds=bounds, n_sigma=n_sigma)

        # What `difference_map` checks before it compares two maps, and the resolution it takes its floor from
        table.meta["sampler"] = self.sampler.kind
        table.meta.update(getattr(self.sampler, "table_meta", {}))
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
            One row per measured point, with the header of `header` as its ``meta``.

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
        Rebuild an experiment from a file written by `save`, or by the earlier format.

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
        if "evaluator" in meta and "schema_version" not in meta:
            meta = _upgrade_legacy_header(meta)

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


def _upgrade_legacy_header(meta: Mapping) -> dict:
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
