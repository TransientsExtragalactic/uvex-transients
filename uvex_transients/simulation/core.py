"""
Monte Carlo sampling of transient populations against a survey schedule.

`SurveySimulator` samples a Monte Carlo realization of every registered transient
type against a `~uvex_transients.surveys.base.SurveySchedule`, producing an
`EventCatalog` (:meth:`SurveySimulator.generate_events`). Real per-event synthetic
photometry is deferred to `~uvex_transients.simulation.event.Event`
(:meth:`~uvex_transients.simulation.event.Event.simulate_photometry`), since it is
too expensive to run over a freshly sampled population dominated by events too faint
to ever matter. The schedule-aware cuts (`SurveySimulator.filter_by_snr` and friends) are
built on `SurveySimulator.iter_epochs`, which yields every observation of every event.
"""

from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, Union

import astropy_healpix as ah
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable, vstack
from astropy.time import Time
from m4opt.fov import contains as fov_contains
from m4opt.missions import Mission
from regions import Regions, SkyRegion
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from uvex_transients.dust import dust_map, log_attenuation
from uvex_transients.utils import config, get_seed_sequence, logger, resolve_healpix_resolution
from uvex_transients.utils.keyed_noise import time_key

from ..surveys.base import SurveySchedule
from ..transients.base import ExtragalacticTransient
from ._registry import combine_metaclasses, make_tagged_registry
from .event_catalog import EventCatalog
from .exposure_catalog import ExposureCatalog
from .photometry_catalog import PhotometryCatalog
from .yield_table import YieldTable

__all__ = ["SurveySimulator", "cut", "action"]

_SeedType = Union[np.random.SeedSequence, int, None]

# ================================== #
# Registry Configuration             #
# ================================== #
# The SurveySimulator class has "cuts" and "actions" which can be registered to it.
# These require a custom metaclass for registration, which is automatically generated here.
cut, _CutRegistryMeta = make_tagged_registry("_cut_name", "_CUT_REGISTRY")
action, _ActionRegistryMeta = make_tagged_registry("_action_name", "_ACTION_REGISTRY")

# We now combine these classes into a single pipeline registry metaclass.
_PipelineRegistryMeta = combine_metaclasses(_CutRegistryMeta, _ActionRegistryMeta)


# ============================================ #
# Utility Functions                            #
# ============================================ #
def _sample_parameters_from_seeds(sed, seeds) -> dict:
    """
    Regenerate each event's own physical SED parameters from its stored `parameter_seed`.

    Draws one independent `numpy.random.Generator` per row, seeded from that row's own
    `parameter_seed`, exactly matching
    `~uvex_transients.simulation.event.Event.sample_parameters` row for row. This
    guarantees the parameters used for screening here are the same ones a kept event's
    own `Event.mag`/`Event.simulate_photometry` will later reconstruct.

    Parameters
    ----------
    sed : ~uvex_transients.models.core.base.SpectralModel
        The transient type's SED model to sample parameters from.
    seeds : array-like of int
        Each event's own `parameter_seed`, in catalog row order.

    Returns
    -------
    dict
        ``{name: samples}``, each with shape ``(len(seeds),)``, in the same row order as
        `seeds`.
    """
    if len(seeds) == 0:
        return {name: np.array([]) for name in sed.sample_parameters(size=0)}
    per_event = [sed.sample_parameters(size=1, rng=np.random.default_rng(int(seed))) for seed in seeds]
    return {name: np.concatenate([sample[name] for sample in per_event]) for name in per_event[0]}


def event_boundaries(event_id) -> tuple[np.ndarray, np.ndarray]:
    """
    Locate each event's block of rows in an array grouped by event.

    `SurveySimulator.iter_epochs` sorts every chunk by ``(event_id, t_obs)``, so each
    event's rows are contiguous. The offsets returned here turn that into per-event
    reductions without a Python loop over rows, for example with
    ``numpy.minimum.reduceat(values, starts)``.

    Parameters
    ----------
    event_id : array-like of int
        A chunk's ``event_id`` column, or any array whose equal values are contiguous.

    Returns
    -------
    starts : numpy.ndarray
        Index of each event's first row, in order of appearance. Empty for empty input.
    stops : numpy.ndarray
        One past the index of each event's last row, so that event's rows are
        ``chunk[starts[i]:stops[i]]``.
    """
    ids = np.asarray(event_id)
    if ids.size == 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)
    starts = np.concatenate([[0], np.flatnonzero(ids[1:] != ids[:-1]) + 1])
    stops = np.append(starts[1:], ids.size)
    return starts, stops


# ============================================ #
# Reducer Functions                            #
# ============================================ #
# A reducer takes one chunk of epochs and the slice of rows belonging to a single event, and returns
# that event's value (or `None` to leave the event out). `SurveySimulator._reduce_events` applies one
# to every event; the cuts then act on the resulting values. The chunk has the columns of
# `SurveySimulator.iter_epochs` plus `detected` and `non_detected` (see
# `SurveySimulator._iter_detection_epochs`).


def _reduce_detection_timing(chunk: QTable, rows: slice) -> tuple[float, float | None, float]:
    """
    Return ``(t_first, min_gap, span)`` in days over an event's detection epochs.

    `t_first` is the earliest detection's time since explosion, `span` is the latest minus
    the earliest, and `min_gap` is the smallest gap between consecutive detections (`None`
    for a single detection). The smallest gap between any two detections is always one
    between consecutive ones, and the largest is always the span.
    """
    t = chunk["t_since_explosion"][rows].to_value(u.day)
    min_gap = float(np.min(np.diff(t))) if len(t) > 1 else None
    return float(t[0]), min_gap, float(t[-1] - t[0])


def _reduce_first_detection(chunk: QTable, rows: slice) -> tuple[Time, str, float, int]:
    """
    Return ``(t_obs, band, snr, observation_index)`` of an event's earliest detection.

    Unlike `_reduce_detection_timing` it keeps the absolute observation time and which
    schedule row it came from, which `SurveySimulator.run_alert_action` needs to find the
    next downlink.
    """
    first = rows.start
    return (
        chunk["t_obs"][first],
        str(chunk["band"][first]),
        float(chunk["snr"][first]),
        int(chunk["observation_index"][first]),
    )


def _reduce_solo_detection(chunk: QTable, rows: slice) -> int | None:
    """
    Return the schedule row of an event's only detection, or `None` if it has several.

    This is what `SurveySimulator.filter_by_first_visit_detected` needs: whether a
    reference image exists only matters for an event that cannot otherwise be confirmed
    across two or more epochs.
    """
    return int(chunk["observation_index"][rows.start]) if rows.stop - rows.start == 1 else None


def _reduce_nondetection_gap(chunk: QTable, rows: slice) -> float | None:
    """
    Return days from an event's last non-detection to its first detection.

    The chunk must keep every epoch (not only detections). The result is ``nan`` if no
    non-detection precedes the first detection, and `None` (leave the event out) if the
    event is never detected.
    """
    detected = np.flatnonzero(chunk["detected"][rows])
    if detected.size == 0:
        return None
    first = rows.start + detected[0]
    earlier = np.flatnonzero(chunk["non_detected"][rows.start : first])
    if earlier.size == 0:
        return np.nan
    t = chunk["t_since_explosion"]
    return float((t[first] - t[rows.start + earlier[-1]]).to_value(u.day))


# ===================================== #
# Survey Simulation Base Class          #
# ===================================== #
class SurveySimulator(metaclass=_PipelineRegistryMeta):
    """
    Samples transient populations against a survey schedule.

    Parameters
    ----------
    survey_schedule : ~uvex_transients.surveys.base.SurveySchedule
        The schedule to sample events against. See :meth:`__init__`.
    transients : dict of str to ExtragalacticTransient, optional
        The transient types to register. See :meth:`__init__`.
    simulation_seed : int, optional
        Root seed for Monte Carlo sampling. See :meth:`__init__`.
    """

    def __init__(
        self,
        survey_schedule: SurveySchedule,
        transients: dict[str, ExtragalacticTransient] | None = None,
        simulation_seed: int | None = None,
    ):
        """
        Store the survey schedule, register any given transient types, and store the root seed.

        Parameters
        ----------
        survey_schedule : ~uvex_transients.surveys.base.SurveySchedule
            The schedule to sample events against.
        transients : dict of str to ExtragalacticTransient, optional
            Transient types to register up front, keyed by name. More may be
            added later via :attr:`transient_collection`.
        simulation_seed : int, optional
            Root seed for Monte Carlo sampling.

        Raises
        ------
        TypeError
            If `survey_schedule` is not a `SurveySchedule`, or `transients`
            contains a value that isn't an `ExtragalacticTransient`.
        ValueError
            If `transients` has a duplicate key.
        """
        # Ensure that the survey schedule is valid.
        if not isinstance(survey_schedule, SurveySchedule):
            raise TypeError(
                f"'survey_schedule' must be an instance of SurveySchedule, got {type(survey_schedule)} instead."
            )

        self._survey_schedule = survey_schedule
        self._simulation_seed = simulation_seed

        # If we are given any transients to start with, we'll add them to the dictionary. Otherwise
        # we'll just pass through.
        self._transients = {}

        if transients is not None:
            for _transient_type_name, _transient_type in transients.items():
                if _transient_type_name in self._transients:
                    raise ValueError(f"Transient type '{_transient_type_name}' is already used.")
                if not isinstance(_transient_type, ExtragalacticTransient):
                    raise TypeError(
                        f"Transient type '{_transient_type_name}' must be an instance of "
                        f"ExtragalacticTransient, got {type(_transient_type)} instead."
                    )

                self._transients[_transient_type_name] = _transient_type

    # ---------------------------------------------- #
    # Properties and Accessors                       #
    # ---------------------------------------------- #
    @property
    def survey_schedule(self) -> SurveySchedule:
        """~uvex_transients.surveys.base.SurveySchedule: The schedule events are sampled against."""
        return self._survey_schedule

    @property
    def transient_collection(self) -> dict[str, ExtragalacticTransient]:
        """Dict of str to ExtragalacticTransient: The registered transient types, keyed by name."""
        return self._transients

    @property
    def simulation_seed(self) -> _SeedType:
        """int, ~numpy.random.SeedSequence, or None: Root seed for Monte Carlo sampling."""
        return self._simulation_seed

    # -------------------------------------------------- #
    # Event Generation                                   #
    # -------------------------------------------------- #
    def _resolve_time_bins(self, time_bins: Time | int) -> Time:
        """
        Resolve `generate_events`'s `time_bins` argument down to a concrete array of edges.

        Parameters
        ----------
        time_bins : ~astropy.time.Time or int
            Either explicit bin edges, or a positive number of equal-width
            bins spanning the whole survey.

        Returns
        -------
        ~astropy.time.Time
            The concrete array of bin edges.

        Raises
        ------
        TypeError
            If `time_bins` is neither a `~astropy.time.Time` array nor an int.
        ValueError
            If `time_bins` is a `Time` array with fewer than 2 edges, or a
            non-positive int.
        """
        if isinstance(time_bins, Time):
            if time_bins.isscalar or time_bins.size < 2:
                raise ValueError("`time_bins`, given as a Time array, must contain at least 2 edges.")
            return time_bins

        if isinstance(time_bins, bool) or not isinstance(time_bins, (int, np.integer)):
            raise TypeError(
                f"`time_bins` must be an astropy Time array of bin edges or a positive int, got {type(time_bins)!r}."
            )

        if time_bins < 1:
            raise ValueError(f"`time_bins`, given as an int, must be a positive number of bins, got {time_bins!r}.")

        schedule = self._survey_schedule
        return schedule.start_time + np.linspace(0.0, 1.0, time_bins + 1) * schedule.duration

    def generate_events(
        self,
        time_bins: Time | int,
        nside: int | None = None,
        order: str | None = None,
        downsample: int | Mapping[str, int] | None = None,
    ) -> EventCatalog:
        """
        Sample a Monte Carlo realization of every registered transient type over a time grid.

        For each transient type and each bin ``[t_k, t_{k+1})`` of `time_bins`, events are
        sampled only within the HEALPix pixels the survey actually observes at some point
        between ``t_k`` and ``t_{k+1} + transient.duration_limit``, i.e. only where an event
        exploding in this bin could plausibly still be caught by an observation before it fades
        below relevance. Explosion times themselves are drawn only within ``[t_k, t_{k+1})``, so
        no event is double-counted across adjacent bins.

        Two columns are computed once here, at generation time, so nothing downstream needs to
        re-derive them: ``luminosity_distance`` (interpolated per transient type off its own
        cached
        :attr:`~uvex_transients.transients.base.ExtragalacticTransient.luminosity_distance_grid`/
        :attr:`~uvex_transients.transients.base.ExtragalacticTransient.redshift_grid`, rather than
        a fresh `cosmology.luminosity_distance` call per event) and ``ebv`` (one vectorized
        Milky Way dust-map query over every sampled position at once).

        Parameters
        ----------
        time_bins : ~astropy.time.Time or int
            Either an explicit, monotonically increasing `Time` array of ``n + 1`` bin edges, or
            a positive int giving the number of evenly-spaced bins to divide
            `survey_schedule`'s full span into.
        nside : int, optional
            HEALPix resolution used both to query the observed footprint and to sample event
            positions. If `None` (the default), uses ``config["healpix.default_nside"]``.
        order : str, optional
            HEALPix pixel ordering scheme (``"nested"`` or ``"ring"``). If `None` (the
            default), uses ``config["healpix.default_order"]``.
        downsample : int or Mapping of str to int, optional
            Downsample the number of events generated, by drawing a random subset
            (without replacement) of each per-bin, per-type table rather than a fixed
            stride, seeded reproducibly off `simulation_seed`. Either a single factor
            applied to every registered transient type, or a ``{type key: factor}``
            mapping giving a different factor per type (a type left out of the mapping
            is not downsampled at all). The default is ``None`` (no downsampling).

        Returns
        -------
        EventCatalog
            One row per sampled event, across every registered transient type and time bin.

        Raises
        ------
        ValueError
            If `downsample` is a mapping naming a key not in `self._transients`.
        """
        # Validate the inputs and ensure that there are actually registered transients to model.
        if not self._transients:
            raise ValueError("No transient types registered in `transient_collection`; nothing to sample.")

        if isinstance(downsample, Mapping):
            unknown = sorted(set(downsample) - set(self._transients))
            if unknown:
                raise ValueError(
                    f"`downsample` names unknown transient key(s) {unknown}; available: {sorted(self._transients)}."
                )

        nside, order = resolve_healpix_resolution(nside, order)

        # Set up the time bins so that we can perform the windowing analysis properly.
        edges = self._resolve_time_bins(time_bins)
        n_bins = len(edges) - 1

        # Set up the RNG. Because the various sampled transients will need to each be assigned a seed
        # from the single seed provided here, we need to create a seed sequence.
        root_seed = get_seed_sequence(self._simulation_seed)
        sorted_names = sorted(self._transients)
        type_seeds = root_seed.spawn(len(sorted_names))

        # --- Begin Iteration Section --- #
        # In this code-section, we iterate through each window (t_i, t_i+1 + duration) and through each
        # of the transient types to construct the sample of events. This is ALL events within the redshift limit
        # which occur within the footprint of the survey. At this stage, the only event reduction is based in the
        # redshift limit and the survey footprint.
        if isinstance(downsample, Mapping):
            if downsample:
                logger.info(f"Downsampling the number of events per-type by {dict(downsample)}.")
        elif downsample is not None:
            logger.info(f"Downsampling the number of events by a factor of {downsample}.")
        tables = []

        with (
            tqdm(total=len(sorted_names) * n_bins, desc="Generating events", unit="bin") as pbar,
            logging_redirect_tqdm(loggers=[logger]),
        ):
            for name, type_seed in zip(sorted_names, type_seeds):
                transient = self._transients[name]
                bin_seeds = type_seed.spawn(n_bins)
                type_downsample = downsample.get(name) if isinstance(downsample, Mapping) else downsample

                for k in range(n_bins):
                    pbar.set_postfix(type=name, bin=f"{k + 1}/{n_bins}")

                    t_start, t_end = edges[k], edges[k + 1]

                    # Determine which set of the healpix IDs intersect at all with the
                    # FOV of the survey.
                    pixel_ids = self._survey_schedule.get_observed_healpix_ids(
                        t_start,
                        t_end + transient.duration_limit,
                        nside=nside,
                        order=order,
                    )

                    # Determine the tiled area for this timestep and provide the debug info to console.
                    N_PIXELS_VISITED = len(pixel_ids)
                    SOLID_ANGLE_VISITED = ah.nside_to_pixel_area(nside) * N_PIXELS_VISITED
                    logger.debug(
                        "Transient Type: %s, Bin: %d/%d, Time Window: %s to %s, "
                        "N Pixels Visited: %d, Solid Angle Visited: %.3f deg^2",
                        name,
                        k + 1,
                        n_bins,
                        t_start.iso,
                        t_end.iso,
                        N_PIXELS_VISITED,
                        SOLID_ANGLE_VISITED.to_value(u.deg**2),
                    )

                    # Extract a random sample of transients which occur within the
                    # relevant footprint.
                    table = transient.sample_events_on_healpix_grid(
                        nside,
                        t_start=t_start,
                        t_end=t_end,
                        pixel_ids=pixel_ids,
                        order=order,
                        seed=bin_seeds[k],
                    )

                    # Interpolate this transient type's own cached D_L(z) grid at each sampled
                    # event's redshift, right here, rather than a fresh `cosmology.luminosity_distance`
                    # call per event later; see `ExtragalacticTransient.luminosity_distance_grid`.
                    if len(table) > 0:
                        table["luminosity_distance"] = (
                            np.interp(
                                table["redshift"],
                                transient.redshift_grid,
                                transient.luminosity_distance_grid.to_value(u.Mpc),
                            )
                            * u.Mpc
                        )
                    else:
                        table["luminosity_distance"] = u.Quantity([], u.Mpc)

                    table["transient_type"] = name
                    table["time_bin"] = k
                    if type_downsample is not None:
                        # Spawned *after* `sample_events_on_healpix_grid` has already
                        # drawn its own 6 children from `bin_seeds[k]` above, so this
                        # doesn't perturb that draw; see its own docstring on why
                        # spawn order off a shared `SeedSequence` matters.
                        downsample_rng = np.random.default_rng(bin_seeds[k].spawn(1)[0])
                        n_keep = -(-len(table) // type_downsample)  # ceiling division
                        keep = np.sort(downsample_rng.choice(len(table), size=n_keep, replace=False))
                        table = table[keep]
                    tables.append(table)
                    pbar.update(1)

        combined = vstack(tables, metadata_conflicts="silent")
        combined.sort("t_explosion")
        combined["event_id"] = np.arange(len(combined), dtype=np.int64)

        # One vectorized Milky Way dust-map query over every sampled position at once, rather
        # than per-event later (see `EventCatalog.ebv`).
        combined["ebv"] = (
            np.asarray(dust_map().query(combined["coord"]), dtype=np.float64)
            if len(combined) > 0
            else np.array([], dtype=np.float64)
        )

        return EventCatalog(
            table=combined,
            nside=nside,
            order=order,
            time_bins=edges,
            seed=self._simulation_seed,
            downsample=dict(downsample) if isinstance(downsample, Mapping) else downsample,
        )

    def compute_effective_exposure(
        self,
        time_bins: Time | int,
        nside: int | None = None,
        order: str | None = None,
    ) -> ExposureCatalog:
        r"""
        Tabulate each registered transient type's effective exposure per time bin.

        For each transient type and each bin ``[t_k, t_{k+1})`` of `time_bins`, reruns
        the footprint query `generate_events` uses and reduces it to a solid angle
        :math:`F(t_k, t_{k+1}+\tau)`. The **effective exposure** for that bin is

        .. math::

            \mathcal E_k = F(t_k, t_{k+1}+\tau)\,(t_{k+1}-t_k).

        Parameters
        ----------
        time_bins : ~astropy.time.Time or int
            Same semantics as `generate_events`.
        nside : int, optional
            Same semantics as `generate_events`.
        order : str, optional
            Same semantics as `generate_events`.

        Returns
        -------
        ExposureCatalog
            One row per ``(transient type, time bin)``, sorted by transient type then
            bin index, with columns ``transient_type``, ``time_bin``, ``t_start``,
            ``t_end``, ``n_pixels_visited``, ``solid_angle``, ``duration``,
            ``effective_exposure``, and ``expected_events``. See
            `ExposureCatalog.total_effective_exposure`/`ExposureCatalog.total_expected_events`
            for the per-type sums over every bin.

        Raises
        ------
        ValueError
            If no transient types are registered in `transient_collection`.

        Notes
        -----
        The *visited* solid angle (not the full :math:`4\pi` sky) times the *bin* width
        (not the padded window used only to decide which pixels are visitable) matches
        the ``solid_angle * duration`` product
        `~uvex_transients.transients.base.ExtragalacticTransient.sample_event_count`
        itself feeds to `numpy.random.Generator.poisson`, so
        ``effective_exposure * transient.integrated_rate`` reproduces the same per-bin
        expected event count `generate_events` actually samples from. Summing
        `expected_events` over every bin for one transient type equals
        ``transient.compute_all_sky_yield`` only when the survey footprint never misses
        any of the sky at any point (:math:`F\equiv4\pi\,\mathrm{sr}`); otherwise this is
        the tighter, footprint-aware quantity that estimator ignores.

        This is deliberately not extracted from a `generate_events` call after the fact,
        since `generate_events` never persists per-bin pixel IDs or solid angles once it
        is done sampling from them. Calling this separately recomputes the footprint
        query, but the expensive part, `SurveySchedule.get_healpix_coverage_index`'s
        whole-schedule rasterization, is cached per ``(nside, order)`` and so is paid for
        at most once, however many times either method (or both) queries it.
        """
        if not self._transients:
            raise ValueError("No transient types registered in `transient_collection`; nothing to tabulate.")

        nside, order = resolve_healpix_resolution(nside, order)
        edges = self._resolve_time_bins(time_bins)
        n_bins = len(edges) - 1
        sorted_names = sorted(self._transients)

        transient_type = []
        time_bin = []
        t_start_col = []
        t_end_col = []
        n_pixels_visited = []
        solid_angle = []
        duration = []

        with (
            tqdm(total=len(sorted_names) * n_bins, desc="Tabulating effective exposure", unit="bin") as pbar,
            logging_redirect_tqdm(loggers=[logger]),
        ):
            for name in sorted_names:
                transient = self._transients[name]

                for k in range(n_bins):
                    pbar.set_postfix(type=name, bin=f"{k + 1}/{n_bins}")

                    t_start, t_end = edges[k], edges[k + 1]

                    pixel_ids = self._survey_schedule.get_observed_healpix_ids(
                        t_start,
                        t_end + transient.duration_limit,
                        nside=nside,
                        order=order,
                    )

                    transient_type.append(name)
                    time_bin.append(k)
                    t_start_col.append(t_start)
                    t_end_col.append(t_end)
                    n_pixels_visited.append(len(pixel_ids))
                    solid_angle.append(ah.nside_to_pixel_area(nside) * len(pixel_ids))
                    duration.append((t_end - t_start).to_value(u.day) * u.day)

                    pbar.update(1)

        table = QTable()
        table["transient_type"] = np.asarray(transient_type)
        table["time_bin"] = np.asarray(time_bin, dtype=np.int64)
        table["t_start"] = Time(t_start_col)
        table["t_end"] = Time(t_end_col)
        table["n_pixels_visited"] = np.asarray(n_pixels_visited, dtype=np.int64)
        table["solid_angle"] = u.Quantity(solid_angle)
        table["duration"] = u.Quantity(duration)
        table["effective_exposure"] = table["solid_angle"] * table["duration"]
        table["expected_events"] = [
            (self._transients[name].integrated_rate * exposure).to_value(u.dimensionless_unscaled)
            for name, exposure in zip(table["transient_type"], table["effective_exposure"])
        ]

        return ExposureCatalog(table=table, nside=nside, order=order, time_bins=edges)

    # -------------------------------------------------- #
    # Filtering                                          #
    # -------------------------------------------------- #
    def _iter_epochs(
        self,
        catalog,
        band_names,
        chunk_size,
        detection_floor,
        selected,
        transient_type,
        progress,
        detector,
        include_snr,
        lookback,
    ) -> Iterator[QTable]:
        """Yield the chunks for `iter_epochs`, which does all the argument validation."""
        table = catalog.table
        schedule = self._survey_schedule

        type_idx = {
            name: np.flatnonzero((transient_type == name) & selected) for name in sorted(set(transient_type[selected]))
        }
        total_chunks = sum(-(-idx.size // chunk_size) for idx in type_idx.values())

        with tqdm(total=total_chunks, desc="Computing epochs", unit="chunk", disable=not progress) as pbar:
            for name, idx in type_idx.items():
                transient = self._transients[name]
                event_id_type = np.asarray(table["event_id"])[idx]
                coord_type = table["coord"][idx]
                t_explosion_type = table["t_explosion"][idx]

                if include_snr:
                    # Each event's own `parameter_seed`, exactly as `Event` uses it, so these SNRs
                    # agree with `Event.simulate_photometry`: it seeds the SED parameters and the
                    # noise of every measurement.
                    seed_type = np.asarray(table["parameter_seed"])[idx]
                    sed_inputs_type = {
                        "redshift": np.asarray(table["redshift"])[idx],
                        "luminosity_distance": table["luminosity_distance"][idx],
                        "ebv": np.asarray(table["ebv"])[idx],
                        **_sample_parameters_from_seeds(transient.sed, seed_type),
                    }

                for start in range(0, idx.size, chunk_size):
                    pbar.set_postfix(type=name)
                    chunk_slice = slice(start, start + chunk_size)
                    coord_chunk = coord_type[chunk_slice]
                    t_explosion_chunk = t_explosion_type[chunk_slice]

                    # Which observations covered each event, from `lookback` before its
                    # explosion (or the start of the schedule) to the end of its active window.
                    if lookback is None:
                        window_start = schedule.start_time + np.zeros(len(coord_chunk)) * u.day
                    else:
                        window_start = t_explosion_chunk - lookback
                    event_index, row_index = schedule.get_observation_indices_of(
                        coord_chunk,
                        nside=catalog.nside,
                        order=catalog.order,
                        start_time=window_start,
                        end_time=t_explosion_chunk + transient.duration_limit,
                    )
                    if len(row_index) == 0:
                        pbar.update(1)
                        continue

                    # One entry per (event, observation) pair, for every column below.
                    observed = schedule.observe_rows[row_index]
                    t_since_explosion = (observed["start_time"] - t_explosion_chunk[event_index]).to(u.day)
                    columns = {
                        "event_id": event_id_type[chunk_slice][event_index],
                        "observation_index": np.asarray(row_index),
                        "t_obs": observed["start_time"],
                        "t_since_explosion": t_since_explosion,
                        "pre_explosion": np.asarray(t_since_explosion.to_value(u.day) < 0),
                    }

                    # Without an SNR there is no floor to apply, so every epoch is kept.
                    meets_floor = np.ones(len(row_index), dtype=bool)

                    if include_snr:
                        # The same measurement `Event.simulate_photometry` makes, for every epoch of
                        # every event in the chunk at once. The noise is keyed on (event seed,
                        # observation start time, band), so each measurement matches the event's
                        # own photometry exactly.
                        flux, flux_err, snr, _ = transient.sed.measure_photometry(
                            t_since_explosion,
                            observed["duration"],
                            detector,
                            coord_chunk[event_index],
                            bands=band_names,
                            observer_location=observed["observer_location"],
                            obstime=observed["start_time"],
                            noise_seed=seed_type[chunk_slice][event_index],
                            noise_observation_keys=time_key(observed["start_time"]),
                            **{key: values[chunk_slice][event_index] for key, values in sed_inputs_type.items()},
                        )

                        # Keep each epoch's best band, so an observation is never counted once per
                        # band. A band that could not be measured (NaN) never wins.
                        best = np.argmax(np.where(np.isnan(snr), -np.inf, snr), axis=0)
                        epochs = np.arange(len(row_index))
                        columns["band"] = np.asarray(band_names)[best]
                        columns["snr"] = snr[best, epochs]
                        columns["flux"] = flux[best, epochs] * u.Jy
                        columns["flux_err"] = flux_err[best, epochs] * u.Jy
                        meets_floor = columns["snr"] >= detection_floor

                    pbar.update(1)
                    if not np.any(meets_floor):
                        continue

                    # Each event's rows are contiguous and time-ordered, so per-event reductions
                    # (see `event_boundaries`) can rely on it.
                    chunk = QTable(columns)[meets_floor]
                    yield chunk[np.lexsort((chunk["t_obs"].jd, chunk["event_id"]))]

    def _iter_detection_epochs(
        self,
        catalog: EventCatalog,
        mission: Mission,
        snr_threshold: float,
        bands: list[str] | None,
        chunk_size: int | None,
        keep_all: bool = False,
        lookback: u.Quantity | None = 0 * u.day,
    ) -> Iterator[QTable]:
        """
        Yield `iter_epochs` chunks with each epoch marked as a detection or not.

        Each chunk gains two columns: ``detected`` (``snr`` above `snr_threshold`, and not before
        the explosion, since a pre-explosion epoch has no source) and ``non_detected`` (``snr`` at
        or below it).

        Parameters
        ----------
        catalog : EventCatalog
            Forwarded to `iter_epochs`.
        mission : m4opt.missions.Mission
            Forwarded to `iter_epochs`.
        snr_threshold : float
            The detection threshold.
        bands : list of str, optional
            Forwarded to `iter_epochs`.
        chunk_size : int, optional
            Forwarded to `iter_epochs`.
        keep_all : bool, optional
            If `False` (the default), epochs below `snr_threshold` are dropped by the
            iterator before they are yielded, since they cannot be detections. If `True`,
            every epoch is kept, e.g. to find non-detections.
        lookback : ~astropy.units.Quantity or None, optional
            Forwarded to `iter_epochs`.

        Yields
        ------
        ~astropy.table.QTable
            One chunk of epochs, with the columns above added.
        """
        epochs = self.iter_epochs(
            catalog,
            mission,
            bands=bands,
            chunk_size=chunk_size,
            detection_floor=-np.inf if keep_all else snr_threshold,
            lookback=lookback,
        )
        with logging_redirect_tqdm(loggers=[logger]):
            for chunk in epochs:
                chunk["detected"] = (chunk["snr"] > snr_threshold) & ~chunk["pre_explosion"]
                chunk["non_detected"] = chunk["snr"] <= snr_threshold
                yield chunk

    def _reduce_events(
        self,
        catalog: EventCatalog,
        mission: Mission,
        snr_threshold: float,
        bands: list[str] | None,
        chunk_size: int | None,
        reducer: Callable[[QTable, slice], Any],
        keep_all: bool = False,
        lookback: u.Quantity | None = 0 * u.day,
    ) -> dict[int, Any]:
        """
        Reduce each event's epochs to a single value in one pass over `iter_epochs`.

        One event's epochs never span two chunks (chunking splits events, not
        observations), so no merging across chunks is needed.

        Parameters
        ----------
        catalog : EventCatalog
            Forwarded to `_iter_detection_epochs`.
        mission : m4opt.missions.Mission
            Forwarded to `_iter_detection_epochs`.
        snr_threshold : float
            The detection threshold.
        bands : list of str, optional
            Forwarded to `_iter_detection_epochs`.
        chunk_size : int, optional
            Forwarded to `_iter_detection_epochs`.
        reducer : callable
            ``reducer(chunk, rows)``, given a chunk (see `_iter_detection_epochs` for its
            columns) and the slice of rows belonging to one event, returns that event's
            value, or `None` to leave the event out.
        keep_all : bool, optional
            If `False` (the default), `reducer` sees only each event's detection epochs, and
            an event with none is never passed to it. If `True` it sees every epoch.
        lookback : ~astropy.units.Quantity or None, optional
            Forwarded to `_iter_detection_epochs`.

        Returns
        -------
        dict of int to Any
            ``{event_id: value}`` for every event `reducer` returned a value for.
        """
        reduced = {}
        for chunk in self._iter_detection_epochs(
            catalog, mission, snr_threshold, bands, chunk_size, keep_all, lookback
        ):
            if not keep_all:
                chunk = chunk[chunk["detected"]]
            event_id = np.asarray(chunk["event_id"])
            for start, stop in zip(*event_boundaries(event_id)):
                value = reducer(chunk, slice(start, stop))
                if value is not None:
                    reduced[int(event_id[start])] = value
        return reduced

    def _mask_out_first_visit_solo_detections(
        self, catalog: EventCatalog, solo: dict[int, int], keep: np.ndarray
    ) -> None:
        """
        Clear `keep` in place for every event in `solo` whose epoch is its field's first visit.

        Shared by `filter_by_first_visit_detected` and `filter_by_snr`'s own
        `exclude_first_visit_detections` option, so the two apply the exact same
        first-visit test rather than each rolling its own.

        Parameters
        ----------
        catalog : EventCatalog
            Supplies the ``event_id`` column `keep` is indexed against.
        solo : dict of int to int
            ``{event_id: observation_index}``, as produced by `_reduce_events` with
            `_reduce_solo_detection`.
        keep : numpy.ndarray
            Boolean array, same length as `catalog`, indexed the same way as
            `catalog.table`; modified in place.
        """
        if not solo:
            return

        first_visit = self.survey_schedule.first_visit_mask
        event_id = np.asarray(catalog.table["event_id"])
        for i, eid in enumerate(event_id):
            obs_index = solo.get(int(eid))
            if obs_index is not None and first_visit[obs_index]:
                keep[i] = False

    def _filtered(self, catalog: EventCatalog, keep: np.ndarray) -> EventCatalog:
        """Return a new `EventCatalog` over `catalog.table[keep]`, carrying every other field unchanged."""
        return EventCatalog(
            table=catalog.table[keep],
            nside=catalog.nside,
            order=catalog.order,
            time_bins=catalog.time_bins,
            seed=catalog.seed,
            downsample=catalog.downsample,
        )

    def _peak_band_flux(
        self,
        catalog: EventCatalog,
        mission: Mission,
        bands: list[str] | None = None,
        n_phase: int | None = None,
        chunk_size: int | None = None,
    ) -> u.Quantity:
        """
        Evaluate the peak (brightest) band-integrated observed flux each event ever reaches.

        Shares `filter_by_limiting_magnitude`'s shared-phase-grid, chunked evaluation
        strategy, but ignores Milky Way dust attenuation (no `log_attenuation`) and the
        survey schedule entirely, and returns the underlying flux `~astropy.units.Quantity`
        itself rather than collapsing it to a keep/discard mask. Shared by
        `filter_by_peak_magnitude` (the AB magnitude of this) and `filter_by_peak_flux`
        (this value directly).

        Parameters
        ----------
        catalog : EventCatalog
            Typically produced by `generate_events`; must already carry the
            `luminosity_distance`/`parameter_seed` columns that method fills in.
        mission : m4opt.missions.Mission
            Supplies the `~m4opt.synphot.Detector` whose named bandpasses `bands` selects
            from.
        bands : list of str, optional
            Which of `mission.detector`'s bandpasses to evaluate. Defaults to every
            bandpass the detector has.
        n_phase : int, optional
            Number of phase-grid samples per transient type, spanning
            ``[0, transient.duration_limit]``. If `None` (the default), uses
            ``config["simulation.filter_by_limiting_magnitude.n_phase"]``.
        chunk_size : int, optional
            Number of events (per transient type) evaluated per `flux_band` call. If
            `None` (the default), uses
            ``config["simulation.filter_by_limiting_magnitude.chunk_size"]``.

        Returns
        -------
        ~astropy.units.Quantity
            One peak flux value (erg/s/cm^2/Hz) per row of `catalog.table`, in row order.

        Raises
        ------
        TypeError
            If `catalog` is not an `EventCatalog`, or `mission` is not a `Mission`.
        ValueError
            If `mission` has no detector, `bands` names an unknown bandpass, `catalog` is
            missing a required column, or contains an unregistered transient type.
        """
        if not isinstance(catalog, EventCatalog):
            raise TypeError(f"'catalog' must be an EventCatalog, got {type(catalog)} instead.")
        if not isinstance(mission, Mission):
            raise TypeError(f"'mission' must be an m4opt.missions.Mission, got {type(mission)} instead.")

        if n_phase is None:
            n_phase = config["simulation.filter_by_limiting_magnitude.n_phase"]
        if chunk_size is None:
            chunk_size = config["simulation.filter_by_limiting_magnitude.chunk_size"]

        detector = mission.detector
        if detector is None:
            raise ValueError(f"Mission {mission.name!r} has no detector configured.")

        band_names = list(detector.bandpasses) if bands is None else list(bands)
        unknown_bands = [band for band in band_names if band not in detector.bandpasses]
        if unknown_bands:
            raise ValueError(f"Unknown bandpass(es) {unknown_bands}; available: {list(detector.bandpasses)}.")

        table = catalog.table
        missing = [col for col in ("luminosity_distance", "parameter_seed") if col not in table.colnames]
        if missing:
            raise ValueError(f"'catalog' is missing column(s) {missing}; regenerate it via `generate_events`.")

        flux_unit = u.erg / u.s / u.cm**2 / u.Hz
        if len(table) == 0:
            return u.Quantity([], flux_unit)

        band_grids = {}
        for band in band_names:
            bp = detector.bandpasses[band]
            wave = bp.waveset
            band_grids[band] = (wave, bp(wave), wave.to(u.Hz, equivalencies=u.spectral()))

        transient_type = np.asarray(table["transient_type"]).astype(str)
        type_names = sorted(set(transient_type) & set(self._transients))
        unknown_types = sorted(set(transient_type) - set(self._transients))
        if unknown_types:
            raise ValueError(f"'catalog' contains transient type(s) {unknown_types} not in `transient_collection`.")

        peak_flux = np.full(len(table), -np.inf)

        type_idx = {name: np.flatnonzero(transient_type == name) for name in type_names}
        n_chunks_by_type = {name: -(-idx.size // chunk_size) for name, idx in type_idx.items()}
        total_chunks = sum(n_chunks_by_type.values())

        with (
            tqdm(total=total_chunks, desc="Evaluating peak band flux", unit="chunk") as pbar,
            logging_redirect_tqdm(loggers=[logger]),
        ):
            for name in type_names:
                transient = self._transients[name]
                idx = type_idx[name]
                n = idx.size
                n_chunks = n_chunks_by_type[name]

                redshift_all = np.asarray(table["redshift"])[idx]
                luminosity_distance_all = table["luminosity_distance"][idx]
                seeds_all = np.asarray(table["parameter_seed"])[idx]
                sed_params_all = _sample_parameters_from_seeds(transient.sed, seeds_all)

                t_grid = np.linspace(0.0, 1.0, n_phase) * transient.duration_limit

                for chunk_num, start in enumerate(range(0, n, chunk_size), start=1):
                    stop = min(start + chunk_size, n)
                    chunk_idx = idx[start:stop]
                    pbar.set_postfix(type=name, chunk=f"{chunk_num}/{n_chunks}")

                    redshift = redshift_all[start:stop]
                    luminosity_distance = luminosity_distance_all[start:stop]
                    sed_params = {param_name: value[start:stop] for param_name, value in sed_params_all.items()}

                    best_flux_by_phase = None
                    for band in band_names:
                        wave, throughput, nu = band_grids[band]
                        flux = transient.sed.flux_band(
                            nu,
                            throughput,
                            t_grid[:, None],
                            redshift=redshift,
                            luminosity_distance=luminosity_distance,
                            **sed_params,
                        )  # shape (n_phase, chunk_size)
                        best_flux_by_phase = (
                            flux if best_flux_by_phase is None else np.maximum(best_flux_by_phase, flux)
                        )

                    peak_flux[chunk_idx] = np.max(best_flux_by_phase.to_value(flux_unit), axis=0)
                    pbar.update(1)

        return u.Quantity(peak_flux, flux_unit)

    def iter_epochs(
        self,
        catalog: EventCatalog,
        mission: Mission | None = None,
        bands: list[str] | None = None,
        chunk_size: int | None = None,
        detection_floor: float = 0.0,
        mask: np.ndarray | None = None,
        progress: bool = True,
        include_snr: bool = True,
        lookback: u.Quantity | None = 0 * u.day,
    ) -> Iterator[QTable]:
        """
        Lazily yield every observation of every event, one chunk of events at a time.

        The core iterator the schedule-aware cuts are built on: any threshold-, visit-count-,
        or cadence-style reduction can be computed from these tables without re-measuring
        anything. For every (event, observation) pair of a chunk it makes the same measurement
        `~uvex_transients.simulation.event.Event.simulate_photometry` makes, with the same noise.

        Each yielded `~astropy.table.QTable` has one row per (event, observation) pair
        for up to `chunk_size` events of a single transient type, sorted by
        ``(event_id, t_obs)`` so each event's rows are contiguous (see `event_boundaries`), with columns:

        - ``event_id``: the event's own ``event_id`` in `catalog`.
        - ``observation_index``: row index into ``survey_schedule.observe_rows``.
        - ``t_obs``: `~astropy.time.Time` the observation started.
        - ``t_since_explosion``: `~astropy.units.Quantity` (day), ``t_obs - t_explosion``;
          negative before the explosion.
        - ``pre_explosion``: bool, ``t_since_explosion < 0``. Such a row has no source, so
          it is measured as pure background and can never be a detection.
        - ``snr``, ``band``, ``flux``, ``flux_err`` (if `include_snr`): the measured SNR of the
          observation in its best band (bands are collapsed to the best one, so an observation
          is never counted once per band), that band, its measured flux density
          (`~astropy.units.Quantity` in Jy), and the one-sigma uncertainty on it.
          ``snr`` is ``flux / flux_err``. The measurement is a Gaussian draw around the true
          flux, keyed on the event's ``parameter_seed``, the observation's start time and the
          band (see `~uvex_transients.utils.keyed_noise.keyed_standard_normal`), so it is the
          same draw `~uvex_transients.simulation.event.Event.simulate_photometry` makes for
          that measurement, whatever else is evaluated alongside it. For a pre-explosion row it
          is a draw from the background noise alone, at that observation's real depth.

        With `include_snr` `False` only the geometry columns are produced, and neither
        `mission` nor the SED is touched.

        A chunk in which no event was observed, or in which every row falls below
        `detection_floor`, yields nothing. An event with no rows at or above
        `detection_floor` therefore never appears in any chunk.

        Parameters
        ----------
        catalog : EventCatalog
            The events to evaluate. Needs ``event_id``, ``coord``, ``t_explosion``,
            ``transient_type`` and ``healpix_id``, plus ``redshift``, ``luminosity_distance``,
            ``ebv`` and ``parameter_seed`` (all filled in by `generate_events`) when
            `include_snr`.
        mission : m4opt.missions.Mission, optional
            Supplies the `~m4opt.synphot.Detector` `bands` selects from. Required unless
            `include_snr` is `False`.
        bands : list of str, optional
            Which of `mission.detector`'s bandpasses to evaluate. Defaults to all of them.
        chunk_size : int, optional
            Number of events (per transient type) evaluated together. If `None`, uses
            ``config["simulation.filter_by_snr.chunk_size"]``.
        detection_floor : float, optional
            Rows with a measured ``snr`` below this are dropped before being yielded. This
            bounds the size of the output, but also means a later reduction can't use a
            threshold below it. The default, 0, keeps every row with a non-negative SNR.
            Ignored when `include_snr` is `False`. Use ``-numpy.inf`` to keep every row,
            e.g. to see non-detections.
        mask : numpy.ndarray, optional
            Restricts evaluation to a subset of `catalog`'s events: either a boolean
            array of length ``len(catalog)`` or an integer index array into
            ``catalog.table``. Masked-out events cost nothing. `None` evaluates them all.
        progress : bool, optional
            Whether to show a progress bar over chunks.
        include_snr : bool, optional
            Whether to measure each observation and yield the ``snr``/``band``/``flux``/
            ``flux_err`` columns. If `False`, only the geometry of which observations covered
            which events is returned, which is much cheaper.
        lookback : ~astropy.units.Quantity or None, optional
            How far before each event's explosion to include observations. The default,
            0 days, keeps only observations that overlap the post-explosion window (this
            can include an exposure that starts just before the explosion, flagged
            ``pre_explosion``). A positive duration also yields the earlier observations of
            the event's position, e.g. to find the last pre-explosion non-detection. `None`
            looks back to the start of the schedule.

        Returns
        -------
        collections.abc.Iterator of ~astropy.table.QTable
            One epoch table per chunk of events, as described above.

        Raises
        ------
        TypeError
            If `catalog` is not an `EventCatalog`, or an SNR is requested without a
            `~m4opt.missions.Mission`.
        ValueError
            If `chunk_size`, `bands`, `mask`, or `lookback` is invalid, or `catalog` lacks
            a required column.
        """
        if not isinstance(catalog, EventCatalog):
            raise TypeError(f"'catalog' must be an EventCatalog, got {type(catalog)} instead.")

        if include_snr and not isinstance(mission, Mission):
            raise TypeError(f"'mission' must be an m4opt.missions.Mission, got {type(mission)} instead.")

        if chunk_size is None:
            chunk_size = config["simulation.filter_by_snr.chunk_size"]
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
            raise ValueError(f"'chunk_size' must be a positive int, got {chunk_size!r}.")

        if lookback is not None:
            try:
                lookback_days = u.Quantity(lookback).to_value(u.day)
            except (u.UnitConversionError, TypeError) as err:
                raise ValueError(f"'lookback' must be a duration or None, got {lookback!r}.") from err
            if np.ndim(lookback_days) != 0 or not np.isfinite(lookback_days) or lookback_days < 0:
                raise ValueError(f"'lookback' must be a finite, non-negative scalar duration, got {lookback!r}.")
            lookback = u.Quantity(lookback)

        detector, band_names = None, []
        if include_snr:
            detector = mission.detector
            if detector is None:
                raise ValueError(f"Mission {mission.name!r} has no detector configured.")

            band_names = list(detector.bandpasses) if bands is None else list(bands)
            if not band_names:
                raise ValueError("'bands' must contain at least one band.")
            unknown_bands = [band for band in band_names if band not in detector.bandpasses]
            if unknown_bands:
                raise ValueError(f"Unknown bandpass(es) {unknown_bands}; available: {list(detector.bandpasses)}.")

        table = catalog.table
        required = ["event_id", "healpix_id"]
        if include_snr:
            required += ["luminosity_distance", "ebv", "parameter_seed"]
        missing = [col for col in required if col not in table.colnames]
        if missing:
            raise ValueError(f"'catalog' is missing column(s) {missing}; regenerate it via `generate_events`.")

        selected = np.ones(len(table), dtype=bool)
        if mask is not None:
            mask = np.asarray(mask)
            if mask.dtype == bool:
                if mask.shape != (len(table),):
                    raise ValueError(f"Boolean 'mask' must have shape ({len(table)},), got {mask.shape}.")
                selected = mask
            else:
                selected = np.zeros(len(table), dtype=bool)
                selected[mask] = True
        transient_type = np.asarray(table["transient_type"]).astype(str)
        unknown_types = sorted(set(transient_type[selected]) - set(self._transients))
        if unknown_types:
            raise ValueError(f"'catalog' contains transient type(s) {unknown_types} not in `transient_collection`.")

        # Validation above runs eagerly, at call time; only the evaluation itself is lazy.
        return self._iter_epochs(
            catalog,
            band_names,
            chunk_size,
            detection_floor,
            selected,
            transient_type,
            progress,
            detector,
            include_snr,
            lookback,
        )

    @classmethod
    def available_cuts(cls) -> tuple[str, ...]:
        """Tuple of str: Every cut name registered on this class via `@cut`, sorted."""
        return tuple(sorted(cls._CUT_REGISTRY))

    def run_cut(self, name: str, catalog: EventCatalog, mission: Mission, **params) -> EventCatalog:
        """
        Run one `@cut`-registered screening method by name.

        A thin dispatch layer over `filter_by_limiting_magnitude`/`filter_by_snr` (and any
        further ``@cut``-decorated methods a subclass adds) so a config-driven caller (see
        `uvex_transients.cli`) can select a cut by name rather than hardcoding which Python
        method to call.

        Parameters
        ----------
        name : str
            One of `available_cuts`.
        catalog : EventCatalog
            The catalog to filter.
        mission : m4opt.missions.Mission
            The mission whose detector(s)/bandpasses the cut evaluates against.
        **params
            Forwarded to the underlying cut method (e.g. `mag_limit` for the
            ``"limiting_magnitude"`` cut, `snr_threshold` for ``"snr"``).

        Returns
        -------
        EventCatalog
            The filtered catalog.
        """
        try:
            method_name = self._CUT_REGISTRY[name]
        except KeyError:
            raise ValueError(f"Unknown cut {name!r}; available: {self.available_cuts()}.") from None
        return getattr(self, method_name)(catalog, mission, **params)

    @cut("limiting_magnitude")
    def filter_by_limiting_magnitude(
        self,
        catalog: EventCatalog,
        mission: Mission,
        mag_limit: float,
        bands: list[str] | None = None,
        n_phase: int | None = None,
        chunk_size: int | None = None,
        n_visits: int = 1,
    ) -> EventCatalog:
        """
        Cheaply cut an `EventCatalog` down to events that could ever plausibly be seen.

        Deliberately not synthetic photometry: no schedule, no background, no SNR formula,
        no per-event noise realization. Each event's brightest band is evaluated over a
        shared phase grid and compared against `mag_limit`; an event survives if at least
        `n_visits` phase samples clear the limit.

        Parameters
        ----------
        catalog : EventCatalog
            Typically produced by `generate_events`; must already carry the
            ``luminosity_distance``/``ebv``/``parameter_seed`` columns that method fills
            in.
        mission : m4opt.missions.Mission
            Supplies the `~m4opt.synphot.Detector` whose named bandpasses `bands` selects
            from.
        mag_limit : float
            AB magnitude limit; a phase sample clears the limit if the event's brightest
            evaluated band is at or below this value at that phase.
        bands : list of str, optional
            Which of `mission.detector`'s bandpasses to evaluate. Defaults to every bandpass
            the detector has.
        n_phase : int, optional
            Number of phase-grid samples per transient type, spanning
            ``[0, transient.duration_limit]``. If `None` (the default), uses
            ``config["simulation.filter_by_limiting_magnitude.n_phase"]`` (50 out of the
            box).
        chunk_size : int, optional
            Number of events (per transient type) evaluated per `flux_band` call. If
            `None` (the default), uses
            ``config["simulation.filter_by_limiting_magnitude.chunk_size"]`` (5000 out of
            the box). Lower it for a very fine `n_phase` or a very densely sampled bandpass.
        n_visits : int, optional
            Minimum number of phase-grid samples that must clear `mag_limit` for an event to
            survive. The default is 1, i.e. an event survives if it is ever bright enough at
            even a single sampled phase.

        Returns
        -------
        EventCatalog
            A new catalog over the surviving rows only (same `nside`/`order`/`time_bins`/
            `seed` as `catalog`; original `event_id` values are preserved, not renumbered).

        See Also
        --------
        filter_by_snr : The schedule-aware cut this one screens events ahead of.
        filter_by_peak_magnitude : A cheaper, schedule- and dust-independent magnitude cut.

        Notes
        -----
        Each event's own physical SED parameters are regenerated from its stored
        `parameter_seed` (see `_sample_parameters_from_seeds`), so a kept event's later
        `simulate_photometry` realization is guaranteed to match what was screened here,
        not an independent draw from the same population.

        `mag_limit` is compared against the same ``linspace(0, duration_limit, n_phase)``
        phase grid for every event of a type, regardless of whether the schedule ever
        actually pointed there at that phase. Raising `n_visits` above 1 is therefore only
        a coarse stand-in for "detected on at least `n_visits` visits", since this method
        never consults the actual schedule; `filter_by_snr` does.

        A freshly sampled catalog is typically dominated by faint, easily rejected events
        and can run into hundreds of thousands of rows. `flux_band` broadcasts every event
        and phase sample into one dense ``(n_phase, n_events, n_wavelength)`` array, so
        evaluating that in a single shot over the whole catalog can exhaust memory;
        `chunk_size` bounds this by evaluating events `chunk_size` at a time, keeping peak
        memory roughly constant regardless of catalog size.

        Examples
        --------
        .. code-block:: python

            detected = simulator.filter_by_limiting_magnitude(
                catalog, mission, mag_limit=22.0, n_visits=2
            )
        """
        if not isinstance(catalog, EventCatalog):
            raise TypeError(f"'catalog' must be an EventCatalog, got {type(catalog)} instead.")
        if not isinstance(mission, Mission):
            raise TypeError(f"'mission' must be an m4opt.missions.Mission, got {type(mission)} instead.")
        if isinstance(n_visits, bool) or not isinstance(n_visits, (int, np.integer)) or n_visits < 1:
            raise ValueError(f"'n_visits' must be a positive int, got {n_visits!r}.")

        if n_phase is None:
            n_phase = config["simulation.filter_by_limiting_magnitude.n_phase"]
        if chunk_size is None:
            chunk_size = config["simulation.filter_by_limiting_magnitude.chunk_size"]

        detector = mission.detector
        if detector is None:
            raise ValueError(f"Mission {mission.name!r} has no detector configured.")

        band_names = list(detector.bandpasses) if bands is None else list(bands)
        unknown_bands = [band for band in band_names if band not in detector.bandpasses]
        if unknown_bands:
            raise ValueError(f"Unknown bandpass(es) {unknown_bands}; available: {list(detector.bandpasses)}.")

        table = catalog.table
        missing = [col for col in ("luminosity_distance", "ebv", "parameter_seed") if col not in table.colnames]
        if missing:
            raise ValueError(f"'catalog' is missing column(s) {missing}; regenerate it via `generate_events`.")

        if len(table) == 0:
            return self._filtered(catalog, np.ones(0, dtype=bool))

        # Bandpass wavelength/throughput/frequency grids are the same for every transient
        # type and event; sampled once here, not per type.
        band_grids = {}
        for band in band_names:
            bp = detector.bandpasses[band]
            wave = bp.waveset
            band_grids[band] = (
                wave,
                bp(wave),
                wave.to(u.Hz, equivalencies=u.spectral()),
            )

        transient_type = np.asarray(table["transient_type"]).astype(str)
        type_names = sorted(set(transient_type) & set(self._transients))
        unknown_types = sorted(set(transient_type) - set(self._transients))
        if unknown_types:
            raise ValueError(f"'catalog' contains transient type(s) {unknown_types} not in `transient_collection`.")

        keep = np.zeros(len(table), dtype=bool)

        # Precompute chunk counts per type up front so the progress bar can report a single
        # total spanning every chunk of every type, not just one tick per type.
        type_idx = {name: np.flatnonzero(transient_type == name) for name in type_names}
        n_chunks_by_type = {name: -(-idx.size // chunk_size) for name, idx in type_idx.items()}
        total_chunks = sum(n_chunks_by_type.values())

        with (
            tqdm(total=total_chunks, desc="Filtering by mag limit", unit="chunk") as pbar,
            logging_redirect_tqdm(loggers=[logger]),
        ):
            for name in type_names:
                transient = self._transients[name]
                idx = type_idx[name]
                n = idx.size
                n_chunks = n_chunks_by_type[name]

                redshift_all = np.asarray(table["redshift"])[idx]
                luminosity_distance_all = table["luminosity_distance"][idx]
                ebv_all = np.asarray(table["ebv"])[idx]
                seeds_all = np.asarray(table["parameter_seed"])[idx]

                # Sampling physical parameters for the whole type at once is cheap (a
                # handful of floats per event); it's only the `flux_band` evaluation below
                # (an (n_phase, n_chunk, n_wavelength) array per band) that can blow up
                # memory for a large catalog, so only *that* is chunked (see `chunk_size`).
                # Each event's own `parameter_seed` (not a shared per-type stream) keeps
                # this screening pass consistent with `Event.mag`/`Event.simulate_photometry`;
                # see `_sample_parameters_from_seeds`.
                sed_params_all = _sample_parameters_from_seeds(transient.sed, seeds_all)

                t_grid = np.linspace(0.0, 1.0, n_phase) * transient.duration_limit

                for chunk_num, start in enumerate(range(0, n, chunk_size), start=1):
                    stop = min(start + chunk_size, n)
                    chunk_idx = idx[start:stop]

                    pbar.set_postfix(type=name, chunk=f"{chunk_num}/{n_chunks}")

                    redshift = redshift_all[start:stop]
                    luminosity_distance = luminosity_distance_all[start:stop]
                    ebv = ebv_all[start:stop]
                    sed_params = {param_name: value[start:stop] for param_name, value in sed_params_all.items()}

                    # Brightest flux across bands *at each sampled phase* (not collapsed
                    # across phase yet), so visits can be counted per phase sample below.
                    best_flux_by_phase = None
                    for band in band_names:
                        wave, throughput, nu = band_grids[band]
                        flux = transient.sed.flux_band(
                            nu,
                            throughput,
                            t_grid[:, None],
                            redshift=redshift,
                            luminosity_distance=luminosity_distance,
                            log_attenuation=log_attenuation(nu, ebv),
                            **sed_params,
                        )  # shape (n_phase, chunk_size)
                        best_flux_by_phase = (
                            flux if best_flux_by_phase is None else np.maximum(best_flux_by_phase, flux)
                        )

                    with np.errstate(invalid="ignore", divide="ignore"):
                        mag_by_phase = best_flux_by_phase.to_value(u.ABmag)  # shape (n_phase, chunk_size)

                    n_visits_cleared = np.count_nonzero(mag_by_phase <= mag_limit, axis=0)  # shape (chunk_size,)
                    keep[chunk_idx[n_visits_cleared >= n_visits]] = True

                    pbar.update(1)

        return self._filtered(catalog, keep)

    @cut("snr")
    def filter_by_snr(
        self,
        catalog: EventCatalog,
        mission: Mission,
        snr_threshold: float,
        bands: list[str] | None = None,
        chunk_size: int | None = None,
        n_visits: int = 1,
        exclude_first_visit_detections: bool = True,
    ) -> EventCatalog:
        """
        Cut an `EventCatalog` down to events the schedule actually detects.

        Unlike `filter_by_limiting_magnitude` (which never consults the schedule),
        this asks the real question: over every observation the schedule actually
        made of an event's position while it was active, is it ever detected above
        `snr_threshold`? An event survives if at least `n_visits` observations clear
        it, in their best band. Each observation is judged on its measured SNR: a simulated
        measurement of the true flux with the detector's noise, the same measurement
        `~uvex_transients.simulation.event.Event.simulate_photometry` makes. Epochs before
        the explosion never count as detections.

        Parameters
        ----------
        catalog : EventCatalog
            Typically the (already magnitude-filtered) output of
            `filter_by_limiting_magnitude`. Running this directly on a freshly sampled
            catalog works too, just more slowly, since every event still costs one
            `get_observations_of` call regardless of how faint it is.
        mission : m4opt.missions.Mission
            Supplies the `~m4opt.synphot.Detector` `bands` selects from.
        snr_threshold : float
            An event survives if at least `n_visits` observations exceed this SNR, in
            their best band.
        bands : list of str, optional
            Which of `mission.detector`'s bandpasses to evaluate. Defaults to every
            bandpass the detector has.
        chunk_size : int, optional
            Number of events (per transient type) whose observations are gathered and
            evaluated together, bounding the size of each flattened `get_snr` batch. If
            `None` (the default), uses ``config["simulation.filter_by_snr.chunk_size"]``
            (2000 out of the box); see `filter_by_limiting_magnitude`'s docstring for
            the same memory-vs-vectorization tradeoff on the evaluation side.
        n_visits : int, optional
            Minimum number of observations that must clear `snr_threshold` (in their
            best band) for an event to survive. The default is 1: an event survives if
            it is ever detected at all.
        exclude_first_visit_detections : bool, optional
            If `True` (the default), also drop an event whose *only* qualifying epoch
            anywhere in `catalog` (regardless of `n_visits`) falls on its field's
            first-ever survey visit -- see `filter_by_first_visit_detected` for why that
            detection has no reference image to be judged against. Pass `False` to keep
            such events here and leave that decision to a separate
            `filter_by_first_visit_detected` call instead.

        Returns
        -------
        EventCatalog
            A new catalog over the surviving rows only (same `nside`/`order`/`time_bins`/
            `seed` as `catalog`; original `event_id` values are preserved, not renumbered).

        Raises
        ------
        TypeError
            If `catalog` is not an `EventCatalog`.
        ValueError
            If `n_visits` is not a positive int.

        See Also
        --------
        filter_by_limiting_magnitude : The cheaper, schedule-independent cut to run first.
        filter_by_first_visit_detected : The standalone cut `exclude_first_visit_detections` wraps.
        iter_epochs : Reuse the underlying per-epoch SNRs for other cuts.

        Notes
        -----
        For each transient type present in `catalog`, events are processed `chunk_size`
        at a time via `iter_epochs` (see its own docstring for the full
        pipeline: schedule lookup, per-event parameter regeneration, and the vectorized
        `get_snr` evaluation), whose per-epoch SNR table this cut then reduces to a
        keep/discard decision by counting, per event, how many rows clear
        `snr_threshold`. Bands are collapsed to the best one before counting, so a
        single observation bright enough in two bands at once still counts as one
        visit, not two, mirroring `filter_by_limiting_magnitude`'s own `n_visits`.

        `exclude_first_visit_detections` reuses this same pass over the epochs rather than
        a second one for `filter_by_first_visit_detected`, which would re-run the whole
        (expensive) SNR evaluation. Each event's detections arrive grouped together (see
        `iter_epochs`), so a group of length 1 is an event with a single detection, and
        its one observation is remembered. One event's detections never span two chunks,
        so that length is the event's entire detection count, not just this chunk's share
        of it, and no merging across chunks is needed.

        Examples
        --------
        .. code-block:: python

            detected = simulator.filter_by_snr(
                catalog, mission, snr_threshold=5.0
            )

            # Keep first-visit solo detections too, e.g. to inspect them separately.
            detected_all = simulator.filter_by_snr(
                catalog,
                mission,
                snr_threshold=5.0,
                exclude_first_visit_detections=False,
            )
        """
        # Validate what only this cut cares about; the catalog, mission, bands and chunk size
        # are validated by `iter_epochs` itself, eagerly, when it is called below.
        if isinstance(n_visits, bool) or not isinstance(n_visits, (int, np.integer)) or n_visits < 1:
            raise ValueError(f"'n_visits' must be a positive int, got {n_visits!r}.")
        if not isinstance(catalog, EventCatalog):
            raise TypeError(f"'catalog' must be an EventCatalog, got {type(catalog)} instead.")

        event_id = np.asarray(catalog.table["event_id"])
        visit_count = np.zeros(len(catalog), dtype=np.int64)
        row_of_event = np.argsort(event_id)
        solo_observation_index: dict[int, int] = {}

        for chunk in self._iter_detection_epochs(catalog, mission, snr_threshold, bands, chunk_size):
            detections = chunk[chunk["detected"]]
            detected_id = np.asarray(detections["event_id"])

            # An event's detections are contiguous, so each block's length is its visit count.
            starts, stops = event_boundaries(detected_id)
            counts = stops - starts
            visit_count[row_of_event[np.searchsorted(event_id[row_of_event], detected_id[starts])]] += counts

            if exclude_first_visit_detections:
                solo = counts == 1
                solo_observation_index.update(
                    zip(
                        detected_id[starts[solo]].tolist(),
                        np.asarray(detections["observation_index"])[starts[solo]].tolist(),
                    )
                )

        keep = visit_count >= n_visits
        if exclude_first_visit_detections:
            self._mask_out_first_visit_solo_detections(catalog, solo_observation_index, keep)

        return self._filtered(catalog, keep)

    @cut("redshift")
    def filter_by_redshift(
        self,
        catalog: EventCatalog,
        mission: Mission,
        min_redshift: float | None = None,
        max_redshift: float | None = None,
    ) -> EventCatalog:
        """
        Cut an `EventCatalog` by its own `redshift` column.

        Parameters
        ----------
        catalog : EventCatalog
            The catalog to filter.
        mission : m4opt.missions.Mission
            Unused; accepted only so every `@cut` shares one call signature (see `run_cut`).
        min_redshift : float, optional
            Keep only rows with `redshift` at or above this value. `None` (the default)
            leaves the lower end unconstrained.
        max_redshift : float, optional
            Keep only rows with `redshift` at or below this value. `None` (the default)
            leaves the upper end unconstrained.

        Returns
        -------
        EventCatalog
            A new catalog over the surviving rows only.

        Raises
        ------
        TypeError
            If `catalog` is not an `EventCatalog`.
        ValueError
            If both `min_redshift` and `max_redshift` are `None`.

        See Also
        --------
        filter_by_transient_type : Cut by transient type instead of redshift.

        Examples
        --------
        .. code-block:: python

            nearby = simulator.filter_by_redshift(
                catalog, mission, max_redshift=0.1
            )
        """
        if not isinstance(catalog, EventCatalog):
            raise TypeError(f"'catalog' must be an EventCatalog, got {type(catalog)} instead.")
        if min_redshift is None and max_redshift is None:
            raise ValueError("At least one of 'min_redshift'/'max_redshift' must be given.")

        redshift = np.asarray(catalog.table["redshift"])
        keep = np.ones(len(catalog), dtype=bool)
        if min_redshift is not None:
            keep &= redshift >= min_redshift
        if max_redshift is not None:
            keep &= redshift <= max_redshift

        return self._filtered(catalog, keep)

    @cut("transient_type")
    def filter_by_transient_type(
        self,
        catalog: EventCatalog,
        mission: Mission,
        types: list[str],
    ) -> EventCatalog:
        """
        Keep only rows whose `transient_type` is one of `types`, dropping every other type.

        Distinct from any cut's own `transient_types:` scoping (see
        `uvex_transients.cli.steps.apply_cut_scoped`), which restricts *which rows a cut
        evaluates*, not which types ultimately survive.

        Parameters
        ----------
        catalog : EventCatalog
            The catalog to filter.
        mission : m4opt.missions.Mission
            Unused; accepted only so every `@cut` shares one call signature.
        types : list of str
            The `transient_type` value(s) to keep.

        Returns
        -------
        EventCatalog
            A new catalog over the surviving rows only.

        Raises
        ------
        TypeError
            If `catalog` is not an `EventCatalog`.
        ValueError
            If `types` is empty.

        See Also
        --------
        filter_by_redshift : Cut by redshift instead of transient type.

        Examples
        --------
        .. code-block:: python

            tdes_only = simulator.filter_by_transient_type(
                catalog, mission, types=["tde"]
            )
        """
        if not isinstance(catalog, EventCatalog):
            raise TypeError(f"'catalog' must be an EventCatalog, got {type(catalog)} instead.")
        if not types:
            raise ValueError("'types' must name at least one transient type.")

        keep = np.isin(np.asarray(catalog.table["transient_type"]), types)
        return self._filtered(catalog, keep)

    @cut("peak_magnitude")
    def filter_by_peak_magnitude(
        self,
        catalog: EventCatalog,
        mission: Mission,
        min_mag: float | None = None,
        max_mag: float | None = None,
        bands: list[str] | None = None,
        n_phase: int | None = None,
        chunk_size: int | None = None,
    ) -> EventCatalog:
        """
        Cut on each event's own peak (brightest) apparent AB magnitude.

        A cheap, purely intrinsic-plus-distance screen on how bright an event ever gets,
        in whichever of `bands` it's brightest in, at its own single brightest sampled
        phase. See `_peak_band_flux`.

        Parameters
        ----------
        catalog : EventCatalog
            Forwarded to `_peak_band_flux`.
        mission : m4opt.missions.Mission
            Forwarded to `_peak_band_flux`.
        min_mag : float, optional
            Keep only events whose peak (numerically lowest, i.e. brightest) AB magnitude
            is at or above this value, excluding events that get implausibly bright.
            `None` (the default) leaves this unconstrained.
        max_mag : float, optional
            Keep only events whose peak AB magnitude is at or below this value, excluding
            events that never get bright enough to matter. `None` (the default) leaves
            this unconstrained.
        bands : list of str, optional
            Forwarded to `_peak_band_flux`.
        n_phase : int, optional
            Forwarded to `_peak_band_flux`.
        chunk_size : int, optional
            Forwarded to `_peak_band_flux`.

        Returns
        -------
        EventCatalog
            A new catalog over the surviving rows only.

        Raises
        ------
        ValueError
            If both `min_mag` and `max_mag` are `None`.

        See Also
        --------
        filter_by_peak_flux : The same cut, compared in flux rather than magnitude.
        filter_by_peak_luminosity : The purely intrinsic (distance-independent) analog.
        filter_by_limiting_magnitude : A schedule-independent cut over a phase grid.

        Notes
        -----
        Unlike `filter_by_limiting_magnitude`, this ignores Milky Way dust attenuation
        and the survey schedule entirely.

        Examples
        --------
        .. code-block:: python

            bright = simulator.filter_by_peak_magnitude(
                catalog, mission, max_mag=24.0
            )
        """
        if min_mag is None and max_mag is None:
            raise ValueError("At least one of 'min_mag'/'max_mag' must be given.")

        flux = self._peak_band_flux(catalog, mission, bands=bands, n_phase=n_phase, chunk_size=chunk_size)
        if len(flux) == 0:
            return self._filtered(catalog, np.zeros(0, dtype=bool))

        with np.errstate(invalid="ignore", divide="ignore"):
            peak_mag = flux.to_value(u.ABmag)

        keep = np.ones(len(peak_mag), dtype=bool)
        if min_mag is not None:
            keep &= peak_mag >= min_mag
        if max_mag is not None:
            keep &= peak_mag <= max_mag

        return self._filtered(catalog, keep)

    @cut("peak_flux")
    def filter_by_peak_flux(
        self,
        catalog: EventCatalog,
        mission: Mission,
        min_flux: float | None = None,
        max_flux: float | None = None,
        bands: list[str] | None = None,
        n_phase: int | None = None,
        chunk_size: int | None = None,
    ) -> EventCatalog:
        """
        Cut on each event's own peak observed flux (erg/s/cm^2/Hz).

        The identical evaluation `filter_by_peak_magnitude` does (see `_peak_band_flux`),
        just compared in flux rather than AB-magnitude units, for a threshold already in
        hand as a flux rather than a magnitude.

        Parameters
        ----------
        catalog : EventCatalog
            Forwarded to `_peak_band_flux`.
        mission : m4opt.missions.Mission
            Forwarded to `_peak_band_flux`.
        min_flux : float, optional
            Keep only events whose peak flux is at or above this value, in erg/s/cm^2/Hz.
            `None` (the default) leaves this unconstrained.
        max_flux : float, optional
            Keep only events whose peak flux is at or below this value, in erg/s/cm^2/Hz.
            `None` (the default) leaves this unconstrained.
        bands : list of str, optional
            Forwarded to `_peak_band_flux`.
        n_phase : int, optional
            Forwarded to `_peak_band_flux`.
        chunk_size : int, optional
            Forwarded to `_peak_band_flux`.

        Returns
        -------
        EventCatalog
            A new catalog over the surviving rows only.

        Raises
        ------
        ValueError
            If both `min_flux` and `max_flux` are `None`.

        See Also
        --------
        filter_by_peak_magnitude : The same cut, compared in AB magnitude rather than flux.

        Examples
        --------
        .. code-block:: python

            bright = simulator.filter_by_peak_flux(
                catalog, mission, min_flux=1e-28
            )
        """
        if min_flux is None and max_flux is None:
            raise ValueError("At least one of 'min_flux'/'max_flux' must be given.")

        flux = self._peak_band_flux(catalog, mission, bands=bands, n_phase=n_phase, chunk_size=chunk_size)
        peak_flux = flux.to_value(u.erg / u.s / u.cm**2 / u.Hz)

        keep = np.ones(len(peak_flux), dtype=bool)
        if min_flux is not None:
            keep &= peak_flux >= min_flux
        if max_flux is not None:
            keep &= peak_flux <= max_flux

        return self._filtered(catalog, keep)

    @cut("peak_luminosity")
    def filter_by_peak_luminosity(
        self,
        catalog: EventCatalog,
        mission: Mission,
        min_luminosity: float | None = None,
        max_luminosity: float | None = None,
        n_phase: int | None = None,
        chunk_size: int | None = None,
    ) -> EventCatalog:
        """
        Cut on each event's own peak bolometric luminosity (erg/s).

        Purely intrinsic: no bands, no distance, no dust, no schedule.
        `~uvex_transients.models.core.base.SpectralModel.eval_bolometric` is evaluated
        over a shared ``[0, transient.duration_limit]`` phase grid (rest-frame time since
        explosion, the same grid `_peak_band_flux` uses as an *observed*-time grid),
        reduced to each event's own maximum.

        Parameters
        ----------
        catalog : EventCatalog
            Typically produced by `generate_events`; must already carry the
            `parameter_seed` column that method fills in.
        mission : m4opt.missions.Mission
            Unused; accepted only so every `@cut` shares one call signature.
        min_luminosity : float, optional
            Keep only events whose peak bolometric luminosity is at or above this value,
            in erg/s. `None` (the default) leaves this unconstrained.
        max_luminosity : float, optional
            Keep only events whose peak bolometric luminosity is at or below this value,
            in erg/s. `None` (the default) leaves this unconstrained.
        n_phase : int, optional
            Number of phase-grid samples per transient type. If `None` (the default),
            uses ``config["simulation.filter_by_limiting_magnitude.n_phase"]``.
        chunk_size : int, optional
            Number of events (per transient type) evaluated per `eval_bolometric` call.
            If `None` (the default), uses
            ``config["simulation.filter_by_limiting_magnitude.chunk_size"]``.

        Returns
        -------
        EventCatalog
            A new catalog over the surviving rows only.

        Raises
        ------
        TypeError
            If `catalog` is not an `EventCatalog`.
        ValueError
            If both `min_luminosity` and `max_luminosity` are `None`, `catalog` is missing
            `parameter_seed`, or contains an unregistered transient type.

        See Also
        --------
        filter_by_peak_magnitude : The distance- and band-dependent analog.
        filter_by_peak_flux : The distance- and band-dependent analog, in flux units.

        Examples
        --------
        .. code-block:: python

            luminous = simulator.filter_by_peak_luminosity(
                catalog, mission, min_luminosity=1e42
            )
        """
        if not isinstance(catalog, EventCatalog):
            raise TypeError(f"'catalog' must be an EventCatalog, got {type(catalog)} instead.")
        if min_luminosity is None and max_luminosity is None:
            raise ValueError("At least one of 'min_luminosity'/'max_luminosity' must be given.")

        if n_phase is None:
            n_phase = config["simulation.filter_by_limiting_magnitude.n_phase"]
        if chunk_size is None:
            chunk_size = config["simulation.filter_by_limiting_magnitude.chunk_size"]

        table = catalog.table
        if "parameter_seed" not in table.colnames:
            raise ValueError("'catalog' is missing column 'parameter_seed'; regenerate it via `generate_events`.")

        if len(table) == 0:
            return self._filtered(catalog, np.zeros(0, dtype=bool))

        transient_type = np.asarray(table["transient_type"]).astype(str)
        type_names = sorted(set(transient_type) & set(self._transients))
        unknown_types = sorted(set(transient_type) - set(self._transients))
        if unknown_types:
            raise ValueError(f"'catalog' contains transient type(s) {unknown_types} not in `transient_collection`.")

        peak_luminosity = np.full(len(table), -np.inf)

        type_idx = {name: np.flatnonzero(transient_type == name) for name in type_names}
        n_chunks_by_type = {name: -(-idx.size // chunk_size) for name, idx in type_idx.items()}
        total_chunks = sum(n_chunks_by_type.values())

        with (
            tqdm(total=total_chunks, desc="Evaluating peak bolometric luminosity", unit="chunk") as pbar,
            logging_redirect_tqdm(loggers=[logger]),
        ):
            for name in type_names:
                transient = self._transients[name]
                idx = type_idx[name]
                n = idx.size
                n_chunks = n_chunks_by_type[name]

                seeds_all = np.asarray(table["parameter_seed"])[idx]
                sed_params_all = _sample_parameters_from_seeds(transient.sed, seeds_all)
                t_grid = np.linspace(0.0, 1.0, n_phase) * transient.duration_limit

                for chunk_num, start in enumerate(range(0, n, chunk_size), start=1):
                    stop = min(start + chunk_size, n)
                    chunk_idx = idx[start:stop]
                    pbar.set_postfix(type=name, chunk=f"{chunk_num}/{n_chunks}")

                    sed_params = {param_name: value[start:stop] for param_name, value in sed_params_all.items()}
                    L_bol = transient.sed.eval_bolometric(t_grid[:, None], **sed_params)  # shape (n_phase, chunk)
                    peak_luminosity[chunk_idx] = np.max(L_bol.to_value(u.erg / u.s), axis=0)
                    pbar.update(1)

        keep = np.ones(len(table), dtype=bool)
        if min_luminosity is not None:
            keep &= peak_luminosity >= min_luminosity
        if max_luminosity is not None:
            keep &= peak_luminosity <= max_luminosity

        return self._filtered(catalog, keep)

    @cut("first_visit_detected")
    def filter_by_first_visit_detected(
        self,
        catalog: EventCatalog,
        mission: Mission,
        snr_threshold: float,
        bands: list[str] | None = None,
        chunk_size: int | None = None,
    ) -> EventCatalog:
        """
        Drop single-epoch detections that fall on their field's very first survey visit.

        A transient with exactly one qualifying detection epoch can only be recognized as
        a transient by differencing against an earlier template image of that field. If
        that one epoch is the field's first-ever visit anywhere in `survey_schedule`, no
        such template exists yet, so the detection isn't actually recoverable. Events
        with zero or two-or-more qualifying epochs are left untouched: a multi-epoch
        event has its own light curve to fall back on even without a template, and a
        zero-epoch event was never a candidate detection in the first place.

        Parameters
        ----------
        catalog : EventCatalog
            Typically an already schedule-aware-filtered (e.g. `filter_by_snr`) catalog.
        mission : m4opt.missions.Mission
            Supplies the `~m4opt.synphot.Detector` `bands` selects from.
        snr_threshold : float
            Same definition as `filter_by_snr`: an epoch qualifies if its measured SNR
            exceeds this value.
        bands : list of str, optional
            Forwarded to `iter_epochs`.
        chunk_size : int, optional
            Forwarded to `iter_epochs`.

        Returns
        -------
        EventCatalog
            A new catalog with every solo-detection, first-visit event removed; every
            other event is kept as-is.

        See Also
        --------
        filter_by_snr : The definition of a qualifying detection this cut builds on.
        uvex_transients.surveys.base.SurveySchedule.first_visit_mask : The schedule lookup this wraps.

        Notes
        -----
        "First visit" is judged against `survey_schedule`'s own earliest observation of
        that field, not just within `catalog`'s sampling window, since a reference image
        would have to predate the survey's first-ever look at that field regardless of
        which events happen to be in this particular catalog.

        Examples
        --------
        .. code-block:: python

            recoverable = (
                simulator.filter_by_first_visit_detected(
                    detected, mission, snr_threshold=5.0
                )
            )
        """
        solo = self._reduce_events(catalog, mission, snr_threshold, bands, chunk_size, _reduce_solo_detection)

        keep = np.ones(len(catalog), dtype=bool)
        self._mask_out_first_visit_solo_detections(catalog, solo, keep)

        return self._filtered(catalog, keep)

    @cut("time_to_first_detection")
    def filter_by_time_to_first_detection(
        self,
        catalog: EventCatalog,
        mission: Mission,
        snr_threshold: float,
        min_delay: float | None = None,
        max_delay: float | None = None,
        bands: list[str] | None = None,
        chunk_size: int | None = None,
    ) -> EventCatalog:
        """
        Cut on the time (days) from explosion to each event's first SNR-qualifying detection.

        Schedule-aware, like `filter_by_snr`: reuses `iter_epochs`'s own
        per-epoch SNR evaluation (see `_reduce_detection_timing`). An event with
        zero qualifying epochs never survives, regardless of `min_delay`/`max_delay`.
        Epochs before the explosion never qualify.

        Parameters
        ----------
        catalog : EventCatalog
            Forwarded to `iter_epochs`.
        mission : m4opt.missions.Mission
            Forwarded to `iter_epochs`.
        snr_threshold : float
            Same definition as `filter_by_snr`.
        min_delay : float, optional
            Minimum time (days) from `t_explosion` to the first qualifying epoch. `None`
            (the default) leaves the lower end unconstrained.
        max_delay : float, optional
            Maximum time (days) from `t_explosion` to the first qualifying epoch. `None`
            (the default) leaves the upper end unconstrained.
        bands : list of str, optional
            Forwarded to `iter_epochs`.
        chunk_size : int, optional
            Forwarded to `iter_epochs`.

        Returns
        -------
        EventCatalog
            A new catalog over the surviving rows only.

        Raises
        ------
        ValueError
            If both `min_delay` and `max_delay` are `None`.

        See Also
        --------
        filter_by_baseline : Cut on the spacing between qualifying detection epochs instead.
        filter_by_snr : The definition of a qualifying detection this cut builds on.

        Examples
        --------
        .. code-block:: python

            early = (
                simulator.filter_by_time_to_first_detection(
                    catalog,
                    mission,
                    snr_threshold=5.0,
                    max_delay=3.0,
                )
            )
        """
        if min_delay is None and max_delay is None:
            raise ValueError("At least one of 'min_delay'/'max_delay' must be given.")

        stats = self._reduce_events(catalog, mission, snr_threshold, bands, chunk_size, _reduce_detection_timing)

        event_id = np.asarray(catalog.table["event_id"])
        keep = np.zeros(len(catalog), dtype=bool)
        for i, eid in enumerate(event_id):
            entry = stats.get(int(eid))
            if entry is None:
                continue
            t_first, _gap, _span = entry
            if min_delay is not None and t_first < min_delay:
                continue
            if max_delay is not None and t_first > max_delay:
                continue
            keep[i] = True

        return self._filtered(catalog, keep)

    @cut("baseline")
    def filter_by_baseline(
        self,
        catalog: EventCatalog,
        mission: Mission,
        snr_threshold: float,
        min_baseline: float | None = None,
        max_baseline: float | None = None,
        bands: list[str] | None = None,
        chunk_size: int | None = None,
    ) -> EventCatalog:
        """
        Cut on the spacing (days) between each event's own SNR-qualifying detection epochs.

        An event survives only if it has at least one pair of qualifying epochs closer
        together than `min_baseline`, and (independently) a full detected time span
        (earliest to latest qualifying epoch) longer than `max_baseline`. For example,
        ``min_baseline=1, max_baseline=10`` requires both some pair of detections under a
        day apart and the detections overall spanning more than ten days.

        Parameters
        ----------
        catalog : EventCatalog
            Forwarded to `iter_epochs`.
        mission : m4opt.missions.Mission
            Forwarded to `iter_epochs`.
        snr_threshold : float
            Same definition as `filter_by_snr`/`filter_by_time_to_first_detection`.
        min_baseline : float, optional
            Require at least one gap between consecutive qualifying epochs shorter than
            this many days. `None` (the default) skips this requirement.
        max_baseline : float, optional
            Require the full first-to-last qualifying-epoch span to exceed this many
            days. `None` (the default) skips this requirement.
        bands : list of str, optional
            Forwarded to `iter_epochs`.
        chunk_size : int, optional
            Forwarded to `iter_epochs`.

        Returns
        -------
        EventCatalog
            A new catalog over the surviving rows only.

        Raises
        ------
        ValueError
            If both `min_baseline` and `max_baseline` are `None`.

        See Also
        --------
        filter_by_time_to_first_detection : Cut on time-to-first-detection instead.
        filter_by_snr : The definition of a qualifying detection this cut builds on.

        Notes
        -----
        An event with fewer than 2 qualifying epochs never satisfies a `min_baseline`
        requirement (there is no gap to compare); one with 0 or 1 never satisfies a
        `max_baseline` requirement (there is no span). See `_reduce_detection_timing`
        for how `min_gap`/`span` are computed.

        Examples
        --------
        .. code-block:: python

            cadence_ok = simulator.filter_by_baseline(
                catalog,
                mission,
                snr_threshold=5.0,
                min_baseline=1.0,
                max_baseline=10.0,
            )
        """
        if min_baseline is None and max_baseline is None:
            raise ValueError("At least one of 'min_baseline'/'max_baseline' must be given.")

        stats = self._reduce_events(catalog, mission, snr_threshold, bands, chunk_size, _reduce_detection_timing)

        event_id = np.asarray(catalog.table["event_id"])
        keep = np.zeros(len(catalog), dtype=bool)
        for i, eid in enumerate(event_id):
            entry = stats.get(int(eid))
            if entry is None:
                continue
            _t_first, gap, span = entry
            if min_baseline is not None and (gap is None or gap >= min_baseline):
                continue
            if max_baseline is not None and span <= max_baseline:
                continue
            keep[i] = True

        return self._filtered(catalog, keep)

    @cut("time_since_last_nondetection")
    def filter_by_time_since_last_nondetection(
        self,
        catalog: EventCatalog,
        mission: Mission,
        snr_threshold: float,
        min_delay: float | None = None,
        max_delay: float | None = None,
        lookback: float | None = None,
        keep_if_no_nondetection: bool = False,
        bands: list[str] | None = None,
        chunk_size: int | None = None,
    ) -> EventCatalog:
        """
        Cut on the time (days) from each event's last non-detection to its first detection.

        This is how tightly the survey brackets the explosion: the first epoch after the
        explosion whose SNR exceeds `snr_threshold` is the first detection, and the latest
        epoch before it whose SNR does not is the last non-detection. Either can be before
        or after the explosion, since a pre-explosion observation of the position is a
        valid non-detection and the lookback below reaches back to it. Schedule-aware,
        like `filter_by_snr`, and judged on the same measured SNR.

        Pre-explosion epochs never count as detections. A pre-explosion epoch that comes
        out above threshold is a noise fluctuation and is ignored altogether. An event
        with no detection after its explosion never survives.

        Parameters
        ----------
        catalog : EventCatalog
            Forwarded to `iter_epochs`.
        mission : m4opt.missions.Mission
            Forwarded to `iter_epochs`.
        snr_threshold : float
            Same definition as `filter_by_snr`.
        min_delay : float, optional
            Minimum time (days) from the last non-detection to the first detection.
            `None` (the default) leaves the lower end unconstrained.
        max_delay : float, optional
            Maximum time (days) from the last non-detection to the first detection, e.g.
            to require the explosion be bracketed to within a few days. `None` (the
            default) leaves the upper end unconstrained.
        lookback : float, optional
            Days before each explosion to search for a non-detection. `None` (the default)
            searches back to the start of the schedule. Smaller values cost less.
        keep_if_no_nondetection : bool, optional
            What to do with an event that has a detection but no earlier non-detection
            within the lookback, so no delay can be measured: its first observation was
            already a detection. `False` (the default) drops it. `True` keeps it,
            ignoring `min_delay`/`max_delay`.
        bands : list of str, optional
            Forwarded to `iter_epochs`.
        chunk_size : int, optional
            Forwarded to `iter_epochs`.

        Returns
        -------
        EventCatalog
            A new catalog over the surviving rows only.

        Raises
        ------
        ValueError
            If both `min_delay` and `max_delay` are `None`.

        See Also
        --------
        filter_by_time_to_first_detection : Cut on the time from explosion instead.
        filter_by_snr : The definition of a qualifying detection this cut builds on.

        Examples
        --------
        .. code-block:: python

            bracketed = simulator.filter_by_time_since_last_nondetection(
                catalog,
                mission,
                snr_threshold=5.0,
                max_delay=3.0,
            )
        """
        if min_delay is None and max_delay is None:
            raise ValueError("At least one of 'min_delay'/'max_delay' must be given.")

        gaps = self._reduce_events(
            catalog,
            mission,
            snr_threshold,
            bands,
            chunk_size,
            _reduce_nondetection_gap,
            keep_all=True,
            lookback=None if lookback is None else lookback * u.day,
        )

        event_id = np.asarray(catalog.table["event_id"])
        keep = np.zeros(len(catalog), dtype=bool)
        for i, eid in enumerate(event_id):
            if int(eid) not in gaps:
                continue
            gap = gaps[int(eid)]
            if np.isnan(gap):
                keep[i] = keep_if_no_nondetection
                continue
            if min_delay is not None and gap < min_delay:
                continue
            if max_delay is not None and gap > max_delay:
                continue
            keep[i] = True

        return self._filtered(catalog, keep)

    @cut("region")
    def filter_by_region(
        self,
        catalog: EventCatalog,
        mission: Mission,
        region: "str | Path | SkyRegion | Regions",
    ) -> EventCatalog:
        """
        Keep only events inside a sky region.

        Uses the same `regions`/`m4opt.fov.contains` convention
        `~uvex_transients.surveys.base.SurveySchedule` uses for its own instrument FOV;
        no WCS is needed.

        Parameters
        ----------
        catalog : EventCatalog
            The catalog to filter.
        mission : m4opt.missions.Mission
            Unused; accepted only so every `@cut` shares one call signature.
        region : str, ~pathlib.Path, ~regions.SkyRegion, or ~regions.Regions
            The region to keep events within. A `str`/`~pathlib.Path` is read via
            `~regions.Regions.read` (any format that library supports, e.g. DS9 ``.reg``);
            a `~regions.SkyRegion`/`~regions.Regions` instance is used directly.

        Returns
        -------
        EventCatalog
            A new catalog over the surviving rows only.

        Raises
        ------
        TypeError
            If `catalog` is not an `EventCatalog`, or `region` is none of the accepted types.

        See Also
        --------
        filter_by_sky_position : A simpler longitude/latitude box test, in any frame.

        Examples
        --------
        .. code-block:: python

            in_field = simulator.filter_by_region(
                catalog, mission, region="field.reg"
            )
        """
        if not isinstance(catalog, EventCatalog):
            raise TypeError(f"'catalog' must be an EventCatalog, got {type(catalog)} instead.")

        if isinstance(region, (str, Path)):
            region = Regions.read(region)
        elif not isinstance(region, (SkyRegion, Regions)):
            raise TypeError(f"'region' must be a str/Path/SkyRegion/Regions, got {type(region)} instead.")

        table = catalog.table
        if len(table) == 0:
            return self._filtered(catalog, np.zeros(0, dtype=bool))

        keep = np.asarray(fov_contains(region, table["coord"]), dtype=bool)
        return self._filtered(catalog, keep)

    @cut("sky_position")
    def filter_by_sky_position(
        self,
        catalog: EventCatalog,
        mission: Mission,
        frame: str,
        phi_min: "float | u.Quantity" = 0.0,
        phi_max: "float | u.Quantity" = 360.0,
        theta_min: "float | u.Quantity" = -90.0,
        theta_max: "float | u.Quantity" = 90.0,
        mode: str = "include",
    ) -> EventCatalog:
        """
        Keep (or drop) events whose position falls in a longitude/latitude box, in any coordinate frame.

        A simple coordinate-box test in whatever frame `frame` names, e.g. Galactic-plane
        avoidance is ``frame="galactic", theta_min=-10, theta_max=10, mode="exclude"``,
        with `phi_min`/`phi_max` left at their full-sky defaults.

        Parameters
        ----------
        catalog : EventCatalog
            The catalog to filter.
        mission : m4opt.missions.Mission
            Unused; accepted only so every `@cut` shares one call signature.
        frame : str
            Any frame name `~astropy.coordinates.SkyCoord.transform_to` accepts (e.g.
            ``"galactic"``, ``"geocentricmeanecliptic"``, ``"icrs"``).
        phi_min, phi_max : float or ~astropy.units.Quantity, optional
            Longitude bounds in `frame` (bare floats are degrees). If `phi_min > phi_max`,
            the bound wraps through 0 (e.g. ``phi_min=350, phi_max=10`` selects longitudes
            near 0). Default: the full ``[0, 360]`` range (no longitude constraint).
        theta_min, theta_max : float or ~astropy.units.Quantity, optional
            Latitude bounds in `frame` (bare floats are degrees; doesn't wrap). Default:
            the full ``[-90, 90]`` range (no latitude constraint).
        mode : {"include", "exclude"}, optional
            ``"include"`` (the default) keeps events inside the box; ``"exclude"`` keeps
            events outside it.

        Returns
        -------
        EventCatalog
            A new catalog over the surviving rows only.

        Raises
        ------
        TypeError
            If `catalog` is not an `EventCatalog`.
        ValueError
            If `mode` isn't ``"include"``/``"exclude"``.

        See Also
        --------
        filter_by_region : Filter against an arbitrary polygon/circle region instead.

        Examples
        --------
        .. code-block:: python

            plane_avoided = simulator.filter_by_sky_position(
                catalog,
                mission,
                frame="galactic",
                theta_min=-10,
                theta_max=10,
                mode="exclude",
            )
        """
        if not isinstance(catalog, EventCatalog):
            raise TypeError(f"'catalog' must be an EventCatalog, got {type(catalog)} instead.")
        if mode not in ("include", "exclude"):
            raise ValueError(f"'mode' must be 'include' or 'exclude', got {mode!r}.")

        table = catalog.table
        if len(table) == 0:
            return self._filtered(catalog, np.zeros(0, dtype=bool))

        phi_min_deg = u.Quantity(phi_min, u.deg).to_value(u.deg)
        phi_max_deg = u.Quantity(phi_max, u.deg).to_value(u.deg)
        theta_min = u.Quantity(theta_min, u.deg).to_value(u.deg)
        theta_max = u.Quantity(theta_max, u.deg).to_value(u.deg)

        transformed = table["coord"].transform_to(frame)
        phi = transformed.spherical.lon.to_value(u.deg)
        theta = transformed.spherical.lat.to_value(u.deg)

        # A >=360 deg-wide span (e.g. the 0/360 full-circle default) is every longitude,
        # regardless of where it starts. Taken care of here, before `% 360.0` below
        # would otherwise collapse a `phi_max=360.0` boundary down to `0.0` and turn "no
        # constraint" into "only exactly phi == 0".
        if phi_max_deg - phi_min_deg >= 360.0:
            in_phi = np.ones(len(table), dtype=bool)
        else:
            phi_min_norm = phi_min_deg % 360.0
            phi_max_norm = phi_max_deg % 360.0
            in_phi = (
                (phi >= phi_min_norm) & (phi <= phi_max_norm)
                if phi_min_norm <= phi_max_norm
                else (phi >= phi_min_norm) | (phi <= phi_max_norm)
            )
        in_theta = (theta >= theta_min) & (theta <= theta_max)
        inside = in_phi & in_theta

        return self._filtered(catalog, inside if mode == "include" else ~inside)

    @cut("query")
    def filter_by_query(self, catalog: EventCatalog, mission: Mission, expr: str) -> EventCatalog:
        """
        Cut an `EventCatalog` by an arbitrary boolean expression over its own columns.

        Evaluated with each of `catalog.table`'s columns bound to its own native type
        (`~astropy.units.Quantity`, `~astropy.coordinates.SkyCoord`, `~astropy.time.Time`,
        or a plain array), so comparisons stay unit- and frame-aware. Combine conditions
        with ``&``/``|``/``~`` (elementwise), not Python's ``and``/``or``/``not``.

        Parameters
        ----------
        catalog : EventCatalog
            The catalog to filter.
        mission : m4opt.missions.Mission
            Unused; accepted only so every `@cut` shares one call signature.
        expr : str
            A boolean expression over `catalog.table`'s column names.

        Returns
        -------
        EventCatalog
            A new catalog over the surviving rows only.

        Raises
        ------
        TypeError
            If `catalog` is not an `EventCatalog`.
        ValueError
            If `expr` fails to evaluate, or doesn't evaluate to a boolean array of the
            catalog's own length.

        See Also
        --------
        run_cut : Dispatch any registered cut, including this one, by name.

        Notes
        -----
        `expr` is evaluated with `eval`, scoped to just `catalog.table`'s own columns plus
        `u` (`astropy.units`), `np` (`numpy`), `SkyCoord`, and `Time`, with an empty
        ``__builtins__`` so a typo can't accidentally reach unrelated Python builtins.
        This is a guard against accidents, not a security sandbox: `expr` should come from
        a config you wrote yourself, the same trust level as this package's own
        ``!prior``/``!astropy_cosmology`` YAML tags, not from an untrusted source.

        Examples
        --------
        .. code-block:: python

            nearby_tdes = simulator.filter_by_query(
                catalog,
                mission,
                expr="(redshift < 0.1) & (transient_type == 'tde')",
            )
        """
        if not isinstance(catalog, EventCatalog):
            raise TypeError(f"'catalog' must be an EventCatalog, got {type(catalog)} instead.")

        table = catalog.table
        namespace = {name: table[name] for name in table.colnames}
        namespace.update(u=u, np=np, SkyCoord=SkyCoord, Time=Time)

        try:
            mask = eval(expr, {"__builtins__": {}}, namespace)
        except Exception as error:
            raise ValueError(f"Failed to evaluate query expression {expr!r}: {error}") from error

        mask = np.asarray(mask)
        if mask.shape != (len(table),) or mask.dtype != bool:
            raise ValueError(
                f"Query expression {expr!r} must evaluate to a boolean array of shape ({len(table)},); "
                f"got {mask.dtype} array of shape {mask.shape}."
            )

        return self._filtered(catalog, mask)

    # -------------------------------------------------- #
    # Actions                                             #
    # -------------------------------------------------- #
    @classmethod
    def available_actions(cls) -> tuple[str, ...]:
        """Tuple of str: Every action name registered on this class via `@action`, sorted."""
        return tuple(sorted(cls._ACTION_REGISTRY))

    def run_action(self, name: str, mission: Mission, **inputs_and_params) -> object:
        """
        Run one `@action`-registered pipeline action by name.

        A thin dispatch layer over `run_photometry_action`/`run_yield_action`/
        `run_detection_counts_action` (and any further ``@action``-decorated methods a
        subclass adds), mirroring `run_cut`'s relationship to `available_cuts`, so a
        config-driven caller (see `uvex_transients.cli.steps`) can select an action by
        name rather than hardcoding which Python method to call.

        Parameters
        ----------
        name : str
            One of `available_actions`.
        mission : m4opt.missions.Mission
            The mission whose detector(s)/bandpasses the action evaluates against.
        **inputs_and_params
            Forwarded to the underlying action method as keyword arguments: both its
            named artifact inputs (e.g. `catalog` for ``"photometry"``; `raw`,
            `detected`, `exposure` for ``"yield"``) and its own parameters (e.g.
            `bands`, `n_sigma`).

        Returns
        -------
        object
            Whatever the underlying action method returns (an `EventCatalog`-adjacent
            artifact, a `~astropy.table.QTable`, ...); the type is declared by the
            action itself, not by this dispatcher.

        Raises
        ------
        ValueError
            If `name` isn't registered.
        TypeError
            If `inputs_and_params` doesn't match the underlying action method's own
            signature (an unknown or missing keyword).
        """
        try:
            method_name = self._ACTION_REGISTRY[name]
        except KeyError:
            raise ValueError(f"Unknown action {name!r}; available: {self.available_actions()}.") from None
        return getattr(self, method_name)(mission=mission, **inputs_and_params)

    @action("photometry")
    def run_photometry_action(
        self,
        catalog: EventCatalog,
        mission: Mission,
        bands: list[str] | None = None,
        n_sigma: float | None = None,
    ) -> QTable:
        """
        Run synthetic photometry over every event in `catalog`.

        A thin wrapper over `~uvex_transients.simulation.event_catalog.EventCatalog.simulate_photometry`,
        supplying this simulator's own `transient_collection`/`survey_schedule` so a
        config-driven step never needs to pass them explicitly.

        Parameters
        ----------
        catalog : EventCatalog
            The catalog of events to simulate photometry for.
        mission : m4opt.missions.Mission
            Supplies the `~m4opt.synphot.Detector` `bands` selects from.
        bands : list of str, optional
            Which of `mission.detector`'s bandpasses to evaluate. Defaults to every band.
        n_sigma : float, optional
            Forwarded to `~uvex_transients.simulation.event.Event.simulate_photometry`.

        Returns
        -------
        ~astropy.table.QTable
            One row per (event, observation, band) synthetic observation.

        See Also
        --------
        run_yield_action : Combine this action's output with raw/detected catalogs into a yield summary.
        run_detection_counts_action : Reduce this action's output to per-type detection counts.

        Examples
        --------
        .. code-block:: python

            photometry = simulator.run_photometry_action(
                catalog=detected, mission=mission
            )
        """
        return catalog.simulate_photometry(
            mission, self.transient_collection, self.survey_schedule, bands=bands, n_sigma=n_sigma
        )

    @action("yield")
    def run_yield_action(
        self,
        raw: EventCatalog,
        detected: EventCatalog,
        exposure: ExposureCatalog,
        mission: Mission,
        confidence: float = 0.9,
    ) -> YieldTable:
        r"""
        Build a per-transient-type yield summary from a raw, a detected, and an exposure catalog.

        A thin wrapper over
        `~uvex_transients.simulation.event_catalog.EventCatalog.compute_yield_summary`,
        supplying this simulator's own `transient_collection`. Unlike the other actions,
        `mission` is accepted (for a uniform `run_action` call signature) but unused,
        since yield is purely a combination of already-computed catalogs/rates.

        Parameters
        ----------
        raw : EventCatalog
            The feasible (pre-cut) catalog.
        detected : EventCatalog
            The detected (post-cut) catalog.
        exposure : ExposureCatalog
            Typically `compute_effective_exposure`'s own output.
        mission : m4opt.missions.Mission
            Unused; accepted only so every `@action` shares one call signature.
        confidence : float, optional
            Confidence level for the Clopper-Pearson binomial bounds. The default is ``0.9``.

        Returns
        -------
        YieldTable
            One row per transient type in `self.transient_collection`.

        See Also
        --------
        compute_effective_exposure : Typically the source of `exposure`.
        run_photometry_action : Typically upstream of `detected`, once further cut.

        Examples
        --------
        .. code-block:: python

            yields = simulator.run_yield_action(
                raw=raw_catalog,
                detected=detected_catalog,
                exposure=exposure,
                mission=mission,
            )
        """
        return raw.compute_yield_summary(detected, exposure, self.transient_collection, confidence=confidence)

    @action("detection_counts")
    def run_detection_counts_action(
        self,
        catalog: EventCatalog,
        exposure: ExposureCatalog,
        photometry: PhotometryCatalog | QTable,
        mission: Mission,
        snr_threshold: float,
        confidence: float = 0.9,
    ) -> QTable:
        """
        Estimate, per transient type, how many events show N_det >= k detected epochs.

        A thin wrapper over
        `~uvex_transients.simulation.photometry_catalog.PhotometryCatalog.compute_detection_count_table`,
        supplying this simulator's own `transient_collection`.

        Parameters
        ----------
        catalog : EventCatalog
            The full per-type event list `photometry` was computed over.
        exposure : ExposureCatalog
            Typically `compute_effective_exposure`'s own output.
        photometry : PhotometryCatalog or ~astropy.table.QTable
            Typically `run_photometry_action`'s own output; a bare `QTable` is wrapped
            automatically.
        mission : m4opt.missions.Mission
            Unused; accepted only so every `@action` shares one call signature.
        snr_threshold : float
            An observation epoch counts as detected if at least one band's measured ``snr``
            in `photometry` exceeds this value.
        confidence : float, optional
            Confidence level for the Clopper-Pearson binomial bounds. The default is ``0.9``.

        Returns
        -------
        ~astropy.table.QTable
            One row per ``(transient_type, n_detections)`` pair.

        See Also
        --------
        run_photometry_action : Typically the source of `photometry`.

        Examples
        --------
        .. code-block:: python

            counts = simulator.run_detection_counts_action(
                catalog=detected,
                exposure=exposure,
                photometry=photometry,
                mission=mission,
                snr_threshold=5.0,
            )
        """
        if not isinstance(photometry, PhotometryCatalog):
            photometry = PhotometryCatalog(table=photometry)
        return photometry.compute_detection_count_table(
            catalog, exposure, self.transient_collection, snr_threshold=snr_threshold, confidence=confidence
        )

    @action("alert")
    def run_alert_action(
        self,
        catalog: EventCatalog,
        mission: Mission,
        snr_threshold: float,
        bands: list[str] | None = None,
        chunk_size: int | None = None,
        processing_delay: u.Quantity = 0 * u.s,
    ) -> QTable:
        r"""
        Build a per-event alert-timing table: first detection, next downlink, and alert time.

        A transient with an explosion time :math:`t_0` may not have the opportunity to relay that information to
        the ground until some :math:`t_{\rm transmission} > t_0` determined by the next downlink time. For fast
        follow up of UVEX detected transients, this delay time may have very important implications for getting on
        target with ground and space-based observatories. This function therefore provides an "alert" with the
        corresponding trigger times to allow effective modeling of this.

        Parameters
        ----------
        catalog : EventCatalog
            Typically an already schedule-aware-filtered (e.g. `filter_by_snr`) catalog;
            events with no qualifying epoch are dropped from the output (see Notes).
        mission : m4opt.missions.Mission
            Supplies the `~m4opt.synphot.Detector` `bands` selects from.
        snr_threshold : float
            Same definition as `filter_by_snr`: an epoch qualifies if its measured SNR
            exceeds this value.
        bands : list of str, optional
            Forwarded to `iter_epochs`.
        chunk_size : int, optional
            Forwarded to `iter_epochs`.
        processing_delay : ~astropy.units.Quantity, optional
            Extra fixed ground-segment latency added on top of the downlink's own
            completion time, e.g. to model processing/distribution time before a real
            alert would actually be issued. The default is ``0 * u.s``: `alert_time` is
            exactly the downlink completion time.

        Returns
        -------
        ~astropy.table.QTable
            One row per event in `catalog` with >= 1 qualifying detection epoch, sorted by
            ``event_id``, with columns:

            - ``event_id``, ``transient_type``
            - ``t_first_detection``: the qualifying epoch itself (absolute time).
            - ``detection_band``, ``detection_snr``: that epoch's best band and its
              measured SNR.
            - ``t_downlink``: the relevant downlink action's *completion* time
              (``start_time + duration``).
            - ``alert_time``: ``t_downlink + processing_delay``.
            - ``alert_delay``: ``alert_time - t_first_detection``, as a
              `~astropy.units.Quantity` in hours.

            ``t_downlink``/``alert_time``/``alert_delay`` are masked for an event whose
            first qualifying detection has no downlink scheduled after it (e.g. it falls
            in the survey's final observing block, after the last downlink).

        See Also
        --------
        filter_by_snr : The definition of a qualifying detection this action builds on.
        uvex_transients.surveys.base.SurveySchedule.next_action_time : The schedule lookup this wraps.

        Notes
        -----
        Events in `catalog` with zero qualifying epochs (e.g. it was cut with a lower
        `snr_threshold` than this call uses) are excluded from the output entirely,
        rather than appearing with masked detection columns too -- there is no detection
        epoch to look up a downlink relative to.

        Examples
        --------
        .. code-block:: python

            alerts = simulator.run_alert_action(
                detected,
                mission,
                snr_threshold=5.0,
                processing_delay=1 * u.hour,
            )
        """
        if not isinstance(catalog, EventCatalog):
            raise TypeError(f"'catalog' must be an EventCatalog, got {type(catalog)} instead.")
        if not isinstance(processing_delay, u.Quantity):
            raise TypeError(f"'processing_delay' must be a Quantity, got {type(processing_delay)}.")

        stats = self._reduce_events(catalog, mission, snr_threshold, bands, chunk_size, _reduce_first_detection)

        event_ids = np.array(sorted(stats), dtype=np.int64)
        if len(event_ids) == 0:
            return QTable(
                {
                    "event_id": event_ids,
                    "transient_type": np.array([], dtype=str),
                    "t_first_detection": Time([], format="jd", scale=self.survey_schedule.table["start_time"].scale),
                    "detection_band": np.array([], dtype=str),
                    "detection_snr": np.array([], dtype=np.float64),
                    "t_downlink": Time([], format="jd", scale=self.survey_schedule.table["start_time"].scale),
                    "alert_time": Time([], format="jd", scale=self.survey_schedule.table["start_time"].scale),
                    "alert_delay": u.Quantity([], u.hour),
                }
            )

        t_obs = Time([stats[eid][0] for eid in event_ids])
        band = np.array([stats[eid][1] for eid in event_ids])
        snr = np.array([stats[eid][2] for eid in event_ids])
        observation_index = np.array([stats[eid][3] for eid in event_ids])

        observe_rows = self.survey_schedule.observe_rows
        t_detection_end = t_obs + observe_rows["duration"][observation_index]

        t_downlink = self.survey_schedule.next_action_time(t_detection_end, "downlink", completion=True)
        alert_time = t_downlink + processing_delay
        alert_delay = (alert_time - t_obs).to(u.hour)

        table_event_id = np.asarray(catalog.table["event_id"])
        transient_type = np.asarray(catalog.table["transient_type"]).astype(str)
        order = np.argsort(table_event_id)
        type_of_event = transient_type[order[np.searchsorted(table_event_id[order], event_ids)]]

        return QTable(
            {
                "event_id": event_ids,
                "transient_type": type_of_event,
                "t_first_detection": t_obs,
                "detection_band": band,
                "detection_snr": snr,
                "t_downlink": t_downlink,
                "alert_time": alert_time,
                "alert_delay": alert_delay,
            }
        )
