r"""
Evaluate the survey's recovery of a canonical transient at one point of a luminosity-half-decline-time map.

`RecoveryEvaluator` is the function `~uvex_transients.phase_space.sampler.PhaseSpaceSampler` calls. At a point
:math:`(\log_{10}L_\mathrm{band}, \log_{10}t_{1/2})` it finds the redshift out to which the event is brighter than a
chosen AB depth (`RedshiftLimit`), draws events volume-weighted out to it on the footprint the survey covers, and counts
how many `SurveySimulator.filter_by_snr` recovers, adding chunks of events until enough are recovered or too many have
been simulated.

It is a thin layer over two parts that do not know about each other:
`~uvex_transients.phase_space.sampling.core.CanonicalSampler`, which answers how many events of `n` at a point are
recovered, and `~uvex_transients.phase_space.sampling.base.StoppingRule`, which decides how many to try. It keeps
the interface the map and the command line were written against.

The recovery fraction :math:`f=k/n` is conditional on the footprint that was sampled, which depends on the number of
explosion-time bins. The noise-free volume-time :math:`\mathcal V T` is the expected number of events over the survey
in that footprint at one event per :math:`\mathrm{Mpc}^3\,\mathrm{yr}`. The product :math:`f\,\mathcal V T` does not
depend on the binning, and is the expected number of recoveries at unit rate, :math:`R\,f\,\mathcal V T` at a rate
:math:`R`.
"""

import numpy as np
from astropy.cosmology import Cosmology
from numpy.typing import ArrayLike, NDArray

from uvex_transients.phase_space.sampling.base import StoppingRule
from uvex_transients.phase_space.sampling.core import CanonicalSampler, RedshiftLimit
from uvex_transients.surveys.base import SurveySchedule
from uvex_transients.utils import get_seed_sequence, logger

__all__ = ["RecoveryEvaluator", "RedshiftLimit"]


# ============================================ #
# Recovery Evaluator                           #
# ============================================ #
class RecoveryEvaluator:
    r"""
    Count the recoveries of a canonical transient at points of the luminosity-half-decline-time plane.

    The transient has a fast-rise, exponential-decay light curve and a power-law spectrum (see
    `~uvex_transients.phase_space.transient._CanonicalTransient`). Instances are called as ``evaluator(x)`` with
    ``x = (log10 L_band / (erg/s), log10 t_half / day)``, the signature
    `~uvex_transients.phase_space.sampler.PhaseSpaceSampler` expects. :math:`L_\mathrm{band}` is the peak rest-frame
    luminosity emitted in the band, :math:`L_\nu\Delta\nu` for the equivalent bandwidth :math:`\Delta\nu` of the band
    with :math:`L_\nu` taken at its effective frequency. One evaluator makes the map of one band.

    Every argument that is `None` (the default) takes ``config["phase_space.recovery.<name>"]``, or
    ``config["phase_space.transient.<name>"]`` for the shape of the transient.

    Parameters
    ----------
    schedule : ~uvex_transients.surveys.base.SurveySchedule
        The survey to recover events against.
    mission : str or m4opt.missions.Mission, optional
        The mission (detector and background), or the name of one.
    band : str, optional
        The detector band that recovers events and in which the luminosity is measured.
    spectral_index : float, optional
        The index :math:`\alpha` of the spectrum, :math:`L_\nu\propto\nu^\alpha`, which sets the K-correction
        :math:`(1+z)^{1+\alpha}`. A spectrum rising steeply to the blue (about 3) has no redshift limit and raises
        `ValueError`.
    rise_ratio : float, optional
        The ratio of the rise to the decay time of the light curve.
    fade_mag : float, optional
        How far below its peak, in magnitudes, an event is followed by the survey.
    snr_threshold : float, optional
        The signal-to-noise a visit must reach.
    n_visits : int, optional
        How many visits must reach it.
    exclude_first_visit_detections : bool, optional
        Whether a detection that is only on a field's first visit is discarded; see `SurveySimulator.filter_by_snr`.
    mag_limit : float, optional
        The AB depth that sets the redshift limit of a point. Only events brighter than it are drawn.
    time_bins : int, optional
        Number of equal explosion-time bins spanning the survey; see `SurveySimulator.sample_N_events`.
    nside, order : optional
        HEALPix resolution of the sampling grid; the package defaults if omitted.
    chunk_size : int, optional
        Events simulated per chunk, and so the fewest at a point.
    k_target : int, optional
        Stop adding chunks to a point once this many events are recovered.
    n_max : int, optional
        Never simulate more than this many events at a point.
    seed : int or numpy.random.SeedSequence, optional
        Root seed. Every chunk draws a fresh child, so chunks and points are independent.
    cosmology : ~astropy.cosmology.Cosmology, optional
        Defaults to the configured cosmology.
    schedule_name : str, optional
        The registry name of `schedule` (see `~uvex_transients.surveys.utils.get_schedule`), recorded in `settings` so
        that `from_settings` can fetch the same schedule. Leave it out for a schedule that is not in the registry.

    Notes
    -----
    The tables the evaluator interpolates (redshift limit, volume, exposure) are sized by
    ``config["phase_space.numerics"]``. They set the precision of the scales, not what is measured, so they are not
    recorded in `settings`.
    """

    def __init__(
        self,
        schedule: SurveySchedule,
        mission=None,
        *,
        band: str | None = None,
        spectral_index: float | None = None,
        rise_ratio: float | None = None,
        fade_mag: float | None = None,
        snr_threshold: float | None = None,
        n_visits: int | None = None,
        exclude_first_visit_detections: bool | None = None,
        mag_limit: float | None = None,
        time_bins: int | None = None,
        nside: int | None = None,
        order: str | None = None,
        chunk_size: int | None = None,
        k_target: int | None = None,
        n_max: int | None = None,
        seed: int | np.random.SeedSequence | None = None,
        cosmology: Cosmology | None = None,
        schedule_name: str | None = None,
    ):
        self._rule = StoppingRule.from_config(chunk_size=chunk_size, k_target=k_target, n_max=n_max)
        self._sampler = CanonicalSampler(
            schedule,
            mission,
            band=band,
            spectral_index=spectral_index,
            rise_ratio=rise_ratio,
            fade_mag=fade_mag,
            snr_threshold=snr_threshold,
            n_visits=n_visits,
            exclude_first_visit_detections=exclude_first_visit_detections,
            mag_limit=mag_limit,
            time_bins=time_bins,
            nside=nside,
            order=order,
            cosmology=cosmology,
            schedule_name=schedule_name,
        )
        self._seed = get_seed_sequence(seed)
        self._seed_setting = int(seed) if isinstance(seed, (int, np.integer)) else None

    def __getattr__(self, name: str):
        # Everything about the physics (the band, the mission, the redshift edge, the tables) is the sampler's
        if name in ("_sampler", "_rule"):
            raise AttributeError(name)
        return getattr(self._sampler, name)

    # -------------------------------------- #
    # What the rule decides                  #
    # -------------------------------------- #
    @property
    def chunk_size(self) -> int:
        """int: Events simulated per chunk, and so the fewest at a point."""
        return self._rule.chunk_size

    @property
    def k_target(self) -> int:
        """int: A point is topped up until this many events are recovered."""
        return self._rule.k_target

    @property
    def n_max(self) -> int:
        """int: The most events simulated at a point."""
        return self._rule.n_max

    @property
    def redshift_limit(self) -> RedshiftLimit:
        """RedshiftLimit: The redshift out to which a source of a luminosity is brighter than the depth."""
        return self._sampler._limit

    # -------------------------------------- #
    # Recording and rebuilding               #
    # -------------------------------------- #
    @property
    def settings(self) -> dict:
        """
        dict: What `from_settings` needs to rebuild this evaluator, as plain types that an ECSV header can hold.

        The mission and schedule are recorded by name. A mission given as an object, or a schedule given without a
        ``schedule_name``, is recorded as `None` and has to be supplied again to `from_settings`.
        """
        settings = {}
        for key, value in self._sampler.settings.items():
            # The rule and the seed sit between the sampling grid and the cosmology, where they always have
            if key == "cosmology":
                settings.update(
                    chunk_size=self._rule.chunk_size,
                    k_target=self._rule.k_target,
                    n_max=self._rule.n_max,
                    seed=self._seed_setting,
                )
            settings[key] = value
        return settings

    @property
    def table_meta(self) -> dict:
        r"""
        dict: What a figure of this evaluator's maps needs to label the luminosity axis.

        ``band``, the unit of the luminosity, and ``absolute_magnitude_zero_point``, the :math:`Z` for which the
        absolute AB magnitude of a source of luminosity :math:`L_\mathrm{band}` is
        :math:`M = Z - 2.5\log_{10}L_\mathrm{band}`. Also the shape of the canonical transient the axes are measured
        on (``spectral_index``, ``rise_ratio``, ``fade_mag``), which
        `~uvex_transients.phase_space.compare.difference_map` checks before it compares two maps. And ``n_max``, the
        most events a point simulates, which sets the smallest recovery fraction a map resolves.
        """
        meta = {}
        for key, value in self._sampler.table_meta.items():
            meta[key] = value
            if key == "fade_mag":
                meta["n_max"] = self._rule.n_max
        return meta

    @classmethod
    def from_settings(
        cls,
        settings: dict,
        *,
        schedule: SurveySchedule | None = None,
        mission=None,
        cosmology: Cosmology | None = None,
        **overrides,
    ) -> "RecoveryEvaluator":
        """
        Rebuild an evaluator from the `settings` of another one.

        Parameters
        ----------
        settings : dict
            The `settings` of the evaluator to rebuild.
        schedule : ~uvex_transients.surveys.base.SurveySchedule, optional
            The schedule. Needed unless `settings` names one in the registry.
        mission : str or m4opt.missions.Mission, optional
            The mission. Needed unless `settings` names one.
        cosmology : ~astropy.cosmology.Cosmology, optional
            By default the configured one, with a warning if its name is not the recorded one.
        **overrides
            Any other constructor argument, replacing the recorded value.

        Returns
        -------
        RecoveryEvaluator
            An evaluator that makes the same measurement, and so reproduces the scales of the recorded one.

        Raises
        ------
        ValueError
            If no schedule or no mission can be found, or the settings are of a map made with the earlier top-hat
            light curve.
        """
        if "rise_ratio" not in settings:
            raise ValueError(
                "These settings are of a map made with a top-hat light curve, which can no longer be rebuilt; "
                "make the map again."
            )
        settings, schedule, mission, schedule_name = CanonicalSampler._rebuild_arguments(
            settings, schedule, mission, cosmology
        )
        settings.update(overrides)
        return cls(schedule, mission, cosmology=cosmology, schedule_name=schedule_name, **settings)

    # -------------------------------------- #
    # Absolute scale                         #
    # -------------------------------------- #
    @staticmethod
    def _points(log_luminosity: ArrayLike, log_half_time: ArrayLike) -> NDArray[np.float64]:
        """Return the coordinates, broadcast against each other, as points with the coordinate on the last axis."""
        log_luminosity, log_half_time = np.broadcast_arrays(
            np.asarray(log_luminosity, dtype=np.float64), np.asarray(log_half_time, dtype=np.float64)
        )
        return np.stack([log_luminosity, log_half_time], axis=-1)

    def volume_time(self, log_luminosity: ArrayLike, log_half_time: ArrayLike) -> NDArray[np.float64]:
        r"""
        Return the effective volume-time :math:`\mathcal V T` of points of the map, in Mpc^3 yr.

        The expected number of events over the survey, inside the footprint and out to the redshift limit, at one
        event per Mpc^3 per yr. It involves no Monte Carlo: the exposure comes from the schedule. Multiply by a rate
        and by the recovery fraction for the expected number of recoveries.

        Parameters
        ----------
        log_luminosity, log_half_time : array-like
            :math:`\log_{10}L_\mathrm{band}` in erg/s and :math:`\log_{10}t_{1/2}` in days, of equal shape.

        Returns
        -------
        numpy.ndarray
            :math:`\mathcal V T`, of the same shape.
        """
        return self._sampler.volume_time(self._points(log_luminosity, log_half_time))

    def coverage(self, log_luminosity: ArrayLike, log_half_time: ArrayLike) -> NDArray[np.float64]:
        r"""
        Return the fraction of the sky and survey window that the sampled footprint covers, for points of the map.

        The effective exposure over that of the whole sky for the whole survey, :math:`\mathcal E/(4\pi\,T)`. It is the
        ratio of the events in the sampled footprint to all events exploding during the survey, so multiplying the
        recovery fraction :math:`f` by it gives `efficiency`.

        Parameters
        ----------
        log_luminosity, log_half_time : array-like
            The coordinates, as in `volume_time`.

        Returns
        -------
        numpy.ndarray
            The coverage, between 0 and 1, of the same shape.
        """
        return self._sampler.coverage(self._points(log_luminosity, log_half_time))

    def all_sky_volume_time(self, log_luminosity: ArrayLike) -> NDArray[np.float64]:
        r"""
        Return the volume-time of the whole sky over the whole survey out to the redshift limit, in Mpc^3 yr.

        The number of events exploding during the survey anywhere on the sky, at one per Mpc^3 per yr. It is
        `volume_time` divided by `coverage`.

        Parameters
        ----------
        log_luminosity : array-like
            :math:`\log_{10}L_\mathrm{band}` in erg/s.

        Returns
        -------
        numpy.ndarray
            The all-sky volume-time, of the same shape.
        """
        return self._sampler.all_sky_volume_time(self._points(log_luminosity, 0.0))

    def efficiency(self, recovery: ArrayLike, log_luminosity: ArrayLike, log_half_time: ArrayLike) -> NDArray:
        r"""
        Convert a recovery fraction in the sampled footprint to one over the whole sky and survey window.

        :math:`\eta = f\,\mathcal E/(4\pi\,T)`: the probability that an event exploding at a random position and time
        during the survey, with a redshift inside the limit, is recovered. Unlike :math:`f` it does not depend on the
        number of time bins, but it does depend on the redshift limit, so it changes with ``mag_limit`` where
        :math:`f\,\mathcal V T` does not.

        Parameters
        ----------
        recovery : array-like
            The recovery fraction :math:`f` in the sampled footprint.
        log_luminosity, log_half_time : array-like
            The coordinates, as in `volume_time`, of the shape of `recovery`.

        Returns
        -------
        numpy.ndarray
            :math:`\eta`, of the same shape.
        """
        return np.asarray(recovery, dtype=np.float64) * self.coverage(log_luminosity, log_half_time)

    # -------------------------------------- #
    # One point                              #
    # -------------------------------------- #
    def __call__(self, x: ArrayLike) -> dict:
        """
        Count the recoveries at one point of the map.

        Parameters
        ----------
        x : array-like
            ``(log10 L_band / (erg/s), log10 t_half / day)``.

        Returns
        -------
        dict
            ``k`` (recoveries), ``n`` (events simulated), ``z_limit``, ``volume_time`` (in Mpc^3 yr), ``coverage``
            (see `coverage`) and ``n_chunks``.
        """
        point = np.array([float(value) for value in x])
        outcome = self._rule.run(self._sampler, point, self._seed)

        # The noise-free scales of the point
        z_limit = float(self._sampler.redshift_limit(point))
        volume_time = float(self._sampler.volume_time(point))
        coverage = float(self._sampler.coverage(point))
        logger.debug(
            "log10 L_band=%.2f, log10 t_half=%.2f: z_limit=%.3g, recovered %d of %d, VT=%.3g Mpc^3 yr.",
            point[0],
            point[1],
            z_limit,
            outcome.k,
            outcome.n,
            volume_time,
        )
        return {
            "k": outcome.k,
            "n": outcome.n,
            "z_limit": z_limit,
            "volume_time": volume_time,
            "coverage": coverage,
            "n_chunks": outcome.extras["n_chunks"],
        }
