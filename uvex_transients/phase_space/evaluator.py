r"""
Evaluate the survey's recovery of a flat-spectrum top-hat transient at one point of a luminosity-duration map.

`RecoveryEvaluator` is the expensive function `~uvex_transients.phase_space.sampler.PhaseSpaceSampler` calls. At a
point :math:`(\log_{10}L_\mathrm{band}, \log_{10}\Delta t)`, with :math:`L_\mathrm{band}` the luminosity in the band
in erg/s, it

1. finds the redshift :math:`z_\mathrm{lim}` out to which the event is brighter than a chosen AB depth
   (`RedshiftLimit`);
2. draws events volume-weighted in redshift out to :math:`z_\mathrm{lim}`, on the footprint the survey actually
   covers, with the same `SurveySimulator.sample_N_events` the population simulations use;
3. counts how many `SurveySimulator.filter_by_snr` recovers, in chunks, until at least `k_target` events are
   recovered or `n_max` have been simulated.

The recovery fraction :math:`f=k/n` is conditional on the event falling in the footprint that was sampled, and
that footprint depends on the number of explosion-time bins: fewer bins pad it with fields that are observed only
long after the explosion, which lowers :math:`f` for short transients. The absolute scale is a separate, noise-free
factor, `RecoveryEvaluator.volume_time`: the expected number of events over the survey, in that same footprint, at a
volumetric rate of one per :math:`\mathrm{Mpc}^3\,\mathrm{yr}`, :math:`\mathcal V T`, in :math:`\mathrm{Mpc}^3\,
\mathrm{yr}`. The product :math:`f\,\mathcal V T` does not depend on the binning, and is the expected number of
recoveries at unit rate; at any other rate :math:`R` it is :math:`R\,f\,\mathcal V T`.
"""

import numpy as np
from astropy import units as u
from astropy.cosmology import Cosmology
from astropy.units import Quantity
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import cumulative_trapezoid

from uvex_transients.missions import get_mission
from uvex_transients.phase_space.transient import _TopHatTransient
from uvex_transients.simulation.core import SurveySimulator
from uvex_transients.surveys.base import SurveySchedule
from uvex_transients.surveys.utils import get_schedule
from uvex_transients.utils import get_seed_sequence, logger, resolve_healpix_resolution
from uvex_transients.utils.cosmology import get_cosmology

__all__ = ["RecoveryEvaluator", "RedshiftLimit"]

_TRANSIENT_KEY = "top_hat"
_EXPOSURE_NODE_SPACING = 0.1  # dex in the window length, between tabulated exposures

# The absolute AB magnitude of a source of spectral luminosity L_nu, M = -2.5 log10(L_nu / (4 pi (10 pc)^2)) - 48.6,
# is M = _M_ZERO - 2.5 log10(L_nu / (erg/s/Hz)).
_M_ZERO = -2.5 * np.log10(1.0 / (4.0 * np.pi * (10.0 * u.pc).to_value(u.cm) ** 2)) - 48.6


def equivalent_bandwidth(bandpass) -> Quantity:
    r"""
    Return the equivalent width, in frequency, of a bandpass: :math:`\int T(\nu)/T_\mathrm{max}\,d\nu`.

    The width of a top hat of the bandpass's peak throughput that passes as much, so that a spectrum of constant
    :math:`L_\nu` emits :math:`L_\nu\,\Delta\nu` erg/s in the band. It is independent of how efficient the detector is.

    Parameters
    ----------
    bandpass : synphot.SpectralElement
        One of a detector's bandpasses.

    Returns
    -------
    ~astropy.units.Quantity
        The width, in Hz.
    """
    wave = bandpass.waveset
    throughput = bandpass(wave).to_value(u.dimensionless_unscaled)
    nu = wave.to(u.Hz, equivalencies=u.spectral()).value
    order = np.argsort(nu)
    return float(np.trapezoid(throughput[order] / throughput.max(), nu[order])) * u.Hz


class RedshiftLimit:
    r"""
    The redshift out to which a flat-spectrum source of a given luminosity is brighter than a depth.

    For a spectrum flat in :math:`L_\nu` the AB magnitude is

    .. math::

        m = -2.5\log_{10}\frac{(1+z)\,L_\nu}{4\pi d_L^2(z)} - 48.6,

    where the factor :math:`1+z` is the K-correction (the bandwidth compression of a spectrum flat in :math:`f_\nu`).
    It does not depend on the band. The luminosity at which :math:`m` equals the depth is tabulated once on a
    redshift grid, increases with :math:`z`, and is inverted by interpolation, so a call is a table lookup.

    With a `bandwidth` the luminosity is the one in the band, :math:`L_\mathrm{band}=L_\nu\Delta\nu` in erg/s,
    which is the same table shifted by :math:`\log_{10}\Delta\nu`.

    Parameters
    ----------
    mag_limit : float
        The AB depth. Events fainter than it, at their own redshift, are never drawn, so recoveries fainter than it
        are not counted. The default of the evaluator is 24.5, about the single-dwell depth of UVEX.
    cosmology : ~astropy.cosmology.Cosmology, optional
        Defaults to the configured cosmology.
    z_min, z_max : float, optional
        The range of the table. A luminosity whose limit lies outside it is clipped to the nearest end.
    n_grid : int, optional
        Number of log-spaced redshifts in the table. The default is 512.
    bandwidth : ~astropy.units.Quantity, optional
        The equivalent width of the band, in frequency (see `equivalent_bandwidth`). If given, luminosities are
        :math:`L_\mathrm{band}` in erg/s. If omitted they are the spectral luminosity :math:`L_\nu` in erg/s/Hz.
    """

    def __init__(
        self,
        mag_limit: float,
        cosmology: Cosmology | None = None,
        z_min: float = 1e-3,
        z_max: float = 10.0,
        n_grid: int = 512,
        bandwidth: Quantity | None = None,
    ):
        cosmology = get_cosmology(cosmology)
        self.mag_limit = float(mag_limit)
        self.z_min, self.z_max = float(z_min), float(z_max)
        z = np.geomspace(self.z_min, self.z_max, n_grid)
        d_l = cosmology.luminosity_distance(z).to_value(u.cm)
        flux = 10.0 ** (-0.4 * (self.mag_limit + 48.6))
        log_offset = 0.0 if bandwidth is None else float(np.log10(bandwidth.to_value(u.Hz)))
        self._log_luminosity = np.log10(4.0 * np.pi * d_l**2 * flux / (1.0 + z)) + log_offset
        self._log_z = np.log10(z)

    def __call__(self, log_luminosity: ArrayLike) -> NDArray[np.float64]:
        r"""
        Return the redshift limit of a luminosity.

        Parameters
        ----------
        log_luminosity : array-like
            :math:`\log_{10}` of the luminosity in the band in erg/s, or, without a `bandwidth`, of the spectral
            luminosity in erg/s/Hz.

        Returns
        -------
        numpy.ndarray
            The redshift at which the source reaches the depth, clipped to ``[z_min, z_max]``.
        """
        return 10.0 ** np.interp(np.asarray(log_luminosity, dtype=np.float64), self._log_luminosity, self._log_z)

    @property
    def log_luminosity_max(self) -> float:
        r"""float: The :math:`\log_{10}` luminosity above which the limit is clipped to ``z_max``, not set by depth."""
        return float(self._log_luminosity[-1])

    def is_clipped(self, log_luminosity: ArrayLike) -> NDArray[np.bool_]:
        r"""
        Return whether the limit of each luminosity lies beyond the table, so that `__call__` clipped it.

        Parameters
        ----------
        log_luminosity : array-like
            :math:`\log_{10}` of the luminosity, in the same units as for `__call__`.

        Returns
        -------
        numpy.ndarray
            Boolean array, true where the luminosity is outside the range the table covers.
        """
        log_luminosity = np.asarray(log_luminosity, dtype=np.float64)
        return (log_luminosity < self._log_luminosity[0]) | (log_luminosity > self._log_luminosity[-1])


class RecoveryEvaluator:
    r"""
    Count the recoveries of a flat-spectrum top-hat transient at points of the luminosity-duration plane.

    Instances are callable as ``evaluator(x)`` with ``x = (log10 L_band / (erg/s), log10 dt / day)``, which is the
    signature `~uvex_transients.phase_space.sampler.PhaseSpaceSampler` expects. :math:`L_\mathrm{band}` is the
    luminosity emitted in the band, :math:`L_\nu\Delta\nu` for the equivalent bandwidth :math:`\Delta\nu` of the
    band (see `equivalent_bandwidth`). One evaluator makes the map of one band.

    Parameters
    ----------
    schedule : ~uvex_transients.surveys.base.SurveySchedule
        The survey to recover events against.
    mission : str or m4opt.missions.Mission, optional
        The mission (detector and background), or the name of one. The default is ``"uvex_fast"``.
    band : str, optional
        The detector band that events are recovered in, and in which the luminosity is measured. The default is
        ``"FUV"``. The flat spectrum makes the other band differ only in sensitivity, so it gets its own map.
    snr_threshold : float, optional
        The signal-to-noise a visit must reach. The default is 5.
    n_visits : int, optional
        How many visits must reach it. The default is 2.
    exclude_first_visit_detections : bool, optional
        Whether a detection that is only on a field's first visit is discarded; see
        `SurveySimulator.filter_by_snr`. The default is `True`.
    mag_limit : float, optional
        The AB depth that sets the redshift limit of a point. The default is 24.5, about the single-dwell depth of
        UVEX. Only events brighter than it are counted, and the survey can recover some fainter ones (10 to 17% of the
        recoveries at the five points tried, for SNR 5 on 2 visits), so the recovered volume-time is lower than for a
        deeper limit. Use a deeper limit, about 25.5, to count them, at the price of more events per point.
    time_bins : int, optional
        Number of equal explosion-time bins spanning the survey; see `SurveySimulator.sample_N_events`. More bins
        waste fewer events on fields not observed near the explosion, which matters for durations short compared with
        a bin, at a cost of about 0.1 s per bin per chunk of events, in footprint queries. The recovery fraction depends
        on it and the product with `volume_time` does not. The default is 12.
    nside, order : optional
        HEALPix resolution of the sampling grid; the package defaults if omitted.
    chunk_size : int, optional
        Events simulated per chunk, and so the minimum ``n`` of a point. The default is 1000.
    k_target : int, optional
        Stop adding chunks to a point once this many events are recovered. The default is 30 (an 18% relative error
        on the recovery).
    n_max : int, optional
        Never simulate more than this many events at a point. The default is 20000.
    seed : int or numpy.random.SeedSequence, optional
        Root seed. Every chunk draws a fresh child, so chunks and points are independent.
    cosmology : ~astropy.cosmology.Cosmology, optional
        Defaults to the configured cosmology.
    z_max : float, optional
        The largest redshift limit a point can have; see `RedshiftLimit`. The default is 2: beyond it the observed bands
        sample the rest-frame spectrum far on the Wien tail, where a flat spectrum is not a realistic transient.
    schedule_name : str, optional
        The registry name of `schedule` (see `~uvex_transients.surveys.utils.get_schedule`), recorded in `settings` so
        that `from_settings` can fetch the same schedule. Leave it out for a schedule that is not in the registry.
    """

    def __init__(
        self,
        schedule: SurveySchedule,
        mission="uvex_fast",
        *,
        band: str = "FUV",
        snr_threshold: float = 5.0,
        n_visits: int = 2,
        exclude_first_visit_detections: bool = True,
        mag_limit: float = 24.5,
        time_bins: int = 12,
        nside: int | None = None,
        order: str | None = None,
        chunk_size: int = 1000,
        k_target: int = 30,
        n_max: int = 20000,
        seed: int | np.random.SeedSequence | None = None,
        cosmology: Cosmology | None = None,
        z_max: float = 2.0,
        schedule_name: str | None = None,
    ):
        if chunk_size < 1 or k_target < 1 or n_max < chunk_size:
            raise ValueError(
                f"Need chunk_size >= 1, k_target >= 1 and n_max >= chunk_size, got {chunk_size}, {k_target}, {n_max}."
            )

        self.schedule = schedule
        self.mission = get_mission(mission) if isinstance(mission, str) else mission
        self._schedule_name = schedule_name
        if band not in self.mission.detector.bandpasses:
            raise ValueError(f"Unknown band {band!r}; the detector has {list(self.mission.detector.bandpasses)}.")
        self.band = str(band)
        self.bandwidth = equivalent_bandwidth(self.mission.detector.bandpasses[self.band])
        self.snr_threshold = float(snr_threshold)
        self.n_visits = int(n_visits)
        self.exclude_first_visit_detections = bool(exclude_first_visit_detections)
        self.time_bins = int(time_bins)
        self.nside, self.order = resolve_healpix_resolution(nside, order)
        self.chunk_size = int(chunk_size)
        self.k_target = int(k_target)
        self.n_max = int(n_max)
        self._seed = get_seed_sequence(seed)
        self.cosmology = get_cosmology(cosmology)
        self.redshift_limit = RedshiftLimit(mag_limit, self.cosmology, z_max=z_max, bandwidth=self.bandwidth)

        self._settings = {
            "schedule_name": schedule_name,
            "mission": mission if isinstance(mission, str) else None,
            "band": self.band,
            "snr_threshold": self.snr_threshold,
            "n_visits": self.n_visits,
            "exclude_first_visit_detections": self.exclude_first_visit_detections,
            "mag_limit": float(mag_limit),
            "time_bins": self.time_bins,
            "nside": int(self.nside),
            "order": str(self.order),
            "chunk_size": self.chunk_size,
            "k_target": self.k_target,
            "n_max": self.n_max,
            "seed": int(seed) if isinstance(seed, (int, np.integer)) else None,
            "z_max": float(z_max),
            "cosmology": str(self.cosmology.name),
        }

        # The instance that is moved from point to point, and a second one only for tabulating exposures.
        self._transient = _TopHatTransient(cosmology=self.cosmology)
        self._probe = _TopHatTransient(cosmology=self.cosmology)
        self._exposure_nodes: dict[float, float] = {}
        self._warned_clipped = False

        # The rate-weighted comoving volume per steradian out to z, at one event per Mpc^3 per yr.
        z = np.concatenate([[0.0], np.geomspace(1e-4, z_max, 4000)])
        dv_dz = self.cosmology.differential_comoving_volume(z).to_value(u.Mpc**3 / u.sr)
        self._z_grid = z
        self._volume = cumulative_trapezoid(dv_dz / (1.0 + z), z, initial=0.0)

        # The window the explosion times are drawn from, over which the all-sky normalizations are taken.
        self._survey_years = float(schedule.duration.to_value(u.yr))

    # ------------------------------------------------------------------ #
    # Recording and rebuilding                                            #
    # ------------------------------------------------------------------ #
    @property
    def settings(self) -> dict:
        """
        dict: What `from_settings` needs to rebuild this evaluator, as plain types that an ECSV header can hold.

        The mission and schedule are recorded by name. A mission given as an object, or a schedule given without a
        ``schedule_name``, is recorded as `None`, and has to be supplied again to `from_settings`.
        """
        return dict(self._settings)

    @property
    def table_meta(self) -> dict:
        r"""
        dict: What a figure of this evaluator's maps needs to label the luminosity axis.

        ``band``, the unit of the luminosity, and ``absolute_magnitude_zero_point``, the :math:`Z` for which the
        absolute AB magnitude of a source of luminosity :math:`L_\mathrm{band}` is
        :math:`M = Z - 2.5\log_{10}L_\mathrm{band}`.
        """
        return {
            "band": self.band,
            "luminosity_unit": "erg/s",
            "bandwidth_hz": float(self.bandwidth.to_value(u.Hz)),
            "absolute_magnitude_zero_point": float(_M_ZERO + 2.5 * np.log10(self.bandwidth.to_value(u.Hz))),
        }

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
            The cosmology. By default the configured one is used, with a warning if its name is not the recorded one.
        **overrides
            Any other constructor argument, replacing the recorded value.

        Returns
        -------
        RecoveryEvaluator
            An evaluator that makes the same measurement, and so reproduces the scales of the recorded one.

        Raises
        ------
        ValueError
            If no schedule or no mission can be found.
        """
        settings = dict(settings)
        recorded_cosmology = settings.pop("cosmology", None)
        schedule_name = settings.pop("schedule_name", None)
        recorded_mission = settings.pop("mission", None)

        if schedule is None:
            if schedule_name is None:
                raise ValueError("The settings name no schedule; pass `schedule`.")
            schedule = get_schedule(name=schedule_name)
        mission = mission if mission is not None else recorded_mission
        if mission is None:
            raise ValueError("The settings name no mission; pass `mission`.")

        if cosmology is None:
            current = str(get_cosmology(None).name)
            if recorded_cosmology is not None and recorded_cosmology != current:
                logger.warning(
                    "The map was made with the %s cosmology but the configured one is %s; pass `cosmology` to match.",
                    recorded_cosmology,
                    current,
                )
        settings.update(overrides)
        return cls(schedule, mission, cosmology=cosmology, schedule_name=schedule_name, **settings)

    # ------------------------------------------------------------------ #
    # Absolute scale                                                      #
    # ------------------------------------------------------------------ #
    def _exposure_at(self, log_window: float) -> float:
        """Return the total effective exposure of the survey for a window length 10**log_window days, in sr yr."""
        key = round(float(log_window), 6)
        if key not in self._exposure_nodes:
            self._probe.duration_limit = 10.0**key * u.day
            simulator = SurveySimulator(self.schedule, {_TRANSIENT_KEY: self._probe})
            catalog = simulator.compute_effective_exposure(self.time_bins, nside=self.nside, order=self.order)
            self._exposure_nodes[key] = catalog.total_effective_exposure[_TRANSIENT_KEY].to_value(u.sr * u.yr)
        return self._exposure_nodes[key]

    def _exposure(self, log_luminosity: ArrayLike, log_duration: ArrayLike) -> tuple[NDArray, NDArray]:
        r"""
        Return the redshift limit and the footprint exposure of points of the map.

        The exposure depends on a point only through the window length :math:`(1+z_\mathrm{lim})\Delta t`, so it is
        tabulated on a grid of that length, 0.1 dex apart, and interpolated.

        Parameters
        ----------
        log_luminosity, log_duration : array-like
            :math:`\log_{10}L_\nu` in erg/s/Hz and :math:`\log_{10}\Delta t` in days.

        Returns
        -------
        redshift_limit : numpy.ndarray
            The redshift limit of each point.
        exposure : numpy.ndarray
            The total effective exposure of the survey for each point, in sr yr.
        """
        log_luminosity, log_duration = np.broadcast_arrays(
            np.asarray(log_luminosity, dtype=np.float64), np.asarray(log_duration, dtype=np.float64)
        )
        z = self.redshift_limit(log_luminosity)
        log_window = np.log10(1.0 + z) + log_duration

        spacing = _EXPOSURE_NODE_SPACING
        low = np.floor(log_window.min() / spacing) * spacing
        high = np.ceil(log_window.max() / spacing) * spacing + spacing
        nodes = np.round(np.arange(low, high + 0.5 * spacing, spacing), 6)
        exposure = np.interp(log_window, nodes, [self._exposure_at(node) for node in nodes])
        return z, exposure

    def volume_time(self, log_luminosity: ArrayLike, log_duration: ArrayLike) -> NDArray[np.float64]:
        r"""
        Return the effective volume-time :math:`\mathcal V T` of points of the map, in Mpc^3 yr.

        This is the expected number of events over the whole survey, inside the footprint and out to the redshift
        limit, at a volumetric rate of one per Mpc^3 per yr. It involves no Monte Carlo: the exposure comes from the
        schedule (see `_exposure`). Multiply by a rate in Mpc^-3 yr^-1 and by the recovery fraction for the expected
        number of recoveries.

        Parameters
        ----------
        log_luminosity, log_duration : array-like
            :math:`\log_{10}L_\nu` in erg/s/Hz and :math:`\log_{10}\Delta t` in days, of equal shape.

        Returns
        -------
        numpy.ndarray
            :math:`\mathcal V T`, of the same shape.
        """
        z, exposure = self._exposure(log_luminosity, log_duration)
        return exposure * np.interp(z, self._z_grid, self._volume)

    def coverage(self, log_luminosity: ArrayLike, log_duration: ArrayLike) -> NDArray[np.float64]:
        r"""
        Return the fraction of the sky and survey window that the sampled footprint covers, for points of the map.

        :math:`\mathcal E/(4\pi\,\mathrm{sr}\,T)`, the effective exposure over that of the whole sky for the whole
        survey. It is the ratio of the events in the sampled footprint to all events exploding during the survey, so
        multiplying the recovery fraction :math:`f` by it gives `efficiency`.

        Parameters
        ----------
        log_luminosity, log_duration : array-like
            :math:`\log_{10}L_\nu` in erg/s/Hz and :math:`\log_{10}\Delta t` in days, of equal shape.

        Returns
        -------
        numpy.ndarray
            The coverage, between 0 and 1, of the same shape.
        """
        _, exposure = self._exposure(log_luminosity, log_duration)
        return exposure / (4.0 * np.pi * self._survey_years)

    def all_sky_volume_time(self, log_luminosity: ArrayLike) -> NDArray[np.float64]:
        r"""
        Return the volume-time of the whole sky over the whole survey out to the redshift limit, in Mpc^3 yr.

        The number of events exploding during the survey, anywhere on the sky and out to the redshift limit, at
        one per Mpc^3 per yr. It is `volume_time` divided by `coverage`.

        Parameters
        ----------
        log_luminosity : array-like
            :math:`\log_{10}L_\nu` in erg/s/Hz.

        Returns
        -------
        numpy.ndarray
            The all-sky volume-time, of the same shape.
        """
        z = self.redshift_limit(log_luminosity)
        return 4.0 * np.pi * self._survey_years * np.interp(z, self._z_grid, self._volume)

    def efficiency(self, recovery: ArrayLike, log_luminosity: ArrayLike, log_duration: ArrayLike) -> NDArray:
        r"""
        Convert a recovery fraction in the sampled footprint to one over the whole sky and survey window.

        :math:`\eta = f\,\mathcal E/(4\pi\,\mathrm{sr}\,T)`: the probability that an event exploding at a random
        position and time during the survey, with a redshift inside the limit, is recovered. Unlike :math:`f` it does
        not depend on the number of time bins. It does depend on the redshift limit, through the volume it is
        averaged over, so it changes with ``mag_limit`` and ``z_max`` where :math:`f\,\mathcal V T` does not.

        Parameters
        ----------
        recovery : array-like
            The recovery fraction :math:`f` in the sampled footprint.
        log_luminosity, log_duration : array-like
            :math:`\log_{10}L_\nu` in erg/s/Hz and :math:`\log_{10}\Delta t` in days, of the shape of `recovery`.

        Returns
        -------
        numpy.ndarray
            :math:`\eta`, of the same shape.
        """
        return np.asarray(recovery, dtype=np.float64) * self.coverage(log_luminosity, log_duration)

    # ------------------------------------------------------------------ #
    # One point                                                           #
    # ------------------------------------------------------------------ #
    def __call__(self, x: ArrayLike) -> dict:
        r"""
        Count the recoveries at one point of the map.

        Parameters
        ----------
        x : array-like
            ``(log10 L_band / (erg/s), log10 dt / day)``.

        Returns
        -------
        dict
            ``k`` (recoveries), ``n`` (events simulated), ``z_limit``, ``volume_time`` (:math:`\mathcal V T`, in
            Mpc^3 yr), ``coverage`` (see `coverage`) and ``n_chunks``.
        """
        log_luminosity, log_duration = (float(value) for value in x)
        z_limit = float(self.redshift_limit(log_luminosity))
        if self.redshift_limit.is_clipped(log_luminosity) and not self._warned_clipped:
            self._warned_clipped = True
            logger.warning(
                "log10 L_band = %.2f is beyond the range the redshift limit table covers, so z_limit is clipped to "
                "%.3g. Every brighter point is clipped too, and its recovered volume-time is a lower bound.",
                log_luminosity,
                z_limit,
            )

        spectral_luminosity = 10.0**log_luminosity * u.erg / u.s / self.bandwidth
        self._transient.set_point(spectral_luminosity, 10.0**log_duration * u.day, z_limit)

        k = n = n_chunks = 0
        while n < self.n_max and k < self.k_target:
            simulator = SurveySimulator(
                self.schedule, {_TRANSIENT_KEY: self._transient}, simulation_seed=self._seed.spawn(1)[0]
            )
            catalog = simulator.sample_N_events(self.chunk_size, self.time_bins, nside=self.nside, order=self.order)
            recovered = simulator.filter_by_snr(
                catalog,
                self.mission,
                self.snr_threshold,
                bands=[self.band],
                n_visits=self.n_visits,
                exclude_first_visit_detections=self.exclude_first_visit_detections,
            )
            k += len(recovered)
            n += len(catalog)
            n_chunks += 1

        _, exposure = self._exposure(log_luminosity, log_duration)
        volume_time = float(exposure * np.interp(z_limit, self._z_grid, self._volume))
        coverage = float(exposure / (4.0 * np.pi * self._survey_years))
        logger.debug(
            "log10 L_band=%.2f, log10 dt=%.2f: z_limit=%.3g, recovered %d of %d, VT=%.3g Mpc^3 yr.",
            log_luminosity,
            log_duration,
            z_limit,
            k,
            n,
            volume_time,
        )
        return {
            "k": k,
            "n": n,
            "z_limit": z_limit,
            "volume_time": volume_time,
            "coverage": coverage,
            "n_chunks": n_chunks,
        }
