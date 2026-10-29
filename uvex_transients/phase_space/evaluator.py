r"""
Evaluate the survey's recovery of a canonical transient at one point of a luminosity-half-decline-time map.

`RecoveryEvaluator` is the expensive function `~uvex_transients.phase_space.sampler.PhaseSpaceSampler` calls. At a point
:math:`(\log_{10}L_\mathrm{band}, \log_{10}t_{1/2})` it finds the redshift out to which the event is brighter than a
chosen AB depth (`RedshiftLimit`), draws events volume-weighted out to it on the footprint the survey covers, and counts
how many `SurveySimulator.filter_by_snr` recovers, adding chunks of events until enough are recovered or too many have
been simulated.

The recovery fraction :math:`f=k/n` is conditional on the footprint that was sampled, which depends on the number of
explosion-time bins. The noise-free volume-time :math:`\mathcal V T` is the expected number of events over the survey
in that footprint at one event per :math:`\mathrm{Mpc}^3\,\mathrm{yr}`. The product :math:`f\,\mathcal V T` does not
depend on the binning, and is the expected number of recoveries at unit rate, :math:`R\,f\,\mathcal V T` at a rate
:math:`R`.
"""

import numpy as np
from astropy import units as u
from astropy.cosmology import Cosmology
from astropy.units import Quantity
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import cumulative_trapezoid

from uvex_transients.missions import get_mission
from uvex_transients.phase_space.transient import FREQUENCY_MAX, _CanonicalTransient
from uvex_transients.simulation.core import SurveySimulator
from uvex_transients.surveys.base import SurveySchedule
from uvex_transients.surveys.utils import get_schedule
from uvex_transients.utils import config, get_seed_sequence, logger, resolve_healpix_resolution
from uvex_transients.utils.cosmology import get_cosmology
from uvex_transients.utils.phot_utils import effective_frequency, equivalent_bandwidth

__all__ = ["RecoveryEvaluator", "RedshiftLimit"]

#: Name the one transient goes by in the simulator.
_TRANSIENT_KEY = "canonical"

#: Absolute AB magnitude of a source of spectral luminosity L_nu, M = -2.5 log10(L_nu / (4 pi (10 pc)^2)) - 48.6, is
#: M = _M_ZERO - 2.5 log10(L_nu / (erg/s/Hz)).
_M_ZERO = -2.5 * np.log10(1.0 / (4.0 * np.pi * (10.0 * u.pc).to_value(u.cm) ** 2)) - 48.6


def _configured(value, key: str):
    """Return `value`, or ``config["phase_space.<key>"]`` if it is `None`."""
    return config[f"phase_space.{key}"] if value is None else value


# ============================================ #
# Redshift Limit                               #
# ============================================ #
class RedshiftLimit:
    r"""
    The redshift out to which a power-law source of a given luminosity is brighter than a depth.

    For a spectrum :math:`L_\nu\propto\nu^\alpha` the AB magnitude in a band at the observed frequency
    :math:`\nu_\mathrm{obs}` is

    .. math::

        m = -2.5\log_{10}\frac{(1+z)^{1+\alpha}\,L_\nu(\nu_\mathrm{obs})}{4\pi d_L^2(z)} - 48.6,

    where :math:`(1+z)^{1+\alpha}` is the K-correction: one factor of :math:`1+z` is the bandwidth compression and
    :math:`(1+z)^\alpha` the shift to the bluer rest-frame frequency :math:`(1+z)\nu_\mathrm{obs}`. The luminosity at
    which :math:`m` equals the depth is tabulated once on a redshift grid, increases with :math:`z`, and is inverted by
    interpolation, so a call is a table lookup. The band is taken as the single frequency :math:`\nu_\mathrm{obs}` at
    which the luminosity is quoted, which is accurate enough for a limit that only bounds where events are drawn.

    With a `bandwidth` the luminosity is the one in the band, :math:`L_\mathrm{band}=L_\nu(\nu_\mathrm{obs})\Delta\nu`
    in erg/s (see `~uvex_transients.utils.phot_utils.equivalent_bandwidth`), the same table shifted by
    :math:`\log_{10}\Delta\nu`.

    Parameters
    ----------
    mag_limit : float
        The AB depth. Events fainter than it, at their own redshift, are never drawn.
    cosmology : ~astropy.cosmology.Cosmology, optional
        Defaults to the configured cosmology.
    z_min, z_max : float, optional
        The range of the table. A luminosity whose limit lies outside it is clipped to the nearest end.
    n_grid : int, optional
        Number of log-spaced redshifts in the table. The default is 512.
    bandwidth : ~astropy.units.Quantity, optional
        The equivalent width of the band, in frequency. If given, luminosities are :math:`L_\mathrm{band}` in erg/s;
        if omitted, the spectral luminosity :math:`L_\nu` in erg/s/Hz.
    spectral_index : float, optional
        The index :math:`\alpha` of the spectrum. The default of 0 is flat.

    Raises
    ------
    ValueError
        If the luminosity that reaches the depth does not increase with redshift, which happens for a spectrum rising
        steeply to the blue (an index of about 3 or more): such a source has no redshift limit.
    """

    def __init__(
        self,
        mag_limit: float,
        cosmology: Cosmology | None = None,
        z_min: float = 1e-3,
        z_max: float = 10.0,
        n_grid: int = 512,
        bandwidth: Quantity | None = None,
        spectral_index: float = 0.0,
    ):
        cosmology = get_cosmology(cosmology)
        self.mag_limit = float(mag_limit)
        self.spectral_index = float(spectral_index)
        self.z_min, self.z_max = float(z_min), float(z_max)

        # The spectral luminosity that gives the limiting flux density at every redshift of the table
        z = np.geomspace(self.z_min, self.z_max, n_grid)
        d_l = cosmology.luminosity_distance(z).to_value(u.cm)
        flux = 10.0 ** (-0.4 * (self.mag_limit + 48.6))
        k_correction = (1.0 + z) ** (1.0 + self.spectral_index)

        # Shifted to the luminosity in the band, if there is a bandwidth to shift it by
        log_offset = 0.0 if bandwidth is None else float(np.log10(bandwidth.to_value(u.Hz)))
        self._log_luminosity = np.log10(4.0 * np.pi * d_l**2 * flux / k_correction) + log_offset
        self._log_z = np.log10(z)

        # The table is inverted by interpolation, so it has to be monotonic
        if np.any(np.diff(self._log_luminosity) <= 0.0):
            raise ValueError(
                f"A source of spectral index {self.spectral_index} needs no more luminosity to reach the depth at a "
                f"greater redshift somewhere below z={self.z_max:g}, so it has no redshift limit; the spectrum is too "
                "blue."
            )

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
        """float: The log10 luminosity above which the limit is clipped to ``z_max``, not set by depth."""
        return float(self._log_luminosity[-1])

    def is_clipped(self, log_luminosity: ArrayLike) -> NDArray[np.bool_]:
        """
        Return whether the limit of each luminosity lies beyond the table, so that `__call__` clipped it.

        Parameters
        ----------
        log_luminosity : array-like
            The log10 luminosity, in the same units as for `__call__`.

        Returns
        -------
        numpy.ndarray
            Boolean array, true where the luminosity is outside the range the table covers.
        """
        log_luminosity = np.asarray(log_luminosity, dtype=np.float64)
        return (log_luminosity < self._log_luminosity[0]) | (log_luminosity > self._log_luminosity[-1])


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
        # ---------------------------------------------- #
        # Settings                                       #
        # ---------------------------------------------- #
        mission = _configured(mission, "recovery.mission")
        chunk_size = int(_configured(chunk_size, "recovery.chunk_size"))
        k_target = int(_configured(k_target, "recovery.k_target"))
        n_max = int(_configured(n_max, "recovery.n_max"))
        if chunk_size < 1 or k_target < 1 or n_max < chunk_size:
            raise ValueError(
                f"Need chunk_size >= 1, k_target >= 1 and n_max >= chunk_size, got {chunk_size}, {k_target}, {n_max}."
            )

        self.schedule = schedule
        self.mission = get_mission(mission) if isinstance(mission, str) else mission
        self.band = str(_configured(band, "recovery.band"))
        if self.band not in self.mission.detector.bandpasses:
            raise ValueError(f"Unknown band {self.band!r}; the detector has {list(self.mission.detector.bandpasses)}.")

        self.spectral_index = float(_configured(spectral_index, "transient.spectral_index"))
        self.snr_threshold = float(_configured(snr_threshold, "recovery.snr_threshold"))
        self.n_visits = int(_configured(n_visits, "recovery.n_visits"))
        self.exclude_first_visit_detections = bool(
            _configured(exclude_first_visit_detections, "recovery.exclude_first_visit_detections")
        )
        self.time_bins = int(_configured(time_bins, "recovery.time_bins"))
        self.nside, self.order = resolve_healpix_resolution(nside, order)
        self.chunk_size, self.k_target, self.n_max = chunk_size, k_target, n_max
        self._seed = get_seed_sequence(seed)
        self.cosmology = get_cosmology(cosmology)
        self._exposure_spacing = float(config["phase_space.numerics.exposure_spacing"])

        # The scale of the luminosity axis: the luminosity in the band, from the spectral one at the band's
        # effective frequency
        bandpass = self.mission.detector.bandpasses[self.band]
        self.reference_frequency = effective_frequency(bandpass)
        self.bandwidth = equivalent_bandwidth(bandpass, self.spectral_index, self.reference_frequency)

        # What `from_settings` needs to rebuild this evaluator, as plain types an ECSV header can hold
        self._settings = {
            "schedule_name": schedule_name,
            "mission": mission if isinstance(mission, str) else None,
            "band": self.band,
            "spectral_index": self.spectral_index,
            "rise_ratio": float(_configured(rise_ratio, "transient.rise_ratio")),
            "fade_mag": float(_configured(fade_mag, "transient.fade_mag")),
            "snr_threshold": self.snr_threshold,
            "n_visits": self.n_visits,
            "exclude_first_visit_detections": self.exclude_first_visit_detections,
            "mag_limit": float(_configured(mag_limit, "recovery.mag_limit")),
            "time_bins": self.time_bins,
            "nside": int(self.nside),
            "order": str(self.order),
            "chunk_size": self.chunk_size,
            "k_target": self.k_target,
            "n_max": self.n_max,
            "seed": int(seed) if isinstance(seed, (int, np.integer)) else None,
            "cosmology": str(self.cosmology.name),
        }
        self._schedule_name = schedule_name

        # ---------------------------------------------- #
        # Scales                                         #
        # ---------------------------------------------- #
        # The redshift limit comes from the depth alone. Its table ends where the band leaves the spectrum, which the
        # model sets at FREQUENCY_MAX in the rest frame, so that no event is drawn where it would be unseen
        self.redshift_edge = float(FREQUENCY_MAX / self.reference_frequency) - 1.0
        self.redshift_limit = RedshiftLimit(
            self._settings["mag_limit"],
            self.cosmology,
            z_min=config["phase_space.numerics.z_table_min"],
            z_max=self.redshift_edge,
            n_grid=config["phase_space.numerics.z_table_points"],
            bandwidth=self.bandwidth,
            spectral_index=self.spectral_index,
        )

        # The rate-weighted comoving volume per steradian out to z, at one event per Mpc^3 per yr
        z = np.concatenate(
            [
                [0.0],
                np.geomspace(
                    config["phase_space.numerics.volume_z_min"],
                    self.redshift_edge,
                    config["phase_space.numerics.volume_points"],
                ),
            ]
        )
        dv_dz = self.cosmology.differential_comoving_volume(z).to_value(u.Mpc**3 / u.sr)
        self._z_grid = z
        self._volume = cumulative_trapezoid(dv_dz / (1.0 + z), z, initial=0.0)

        # The window the explosion times are drawn from, over which the all-sky normalizations are taken
        self._survey_years = float(schedule.duration.to_value(u.yr))

        # The instance that is moved from point to point, and a second one only for tabulating exposures
        self._transient = self.make_transient()
        self._probe = self.make_transient()
        self._exposure_nodes: dict[float, float] = {}
        self._warned_clipped = False

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
        return dict(self._settings)

    @property
    def table_meta(self) -> dict:
        r"""
        dict: What a figure of this evaluator's maps needs to label the luminosity axis.

        ``band``, the unit of the luminosity, and ``absolute_magnitude_zero_point``, the :math:`Z` for which the
        absolute AB magnitude of a source of luminosity :math:`L_\mathrm{band}` is
        :math:`M = Z - 2.5\log_{10}L_\mathrm{band}`.
        """
        bandwidth_hz = float(self.bandwidth.to_value(u.Hz))
        return {
            "band": self.band,
            "luminosity_unit": "erg/s",
            "spectral_index": self.spectral_index,
            "bandwidth_hz": bandwidth_hz,
            "absolute_magnitude_zero_point": float(_M_ZERO + 2.5 * np.log10(bandwidth_hz)),
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
        settings = dict(settings)
        if "rise_ratio" not in settings:
            raise ValueError(
                "These settings are of a map made with a top-hat light curve, which can no longer be rebuilt; "
                "make the map again."
            )
        recorded_cosmology = settings.pop("cosmology", None)
        schedule_name = settings.pop("schedule_name", None)
        recorded_mission = settings.pop("mission", None)

        # The schedule and the mission are the two things a file can only name
        if schedule is None:
            if schedule_name is None:
                raise ValueError("The settings name no schedule; pass `schedule`.")
            schedule = get_schedule(name=schedule_name)
        mission = mission if mission is not None else recorded_mission
        if mission is None:
            raise ValueError("The settings name no mission; pass `mission`.")

        # A different cosmology rescales every volume, so say so instead of quietly rebuilding a different map
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

    def make_transient(self) -> _CanonicalTransient:
        """
        Return a transient of this evaluator's spectrum, whose luminosities are taken at `reference_frequency`.

        Returns
        -------
        _CanonicalTransient
            A new instance, not yet placed at a point of the map.
        """
        return _CanonicalTransient(
            cosmology=self.cosmology,
            spectral_index=self.spectral_index,
            reference_frequency=self.reference_frequency,
            rise_ratio=self._settings["rise_ratio"],
            fade_mag=self._settings["fade_mag"],
        )

    # -------------------------------------- #
    # Absolute scale                         #
    # -------------------------------------- #
    def _exposure_at(self, log_window: float) -> float:
        """Return the effective exposure of the survey, in sr yr, for a window of length 10**log_window days."""
        key = round(float(log_window), 6)
        if key not in self._exposure_nodes:
            self._probe.duration_limit = 10.0**key * u.day
            simulator = SurveySimulator(self.schedule, {_TRANSIENT_KEY: self._probe})
            catalog = simulator.compute_effective_exposure(self.time_bins, nside=self.nside, order=self.order)
            self._exposure_nodes[key] = catalog.total_effective_exposure[_TRANSIENT_KEY].to_value(u.sr * u.yr)
        return self._exposure_nodes[key]

    def _exposure(self, log_luminosity: ArrayLike, log_half_time: ArrayLike) -> tuple[NDArray, NDArray]:
        """
        Return the redshift limit and the footprint exposure of points of the map.

        The exposure depends on a point only through the window length over which an event is followed, so it is
        tabulated on a grid of that length and interpolated.

        Parameters
        ----------
        log_luminosity, log_half_time : array-like
            The coordinates of the points, of equal shape once broadcast.

        Returns
        -------
        redshift_limit, exposure : numpy.ndarray
            The redshift limit of each point, and the effective exposure of the survey for it in sr yr.
        """
        log_luminosity, log_half_time = np.broadcast_arrays(
            np.asarray(log_luminosity, dtype=np.float64), np.asarray(log_half_time, dtype=np.float64)
        )
        z = self.redshift_limit(log_luminosity)

        # The window is the half-decline time, stretched to the fade and by time dilation at the farthest event
        log_window = np.log10(1.0 + z) + np.log10(self._transient.window_factor) + log_half_time

        # Tabulate the exposure on nodes that bracket the windows asked for
        spacing = self._exposure_spacing
        low = np.floor(log_window.min() / spacing) * spacing
        high = np.ceil(log_window.max() / spacing) * spacing + spacing
        nodes = np.round(np.arange(low, high + 0.5 * spacing, spacing), 6)
        exposure = np.interp(log_window, nodes, [self._exposure_at(node) for node in nodes])
        return z, exposure

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
        z, exposure = self._exposure(log_luminosity, log_half_time)
        return exposure * np.interp(z, self._z_grid, self._volume)

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
        _, exposure = self._exposure(log_luminosity, log_half_time)
        return exposure / (4.0 * np.pi * self._survey_years)

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
        z = self.redshift_limit(log_luminosity)
        return 4.0 * np.pi * self._survey_years * np.interp(z, self._z_grid, self._volume)

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
    def _count_recoveries(self) -> tuple[int, int, int]:
        """Simulate chunks of events at the current point of the transient, and return ``(k, n, n_chunks)``."""
        k = n = n_chunks = 0

        # Top the point up until it has enough recoveries, or has used all the events it is allowed
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
        return k, n, n_chunks

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
        log_luminosity, log_half_time = (float(value) for value in x)
        z_limit = float(self.redshift_limit(log_luminosity))

        # Every point brighter than this one is clipped too, so say so once
        if self.redshift_limit.is_clipped(log_luminosity) and not self._warned_clipped:
            self._warned_clipped = True
            logger.warning(
                "log10 L_band = %.2f is so bright that its redshift limit is beyond the one at which the band leaves "
                "the spectrum, so z_limit is clipped to %.3g. Every brighter point is clipped too, and its recovered "
                "volume-time is a lower bound.",
                log_luminosity,
                z_limit,
            )

        # Move the transient to the point and simulate it
        spectral_luminosity = 10.0**log_luminosity * u.erg / u.s / self.bandwidth
        self._transient.set_point(spectral_luminosity, 10.0**log_half_time * u.day, z_limit)
        k, n, n_chunks = self._count_recoveries()

        # The noise-free scales of the point
        _, exposure = self._exposure(log_luminosity, log_half_time)
        volume_time = float(exposure * np.interp(z_limit, self._z_grid, self._volume))
        coverage = float(exposure / (4.0 * np.pi * self._survey_years))
        logger.debug(
            "log10 L_band=%.2f, log10 t_half=%.2f: z_limit=%.3g, recovered %d of %d, VT=%.3g Mpc^3 yr.",
            log_luminosity,
            log_half_time,
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
