r"""
The canonical sampler: a fast-rise, exponential-decay transient with a power-law spectrum.

`CanonicalSampler` is the sampler of the canonical recovery map. A point of its space is the base-10 logarithm of
:math:`L_\mathrm{band}` in erg/s and of :math:`t_{1/2}` in days. :math:`L_\mathrm{band}` is the peak rest-frame
luminosity emitted in the band, :math:`L_\nu\Delta\nu` for the equivalent bandwidth :math:`\Delta\nu` of the band
with :math:`L_\nu` taken at its effective frequency, and :math:`t_{1/2}` is the rest-frame time from the peak until the
luminosity has fallen to half. Any other SED is placed on the same map by measuring the same two quantities on it.
The transient is `~uvex_transients.phase_space.transient._CanonicalTransient`. One sampler makes the map of one band.

`RedshiftLimit` is the redshift out to which a power-law source of a given luminosity is brighter than a depth,
tabulated once and inverted by interpolation.
"""

from collections.abc import Mapping

import numpy as np
from astropy import units as u
from astropy.cosmology import Cosmology
from astropy.units import Quantity
from numpy.typing import ArrayLike, NDArray

from uvex_transients.phase_space.samplers.survey import SurveySampler, _configured
from uvex_transients.phase_space.space import Axis, Space
from uvex_transients.phase_space.transient import FREQUENCY_MAX, _CanonicalTransient
from uvex_transients.surveys.base import SurveySchedule
from uvex_transients.utils import config, logger
from uvex_transients.utils.cosmology import get_cosmology
from uvex_transients.utils.phot_utils import effective_frequency, equivalent_bandwidth

__all__ = ["CanonicalSampler", "RedshiftLimit"]

#: Absolute AB magnitude of a source of spectral luminosity L_nu, M = -2.5 log10(L_nu / (4 pi (10 pc)^2)) - 48.6, is
#: M = _M_ZERO - 2.5 log10(L_nu / (erg/s/Hz)).
_M_ZERO = -2.5 * np.log10(1.0 / (4.0 * np.pi * (10.0 * u.pc).to_value(u.cm) ** 2)) - 48.6


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
# Canonical sampler                            #
# ============================================ #
class CanonicalSampler(SurveySampler, kind="canonical"):
    r"""
    Count the recoveries of a canonical transient at points of the luminosity-half-decline-time plane.

    The transient has a fast-rise, exponential-decay light curve and a power-law spectrum (see
    `~uvex_transients.phase_space.transient._CanonicalTransient`). A point is
    ``x = (log10 L_band / (erg/s), log10 t_half / day)``. :math:`L_\mathrm{band}` is the peak rest-frame luminosity
    emitted in the band, :math:`L_\nu\Delta\nu` for the equivalent bandwidth :math:`\Delta\nu` of the band with
    :math:`L_\nu` taken at its effective frequency. One sampler makes the map of one band.

    Both axes are derived quantities, not parameters of the SED, so this sampler places a point itself instead of
    binding axes to parameters.

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
    cosmology : ~astropy.cosmology.Cosmology, optional
        Defaults to the configured cosmology.
    schedule_name : str, optional
        The registry name of `schedule`, recorded in `settings` so that `from_settings` can fetch the same schedule.
    """

    TRANSIENT_KEY = "canonical"

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
        cosmology: Cosmology | None = None,
        schedule_name: str | None = None,
    ):
        super().__init__(
            schedule,
            mission,
            band=band,
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
        self.spectral_index = float(_configured(spectral_index, "transient.spectral_index"))

        # The scale of the luminosity axis: the luminosity in the band, from the spectral one at the band's
        # effective frequency
        bandpass = self.mission.detector.bandpasses[self.band]
        self.reference_frequency = effective_frequency(bandpass)
        self.bandwidth = equivalent_bandwidth(bandpass, self.spectral_index, self.reference_frequency)

        # What `from_settings` needs to rebuild this sampler, as plain types an ECSV header can hold
        self._settings = {
            "schedule_name": schedule_name,
            "mission": self._mission_name,
            "band": self.band,
            "spectral_index": self.spectral_index,
            "rise_ratio": float(_configured(rise_ratio, "transient.rise_ratio")),
            "fade_mag": float(_configured(fade_mag, "transient.fade_mag")),
            "snr_threshold": self.snr_threshold,
            "n_visits": self.n_visits,
            "exclude_first_visit_detections": self.exclude_first_visit_detections,
            "mag_limit": self.mag_limit,
            "time_bins": self.time_bins,
            "nside": int(self.nside),
            "order": str(self.order),
            "cosmology": str(self.cosmology.name),
        }

        # The redshift limit comes from the depth alone. Its table ends where the band leaves the spectrum, which the
        # model sets at FREQUENCY_MAX in the rest frame, so that no event is drawn where it would be unseen
        redshift_edge = float(FREQUENCY_MAX / self.reference_frequency) - 1.0
        self._limit = RedshiftLimit(
            self.mag_limit,
            self.cosmology,
            z_min=config["phase_space.numerics.z_table_min"],
            z_max=redshift_edge,
            n_grid=config["phase_space.numerics.z_table_points"],
            bandwidth=self.bandwidth,
            spectral_index=self.spectral_index,
        )

        self._space = Space(
            [
                Axis(
                    "log_luminosity",
                    *config["phase_space.map.log_luminosity"],
                    label=rf"$L_\mathrm{{{self.band}}}$",
                    unit="erg/s",
                ),
                Axis(
                    "log_half_time",
                    *config["phase_space.map.log_half_time"],
                    label=r"rest-frame half-decline time $t_{1/2}$",
                    unit="day",
                ),
            ]
        )
        self._build_scales(redshift_edge)

    # -------------------------------------- #
    # Recording and rebuilding               #
    # -------------------------------------- #
    @property
    def table_meta(self) -> dict:
        r"""
        dict: What a figure of this sampler's maps needs to label the luminosity axis.

        ``band``, the unit of the luminosity, and ``absolute_magnitude_zero_point``, the :math:`Z` for which the
        absolute AB magnitude of a source of luminosity :math:`L_\mathrm{band}` is
        :math:`M = Z - 2.5\log_{10}L_\mathrm{band}`. Also the shape of the canonical transient the axes are measured
        on (``spectral_index``, ``rise_ratio``, ``fade_mag``), which
        `~uvex_transients.phase_space.compare.difference_map` checks before it compares two maps.
        """
        bandwidth_hz = float(self.bandwidth.to_value(u.Hz))
        return {
            "band": self.band,
            "luminosity_unit": "erg/s",
            "spectral_index": self.spectral_index,
            "rise_ratio": self._settings["rise_ratio"],
            "fade_mag": self._settings["fade_mag"],
            "bandwidth_hz": bandwidth_hz,
            "absolute_magnitude_zero_point": float(_M_ZERO + 2.5 * np.log10(bandwidth_hz)),
        }

    @classmethod
    def from_settings(
        cls,
        settings: Mapping,
        *,
        schedule: SurveySchedule | None = None,
        mission=None,
        cosmology: Cosmology | None = None,
        **overrides,
    ) -> "CanonicalSampler":
        """
        Rebuild a sampler from the `settings` of another one.

        Parameters
        ----------
        settings : Mapping
            The `settings` of the sampler to rebuild.
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
        CanonicalSampler
            A sampler that makes the same measurement, and so reproduces the scales of the recorded one.

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
        settings, schedule, mission, schedule_name = cls._rebuild_arguments(settings, schedule, mission, cosmology)
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
    # The point of the space                 #
    # -------------------------------------- #
    def redshift_limit(self, x: ArrayLike) -> NDArray[np.float64]:
        """
        Return the redshift out to which events are drawn at points of the space.

        It depends on the luminosity alone, so the half-decline time is not used.

        Parameters
        ----------
        x : array-like
            Points ``(log10 L_band, log10 t_half)``, shape ``(..., 2)``.

        Returns
        -------
        numpy.ndarray
            The redshift limit of each point, of shape ``(...)``.
        """
        return self._limit(self._points(x)[..., 0])

    def _place(self, x: NDArray[np.float64]) -> None:
        """
        Move the transient to ``(log10 L_band / (erg/s), log10 t_half / day)``.

        Parameters
        ----------
        x : numpy.ndarray
            The point, shape ``(2,)``.
        """
        log_luminosity, log_half_time = float(x[0]), float(x[1])
        z_limit = float(self.redshift_limit(x))

        # Every point brighter than this one is clipped too, so say so once
        if self._limit.is_clipped(log_luminosity) and not self._warned_clipped:
            self._warned_clipped = True
            logger.warning(
                "log10 L_band = %.2f is so bright that its redshift limit is beyond the one at which the band leaves "
                "the spectrum, so z_limit is clipped to %.3g. Every brighter point is clipped too, and its recovered "
                "volume-time is a lower bound.",
                log_luminosity,
                z_limit,
            )

        spectral_luminosity = 10.0**log_luminosity * u.erg / u.s / self.bandwidth
        self._transient.set_point(spectral_luminosity, 10.0**log_half_time * u.day, z_limit)

    def _log_window(self, z: ArrayLike, x: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Return the log10 of the window in days: the half-decline time, stretched to the fade and by time dilation.

        Parameters
        ----------
        z : array-like
            The redshift limit of each point.
        x : numpy.ndarray
            The points, shape ``(..., 2)``.

        Returns
        -------
        numpy.ndarray
            :math:`\log_{10}` of the window in days, of the shape of `z`.
        """
        return np.log10(1.0 + z) + np.log10(self._transient.window_factor) + x[..., 1]
