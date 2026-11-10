r"""
The shock breakout sampler: the survey's detection of a breakout over its peak luminosity and time scale.

`BreakoutSampler` is the second scenario of the phase-space machinery, and the reason it is built the way it is. Its
transient is the `~uvex_transients.models.shock_cooling.inverse_gamma.InverseGammaBlackbodySED`: a bolometric light
curve that rises, peaks and declines on a time scale :math:`t_0`, with a blackbody on a photosphere that expands from
the breakout radius, so that the colour evolves. A point of its space is ``(log10 L_0 / (erg/s), log10 t_0 / s)``, and
both axes are parameters of the SED, so they are bound to it by name
(`~uvex_transients.phase_space.samplers.survey.ParameterAxis`) and no placing code is written.

**What counts as a detection.** A breakout is detected if at least one exposure reaches the signal-to-noise threshold,
and only an exposure that starts early enough counts: an event is followed for the window out to the *bolometric*
half-decline of its light curve, :math:`x_{1/2}\,t_0` after breakout with :math:`x_{1/2}` a constant of the fixed shape
(about 3.6), stretched by time dilation. An exposure that starts later would be seeing the cooling emission that
follows, not the breakout, so it does not count. The UVEX dwell of 900 s is long compared to a breakout of seconds, so
an exposure measures the photons that arrive over the whole of it, and an exposure that starts before the explosion and
runs into it is a measurement of the source (see
`~uvex_transients.transients.base.TransientBase.AVERAGE_OVER_EXPOSURE`). A short pulse is therefore averaged over its
dwell, which is what a detector does.

**A caveat, accepted.** The window limits which exposures count by when they *start*, but a 900 s exposure integrates
whatever arrives over the whole of it, including emission after the window. For a time scale below about 100 s most of
the photons of an exposure that starts at breakout arrive after the window (the ultraviolet keeps rising for hours as
the photosphere expands and cools), so a detection there can be driven by the cooling emission. The map is the
sensitivity to the transient as defined, not to its first seconds alone.

**The redshift limit.** Events are drawn out to the redshift at which the *instantaneous* band magnitude at the
brightest point of the window reaches the depth, a bound that no exposure beats. The temperature depends on the
luminosity and the time scale together, so the limit is a function of both and is tabulated over the axes, once, with
the same band weights the detection horizon uses.
"""

from typing import Union

import numpy as np
from astropy import units as u
from astropy.cosmology import Cosmology
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq

from uvex_transients.models._utils import to_cgs_value
from uvex_transients.models.core.detection_horizon import peak_band_magnitudes, resolve_bandpasses
from uvex_transients.models.core.priors import ConstantPrior
from uvex_transients.models.lightcurves.generic import InverseGammaBrokenPowerLawLightcurve
from uvex_transients.models.shock_cooling import InverseGammaBlackbodySED
from uvex_transients.phase_space.samplers.survey import ParameterAxis, SurveySampler, _configured
from uvex_transients.phase_space.space import Axis, Space
from uvex_transients.surveys.base import SurveySchedule
from uvex_transients.transients.base import ExtragalacticTransient
from uvex_transients.utils import logger
from uvex_transients.utils.cosmology import resolve_cosmological_distances

__all__ = ["BreakoutSampler", "half_decline_time"]

_SECONDS_PER_DAY = 86400.0

#: The smallest redshift of the table of the redshift limit.
_Z_MIN = 1e-3
#: Points of the redshift grid, and of the grid of times in the window, that the limit is evaluated on.
_N_REDSHIFT = 96
_N_TIME = 96
#: The earliest time in the window, in units of the time scale. The light curve vanishes well before it.
_X_MIN = 1e-2


# ============================================ #
# The window                                   #
# ============================================ #
def half_decline_time(p: float, q: float, x_break: float, smoothness: float) -> tuple[float, float]:
    r"""
    Return the time of the peak and of the half-decline of the light curve, in units of its time scale :math:`t_0`.

    The light curve is that of
    `~uvex_transients.models.lightcurves.generic.InverseGammaBrokenPowerLawLightcurve`. The half-decline is the time
    after the peak at which the luminosity has fallen to half of its maximum.

    Parameters
    ----------
    p, q, x_break, smoothness : float
        The shape parameters of the light curve.

    Returns
    -------
    x_peak : float
        The time of the peak since breakout, in units of :math:`t_0`.
    x_half : float
        The time of the half-decline since breakout, in units of :math:`t_0`.

    Raises
    ------
    ValueError
        If the light curve never falls to half of its peak.
    """
    shape = np.float64(p), np.float64(q), np.float64(x_break), np.float64(smoothness)
    log_peak, log_shape_peak = InverseGammaBrokenPowerLawLightcurve._peak(*shape)

    def excess(log_x: float) -> float:
        """
        Return how far the log of the light curve is above half its peak.

        Parameters
        ----------
        log_x : float
            The natural log of the time since breakout, in units of :math:`t_0`.

        Returns
        -------
        float
            Positive before the half-decline and negative after it.
        """
        return float(InverseGammaBrokenPowerLawLightcurve._log_shape(log_x, *shape) - log_shape_peak - np.log(0.5))

    upper = float(log_peak) + 1.0
    while excess(upper) > 0.0:
        upper += 2.0
        if upper > 60.0:
            raise ValueError("The light curve never falls to half of its peak, so it has no half-decline time.")
    return float(np.exp(log_peak)), float(np.exp(brentq(excess, float(log_peak), upper)))


# ============================================ #
# The transient                                #
# ============================================ #
class _BreakoutTransient(ExtragalacticTransient):
    """
    A shock breakout with a blackbody photosphere, at a constant comoving rate of one event per Mpc^3 per year.

    `BreakoutSampler` places it at a point by fixing the amplitude and the time scale of its SED. It is measured over
    the whole of each exposure, since a breakout can be short compared to one. The class name starts with an underscore
    and the module is not one the CLI imports, which keeps it out of ``full_run.yaml`` and the checks that iterate over
    shipped classes.
    """

    DEFAULT_MODEL = InverseGammaBlackbodySED
    DEFAULT_DURATION = 1 * u.day
    DEFAULT_Z_LIM = 1.0
    HAS_STORED_HORIZON_CURVE = False
    AVERAGE_OVER_EXPOSURE = True

    @property
    def rate(self) -> u.Quantity:
        """~astropy.units.Quantity: One event per cubic megaparsec per year, so counts are volume-times."""
        return 1.0 * u.Mpc**-3 * u.yr**-1

    def rate_shape(self, z: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
        """
        Return the constant rate shape, ones the shape of `z`.

        Parameters
        ----------
        z : float or numpy.ndarray
            Redshift(s).

        Returns
        -------
        float or numpy.ndarray
            Ones, the same shape as `z`.
        """
        z = np.asarray(z)
        shape = np.ones_like(z, dtype=np.float64)
        return shape if z.ndim > 0 else shape.item()


# ============================================ #
# The sampler                                  #
# ============================================ #
class BreakoutSampler(SurveySampler, kind="breakout"):
    r"""
    Count the detections of a shock breakout at points of the peak-luminosity and time-scale plane.

    A point is ``x = (log10 L_0 / (erg/s), log10 t_0 / s)``: the peak bolometric luminosity :math:`L_0` and the time
    scale :math:`t_0` of the light curve, which peaks at about :math:`t_0`. The other parameters of the SED are fixed at
    the defaults of `~uvex_transients.models.shock_cooling.inverse_gamma.InverseGammaBlackbodySED`. One sampler makes
    the map of one band. See the module documentation for what counts as a detection.

    Every argument that is `None` (the default) takes ``config["phase_space.breakout.<name>"]`` or
    ``config["phase_space.recovery.<name>"]``.

    Parameters
    ----------
    schedule : ~uvex_transients.surveys.base.SurveySchedule
        The survey to recover events against.
    mission : str or m4opt.missions.Mission, optional
        The mission (detector and background), or the name of one.
    band : str, optional
        The detector band that recovers events.
    log_L0, log_t0 : sequence of float, optional
        The ranges ``(lo, hi)`` of the axes, which are also what the redshift limit is tabulated over.
    snr_threshold : float, optional
        The signal-to-noise an exposure must reach.
    n_visits : int, optional
        How many exposures must reach it. One, by default, since only the first exposures count.
    exclude_first_visit_detections : bool, optional
        Whether a detection that is only on a field's first visit is discarded; see `SurveySimulator.filter_by_snr`.
    mag_limit : float, optional
        The AB depth that sets the redshift limit of a point. Only events brighter than it are drawn.
    time_bins : int, optional
        Number of equal explosion-time bins spanning the survey; see `SurveySimulator.sample_N_events`. Many, since a
        breakout is short compared to a bin.
    nside, order : optional
        HEALPix resolution of the sampling grid; the package defaults if omitted.
    z_max : float, optional
        The redshift the limit is tabulated to, and beyond which no event is drawn.
    table_points : int, optional
        Points per axis of the table of the redshift limit.
    cosmology : ~astropy.cosmology.Cosmology, optional
        Defaults to the configured cosmology.
    schedule_name : str, optional
        The registry name of `schedule`, recorded in `settings` so that `from_settings` can fetch the same schedule.

    Raises
    ------
    ValueError
        If the shape parameters of the SED are not fixed, which the window of the detection assumes.
    """

    TRANSIENT_KEY = "breakout"

    def __init__(
        self,
        schedule: SurveySchedule,
        mission=None,
        *,
        band: str | None = None,
        log_L0=None,
        log_t0=None,
        snr_threshold: float | None = None,
        n_visits: int | None = None,
        exclude_first_visit_detections: bool | None = None,
        mag_limit: float | None = None,
        time_bins: int | None = None,
        nside: int | None = None,
        order: str | None = None,
        z_max: float | None = None,
        table_points: int | None = None,
        cosmology: Cosmology | None = None,
        schedule_name: str | None = None,
    ):
        super().__init__(
            schedule,
            mission,
            band=band,
            snr_threshold=snr_threshold,
            n_visits=_configured(n_visits, "breakout.n_visits"),
            exclude_first_visit_detections=exclude_first_visit_detections,
            mag_limit=mag_limit,
            time_bins=_configured(time_bins, "breakout.time_bins"),
            nside=nside,
            order=order,
            cosmology=cosmology,
            schedule_name=schedule_name,
        )
        log_L0 = [float(v) for v in _configured(log_L0, "breakout.log_L0")]
        log_t0 = [float(v) for v in _configured(log_t0, "breakout.log_t0")]
        self.z_max = float(_configured(z_max, "breakout.z_max"))
        self.table_points = int(_configured(table_points, "breakout.table_points"))

        self._space = Space(
            [
                Axis("log_L0", *log_L0, label=r"peak luminosity $L_0$", unit="erg/s"),
                Axis("log_t0", *log_t0, label=r"time scale $t_0$", unit="s"),
            ]
        )
        self.parameter_axes = (
            ParameterAxis("log_L0", "amplitude", u.erg / u.s),
            ParameterAxis("log_t0", "t_scale", u.s),
        )
        self._settings = {
            "schedule_name": schedule_name,
            "mission": self._mission_name,
            "band": self.band,
            "log_L0": log_L0,
            "log_t0": log_t0,
            "snr_threshold": self.snr_threshold,
            "n_visits": self.n_visits,
            "exclude_first_visit_detections": self.exclude_first_visit_detections,
            "mag_limit": self.mag_limit,
            "time_bins": self.time_bins,
            "nside": int(self.nside),
            "order": str(self.order),
            "z_max": self.z_max,
            "table_points": self.table_points,
            "cosmology": str(self.cosmology.name),
        }

        # The window is a constant of the shape of the light curve, so the shape has to be fixed
        self.x_peak, self.x_half = half_decline_time(*self._fixed_shape())
        self._log_redshift_table = None
        self._clipped_warning = False
        self._build_scales(self.z_max)

    def _fixed_shape(self) -> tuple[float, float, float, float]:
        """Return the shape parameters of the light curve, ``(p, q, x_break, smoothness)``, which have to be fixed."""
        sed = InverseGammaBlackbodySED()
        values = []
        for name in ("p", "q", "x_break", "smoothness"):
            if not isinstance(sed[name].prior, ConstantPrior):
                raise ValueError(f"The breakout window assumes a fixed shape, but {name!r} is drawn from a prior.")
            values.append(float(np.asarray(to_cgs_value(sed[name].sample(size=1, rng=0))).ravel()[0]))
        return tuple(values)

    # -------------------------------------- #
    # The transient                          #
    # -------------------------------------- #
    def make_transient(self) -> _BreakoutTransient:
        """
        Return a new breakout, not yet placed at a point of the space.

        Returns
        -------
        _BreakoutTransient
            An instance that `_place` can move.
        """
        return _BreakoutTransient(cosmology=self.cosmology)

    def _place(self, x: NDArray[np.float64]) -> None:
        """
        Move the transient to ``(log10 L_0 / (erg/s), log10 t_0 / s)``.

        The amplitude and the time scale are fixed, the redshift limit is the point's, and an event is followed for the
        window out to the half-decline of its light curve, stretched by time dilation at the farthest event.

        Parameters
        ----------
        x : numpy.ndarray
            The point, shape ``(2,)``.
        """
        self._fix_parameters(x)
        z_limit = float(self.redshift_limit(x))
        if z_limit >= self.z_max * (1.0 - 1e-6) and not self._clipped_warning:
            self._clipped_warning = True
            logger.warning(
                "The redshift limit of log10 L_0 = %.2f, log10 t_0 = %.2f reaches z_max = %g, so it is clipped and its "
                "recovered volume-time is a lower bound. Every brighter point is clipped too.",
                x[0],
                x[1],
                self.z_max,
            )
        self._transient.redshift_limit = z_limit
        self._transient.duration_limit = (1.0 + z_limit) * self.x_half * 10.0 ** float(x[1]) * u.s

    def _log_window(self, z: ArrayLike, x: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Return the log10 of the window in days: the half-decline time, stretched by time dilation.

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
        return np.log10(1.0 + z) + np.log10(self.x_half) + x[..., 1] - np.log10(_SECONDS_PER_DAY)

    # -------------------------------------- #
    # The redshift limit                     #
    # -------------------------------------- #
    def _build_redshift_table(self) -> None:
        """
        Tabulate the redshift limit over the axes.

        At each point of a grid in ``(log L_0, log t_0)`` the instantaneous band magnitude of the SED is found at every
        redshift of a grid, at the brightest time of the window (from the start out to the half-decline, at the
        redshift). The limit is the redshift at which that reaches the depth. The SED is evaluated once for each point
        and every redshift is then a change of band weights, which is why the table is cheap.
        """
        sed = InverseGammaBlackbodySED()
        _, band_grids = resolve_bandpasses({self.band: self.mission.detector.bandpasses[self.band]})
        z_grid = np.geomspace(_Z_MIN, self.z_max, _N_REDSHIFT)
        luminosity_distance = resolve_cosmological_distances(redshift=z_grid, cosmology=self.cosmology)[
            "luminosity_distance"
        ].cgs.value
        x_window = np.geomspace(_X_MIN, self.x_half, _N_TIME)

        (lo_l, hi_l), (lo_t, hi_t) = self._space.bounds
        log_l = np.linspace(lo_l, hi_l, self.table_points)
        log_t = np.linspace(lo_t, hi_t, self.table_points)
        constants = {name: to_cgs_value(value) for name, value in sed.sample_parameters(size=len(log_l), rng=0).items()}

        table = np.empty((len(log_l), len(log_t)))
        for j, log_t0 in enumerate(log_t):
            t0 = 10.0**log_t0
            parameters = {**constants, "amplitude": 10.0**log_l, "t_scale": np.full(len(log_l), t0)}
            peak = peak_band_magnitudes(
                sed._eval, parameters, z_grid, luminosity_distance, band_grids, x_window * t0, progress=False
            )[:, 0, :]
            for i in range(len(log_l)):
                # Fainter with distance, up to the wobble of the K-correction, which a limit does not follow
                magnitude = np.maximum.accumulate(peak[i])
                table[i, j] = np.interp(self.mag_limit, magnitude, np.log10(z_grid))

        # Where even the nearest redshift is too faint, or the farthest too bright, `interp` holds the end
        self._log_redshift_table = RegularGridInterpolator((log_l, log_t), table, method="linear")
        self._table_bounds = np.array([[log_l[0], log_l[-1]], [log_t[0], log_t[-1]]])

    def redshift_limit(self, x: ArrayLike) -> NDArray[np.float64]:
        """
        Return the redshift out to which events are drawn at points of the space.

        It is the redshift at which the instantaneous band magnitude, at the brightest time of the window, reaches the
        depth, from a table over the axes that is built on first use. A point outside the axes' ranges takes the limit
        of the nearest point inside.

        Parameters
        ----------
        x : array-like
            Points ``(log10 L_0, log10 t_0)``, shape ``(..., 2)``.

        Returns
        -------
        numpy.ndarray
            The redshift limit of each point, of shape ``(...)``.
        """
        if self._log_redshift_table is None:
            self._build_redshift_table()
        x = self._points(x)
        clipped = np.clip(x, self._table_bounds[:, 0], self._table_bounds[:, 1])
        return 10.0 ** self._log_redshift_table(clipped.reshape(-1, 2)).reshape(x.shape[:-1])
