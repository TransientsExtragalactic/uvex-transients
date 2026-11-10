r"""
Survey-backed samplers: the machinery shared by every sampler that simulates events against a survey schedule.

A `SurveySampler` answers "how many detections in n tries at a point?" by placing a transient at the point, drawing `n`
events volume-weighted out to its redshift limit on the footprint the survey covers, and counting how many
`~uvex_transients.simulation.core.SurveySimulator.filter_by_snr` recovers. What differs between the kinds of map is only
the transient, how a point of the space places it, and the redshift limit, so those are what a subclass supplies. The
rest is here: the detection definition, the survey and its settings, the noise-free scales (the exposure, volume-time
and coverage of a point) and the products made from them.

**Axes and parameters.** A point of the space is turned into the parameters of the transient's SED in one of two ways.
An axis can be bound to a parameter by name with a `ParameterAxis`: the coordinate is then :math:`\log_{10}` of the
parameter in a unit (or the parameter itself, on a linear axis), and the sampler fixes it with no physics code. Or an
axis can be a derived quantity, such as a luminosity in a band, and the subclass places the point itself. The link is
declared here, on the sampler, so that a `~uvex_transients.phase_space.space.Space` stays free of any model.

The recovery fraction of a point is conditional on the footprint that was sampled, which depends on the number of
explosion-time bins. The noise-free volume-time :math:`\mathcal V T` is the expected number of events over the survey in
that footprint at one event per :math:`\mathrm{Mpc}^3\,\mathrm{yr}`. The product :math:`f\,\mathcal V T` does not depend
on the binning, and is the expected number of recoveries at unit rate.
"""

from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from astropy import units as u
from astropy.cosmology import Cosmology
from astropy.units import Quantity
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import cumulative_trapezoid

from uvex_transients.missions import get_mission
from uvex_transients.phase_space.samplers.base import Sampler
from uvex_transients.phase_space.samplers.products import Product, Scale, Weighted
from uvex_transients.simulation.core import SurveySimulator
from uvex_transients.surveys.base import SurveySchedule
from uvex_transients.surveys.utils import get_schedule
from uvex_transients.utils import config, logger, resolve_healpix_resolution
from uvex_transients.utils.cosmology import get_cosmology

__all__ = ["ParameterAxis", "SurveySampler"]


def _configured(value, key: str):
    """Return `value`, or ``config["phase_space.<key>"]`` if it is `None`."""
    return config[f"phase_space.{key}"] if value is None else value


# ============================================ #
# Parameter axes                               #
# ============================================ #
@dataclass(frozen=True)
class ParameterAxis:
    r"""
    The link from one axis of a sampler's space to one parameter of its transient's SED.

    The coordinate on a ``"log10"`` axis is :math:`\log_{10}(\mathrm{value}/\mathrm{unit})`, and on a ``"linear"`` axis
    it is :math:`\mathrm{value}/\mathrm{unit}`.

    Raises
    ------
    ValueError
        If `unit` is not a unit.
    """

    axis: str
    """str: The name of the axis, one of the names of the sampler's space."""

    parameter: str
    """str: The name of the parameter on the SED of the transient, for example ``"amplitude"``."""

    unit: u.UnitBase
    """~astropy.units.UnitBase: The unit the coordinate is measured in. It has to be compatible with the parameter."""

    def __post_init__(self):
        object.__setattr__(self, "unit", u.Unit(self.unit))

    def value(self, coordinate: float, scale: str) -> Quantity:
        """
        Return the parameter value that a coordinate stands for.

        Parameters
        ----------
        coordinate : float
            The coordinate on the axis.
        scale : {"log10", "linear"}
            The scale of the axis, which is how the coordinate is read.

        Returns
        -------
        ~astropy.units.Quantity
            The value of the parameter, in `unit`.
        """
        return (10.0**coordinate if scale == "log10" else coordinate) * self.unit


# ============================================ #
# Survey sampler                               #
# ============================================ #
class SurveySampler(Sampler):
    r"""
    Count the recoveries of a transient simulated against a survey: what every survey-backed sampler shares.

    A subclass is built in its own ``__init__``, which calls ``super().__init__``, sets ``_space`` and ``_settings``,
    and calls `_build_scales`. It supplies `make_transient`, `redshift_limit`, `_place` and `_log_window`. If its axes
    are parameters of the SED, it lists them in ``parameter_axes``, a tuple of `ParameterAxis`, and `_place` calls
    `_fix_parameters` for them and only sets the window and the redshift limit itself.

    Every argument that is `None` (the default) takes ``config["phase_space.recovery.<name>"]``.

    Parameters
    ----------
    schedule : ~uvex_transients.surveys.base.SurveySchedule
        The survey to recover events against.
    mission : str or m4opt.missions.Mission, optional
        The mission (detector and background), or the name of one.
    band : str, optional
        The detector band that recovers events.
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
        The registry name of `schedule` (see `~uvex_transients.surveys.utils.get_schedule`), recorded in `settings` so
        that `from_settings` can fetch the same schedule. Leave it out for a schedule that is not in the registry.

    Notes
    -----
    The tables a sampler interpolates (volume, exposure) are sized by ``config["phase_space.numerics"]``. They set the
    precision of the scales, not what is measured, so they are not recorded in `settings`.
    """

    TRANSIENT_KEY: ClassVar[str] = "transient"
    """str: The name the one transient goes by in the simulator."""

    parameter_axes: tuple[ParameterAxis, ...] = ()
    """tuple of ParameterAxis: The axes that are parameters of the SED, which `parameters_at` uses to place a point."""

    def __init__(
        self,
        schedule: SurveySchedule,
        mission=None,
        *,
        band: str | None = None,
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
        mission = _configured(mission, "recovery.mission")
        self.schedule = schedule
        self.mission = get_mission(mission) if isinstance(mission, str) else mission
        self.band = str(_configured(band, "recovery.band"))
        if self.band not in self.mission.detector.bandpasses:
            raise ValueError(f"Unknown band {self.band!r}; the detector has {list(self.mission.detector.bandpasses)}.")

        self.snr_threshold = float(_configured(snr_threshold, "recovery.snr_threshold"))
        self.n_visits = int(_configured(n_visits, "recovery.n_visits"))
        self.exclude_first_visit_detections = bool(
            _configured(exclude_first_visit_detections, "recovery.exclude_first_visit_detections")
        )
        self.mag_limit = float(_configured(mag_limit, "recovery.mag_limit"))
        self.time_bins = int(_configured(time_bins, "recovery.time_bins"))
        self.nside, self.order = resolve_healpix_resolution(nside, order)
        self.cosmology = get_cosmology(cosmology)
        self._exposure_spacing = float(config["phase_space.numerics.exposure_spacing"])

        # What a subclass records in `settings`, as plain types that an ECSV header can hold
        self._mission_name = mission if isinstance(mission, str) else None
        self._schedule_name = schedule_name

    def _build_scales(self, redshift_edge: float) -> None:
        """
        Build what the noise-free scales and the simulation need, once a subclass has set ``_space`` and its limit.

        Parameters
        ----------
        redshift_edge : float
            The redshift out to which the volume is tabulated, and beyond which no event is drawn.
        """
        self.redshift_edge = float(redshift_edge)

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
        self._survey_years = float(self.schedule.duration.to_value(u.yr))

        # The instance that is moved from point to point, and a second one only for tabulating exposures
        self._transient = self.make_transient()
        self._probe = self.make_transient()
        self._exposure_nodes: dict[float, float] = {}
        self._warned_clipped = False
        self._check_parameter_axes()

    # -------------------------------------- #
    # What a subclass supplies               #
    # -------------------------------------- #
    @abstractmethod
    def make_transient(self):
        """
        Return a new transient of this sampler, not yet placed at a point of the space.

        Returns
        -------
        ~uvex_transients.transients.base.ExtragalacticTransient
            An instance that `_place` can move.
        """

    @abstractmethod
    def redshift_limit(self, x: ArrayLike) -> NDArray[np.float64]:
        """
        Return the redshift out to which events are drawn at points of the space.

        Parameters
        ----------
        x : array-like
            Points in the coordinates of `space`, shape ``(..., ndim)``.

        Returns
        -------
        numpy.ndarray
            The redshift limit of each point, of shape ``(...)``.
        """

    @abstractmethod
    def _place(self, x: NDArray[np.float64]) -> None:
        """
        Move the transient to a point of the space.

        This sets the parameters of the SED (`_fix_parameters` does it for the axes that are bound to a parameter), the
        redshift limit of the transient and its ``duration_limit``, the window of observations that can see an event.

        Parameters
        ----------
        x : numpy.ndarray
            The point, in the coordinates of `space`, shape ``(ndim,)``.
        """

    @abstractmethod
    def _log_window(self, z: ArrayLike, x: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Return the log10 of the window, in days, that an event is followed for, for the farthest events of points.

        Parameters
        ----------
        z : array-like
            The redshift limit of each point.
        x : numpy.ndarray
            The points, shape ``(..., ndim)``.

        Returns
        -------
        numpy.ndarray
            :math:`\log_{10}` of the window in days, in the observer frame, of the shape of `z`.
        """

    # -------------------------------------- #
    # Axes and parameters                    #
    # -------------------------------------- #
    def parameters_at(self, x: ArrayLike) -> dict[str, Quantity]:
        """
        Return the SED parameters that a point of the space sets, from the axes bound to a parameter.

        Parameters
        ----------
        x : array-like
            The point, in the coordinates of `space`, shape ``(ndim,)``.

        Returns
        -------
        dict of str to ~astropy.units.Quantity
            The value of each bound parameter at the point.
        """
        x = self._points(x)
        return {
            binding.parameter: binding.value(float(x[self.space.index(binding.axis)]), self.space[binding.axis].scale)
            for binding in self.parameter_axes
        }

    def _fix_parameters(self, x: ArrayLike) -> None:
        """
        Fix the bound parameters of the transient's SED at a point, leaving every other parameter to its prior.

        Parameters
        ----------
        x : array-like
            The point, in the coordinates of `space`, shape ``(ndim,)``.
        """
        for name, value in self.parameters_at(x).items():
            self._transient.sed[name].fix(value)

    def _check_parameter_axes(self) -> None:
        """
        Check that every bound axis and parameter exists, and that the units agree.

        Raises
        ------
        ValueError
            If a binding names an axis that is not in the space, a parameter that is not on the SED, or a unit that does
            not match the parameter's, or two bindings name the same axis or parameter.
        """
        seen_axes, seen_parameters = set(), set()
        for binding in self.parameter_axes:
            if binding.axis not in self.space.names:
                raise ValueError(
                    f"Parameter axis {binding.axis!r} is not an axis of the space {list(self.space.names)}."
                )
            if binding.parameter not in self._transient.sed:
                raise ValueError(
                    f"Parameter axis {binding.axis!r} is bound to {binding.parameter!r}, which is not a parameter "
                    f"of the SED; it has {list(self._transient.sed)}."
                )
            if binding.axis in seen_axes or binding.parameter in seen_parameters:
                raise ValueError(
                    f"Parameter axis {binding.axis!r} -> {binding.parameter!r} repeats an axis or a parameter."
                )
            seen_axes.add(binding.axis)
            seen_parameters.add(binding.parameter)

            scale = self._transient.sed[binding.parameter].scale
            parameter_unit = u.Unit(getattr(scale, "unit", u.dimensionless_unscaled))
            if not binding.unit.is_equivalent(parameter_unit):
                raise ValueError(
                    f"Parameter axis {binding.axis!r} is in {binding.unit}, which is not a unit of "
                    f"{binding.parameter!r} ({parameter_unit})."
                )

    # -------------------------------------- #
    # Description                            #
    # -------------------------------------- #
    @property
    def space(self):
        """~uvex_transients.phase_space.space.Space: The axes of the sampler, with their default ranges."""
        return self._space

    @property
    def settings(self) -> dict:
        """
        dict: What `from_settings` needs to rebuild this sampler, as plain types that an ECSV header can hold.

        The mission and schedule are recorded by name. A mission given as an object, or a schedule given without a
        ``schedule_name``, is recorded as `None` and has to be supplied again to `from_settings`.
        """
        return dict(self._settings)

    @classmethod
    def _rebuild_arguments(
        cls,
        settings: Mapping,
        schedule: SurveySchedule | None,
        mission,
        cosmology: Cosmology | None,
    ) -> tuple[dict, SurveySchedule, object, str | None]:
        """
        Resolve what `from_settings` takes from outside the recorded settings: the schedule, mission and cosmology.

        Parameters
        ----------
        settings : Mapping
            The recorded settings.
        schedule, mission, cosmology : optional
            As for `from_settings`.

        Returns
        -------
        settings : dict
            The settings without the entries that are resolved here.
        schedule : ~uvex_transients.surveys.base.SurveySchedule
            The schedule, fetched by name if it was not given.
        mission : str or m4opt.missions.Mission
            The mission, from the argument or the settings.
        schedule_name : str or None
            The recorded registry name of the schedule.

        Raises
        ------
        ValueError
            If no schedule or no mission can be found.
        """
        settings = dict(settings)
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
        return settings, schedule, mission, schedule_name

    def _points(self, x: ArrayLike) -> NDArray[np.float64]:
        """Return `x` as floats, after checking that its last axis runs over the axes of the space."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 0 or x.shape[-1] != self.space.ndim:
            raise ValueError(
                f"Points of this space have {self.space.ndim} coordinates, got an array of shape {x.shape}."
            )
        return x

    # -------------------------------------- #
    # The tries                              #
    # -------------------------------------- #
    def draw(self, x: ArrayLike, n: int, seed: np.random.SeedSequence) -> int:
        """
        Simulate `n` events at a point and return how many the survey recovers.

        Parameters
        ----------
        x : array-like
            The point, in the coordinates of `space`, shape ``(ndim,)``.
        n : int
            The number of events.
        seed : numpy.random.SeedSequence
            The source of the randomness of the simulation.

        Returns
        -------
        int
            The number of events recovered, between 0 and `n`.

        Raises
        ------
        RuntimeError
            If the simulation does not return `n` events, which would make the count mean something else.
        """
        self._place(self._points(x))
        simulator = SurveySimulator(self.schedule, {self.TRANSIENT_KEY: self._transient}, simulation_seed=seed)
        catalog = simulator.sample_N_events(int(n), self.time_bins, nside=self.nside, order=self.order)
        if len(catalog) != n:
            raise RuntimeError(f"Asked the simulator for {n} events and got {len(catalog)}.")
        recovered = simulator.filter_by_snr(
            catalog,
            self.mission,
            self.snr_threshold,
            bands=[self.band],
            n_visits=self.n_visits,
            exclude_first_visit_detections=self.exclude_first_visit_detections,
        )
        return len(recovered)

    # -------------------------------------- #
    # Absolute scale                         #
    # -------------------------------------- #
    def _exposure_at(self, log_window: float) -> float:
        """Return the effective exposure of the survey, in sr yr, for a window of length 10**log_window days."""
        key = round(float(log_window), 6)
        if key not in self._exposure_nodes:
            self._probe.duration_limit = 10.0**key * u.day
            simulator = SurveySimulator(self.schedule, {self.TRANSIENT_KEY: self._probe})
            catalog = simulator.compute_effective_exposure(self.time_bins, nside=self.nside, order=self.order)
            self._exposure_nodes[key] = catalog.total_effective_exposure[self.TRANSIENT_KEY].to_value(u.sr * u.yr)
        return self._exposure_nodes[key]

    def _exposure(self, x: ArrayLike) -> tuple[NDArray, NDArray]:
        """
        Return the redshift limit and the footprint exposure of points of the space.

        The exposure depends on a point only through the window length over which an event is followed, so it is
        tabulated on a grid of that length and interpolated.

        Parameters
        ----------
        x : array-like
            Points in the coordinates of `space`, shape ``(..., ndim)``.

        Returns
        -------
        redshift_limit, exposure : numpy.ndarray
            The redshift limit of each point, and the effective exposure of the survey for it in sr yr.
        """
        x = self._points(x)
        z = self.redshift_limit(x)
        log_window = self._log_window(z, x)

        # Tabulate the exposure on nodes that bracket the windows asked for
        spacing = self._exposure_spacing
        low = np.floor(log_window.min() / spacing) * spacing
        high = np.ceil(log_window.max() / spacing) * spacing + spacing
        nodes = np.round(np.arange(low, high + 0.5 * spacing, spacing), 6)
        exposure = np.interp(log_window, nodes, [self._exposure_at(node) for node in nodes])
        return z, exposure

    def volume_time(self, x: ArrayLike) -> NDArray[np.float64]:
        r"""
        Return the effective volume-time :math:`\mathcal V T` of points of the space, in Mpc^3 yr.

        The expected number of events over the survey, inside the footprint and out to the redshift limit, at one
        event per Mpc^3 per yr. It involves no Monte Carlo: the exposure comes from the schedule. Multiply by a rate
        and by the recovery fraction for the expected number of recoveries.

        Parameters
        ----------
        x : array-like
            Points in the coordinates of `space`, shape ``(..., ndim)``.

        Returns
        -------
        numpy.ndarray
            :math:`\mathcal V T`, of shape ``(...)``.
        """
        z, exposure = self._exposure(x)
        return exposure * np.interp(z, self._z_grid, self._volume)

    def coverage(self, x: ArrayLike) -> NDArray[np.float64]:
        r"""
        Return the fraction of the sky and survey window that the sampled footprint covers, for points of the space.

        The effective exposure over that of the whole sky for the whole survey, :math:`\mathcal E/(4\pi\,T)`. It is the
        ratio of the events in the sampled footprint to all events exploding during the survey, so multiplying the
        recovery fraction :math:`f` by it gives the efficiency.

        Parameters
        ----------
        x : array-like
            Points in the coordinates of `space`, shape ``(..., ndim)``.

        Returns
        -------
        numpy.ndarray
            The coverage, between 0 and 1, of shape ``(...)``.
        """
        _, exposure = self._exposure(x)
        return exposure / (4.0 * np.pi * self._survey_years)

    def all_sky_volume_time(self, x: ArrayLike) -> NDArray[np.float64]:
        r"""
        Return the volume-time of the whole sky over the whole survey out to the redshift limit, in Mpc^3 yr.

        The number of events exploding during the survey anywhere on the sky, at one per Mpc^3 per yr. It is
        `volume_time` divided by `coverage`.

        Parameters
        ----------
        x : array-like
            Points in the coordinates of `space`, shape ``(..., ndim)``.

        Returns
        -------
        numpy.ndarray
            The all-sky volume-time, of shape ``(...)``.
        """
        z = self.redshift_limit(self._points(x))
        return 4.0 * np.pi * self._survey_years * np.interp(z, self._z_grid, self._volume)

    @property
    def products(self) -> tuple[Product, ...]:
        """
        Tuple of Product: The quantities of a map of this sampler beyond the success fraction.

        The redshift limit, coverage, volume-time and all-sky volume-time of each point, and the efficiency (the
        success fraction times the coverage) and the recovered volume-time (the fraction times the volume-time).
        """
        return (
            Scale("z_limit", self.redshift_limit, label=r"$z_\mathrm{lim}$"),
            Scale("coverage", self.coverage, label="coverage"),
            Scale("volume_time", self.volume_time, label=r"$\mathcal{V}T$", unit="Mpc3 yr"),
            Scale("all_sky_volume_time", self.all_sky_volume_time, label=r"all-sky $\mathcal{V}T$", unit="Mpc3 yr"),
            Weighted("efficiency", self.coverage, label=r"efficiency $\eta$"),
            Weighted("recovered_volume_time", self.volume_time, label=r"$f\,\mathcal{V}T$", unit="Mpc3 yr"),
        )
