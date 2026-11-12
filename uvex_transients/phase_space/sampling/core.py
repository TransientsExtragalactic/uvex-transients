r"""
The generic samplers: a toy with a known answer, the survey-backed machinery, and the canonical sampler.

The shock breakout sampler, which is built on `SurveySampler`, is in `~uvex_transients.phase_space.sampling.breakout`.

A sampler with a known answer, for testing the machinery that sits on it.

`LogisticSampler` succeeds with a probability that is a logistic function of the point, :math:`p(x)=\mathrm{expit}
[\sum_i s_i (x_i - c_i)]`, so the success fraction of a map of it is known exactly. That is what lets the map, the
acquisition and the campaign be tested against the truth without running a survey simulation.

Survey-backed samplers: the machinery shared by every sampler that simulates events against a survey schedule.

A `SurveySampler` answers "how many detections in n tries at a point?" by placing a transient at the point, drawing `n`
events volume-weighted out to its redshift limit on the footprint the survey covers, and counting how many
`~uvex_transients.simulation.core.SurveySimulator.filter_by_snr` recovers. What differs between the kinds of map is only
the transient, how a point of the space places it, and the redshift limit, so those are what a subclass supplies. The
rest is here: the detection definition, the survey and its settings, the noise-free scales (the exposure, volume-time
and coverage of a point) and the products made from them.

**Axes and parameters.** A point of the space is turned into the parameters of the transient"s SED in one of two ways.
An axis can be bound to a parameter by name with a `ParameterAxis`: the coordinate is then :math:`\log_{10}` of the
parameter in a unit (or the parameter itself, on a linear axis), and the sampler fixes it with no physics code. Or an
axis can be a derived quantity, such as a luminosity in a band, and the subclass places the point itself. The link is
declared here, on the sampler, so that a `~uvex_transients.phase_space.space.Space` stays free of any model.

The recovery fraction of a point is conditional on the footprint that was sampled, which depends on the number of
explosion-time bins. The noise-free volume-time :math:`\mathcal V T` is the expected number of events over the survey in
that footprint at one event per :math:`\mathrm{Mpc}^3\,\mathrm{yr}`. The product :math:`f\,\mathcal V T` does not depend
on the binning, and is the expected number of recoveries at unit rate.

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

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from astropy import units as u
from astropy.cosmology import Cosmology
from astropy.units import Quantity
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import cumulative_trapezoid
from scipy.special import expit

from uvex_transients.missions import get_mission
from uvex_transients.phase_space.sampling.base import Product, Sampler, Scale, Weighted
from uvex_transients.phase_space.space import Axis, Space
from uvex_transients.phase_space.transient import FREQUENCY_MAX, _CanonicalTransient
from uvex_transients.simulation.core import SurveySimulator
from uvex_transients.surveys.base import SurveySchedule
from uvex_transients.surveys.utils import get_schedule
from uvex_transients.utils import config, logger, resolve_healpix_resolution
from uvex_transients.utils.cosmology import get_cosmology
from uvex_transients.utils.phot_utils import effective_frequency, equivalent_bandwidth

__all__ = ["CanonicalSampler", "LogisticSampler", "ParameterAxis", "RedshiftLimit", "SurveySampler"]


# ============================================ #
# A sampler with a known answer                #
# ============================================ #


class LogisticSampler(Sampler, kind="logistic"):
    r"""
    Succeed with probability :math:`p(x)=\mathrm{expit}[\sum_i s_i (x_i - c_i)]`.

    The boundary :math:`p=1/2` is the plane through `center` that is perpendicular to `slope`, and `slope` sets how
    sharp it is.

    Parameters
    ----------
    space : Space
        The axes of the experiment.
    center : sequence of float
        A point on the boundary, one coordinate per axis.
    slope : sequence of float
        How fast the logit of the probability rises along each axis.

    Raises
    ------
    ValueError
        If `center` or `slope` does not have one entry per axis.
    """

    def __init__(self, space: Space, center: Sequence[float], slope: Sequence[float]):
        center = np.asarray(center, dtype=np.float64)
        slope = np.asarray(slope, dtype=np.float64)
        if center.shape != (space.ndim,) or slope.shape != (space.ndim,):
            raise ValueError(
                f"`center` and `slope` need one entry for each of the {space.ndim} axes, "
                f"got shapes {center.shape} and {slope.shape}."
            )
        self._space = space
        self._center = center
        self._slope = slope

    @property
    def space(self) -> Space:
        """Space: The axes of the experiment."""
        return self._space

    def probability(self, x: ArrayLike) -> NDArray[np.float64]:
        """
        Return the probability of success at points of the space.

        Parameters
        ----------
        x : array-like
            Points in the coordinates of the space, shape ``(..., ndim)``.

        Returns
        -------
        numpy.ndarray
            The probability, of shape ``(...)``.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 0 or x.shape[-1] != self._space.ndim:
            raise ValueError(f"Points of this space have {self._space.ndim} coordinates, got shape {x.shape}.")
        return expit(np.sum(self._slope * (x - self._center), axis=-1))

    def draw(self, x: ArrayLike, n: int, seed: np.random.SeedSequence) -> int:
        """
        Make `n` tries at a point.

        Parameters
        ----------
        x : array-like
            The point, shape ``(ndim,)``.
        n : int
            The number of tries.
        seed : numpy.random.SeedSequence
            The source of the randomness.

        Returns
        -------
        int
            The number of successes.
        """
        return int(np.random.default_rng(seed).binomial(int(n), float(self.probability(x))))

    @property
    def products(self) -> tuple[Product, ...]:
        """Tuple of Product: The true probability, which a fitted map can be compared to."""
        return (Scale("true_probability", self.probability, label=r"true $p$"),)

    @property
    def settings(self) -> dict:
        """dict: The space, `center` and `slope`, as plain types."""
        return {"space": self._space.to_dict(), "center": self._center.tolist(), "slope": self._slope.tolist()}

    @classmethod
    def from_settings(cls, settings: Mapping, **context) -> "LogisticSampler":
        """
        Rebuild a sampler from `settings`.

        Parameters
        ----------
        settings : Mapping
            The `settings` of the sampler to rebuild.
        **context
            Ignored: this sampler needs nothing a file cannot record.

        Returns
        -------
        LogisticSampler
            A sampler with the same space, `center` and `slope`.
        """
        return cls(Space.from_dict(settings["space"]), settings["center"], settings["slope"])


# ============================================ #
# Survey-backed samplers                       #
# ============================================ #


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
    @classmethod
    def from_settings(
        cls,
        settings: Mapping,
        *,
        schedule: SurveySchedule | None = None,
        mission=None,
        cosmology: Cosmology | None = None,
        **overrides,
    ) -> "SurveySampler":
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
        SurveySampler
            A sampler that makes the same measurement, and so reproduces the scales of the recorded one.

        Raises
        ------
        ValueError
            If no schedule or no mission can be found.
        """
        settings, schedule, mission, schedule_name = cls._rebuild_arguments(settings, schedule, mission, cosmology)
        settings.update(overrides)
        return cls(schedule, mission, cosmology=cosmology, schedule_name=schedule_name, **settings)

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

    def rate_products(self, rate: Quantity) -> tuple[Product, ...]:
        r"""
        Return the products that need a volumetric rate: the expected number of recoveries over the survey.

        This is the recovered volume-time times a rate that is constant in redshift, for a population of the luminosity
        and duration of the point.

        Parameters
        ----------
        rate : ~astropy.units.Quantity
            The comoving volumetric rate, with units of inverse volume per time, for example ``10 / u.Gpc**3 / u.yr``.

        Returns
        -------
        tuple of Product
            ``expected_recoveries``, with its interval.
        """
        per_mpc3_yr = u.Quantity(rate).to_value(u.Mpc**-3 * u.yr**-1)
        return (
            Weighted(
                "expected_recoveries",
                lambda x: per_mpc3_yr * self.volume_time(x),
                label="expected recoveries",
            ),
        )

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


# ============================================ #
# The canonical sampler                        #
# ============================================ #

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
