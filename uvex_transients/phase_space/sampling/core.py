r"""
Survey-backed samplers: the machinery shared by every sampler that simulates events against a survey schedule.

A `SurveySampler` answers "how many detections in n tries at a point?" by placing a transient at the point, drawing `n`
events volume-weighted out to its redshift limit on the footprint the survey covers, and counting how many
:meth:`~uvex_transients.simulation.core.SurveySimulator.filter_by_snr` recovers. What differs between the kinds of map
is only the transient, how a point of the space places it, and the redshift limit, so those are what a subclass
supplies. The rest is here: the detection definition, the survey and its settings, the noise-free scales (the exposure,
volume-time and coverage of a point) and the products made from them.

**Axes and parameters.** A point of the space is turned into the parameters of the transient's SED in one of two ways.
An axis can be bound to a parameter by name with a `ParameterAxis`: the coordinate is then :math:`\log_{10}` of the
parameter in a unit (or the parameter itself, on a linear axis), and the sampler fixes it with no physics code. Or an
axis can be a derived quantity, such as a luminosity in a band, and the subclass places the point itself. The link is
declared here, on the sampler, so that a :class:`~uvex_transients.phase_space.space.Space` stays free of any model.

**What counts as a detection.** By default a recovery is at least `n_visits` observations above `snr_threshold` in the
band (:meth:`~uvex_transients.simulation.core.SurveySimulator.filter_by_snr`). Any other definition is a `detection`: an
ordered list of the simulator's registered cuts, the same ones a run-config's ``steps:`` use, applied to the events a
point draws, and the events that survive them are the recoveries. A subclass whose rule is not a sequence of cuts
overrides `SurveySampler.detect`. The noise-free scales do not depend on the rule, only on the window events are
followed for, so a rule that needs observations beyond that window has to be given a longer one.

The recovery fraction of a point is conditional on the footprint that was sampled, which depends on the number of
explosion-time bins. The noise-free volume-time :math:`\mathcal V T` is the expected number of events over the survey in
that footprint at one event per :math:`\mathrm{Mpc}^3\,\mathrm{yr}`. The product :math:`f\,\mathcal V T` does not depend
on the binning, and is the expected number of recoveries at unit rate.

The samplers built on it are in their own modules: :mod:`~uvex_transients.phase_space.sampling.canonical` (the canonical
map) and :mod:`~uvex_transients.phase_space.sampling.breakout` (shock breakout).
"""

import inspect
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

from uvex_transients.missions import get_mission
from uvex_transients.phase_space._config import configured, default
from uvex_transients.phase_space.sampling.base import Product, Sampler, Scale, Weighted
from uvex_transients.phase_space.space import Space
from uvex_transients.simulation.core import SurveySimulator
from uvex_transients.simulation.event_catalog import EventCatalog
from uvex_transients.surveys.base import SurveySchedule
from uvex_transients.surveys.utils import get_schedule
from uvex_transients.utils import config, logger, resolve_healpix_resolution
from uvex_transients.utils.cosmology import get_cosmology

__all__ = ["ParameterAxis", "SurveySampler", "SurveySettings", "check_detection", "describe_detection"]


# ============================================ #
# What counts as a detection                   #
# ============================================ #
def _cut_parameters(function) -> tuple[str, ...]:
    """Return the names of the parameters of a cut, after the simulator, the catalog and the mission."""
    return tuple(inspect.signature(function).parameters)[3:]


def _plain(value):
    """Return `value` as plain types (numbers, strings, booleans, `None`, lists and mappings of them) or raise."""
    # Subclasses of the builtin types (a value read from YAML, say) are converted, since a header cannot hold them
    if isinstance(value, np.generic):
        return value.item()
    if value is None:
        return None
    for kind in (bool, int, float, str):
        if isinstance(value, kind):
            return kind(value)
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    raise TypeError(f"{value!r} is not a plain value (number, string, boolean, list or mapping).")


def check_detection(detection: Sequence[Mapping] | None) -> tuple[dict, ...] | None:
    """
    Check a definition of a detection and return it as plain types.

    A definition is a list of cuts. Each entry is a mapping with the name of a cut registered on
    :class:`~uvex_transients.simulation.core.SurveySimulator` as ``cut``, and the arguments of that cut as ``params``,
    as in the ``steps:`` of a run-config::

        [
            {"cut": "snr", "params": {"n_visits": 2}},
            {
                "cut": "time_since_last_nondetection",
                "params": {"max_delay": 1.0},
            },
        ]

    Parameters
    ----------
    detection : sequence of Mapping or None
        The entries, applied in order, or `None` for the sampler's default.

    Returns
    -------
    tuple of dict or None
        The entries as plain dicts (``params`` left out when empty), or `None`.

    Raises
    ------
    ValueError
        If `detection` is empty or not a list of mappings, an entry has a key other than ``cut`` and ``params``, a cut
        is not registered, or a cut does not take one of its ``params``.
    """
    if detection is None:
        return None
    if isinstance(detection, (str, Mapping)) or not isinstance(detection, Sequence):
        raise ValueError(f"`detection` must be a list of cuts, each with a 'cut' name, got {detection!r}.")
    if not len(detection):
        raise ValueError("`detection` needs at least one cut; leave it out for the default.")

    steps = []
    for index, step in enumerate(detection):
        where = f"detection[{index}]"
        if not isinstance(step, Mapping) or "cut" not in step:
            raise ValueError(f"{where}: an entry is a mapping with a 'cut' name and optional 'params', got {step!r}.")
        unknown = sorted(set(step) - {"cut", "params"})
        if unknown:
            raise ValueError(f"{where}: unknown key(s) {unknown}; an entry has 'cut' and optionally 'params'.")
        name = str(step["cut"])
        try:
            function = SurveySimulator.get_cut(name)
        except ValueError as error:
            raise ValueError(f"{where}: {error}") from error

        params = step.get("params") or {}
        if not isinstance(params, Mapping):
            raise ValueError(f"{where}: 'params' must be a mapping, got {params!r}.")
        accepted = _cut_parameters(function)
        unknown = sorted(set(params) - set(accepted))
        if unknown:
            raise ValueError(f"{where}: the cut {name!r} takes no parameter(s) {unknown}; it takes {list(accepted)}.")
        try:
            entry = {"cut": name, **({"params": _plain(params)} if params else {})}
        except TypeError as error:
            raise ValueError(f"{where}: {error}") from error
        steps.append(entry)
    return tuple(steps)


def describe_detection(settings: Mapping) -> str:
    """
    Describe what a map counts as a detection, in a few words.

    Parameters
    ----------
    settings : Mapping
        The settings of a survey-backed sampler, with ``snr_threshold`` and ``n_visits`` and, if it was given a
        definition of a detection, ``detection``.

    Returns
    -------
    str
        For the default, the threshold and the number of visits. For a list of cuts, the threshold and the cuts, in
        order, each with the arguments it was given.
    """
    steps = settings.get("detection")
    if not steps:
        return f"SNR >= {settings['snr_threshold']:g} on {settings['n_visits']} visit(s)"
    parts = []
    for step in steps:
        params = ", ".join(f"{key}={value}" for key, value in step.get("params", {}).items())
        parts.append(f"{step['cut']}({params})" if params else step["cut"])
    return f"SNR >= {settings['snr_threshold']:g}: " + " + ".join(parts)


# ============================================ #
# Survey settings                              #
# ============================================ #
@dataclass(frozen=True)
class SurveySettings:
    """
    What every survey sampler decides about a recovery and its simulation, resolved against the configured defaults.

    This is the one place the defaults of these settings are applied and validated: a sampler, the run-config and the
    dry run all go through `resolve` and `default`. The fields are the keys of the ``detection:`` (and its ``cuts``)
    and ``simulation:`` sections of a run-config, and of the same sections of the package configuration.
    """

    band: str
    """str: The detector band that recovers the events and defines the luminosity."""

    snr_threshold: float
    """float: The signal-to-noise a visit must reach."""

    n_visits: int
    """int: How many visits must reach it."""

    exclude_first_visit_detections: bool
    """bool: Whether a detection that is only on a field's first visit is discarded."""

    mag_limit: float
    """float: The AB depth that sets the redshift limit of a point."""

    time_bins: int
    """int: Number of equal explosion-time bins spanning the survey."""

    nside: int
    """int: HEALPix resolution of the sampling grid."""

    order: str
    """str: HEALPix ordering of the sampling grid."""

    detection: tuple[dict, ...] | None = None
    """tuple of dict or None: The cuts that define a recovery (see `check_detection`), or `None` for the default."""

    #: The shared section of the configuration each setting is a default of
    SECTIONS: ClassVar[dict[str, str]] = {
        "band": "detection",
        "snr_threshold": "detection",
        "n_visits": "detection",
        "exclude_first_visit_detections": "detection",
        "mag_limit": "simulation",
        "time_bins": "simulation",
    }

    @classmethod
    def default(cls, key: str, kinds: Sequence[str] = ()):
        """
        Return the configured default of one of the settings for a kind of sampler.

        Parameters
        ----------
        key : str
            One of ``SECTIONS``.
        kinds : sequence of str, optional
            The kinds of the sampler, most specific first (see `Sampler.kinds`).

        Returns
        -------
        object
            The value in the section of the first kind that has the key, else in the shared section.
        """
        return default(cls.SECTIONS[key], key, kinds)

    @classmethod
    def resolve(cls, kinds: Sequence[str] = (), **given) -> "SurveySettings":
        """
        Build the settings, taking every one that is not given (or is `None`) from the configuration.

        Parameters
        ----------
        kinds : sequence of str, optional
            The kinds of the sampler, whose own sections of the configuration take precedence.
        **given
            Any of the fields, and ``detection`` as a list of cuts. `nside` and `order` default to the package's
            HEALPix resolution.

        Returns
        -------
        SurveySettings
            The settings, as plain types.

        Raises
        ------
        TypeError
            If a keyword is not one of the settings.
        ValueError
            If ``detection`` is not a valid list of cuts.
        """
        unknown = sorted(set(given) - set(cls.SECTIONS) - {"nside", "order", "detection"})
        if unknown:
            raise TypeError(f"Unknown survey setting(s) {unknown}.")
        values = {name: given.get(name) for name in cls.SECTIONS}
        values = {name: cls.default(name, kinds) if value is None else value for name, value in values.items()}
        nside, order = resolve_healpix_resolution(given.get("nside"), given.get("order"))
        return cls(
            band=str(values["band"]),
            snr_threshold=float(values["snr_threshold"]),
            n_visits=int(values["n_visits"]),
            exclude_first_visit_detections=bool(values["exclude_first_visit_detections"]),
            mag_limit=float(values["mag_limit"]),
            time_bins=int(values["time_bins"]),
            nside=int(nside),
            order=str(order),
            detection=check_detection(given.get("detection")),
        )

    def as_dict(self) -> dict:
        """
        Return the settings a map records, as plain types.

        Returns
        -------
        dict
            Every field but ``detection``, which a sampler records only when it was given one.
        """
        return {name: getattr(self, name) for name in (*self.SECTIONS, "nside", "order")}


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

    A subclass is built in its own ``__init__``, which calls ``super().__init__`` and then `_configure` once, with its
    space, the redshift out to which the volume is tabulated and the settings it adds to the ones recorded here. It
    supplies `make_transient`, `redshift_limit`, `_place` and `_log_window`. If its axes are parameters of the SED, it
    lists them in ``parameter_axes``, a tuple of `ParameterAxis`, and `_place` calls `_fix_parameters` for them and only
    sets the window and the redshift limit itself.

    Every setting that is `None` (the default) takes its default from the configuration, which is resolved once by
    `SurveySettings`: ``config["phase_space.<kind>.<name>"]`` if the kind of the sampler (or a kind it is built on) has
    the key, else ``config["phase_space.detection.<name>"]`` or ``config["phase_space.simulation.<name>"]``. A subclass
    takes these as ``**survey`` and passes them on, so a new survey setting is added in this one place.

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
    detection : sequence of Mapping, optional
        What counts as a recovery, as a list of the simulator's registered cuts (see `check_detection`). By default
        one ``snr`` cut, which is the three settings above. A cut that takes `snr_threshold`, `n_visits`,
        `exclude_first_visit_detections` or ``bands`` and is not given them gets the sampler's, with ``bands`` the
        band of the map. A rule that is not a list of cuts is written by overriding `detect`.
    mag_limit : float, optional
        The AB depth that sets the redshift limit of a point. Only events brighter than it are drawn.
    time_bins : int, optional
        Number of equal explosion-time bins spanning the survey; see `SurveySimulator.sample_N_events`.
    nside, order : optional
        HEALPix resolution of the sampling grid; the package defaults if omitted.
    cosmology : ~astropy.cosmology.Cosmology, optional
        Defaults to the configured cosmology.
    schedule_name : str, optional
        The registry name of `schedule` (see :func:`~uvex_transients.surveys.utils.get_schedule`), recorded in
        `settings` so that `from_settings` can fetch the same schedule. Leave it out for a schedule that is not in the
        registry.

    Notes
    -----
    The tables a sampler interpolates (volume, exposure) are sized by ``config["phase_space.numerics"]``. They set the
    precision of the scales, not what is measured, so they are not recorded in `settings`.
    """

    TRANSIENT_KEY: ClassVar[str] = "transient"
    """str: The name the one transient goes by in the simulator."""

    yield_weight = "volume_time"

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
        detection: Sequence[Mapping] | None = None,
        mag_limit: float | None = None,
        time_bins: int | None = None,
        nside: int | None = None,
        order: str | None = None,
        cosmology: Cosmology | None = None,
        schedule_name: str | None = None,
    ):
        mission = configured(mission, "mission")
        self.schedule = schedule
        self.mission = get_mission(mission) if isinstance(mission, str) else mission
        self.survey = SurveySettings.resolve(
            self.kinds(),
            band=band,
            snr_threshold=snr_threshold,
            n_visits=n_visits,
            exclude_first_visit_detections=exclude_first_visit_detections,
            detection=detection,
            mag_limit=mag_limit,
            time_bins=time_bins,
            nside=nside,
            order=order,
        )
        """SurveySettings: What counts as a recovery and how a point is simulated, with the defaults applied."""
        if self.survey.band not in self.mission.detector.bandpasses:
            raise ValueError(
                f"Unknown band {self.survey.band!r}; the detector has {list(self.mission.detector.bandpasses)}."
            )

        # The settings are attributes too, which is how the sampler and its subclasses read them
        self.band = self.survey.band
        self.snr_threshold = self.survey.snr_threshold
        self.n_visits = self.survey.n_visits
        self.exclude_first_visit_detections = self.survey.exclude_first_visit_detections
        self.detection = self.survey.detection
        """tuple of dict or None: The cuts that define a recovery, or `None` for the default (see `detect`)."""
        self.mag_limit = self.survey.mag_limit
        self.time_bins = self.survey.time_bins
        self.nside, self.order = self.survey.nside, self.survey.order
        self.cosmology = get_cosmology(cosmology)
        self._exposure_spacing = float(config["phase_space.numerics.exposure_spacing"])

        # What `settings` records, as plain types that an ECSV header can hold. A subclass adds its own in `_configure`
        self._mission_name = mission if isinstance(mission, str) else None
        self._schedule_name = schedule_name
        self._settings: dict = {
            "schedule_name": schedule_name,
            "mission": self._mission_name,
            **self.survey.as_dict(),
            "cosmology": str(self.cosmology.name),
        }
        self._space: Space | None = None

    def _configure(self, space: Space, redshift_edge: float, **settings) -> None:
        """
        Finish building a sampler: give it its space and settings, then build what the scales and the simulation need.

        A subclass calls this once, at the end of its ``__init__``, after the settings it records are resolved. It
        takes the place of setting ``_space`` and ``_settings`` and calling `_build_scales` by hand.

        Parameters
        ----------
        space : ~uvex_transients.phase_space.space.Space
            The axes of the sampler, with their default ranges.
        redshift_edge : float
            The redshift out to which the volume is tabulated, and beyond which no event is drawn.
        **settings
            What the subclass records in `settings` beyond the ones recorded here, as plain types. They are available to
            `make_transient`, which this builds.
        """
        self._space = space
        self._settings.update(settings)
        self._build_scales(redshift_edge)

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

    @classmethod
    def from_run_config(cls, run, **options) -> "SurveySampler":
        """
        Build a sampler from the arguments of a map in a run-config, for the survey and mission the run-config names.

        Parameters
        ----------
        run : object
            The parsed run-config, with ``schedule``, ``mission_name`` and ``schedule_name``.
        **options
            The arguments of the sampler: the detection, simulation and ``settings:`` sections of the map, merged.

        Returns
        -------
        SurveySampler
            The sampler.

        Raises
        ------
        ValueError
            If the sampler does not take one of the `options`.
        """
        cls._check_options(None, None, schedule_name=None, **options)
        return cls(run.schedule, run.mission_name, schedule_name=run.schedule_name, **options)

    @classmethod
    def describe(cls, settings: Mapping) -> str:
        """
        Describe the map of a survey sampler: its band and what it counts as a detection.

        Parameters
        ----------
        settings : Mapping
            The `settings` of a sampler of this class.

        Returns
        -------
        str
            For example ``"FUV: SNR >= 5 on 2 visit(s)"``.
        """
        return f"{settings['band']}: {describe_detection(settings)}"

    @property
    def space(self):
        """~uvex_transients.phase_space.space.Space: The axes of the sampler, with their default ranges."""
        if self._space is None:
            raise RuntimeError(f"{type(self).__name__} has no space yet: its __init__ has to call `_configure`.")
        return self._space

    @property
    def settings(self) -> dict:
        """
        dict: What `from_settings` needs to rebuild this sampler, as plain types that an ECSV header can hold.

        The mission and schedule are recorded by name. A mission given as an object, or a schedule given without a
        ``schedule_name``, is recorded as `None` and has to be supplied again to `from_settings`.
        """
        settings = dict(self._settings)
        if self.detection is not None:
            settings["detection"] = [dict(step) for step in self.detection]
        return settings

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
        return len(self.detect(simulator, catalog))

    def detect(self, simulator: SurveySimulator, catalog: EventCatalog) -> EventCatalog:
        """
        Return the events of a catalog that count as recovered.

        This is what a recovery is. It applies each cut of `detection` in turn to the events that survived the one
        before, and with no `detection` it is the one ``snr`` cut, so that an event is recovered when at least
        `n_visits` of its observations reach `snr_threshold` in the band. A cut is given the sampler's `snr_threshold`,
        `n_visits`, `exclude_first_visit_detections` and ``bands`` (the band of the map) for each of those it takes and
        its entry leaves out.

        Override it for a rule that is not a sequence of cuts. It is called once per draw, and has to return the
        rows of `catalog` that are recovered. The epochs of the events (each observation, with its time and its
        measured SNR) are from :meth:`~uvex_transients.simulation.core.SurveySimulator.iter_epochs`, and a boolean mask
        over the rows of the catalog is applied with ``dataclasses.replace(catalog, table=catalog.table[mask])``.

        Parameters
        ----------
        simulator : ~uvex_transients.simulation.core.SurveySimulator
            The simulator that made `catalog`, which the cuts and the epochs of the events are taken from.
        catalog : ~uvex_transients.simulation.event_catalog.EventCatalog
            The `n` events drawn at a point.

        Returns
        -------
        ~uvex_transients.simulation.event_catalog.EventCatalog
            The events of `catalog` that are recovered.
        """
        defaults = {
            "snr_threshold": self.snr_threshold,
            "bands": [self.band],
            "n_visits": self.n_visits,
            "exclude_first_visit_detections": self.exclude_first_visit_detections,
        }
        for step in self.detection or ({"cut": "snr"},):
            function = SurveySimulator.get_cut(step["cut"])
            accepted = _cut_parameters(function)
            params = {key: value for key, value in defaults.items() if key in accepted}
            params.update(step.get("params", {}))
            catalog = function(simulator, catalog, self.mission, **params)
            if not len(catalog):
                break
        return catalog

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
