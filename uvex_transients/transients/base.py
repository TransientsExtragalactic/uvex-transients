r"""
Base classes for transient population models.

A :class:`TransientBase` subclass is one type of transient to simulate in the
survey. It is a thin container pairing a single
:class:`~uvex_transients.models.core.base.SpectralModel` (the full, self-contained
:math:`L_\\nu(\\nu, t)` model -- flux, magnitude, band-averaged photometry, and
rest-frame spectrum generation, including cosmological redshift/distance
resolution, all live on the SED itself now; see that module) with the metadata
needed to run and window a Monte Carlo survey simulation: a duration limit and,
for :class:`ExtragalacticTransient`, a cosmological volumetric event rate. Milky
Way foreground reddening is likewise no longer a `TransientBase` concern -- see
:class:`~m4opt.synphot.extinction.DustExtinction` -- since it's folded into a flux/magnitude call via
the SED's own ``log_attenuation`` keyword (``ebv``/``dust_law`` for
``as_source_spectrum``/``simulate_photometry``) rather than wrapped here.

:class:`ExtragalacticTransient` adds cosmological volumetric-rate sampling: given a
per-class comoving event-rate density (events / Mpc^3 / yr as a function of redshift),
it can draw a Monte Carlo realization of events over a sky patch and time window,
either as individual ``(RA, DEC, z, t_explosion, parameter_seed)`` tuples or, more
compactly, as ``(healpix_id, time_id, z, seed)`` tuples against a shared discretization
grid. It caches its redshift grid's corresponding luminosity-distance values
(:attr:`~ExtragalacticTransient.luminosity_distance_grid`) alongside the redshift grid
itself (:attr:`~ExtragalacticTransient.redshift_grid`), so callers needing :math:`D_L(z)`
for a batch of events have it on hand without a second cosmology lookup.
"""

import warnings
from abc import ABC, abstractmethod
from typing import ClassVar, Union

import astropy_healpix as ah
import numpy as np
from astropy import units as u
from astropy.coordinates import ICRS, SkyCoord
from astropy.cosmology import Cosmology
from astropy.table import QTable
from astropy.time import Time
from astropy.units import Quantity
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid

from uvex_transients.models import SpectralModel
from uvex_transients.utils import get_rng, get_seed_sequence, logger, spawn_seeds, split_root_seed
from uvex_transients.utils.cosmology import get_cosmology

_SeedType = Union[np.random.SeedSequence, int]


# =========================================================================== #
# Transient Base Class                                                        #
# =========================================================================== #
class TransientBase(ABC):
    """
    Abstract base class for transient types.

    A container: one :class:`~uvex_transients.models.core.base.SpectralModel` instance
    (:attr:`sed`) plus the metadata needed to window a survey simulation around it
    (:attr:`duration_limit`). All flux/magnitude/spectrum evaluation is the SED's
    own job -- see :class:`~uvex_transients.models.core.base.SpectralModel` and, for a
    composed lightcurve + spectral-shape SED, :class:`~uvex_transients.models.core.base.ComposedSpectralModel`.

    Parameters
    ----------
    cosmology : ~astropy.cosmology.Cosmology, optional
        The cosmology to use for this transient class. See :meth:`__init__`.
    **_
        Ignored. See :meth:`__init__`.
    """

    # ------------------------------ #
    # Class Variables                #
    # ------------------------------ #
    # Concrete subclasses (e.g. `TypeIIPSNe`) must override both of these with
    # real values.
    DEFAULT_MODEL: ClassVar[type[SpectralModel] | None] = None
    """type[SpectralModel]: The SED model class associated with this transient type. Must be set by subclasses."""
    DEFAULT_DURATION: ClassVar[Quantity | None] = None
    """ ~astropy.units.Quantity: The duration of this transient.

    This parameter is used when doing windowing to determine the relevant observations which could detect a given
    transient and should be a strong upper bound on the total duration of the relevant transient.
    """

    _REGISTRY: ClassVar[dict[str, type["TransientBase"]]] = {}
    """dict[str, type[TransientBase]]: Every concrete subclass, keyed by class name; see `registry`.

    Populated automatically by `__init_subclass__` -- a class only appears here once its
    defining module has actually been imported (this registry does not, by itself, import
    anything).
    """

    # ------------------------------ #
    # Instantiation                  #
    # ------------------------------ #

    def __init_subclass__(cls, **kwargs):
        """
        Register a concrete subclass in `_REGISTRY`, enforcing that it set its class variables.

        Parameters
        ----------
        **kwargs
            Forwarded to :meth:`object.__init_subclass__` unchanged; this
            class declares no class-keyword-argument options of its own.

        Raises
        ------
        TypeError
            If a concrete (non-abstract) subclass leaves `DEFAULT_MODEL` or
            `DEFAULT_DURATION` unset.
        """
        super().__init_subclass__(**kwargs)

        # Concrete subclasses must supply real values for both class variables (see the
        # docstrings above). Enforcing this here, at class-definition time, turns a
        # would-be `AttributeError`/`TypeError` buried inside `__init__` (or worse,
        # inside a Monte Carlo loop) into an immediate, actionable error at import time.
        if ABC in cls.__bases__:
            # `cls` is itself still abstract (e.g. `ExtragalacticTransient`) -- it's
            # not required to have set concrete defaults yet; its own subclasses are.
            return

        missing = [name for name in ("DEFAULT_MODEL", "DEFAULT_DURATION") if getattr(cls, name) is None]
        if missing:
            raise TypeError(f"{cls.__name__} must override {missing} with real values.")

        TransientBase._REGISTRY[cls.__name__] = cls

    @classmethod
    def registry(cls) -> dict[str, type["TransientBase"]]:
        """dict[str, type[TransientBase]]: A copy of every concrete subclass imported so far, keyed by class name."""
        return dict(cls._REGISTRY)

    def __init__(self, cosmology: Union[Cosmology, None] = None, **_):
        """
        Instantiate the transient class.

        Parameters
        ----------
        cosmology : ~astropy.cosmology.Cosmology, optional
            The cosmology to use for this transient class. This is used to determine
            the relevant luminosity distances of objects and to account for cosmological
            volume corrections. If ``None`` (default), the configured default cosmology
            is used; see `uvex_transients.utils.cosmology.get_cosmology`.
        **_
            Ignored. Lets subclasses (e.g. `ExtragalacticTransient`) forward extra
            constructor arguments through a shared call signature without this base
            `__init__` needing to know about them.
        """
        # Resolve the cosmology (falling back to the configured default), then
        # validate it before assigning it privately.
        cosmology = get_cosmology(cosmology)
        if not isinstance(cosmology, Cosmology):
            raise TypeError(
                f"Parameter 'cosmology' must be an instance of astropy.cosmology.Cosmology, not {type(cosmology)}."
            )

        self._cosmology: Cosmology = cosmology

        # Instantiate the SED and duration limit fresh for this instance.
        # `__init_subclass__` guarantees both class variables are non-`None` by the time
        # this runs.
        self._sed = self.__class__.DEFAULT_MODEL()
        self._duration_limit = self.__class__.DEFAULT_DURATION

        # `Event.simulate_photometry`'s windowing defaults: no pre-explosion photometry,
        # and a post-explosion window that tracks `duration_limit` (via the
        # `photometry_post_window` getter) until explicitly overridden.
        self._photometry_pre_window: Quantity = 0 * u.day
        self._photometry_post_window: Quantity | None = None

    # ------------------------------ #
    # Properties                     #
    # ------------------------------ #
    @property
    def cosmology(self) -> Cosmology:
        """
        ~astropy.cosmology.Cosmology: The cosmology used for luminosity-distance/volume calculations.

        Reassigning this only replaces the `Cosmology` instance itself; it does not
        by itself invalidate any cached, cosmology-dependent quantities on subclasses
        (see `ExtragalacticTransient.integrated_rate`/`ExtragalacticTransient.luminosity_distance_grid`,
        which cache by cosmology and rebuild automatically when this changes).
        """
        return self._cosmology

    @cosmology.setter
    def cosmology(self, cosmology: Cosmology):
        """
        Set the cosmology used for luminosity-distance/volume calculations.

        Parameters
        ----------
        cosmology : ~astropy.cosmology.Cosmology
            The new cosmology.

        Raises
        ------
        TypeError
            If `cosmology` is not an `~astropy.cosmology.Cosmology` instance.
        """
        if not isinstance(cosmology, Cosmology):
            raise TypeError(
                f"Parameter 'cosmology' must be an instance of astropy.cosmology.Cosmology, not {type(cosmology)}."
            )
        self._cosmology = cosmology

    @property
    def sed(self) -> SpectralModel:
        """
        SpectralModel: This instance's SED -- flux, magnitude, and spectrum evaluation all live here.

        See :class:`~uvex_transients.models.core.base.SpectralModel` for the full API
        (``flux``/``flux_bolometric``/``flux_band``, their ``mag*`` counterparts, and
        ``generate_spectrum``), each of which resolves redshift/distance from a
        cosmology directly and accepts an optional ``log_attenuation`` (or, for
        ``as_source_spectrum``/``simulate_photometry``, ``ebv``/``dust_law``) for Milky
        Way foreground reddening (see :class:`~m4opt.synphot.extinction.DustExtinction`).
        """
        return self._sed

    @property
    def duration_limit(self) -> Quantity:
        """
        ~astropy.units.Quantity: Upper bound on this transient's total duration.

        Defaults to `DEFAULT_DURATION`. Used for windowing -- determining which
        observations could plausibly have detected a given transient -- so this
        should be a strict upper bound on the time between explosion and the
        transient fading below any relevant detection threshold, not a typical
        or characteristic duration.
        """
        return self._duration_limit

    @duration_limit.setter
    def duration_limit(self, duration_limit: Quantity) -> None:
        """
        Set the upper bound on this transient's total duration.

        Parameters
        ----------
        duration_limit : ~astropy.units.Quantity
            The new duration limit; see :attr:`duration_limit`.
        """
        self._duration_limit = self._validate_duration_limit(duration_limit)

    @staticmethod
    def _validate_duration_limit(duration_limit: Quantity) -> Quantity:
        """
        Validate a candidate `duration_limit`, shared by the setter and (implicitly) `__init__`.

        Parameters
        ----------
        duration_limit : ~astropy.units.Quantity
            The candidate value to validate.

        Returns
        -------
        ~astropy.units.Quantity
            `duration_limit`, unchanged.

        Raises
        ------
        TypeError
            If `duration_limit` is not a Quantity with time units.
        ValueError
            If `duration_limit` is not finite and positive.
        """
        if not isinstance(duration_limit, Quantity) or duration_limit.unit.physical_type != "time":
            raise TypeError(
                f"`duration_limit` must be an astropy Quantity with time units, not {type(duration_limit)!r}."
            )

        if not np.isfinite(duration_limit) or duration_limit <= 0:
            raise ValueError(f"`duration_limit` must be finite and positive, got {duration_limit!r}.")

        return duration_limit

    @property
    def photometry_pre_window(self) -> Quantity:
        """
        ~astropy.units.Quantity: How far before explosion `Event.simulate_photometry` still generates photometry.

        Passed straight through to
        `~uvex_transients.simulation.event.Event.__init__`'s `photometry_pre_window`:
        observations in this window never reach `sed` (outside its valid domain,
        ``t >= 0``) and instead get pure background/non-detection photometry. Defaults
        to ``0 * u.day`` -- no pre-explosion photometry -- since that's a strict
        addition to how much of the schedule an event's photometry covers, unlike
        `duration_limit`/`photometry_post_window`, which bound something that must
        already be supplied.
        """
        return self._photometry_pre_window

    @photometry_pre_window.setter
    def photometry_pre_window(self, photometry_pre_window: Quantity) -> None:
        """
        Set how far before explosion `Event.simulate_photometry` still generates photometry.

        Parameters
        ----------
        photometry_pre_window : ~astropy.units.Quantity
            The new pre-explosion window; see :attr:`photometry_pre_window`.
        """
        self._photometry_pre_window = self._validate_photometry_window(
            photometry_pre_window, name="photometry_pre_window", allow_zero=True
        )

    @property
    def photometry_post_window(self) -> Quantity:
        """
        ~astropy.units.Quantity: How far after explosion `Event.simulate_photometry` evaluates `sed`.

        Passed straight through to
        `~uvex_transients.simulation.event.Event.__init__`'s `photometry_post_window`;
        observations past it get background/non-detection photometry instead of `sed`
        evaluation, the same as before `t=0`. Tracks `duration_limit` (including any
        later change to it) until set explicitly here, at which point it's decoupled --
        the same relationship `Event.__init__` itself has between the two parameters.
        """
        return self._duration_limit if self._photometry_post_window is None else self._photometry_post_window

    @photometry_post_window.setter
    def photometry_post_window(self, photometry_post_window: Quantity) -> None:
        """
        Set how far after explosion `Event.simulate_photometry` evaluates `sed`.

        Parameters
        ----------
        photometry_post_window : ~astropy.units.Quantity
            The new post-explosion window; see :attr:`photometry_post_window`.
        """
        self._photometry_post_window = self._validate_photometry_window(
            photometry_post_window, name="photometry_post_window", allow_zero=False
        )

    @staticmethod
    def _validate_photometry_window(window: Quantity, *, name: str, allow_zero: bool) -> Quantity:
        """
        Validate a candidate `photometry_pre_window`/`photometry_post_window`, shared by both setters.

        Parameters
        ----------
        window : ~astropy.units.Quantity
            The candidate value to validate.
        name : str
            Which property this is for, used only to phrase the error message.
        allow_zero : bool
            Whether ``0`` is an acceptable value (true for `photometry_pre_window`,
            since "no pre-explosion window" is its own default; false for
            `photometry_post_window`, which -- like `duration_limit` -- must bound a
            strictly positive span).

        Returns
        -------
        ~astropy.units.Quantity
            `window`, unchanged.

        Raises
        ------
        TypeError
            If `window` is not a Quantity with time units.
        ValueError
            If `window` is not finite, or isn't within the bound `allow_zero` implies.
        """
        if not isinstance(window, Quantity) or window.unit.physical_type != "time":
            raise TypeError(f"`{name}` must be an astropy Quantity with time units, not {type(window)!r}.")

        if not np.isfinite(window) or (window < 0 if allow_zero else window <= 0):
            bound = "non-negative" if allow_zero else "positive"
            raise ValueError(f"`{name}` must be finite and {bound}, got {window!r}.")

        return window


# =========================================================================== #
# Extragalactic Transient                                                     #
# =========================================================================== #
class ExtragalacticTransient(TransientBase, ABC):
    """
    Extragalactic transient with a cosmological volumetric event rate; see the module docstring.

    Parameters
    ----------
    cosmology : ~astropy.cosmology.Cosmology, optional
        The cosmology to use for this transient class. See :meth:`__init__`.
    """

    DEFAULT_Z_LIM = 10
    DEFAULT_Z_GRID_SIZE = 100

    RATE_CI: ClassVar[tuple[float, float] | None] = None
    """tuple[float, float] | None: Multiplicative (lower, upper) 90% confidence factors on `rate`.

    If a publication reports a rate :math:`R_0 {}^{+\\Delta R_+}_{-\\Delta R_-}`, this is
    ``((R_0 - dR_minus) / R_0, (R_0 + dR_plus) / R_0)`` -- see :ref:`yield-statistics`.
    Expressed as multiplicative factors (rather than absolute `Quantity` bounds) so that it
    applies unchanged to a `rate` that is itself cosmology-dependent (e.g. a core-collapse
    SNe subtype, whose `rate` scales with the instance's `cosmology`). `None` (the default)
    means no rate uncertainty has been sourced for this class yet; `rate_ci` then degenerates
    to the point value `rate` twice over.
    """

    def __init__(self, cosmology: Union[Cosmology, None] = None):
        """
        Instantiate the transient, and set up (but do not yet build) its lazy rate cache.

        Parameters
        ----------
        cosmology : ~astropy.cosmology.Cosmology, optional
            The cosmology to use for this transient class. See :meth:`TransientBase.__init__`.
        """
        # Instantiate the parent class (resolves `None` to the configured default).
        super().__init__(cosmology=cosmology)

        # Assign the z grid parameters.
        self._redshift_limit = self.DEFAULT_Z_LIM
        self._redshift_grid_size = self.DEFAULT_Z_GRID_SIZE

        # Ensure that the rate cache is cleared before proceeding. This
        # also generates the appropriate attributes of the class for the
        # rate caching, which is loaded lazily as needed.
        self._invalidate_rate_cache()

    def _invalidate_rate_cache(self) -> None:
        """Clear the lazily-computed rate table; the next access rebuilds it."""
        self._redshift_grid: NDArray[np.float64] | None = None
        self._luminosity_distance_grid: Quantity | None = None
        self._z_cdf: NDArray[np.float64] | None = None
        self._integrated_rate: Quantity | None = None

    # ---------------------------------------- #
    # Properties                               #
    # ---------------------------------------- #
    @property
    def cosmology(self) -> Cosmology:
        """
        ~astropy.cosmology.Cosmology: The cosmology used for luminosity-distance/volume calculations.

        Reassigning this invalidates the cached rate table -- see
        `TransientBase.cosmology` and `integrated_rate`/`luminosity_distance_grid`.
        """
        return self._cosmology

    @cosmology.setter
    def cosmology(self, value: Cosmology) -> None:
        """
        Set the cosmology used for luminosity-distance/volume calculations, invalidating the rate cache.

        Parameters
        ----------
        value : ~astropy.cosmology.Cosmology
            The new cosmology.

        Raises
        ------
        TypeError
            If `value` is not an `~astropy.cosmology.Cosmology` instance.
        """
        if not isinstance(value, Cosmology):
            raise TypeError(
                f"Parameter 'cosmology' must be an instance of astropy.cosmology.Cosmology, not {type(value)}."
            )
        self._cosmology = value
        self._invalidate_rate_cache()

    @property
    def redshift_limit(self) -> float:
        """float: Upper redshift bound of `redshift_grid`; reassigning invalidates the rate cache."""
        return self._redshift_limit

    @redshift_limit.setter
    def redshift_limit(self, value: float) -> None:
        """
        Set the upper redshift bound of `redshift_grid`, invalidating the rate cache.

        Parameters
        ----------
        value : float
            The new redshift limit.
        """
        self._redshift_limit = self._validate_redshift_limit(value)
        self._invalidate_rate_cache()

    @property
    def redshift_grid_size(self) -> int:
        """int: Number of points in `redshift_grid`; reassigning invalidates the rate cache."""
        return self._redshift_grid_size

    @redshift_grid_size.setter
    def redshift_grid_size(self, value: int) -> None:
        """
        Set the number of points in `redshift_grid`, invalidating the rate cache.

        Parameters
        ----------
        value : int
            The new grid size.
        """
        self._redshift_grid_size = self._validate_redshift_grid_size(value)
        self._invalidate_rate_cache()

    @property
    def redshift_grid(self) -> NDArray[np.float64]:
        """numpy.ndarray: The cached redshift grid backing `integrated_rate`/sampling."""
        self._ensure_rate_table()
        return self._redshift_grid

    @property
    def luminosity_distance_grid(self) -> Quantity:
        r"""
        ~astropy.units.Quantity: :math:`D_L(z)` at each point of `redshift_grid`.

        Cached alongside `redshift_grid` (built by the same `_ensure_rate_table` call,
        against the same `cosmology`), so a caller with a batch of sampled redshifts
        can get :math:`D_L` for all of them via ``numpy.interp(z, transient.redshift_grid,
        transient.luminosity_distance_grid.value) * transient.luminosity_distance_grid.unit``
        instead of a second, separate `cosmology.luminosity_distance` call.
        """
        self._ensure_rate_table()
        return self._luminosity_distance_grid

    # ======================================= #
    # Rates and Statistics                    #
    # ======================================= #

    @property
    def integrated_rate(self) -> Quantity:
        r"""
        ~astropy.units.Quantity: Expected event rate per observer time and solid angle.

        Integrate the intrinsic volumetric rate :math:`R(z)` over the population's
        redshift domain:

        .. math::

            \mathcal R_\Omega
            = \int_0^{z_{\max}}
              \frac{R(z)}{1+z}
              \frac{dV_c}{dz\,d\Omega}\,dz.

        Here, :math:`R(z)` is defined per comoving volume and source-frame time.
        The factor :math:`(1+z)^{-1}` converts the rate to observer-frame time.

        This rate includes no survey footprint or detection selection. Its units
        are equivalent to :math:`\mathrm{yr}^{-1}\,\mathrm{sr}^{-1}`.
        """
        self._ensure_rate_table()
        return self._integrated_rate

    @property
    def integrated_rate_ci(self) -> tuple[Quantity, Quantity]:
        r"""
        Tuple of ~astropy.units.Quantity: Rate-only bounds on :attr:`integrated_rate`.

        Assume the volumetric rate has a fixed redshift dependence,
        :math:`R(z;A)=A f(z)`, with uncertainty only in its normalization.
        For a fiducial normalization :math:`R_0`, ``RATE_CI`` contains the
        dimensionless endpoint multipliers

        .. math::

            b_{\mathrm L} = \frac{R_{\mathrm L}}{R_0},
            \qquad
            b_{\mathrm U} = \frac{R_{\mathrm U}}{R_0}.

        The returned lower and upper bounds are

        .. math::

            [\mathcal R_{\Omega,\mathrm L},\mathcal R_{\Omega,\mathrm U}]
            = [b_{\mathrm L}\mathcal R_\Omega,
               b_{\mathrm U}\mathcal R_\Omega].

        The multipliers specify absolute endpoints relative to the fiducial
        rate, not fractional error magnitudes. The bounds retain the confidence
        level assigned to ``RATE_CI``.

        If ``RATE_CI`` is ``None``, both bounds equal :attr:`integrated_rate`.
        This convention indicates that rate uncertainty is not represented;
        it does not establish that the physical rate is known exactly.
        """
        if self.RATE_CI is None:
            integrated = self.integrated_rate
            return (integrated, integrated)
        lower, upper = self.RATE_CI
        integrated = self.integrated_rate
        return (integrated * lower, integrated * upper)

    @property
    def all_sky_rate(self) -> Quantity:
        r"""
        ~astropy.units.Quantity: Expected all-sky event rate per observer year.

        For an isotropic population, integrate :attr:`integrated_rate` over
        the full sky:

        .. math::

            \mathcal R = (4\pi\,\mathrm{sr})\,\mathcal R_\Omega.

        The result includes events throughout the population's redshift domain,
        before applying any survey footprint or detection selection, and is
        returned in inverse years. See :ref:`yield-statistics`.
        """
        return (self.integrated_rate * 4 * np.pi * u.sr).to(u.yr**-1)

    @property
    def all_sky_rate_ci(self) -> tuple[Quantity, Quantity]:
        r"""
        Tuple of ~astropy.units.Quantity: Rate-only bounds on :attr:`all_sky_rate`.

        Multiply each endpoint of :attr:`integrated_rate_ci` by the full-sky
        solid angle:

        .. math::

            [\mathcal R_{\mathrm L},\mathcal R_{\mathrm U}]
            = (4\pi\,\mathrm{sr})
              [\mathcal R_{\Omega,\mathrm L},\mathcal R_{\Omega,\mathrm U}].

        Return the lower and upper bounds in inverse years, preserving the
        confidence level and uncertainty convention of :attr:`integrated_rate_ci`.
        These bounds describe uncertainty in the expected rate, not fluctuations
        in a realized event count.
        """
        lower, upper = self.integrated_rate_ci
        factor = 4 * np.pi * u.sr
        return ((lower * factor).to(u.yr**-1), (upper * factor).to(u.yr**-1))

    def compute_all_sky_yield(self, duration: Quantity) -> float:
        r"""
        Compute the expected intrinsic event count over the full sky.

        Multiply :attr:`all_sky_rate` by the observer-frame sampling duration:

        .. math::

            \mu_0 = \mathcal R T.

        No survey footprint or detection selection is applied.

        Parameters
        ----------
        duration : ~astropy.units.Quantity
            Nonnegative observer-frame sampling-window duration, convertible
            to time units. Include any temporal padding used to generate the
            event catalog.

        Returns
        -------
        float
            Expected number of intrinsic events within the population's
            redshift domain and the specified time window. This expectation
            need not be an integer.

        Notes
        -----
        Cosmological time dilation is already included in :attr:`all_sky_rate`;
        no additional redshift factor is applied to ``duration``.

        When using this count to normalize a simulated catalog, the duration
        must match the catalog's sampling window. See :ref:`yield-statistics`.
        """
        return (self.all_sky_rate * duration).to_value(u.dimensionless_unscaled)

    def compute_all_sky_yield_ci(self, duration: Quantity) -> tuple[float, float]:
        r"""
        Compute rate-only bounds on the expected all-sky intrinsic event count.

        For a fixed observer-frame duration, transform the bounds on
        :attr:`all_sky_rate` as

        .. math::

            [\mu_{\mathrm L},\mu_{\mathrm U}]
            = [T\mathcal R_{\mathrm L},T\mathcal R_{\mathrm U}].

        Parameters
        ----------
        duration : ~astropy.units.Quantity
            Nonnegative observer-frame sampling-window duration, convertible
            to time units. Use the same window as for
            :meth:`compute_all_sky_yield`.

        Returns
        -------
        tuple of float
            Lower and upper bounds on the expected intrinsic event count.
            The endpoints need not be integers and retain the confidence
            level assigned to the rate bounds.

        Notes
        -----
        These bounds propagate only the uncertainty represented by ``RATE_CI``.
        They include neither survey selection nor Poisson fluctuations in a
        realized count, and are not a prediction interval for a future catalog.

        If ``RATE_CI`` is ``None``, both endpoints equal the expected count.
        See :ref:`yield-statistics`.
        """
        lower, upper = self.all_sky_rate_ci
        return (
            (lower * duration).to_value(u.dimensionless_unscaled),
            (upper * duration).to_value(u.dimensionless_unscaled),
        )

    @staticmethod
    def _validate_redshift_limit(z_max: float) -> float:
        """
        Validate a candidate `redshift_limit`, shared by the setter and `__init__`.

        Parameters
        ----------
        z_max : float
            The candidate value to validate.

        Returns
        -------
        float
            `z_max`, coerced to a plain Python ``float``.

        Raises
        ------
        ValueError
            If `z_max` is not finite and positive.
        """
        if not np.isfinite(z_max) or z_max <= 0:
            raise ValueError(f"`redshift_limit` must be finite and positive, got {z_max!r}.")
        return float(z_max)

    @staticmethod
    def _validate_redshift_grid_size(n_grid: int) -> int:
        """
        Validate a candidate `redshift_grid_size`, shared by the setter and `__init__`.

        Parameters
        ----------
        n_grid : int
            The candidate value to validate.

        Returns
        -------
        int
            `n_grid`, coerced to a plain Python ``int``.

        Raises
        ------
        ValueError
            If `n_grid` is not an integer >= 2.
        """
        if isinstance(n_grid, bool) or not isinstance(n_grid, (int, np.integer)) or n_grid < 2:
            raise ValueError(f"`redshift_grid_size` must be an integer >= 2, got {n_grid!r}.")
        return int(n_grid)

    # ------------------------------ #
    # Event Rate Computations        #
    # ------------------------------ #
    @property
    @abstractmethod
    def rate(self) -> Quantity:
        r"""
        ~astropy.units.Quantity: The fiducial rate normalization :math:`R_0` (this class's :math:`A`).

        Concrete subclasses implement this -- and `rate_shape` below -- instead of `event_rate`
        directly: ``event_rate(z) = rate * rate_shape(z)``, i.e. :math:`R(z;A)=A f(z)`; see
        :ref:`yield-statistics`. May depend on `cosmology` (e.g. a core-collapse SNe subtype's
        rate, whose overall coefficient scales with :math:`h^2`) -- it is a property, not a
        `ClassVar`, precisely to allow that.
        """

    @abstractmethod
    def rate_shape(self, z: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
        r"""
        Compute the dimensionless redshift shape :math:`f(z)` of the comoving event rate.

        Must be NumPy-vectorized (accept and return an array elementwise when `z` is
        an array) -- `event_rate` (and, through it, `integrated_rate`) evaluates this once,
        across the whole `redshift_grid`, not in a per-point loop. By convention (not
        enforced) :math:`f(0)=1`, making `rate` the local volumetric rate; see
        :ref:`yield-statistics`.

        Parameters
        ----------
        z : float or numpy.ndarray
            Redshift(s) at which to evaluate the rate shape.

        Returns
        -------
        float or numpy.ndarray
            The dimensionless rate shape, :math:`f(z)`.
        """

    def event_rate(self, z: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
        r"""
        Compute the comoving event rate density at redshift(s) `z`, :math:`R(z;A)=A f(z)`.

        A thin product of `rate` (:math:`A`) and `rate_shape` (:math:`f(z)`); see those
        properties/methods, which concrete subclasses implement instead of this one.

        Parameters
        ----------
        z : float or numpy.ndarray
            Redshift(s) at which to evaluate the event rate.

        Returns
        -------
        float or numpy.ndarray
            The event rate, in events / Mpc^3 / yr.
        """
        z = np.asarray(z, dtype=float)
        shape = np.asarray(self.rate_shape(z), dtype=float)
        rate_value = self.rate.to_value(u.Mpc**-3 * u.yr**-1)
        result = rate_value * shape
        return result if z.ndim > 0 else result.item()

    @property
    def rate_ci(self) -> tuple[Quantity, Quantity]:
        """
        tuple[~astropy.units.Quantity, ~astropy.units.Quantity]: ``(R_L, R_U)``, the confidence bounds on `rate`.

        Derived from `rate` and `RATE_CI`; see `RATE_CI` for the confidence level and for what
        it means for this to be `(rate, rate)` when `RATE_CI` is unset.
        """
        rate = self.rate
        if self.RATE_CI is None:
            return (rate, rate)
        lower, upper = self.RATE_CI
        return (rate * lower, rate * upper)

    @property
    def effective_volume(self) -> Quantity:
        r"""
        ~astropy.units.Quantity: :math:`\mathcal V`, the all-sky rate-weighted comoving volume.

        Depends only on `rate_shape`, `cosmology`, and `redshift_limit` -- not on `rate` -- so,
        unlike `rate`, it carries no rate-normalization uncertainty; see :ref:`yield-statistics`.
        Derived from `integrated_rate` (the per-steradian, :math:`R_0`-normalized quantity) by
        dividing out `rate` and restoring the :math:`4\pi` sky factor, rather than retabulating.
        """
        return (self.integrated_rate / self.rate * 4 * np.pi * u.sr).to(u.Mpc**3)

    def _ensure_rate_table(self) -> None:
        r"""
        Lazily (re)build the redshift grid, luminosity-distance grid, CDF, and integrated rate, if invalidated.

        Tabulates :math:`w(z) = R(z) \cdot (dV_c/dz) / (1+z)` once on `redshift_grid` (`R`
        being `event_rate`; the :math:`1/(1+z)` is the cosmological rate-dilation
        correction between rest-frame event rate and observer-frame duration), then
        derives `integrated_rate` (the total, un-normalized integral, scaled to
        events per steradian per unit observer time), the CDF used for inversion
        sampling from that single tabulated array, and `luminosity_distance_grid`
        (:math:`D_L(z)` at the same grid points, for reuse by callers) -- `event_rate`
        is evaluated exactly once per rebuild, regardless of how many subsequent
        events are sampled from it. Explicit tabulate-and-invert (rather than treating
        `event_rate` as a `~uvex_transients.models.core.priors.Prior`) is used
        deliberately, since `event_rate` may be an arbitrary, non-smooth function for
        which the polynomial-inversion machinery `Prior` relies on has no particular
        guarantees.
        """
        if self._redshift_grid is not None:
            return

        logger.debug(
            "%s: (re)building rate table over z in [0, %s] (%d grid points, cosmology=%s).",
            type(self).__name__,
            self._redshift_limit,
            self._redshift_grid_size,
            self._cosmology.name,
        )

        z_grid = np.linspace(0.0, self._redshift_limit, self._redshift_grid_size)
        rate = np.asarray(self.event_rate(z_grid), dtype=float)
        dVc_dz = self._cosmology.differential_comoving_volume(z_grid).to_value(u.Mpc**3 / u.sr)
        weight = rate * dVc_dz / (1.0 + z_grid)

        cumulative = cumulative_trapezoid(weight, z_grid, initial=0.0)
        total = cumulative[-1]

        if not np.isfinite(total) or total <= 0:
            n_nonfinite = int(np.sum(~np.isfinite(rate)))
            raise ValueError(
                f"The integrated event rate must be finite and positive, got {total!r}; check `event_rate`. "
                f"`event_rate(z)` over `redshift_grid` (z in [0, {self._redshift_limit}]) returned values in "
                f"[{np.nanmin(rate)!r}, {np.nanmax(rate)!r}] with {n_nonfinite} non-finite entries."
            )

        self._redshift_grid = z_grid
        self._luminosity_distance_grid = self._cosmology.luminosity_distance(z_grid)
        self._z_cdf = cumulative / total
        # events / Mpc^3 / yr (implicit units of `event_rate`) * Mpc^3/sr (from
        # `differential_comoving_volume`) integrated over dz -> events / sr / yr.
        self._integrated_rate = total / (u.sr * u.yr)

        logger.debug("%s: rate table built; integrated_rate=%s.", type(self).__name__, self._integrated_rate)

    @staticmethod
    def _resolve_duration(duration: Quantity = None, t_start: Time = None, t_end: Time = None) -> Quantity:
        """
        Resolve a sampling-window length from either an explicit `duration` or a `t_start`/`t_end` pair.

        Parameters
        ----------
        duration : ~astropy.units.Quantity, optional
            The window length, given directly. Takes priority over `t_start`/`t_end`
            if both are supplied (with a warning; see below).
        t_start, t_end : ~astropy.time.Time, optional
            The start/end of the window; `duration` is computed as ``t_end - t_start``
            if `duration` itself isn't given.

        Returns
        -------
        ~astropy.units.Quantity
            The resolved window length.

        Raises
        ------
        TypeError
            If `duration` is not a Quantity, or `t_start`/`t_end` are not Time objects.
        ValueError
            If none of `duration` or the `t_start`/`t_end` pair is fully supplied.
        """
        # If duration is provided ab-initio, we just utilize that and proceed. We warn
        # if `t_end` was *also* supplied, since -- unlike `t_start`, which callers may
        # still need independently (e.g. `sample_events_on_healpix_grid` uses it to
        # anchor sampled explosion times, not just to derive a duration) -- `t_end` is
        # only ever used here, to compute a duration, so it's silently ignored once
        # `duration` is given directly.
        if duration is not None:
            # Ensure duration is a quantity.
            if not isinstance(duration, u.Quantity):
                raise TypeError(f"`duration` must be a Quantity, not {type(duration)}.")

            if t_end is not None:
                warnings.warn("Parameter 't_end' ignored in favor of 'duration'.", stacklevel=2)

            # Return the duration.
            return duration

        # In this case we can determine the duration from the
        # t_start and the t_end.
        elif (t_start is not None) and (t_end is not None):
            # Validate that the types are correct.
            if not isinstance(t_start, Time):
                raise TypeError(f"`t_start` must be a Time, not {type(t_start)}.")
            if not isinstance(t_end, Time):
                raise TypeError(f"`t_end` must be a Time, not {type(t_end)}.")

            # Compute the duration. The redundant conversion here is to ensure
            # that we go from a TimeDelta object to a quantity.
            _computed_duration = (t_end - t_start).to_value(u.day) * u.day

            return _computed_duration

        else:
            raise ValueError("Either 'duration' or both 't_start' and 't_end' must be provided.")

    @staticmethod
    def _resolve_pixel_index_array(
        nside: int,
        pixel_mask: NDArray[np.bool_] | None,
        pixel_ids: NDArray[np.integer] | None,
    ) -> NDArray[np.int64]:
        """
        Resolve a mask or explicit pixel list into a sorted array of unique pixel indices.

        Parameters
        ----------
        nside : int
            HEALPix resolution parameter of the grid `pixel_mask`/`pixel_ids` are
            defined against.
        pixel_mask : numpy.ndarray of bool, optional
            Boolean mask, shape ``(ah.nside_to_npix(nside),)``, selecting pixels.
            Mutually exclusive with `pixel_ids`.
        pixel_ids : numpy.ndarray of int, optional
            Explicit pixel indices, each in ``[0, ah.nside_to_npix(nside))``.
            Mutually exclusive with `pixel_mask`.

        Returns
        -------
        numpy.ndarray
            Sorted, deduplicated ``int64`` pixel indices. If neither `pixel_mask` nor
            `pixel_ids` is given, every pixel in the grid is returned.

        Raises
        ------
        ValueError
            If both `pixel_mask` and `pixel_ids` are supplied, if `pixel_mask` has the
            wrong shape, or if `pixel_ids` contains an out-of-range or non-1D value.
        TypeError
            If `pixel_mask` is not Boolean, or `pixel_ids` is not integer-typed.
        """
        if pixel_mask is not None and pixel_ids is not None:
            raise ValueError("Provide at most one of `pixel_mask` and `pixel_ids`.")

        npix = ah.nside_to_npix(nside)

        if pixel_mask is not None:
            mask = np.asarray(pixel_mask)

            if mask.dtype != np.bool_:
                raise TypeError("`pixel_mask` must have Boolean dtype.")

            if mask.shape != (npix,):
                raise ValueError(f"`pixel_mask` must have shape ({npix},), got {mask.shape}.")

            return np.flatnonzero(mask).astype(np.int64)

        if pixel_ids is not None:
            indices = np.asarray(pixel_ids)

            if not np.issubdtype(indices.dtype, np.integer):
                raise TypeError("`pixel_ids` must contain integers.")

            indices = indices.astype(np.int64, copy=False)

            if indices.ndim != 1:
                raise ValueError("`pixel_ids` must be one-dimensional.")

            if np.any(indices < 0) or np.any(indices >= npix):
                raise ValueError(f"`pixel_ids` must lie between 0 and {npix - 1}.")

            return np.unique(indices)

        return np.arange(npix)

    @staticmethod
    def _create_event_table_buffer(
        size: int,
        *,
        time_scale: str = "utc",
        frame=ICRS(),
    ) -> QTable:
        """
        Create an empty, fixed-size event table for later population.

        Parameters
        ----------
        size : int
            Number of event rows to allocate.
        time_scale : str, optional
            Astropy time scale for the explosion-time column.
        frame : astropy.coordinates.BaseCoordinateFrame, optional
            Coordinate frame for the event positions.

        Returns
        -------
        astropy.table.QTable
            Preallocated event table.
        """
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)):
            raise TypeError(f"`size` must be an integer, got {type(size).__name__}.")

        if size < 0:
            raise ValueError(f"`size` must be non-negative, got {size}.")

        table = QTable()

        table["healpix_id"] = np.full(
            size,
            -1,
            dtype=np.int64,
        )

        table["healpix_dx"] = np.full(
            size,
            np.nan,
            dtype=np.float64,
        )

        table["healpix_dy"] = np.full(
            size,
            np.nan,
            dtype=np.float64,
        )

        # Positional args (rather than `lon=`/`lat=` kwargs, which `SkyCoord` doesn't
        # accept) so this works regardless of `frame`'s component names (`ra`/`dec`
        # for `ICRS`, `l`/`b` for `Galactic`, etc.).
        table["coord"] = SkyCoord(
            np.full(size, np.nan) * u.deg,
            np.full(size, np.nan) * u.deg,
            frame=frame,
        )

        table["redshift"] = np.full(
            size,
            np.nan,
            dtype=np.float64,
        )

        # Unlike the other columns, `Time` rejects non-finite values outright (for
        # every numeric format, not just `jd`) -- `np.nan` isn't an option here. `jd=0`
        # (4713 BCE) can't collide with any real survey date, so it serves the same
        # "not yet populated" sentinel role as `healpix_id`'s `-1`.
        table["t_explosion"] = Time(
            np.zeros(size),
            format="jd",
            scale=time_scale,
        )

        table["parameter_seed"] = np.zeros(
            size,
            dtype=np.uint64,
        )

        return table

    def sample_event_count(
        self,
        solid_angle: Quantity,
        duration: Quantity = None,
        t_start: Time = None,
        t_end: Time = None,
        seed: _SeedType | None = None,
    ) -> int:
        """
        Sample the number of events in a given solid angle and duration.

        Parameters
        ----------
        solid_angle : ~astropy.units.Quantity
            The solid angle over which to sample events.
        duration : ~astropy.units.Quantity, optional
            The duration over which to sample events. Mutually exclusive with
            `t_start`/`t_end`; see `_resolve_duration`.
        t_start, t_end : ~astropy.time.Time, optional
            The start/end of the sampling window, used to derive `duration` if it
            isn't supplied directly. See `_resolve_duration`.
        seed : numpy.random.SeedSequence, int, or None, optional
            Root seed for reproducibility; see `uvex_transients.utils.split_root_seed`.

        Returns
        -------
        int
            The sampled number of events.
        """
        # Validate the duration.
        duration = self._resolve_duration(duration, t_start, t_end)

        # Ensure that the rate table has been computed so that we can extract the
        # correct integrated rate.
        self._ensure_rate_table()

        # Determine the rate in this time period and solid angle.
        n_expected = (self.integrated_rate * solid_angle * duration).to_value(u.dimensionless_unscaled)

        # Generate the random realization.
        rng, _ = split_root_seed(seed)
        return int(rng.poisson(n_expected))

    def sample_event_redshift(
        self,
        n_samples: int,
        *,
        rng: Union[np.random.Generator, int, None] = None,
    ) -> NDArray[np.float64]:
        r"""
        Draw `n_samples` redshifts from the rate-weighted redshift distribution.

        Uses inverse-transform sampling against the tabulated CDF built by
        `_ensure_rate_table` -- see that method's docstring for why the CDF is
        tabulated once and inverted, rather than treating `event_rate` as a
        `~uvex_transients.models.core.priors.Prior`.

        Parameters
        ----------
        n_samples : int
            Number of redshifts to draw.
        rng : numpy.random.Generator, int, or None, optional
            Random-number source; see `uvex_transients.utils.get_rng`. Unlike
            `sample_event_count`, this takes an already-materialized generator
            (not a root seed to split) since it's meant to be handed one of the
            independent streams a caller has already spawned (see
            `sample_events_on_healpix_grid`).

        Returns
        -------
        numpy.ndarray
            ``float64`` array of shape ``(n_samples,)``.
        """
        self._ensure_rate_table()
        rng = get_rng(rng)

        # Draw uniform CDF values and invert them against the tabulated
        # (redshift_grid, z_cdf) pairs -- `redshift_grid` is monotonically
        # increasing in `z`, and `_ensure_rate_table` guarantees `z_cdf` is
        # monotonically increasing too, so linear interpolation is a valid inverse.
        u_samples = rng.random(n_samples)
        return np.interp(u_samples, self._z_cdf, self._redshift_grid)

    def sample_events_on_healpix_grid(
        self,
        nside: int,
        *,
        t_start: Time,
        t_end: Time | None = None,
        duration: Quantity | None = None,
        pixel_mask: NDArray[np.bool_] | None = None,
        pixel_ids: NDArray[np.int64] | None = None,
        order: str = "nested",
        jitter: bool = True,
        seed: _SeedType | None = None,
    ) -> QTable:
        """
        Draw a Monte Carlo realization of events on a HEALPix grid, over a time window.

        Each sampled event is placed uniformly at random within one of the selected
        HEALPix pixels (optionally jittered to a sub-pixel position; see `jitter`),
        assigned a redshift drawn from `sample_event_redshift`, an explosion time drawn
        uniformly across ``[t_start, t_start + duration)``, and a per-event parameter
        seed (see `spawn_seeds`) rather than fully-sampled physical parameters -- those
        are meant to be regenerated lazily, on demand, from that seed (via
        `SpectralModel.sample_parameters`).

        Parameters
        ----------
        nside : int
            HEALPix resolution parameter of the sampling grid.
        t_start : ~astropy.time.Time
            Start of the sampling window.
        t_end : ~astropy.time.Time, optional
            End of the sampling window. Mutually exclusive with `duration`; see
            `_resolve_duration`.
        duration : ~astropy.units.Quantity, optional
            Length of the sampling window. Mutually exclusive with `t_end`; see
            `_resolve_duration`.
        pixel_mask : numpy.ndarray of bool, optional
            Boolean mask, shape ``(ah.nside_to_npix(nside),)``, selecting which pixels
            to sample events in. Mutually exclusive with `pixel_ids`. If neither is
            given, every pixel in the grid is eligible.
        pixel_ids : numpy.ndarray of int, optional
            Explicit pixel indices to sample events in. Mutually exclusive with
            `pixel_mask`.
        order : str, optional
            HEALPix pixel ordering scheme (``"nested"`` or ``"ring"``), consistent with
            whatever ordering `pixel_mask`/`pixel_ids` were defined against. The
            default is ``"nested"``.
        jitter : bool, optional
            If `True` (the default), place each event at a random sub-pixel position.
            If `False`, place every event at its pixel's center.
        seed : numpy.random.SeedSequence, int, or None, optional
            Root seed for reproducibility. A single tree of children is spawned from
            this seed for every stochastic draw in this call -- including the event
            count itself -- so that no two draws (however many events end up being
            sampled) ever share a stream.

        Returns
        -------
        astropy.table.QTable
            Event table with one row per sampled event; see `_create_event_table_buffer`
            for its columns. Empty (but still correctly typed) if zero events were
            sampled.
        """
        # Validate the duration and resolve the pixel selection down to a concrete
        # array of pixel indices.
        duration = self._resolve_duration(duration, t_start, t_end)
        pixel_indices = self._resolve_pixel_index_array(nside, pixel_mask, pixel_ids)

        # Ensure that the rate table is loaded for our use.
        self._ensure_rate_table()

        # Using nside, we now want to determine the solid angular size of a single pixel
        # in this healpix discretization, then the total solid angle of the pixels
        # we're actually sampling.
        solid_angle_per_pix = ah.nside_to_pixel_area(nside)
        total_solid_angle = len(pixel_indices) * solid_angle_per_pix

        # Spawn one tree of independent random streams from the root seed -- one per
        # stochastic draw below, plus a spawn point for the per-event parameter seeds
        # -- rather than splitting the root seed separately for each consumer. This is
        # what guarantees `count_seed` (below) can never collide with, e.g., `pixel_seed`,
        # even though both ultimately derive from the same `seed` argument.
        root_seed = get_seed_sequence(seed)
        (
            count_seed,
            pixel_seed,
            jitter_seed,
            time_seed,
            redshift_seed,
            parameter_spawn_seed,
        ) = root_seed.spawn(6)

        # Compute the expected number of events that we will be generating.
        number_of_events = self.sample_event_count(
            solid_angle=total_solid_angle,
            duration=duration,
            seed=count_seed,
        )

        logger.debug(
            "%s: sampled %d events over %d pixels (nside=%d), solid angle %s, duration %s.",
            type(self).__name__,
            number_of_events,
            len(pixel_indices),
            nside,
            total_solid_angle,
            duration,
        )

        event_buffer = self._create_event_table_buffer(
            number_of_events,
            time_scale=t_start.scale,
        )

        if number_of_events == 0:
            return event_buffer

        pixel_rng = np.random.default_rng(pixel_seed)
        jitter_rng = np.random.default_rng(jitter_seed)
        time_rng = np.random.default_rng(time_seed)
        redshift_rng = np.random.default_rng(redshift_seed)

        # Sample redshifts and per-event host pixels.
        redshifts = self.sample_event_redshift(number_of_events, rng=redshift_rng)
        event_pixel = pixel_rng.choice(
            pixel_indices,
            size=number_of_events,
            replace=True,
        )

        # Sample positions within each pixel.
        if jitter:
            event_jitter = jitter_rng.random((number_of_events, 2))
        else:
            event_jitter = np.full(
                (number_of_events, 2),
                0.5,
                dtype=float,
            )

        lon, lat = ah.healpix_to_lonlat(
            event_pixel,
            nside=nside,
            dx=event_jitter[:, 0],
            dy=event_jitter[:, 1],
            order=order,
        )

        event_coords = SkyCoord(
            lon,
            lat,
            frame=ICRS(),
        )

        # Sample explosion times uniformly across the sampling window.
        event_times = t_start + time_rng.random(number_of_events) * duration

        # Draw independent, individually storable per-event seeds -- for regenerating
        # each event's physical lightcurve parameters later, on demand, without needing
        # to replay the draws for any other event (see `spawn_seeds`).
        event_seed = spawn_seeds(parameter_spawn_seed, number_of_events)

        # Populate the table buffer.
        event_buffer["healpix_id"] = event_pixel
        event_buffer["healpix_dx"] = event_jitter[:, 0]
        event_buffer["healpix_dy"] = event_jitter[:, 1]
        event_buffer["coord"] = event_coords
        event_buffer["redshift"] = redshifts
        event_buffer["t_explosion"] = event_times
        event_buffer["parameter_seed"] = event_seed

        event_buffer.meta.update(
            {
                "healpix_nside": int(nside),
                "healpix_order": order,
                "coordinate_frame": "icrs",
                "time_scale": t_start.scale,
                "time_frame": "observer",
            }
        )

        return event_buffer
