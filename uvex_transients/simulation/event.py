"""
A single sampled transient event, reconstructed from one `EventCatalog` row.

An :class:`Event` wraps everything needed to evaluate one sampled transient's
detectability against a real survey schedule: its position, redshift, the
luminosity distance and E(B-V) already cached in the catalog at generation time
(see :meth:`~uvex_transients.simulation.core.SurveySimulator.generate_events` --
neither is re-derived here), its SED parameter seed, and the schedule's own record
of which observations actually covered its position during its active window.

Normally reconstructed via :meth:`~uvex_transients.simulation.event_catalog.EventCatalog.get_events`,
not constructed directly.
"""

from collections.abc import Hashable

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable, vstack
from astropy.time import Time
from astropy.units import Quantity
from m4opt.missions import Mission

from uvex_transients.dust import log_attenuation
from uvex_transients.models._utils import simulate_flat_photometry
from uvex_transients.utils import get_rng

from ..surveys.base import SurveySchedule
from ..transients.base import TransientBase


class Event:
    """
    One sampled transient event, tied to a real survey schedule.

    Building an `Event` runs exactly one query against `schedule`
    (:meth:`~uvex_transients.surveys.base.SurveySchedule.get_observations_of`) to find
    which scheduled observations actually covered this event's position during
    ``[t_explosion, t_explosion + transient.duration_limit)`` -- no photometry is done
    here. :meth:`simulate_photometry` does the (comparatively expensive) per-observation,
    per-band synthetic photometry, on demand.

    Parameters
    ----------
    event_id : int
        See :meth:`__init__`.
    schedule : ~uvex_transients.surveys.base.SurveySchedule
        See :meth:`__init__`.
    transient : ~uvex_transients.transients.base.TransientBase
        See :meth:`__init__`.
    coord : ~astropy.coordinates.SkyCoord
        See :meth:`__init__`.
    redshift : float
        See :meth:`__init__`.
    t_explosion : ~astropy.time.Time
        See :meth:`__init__`.
    seed : int
        See :meth:`__init__`.
    luminosity_distance : ~astropy.units.Quantity, optional
        See :meth:`__init__`.
    ebv : float, optional
        See :meth:`__init__`.
    transient_type : str, optional
        See :meth:`__init__`.
    photometry_pre_window : ~astropy.units.Quantity, optional
        See :meth:`__init__`.
    photometry_post_window : ~astropy.units.Quantity, optional
        See :meth:`__init__`.
    """

    def __init__(
        self,
        event_id: int,
        schedule: SurveySchedule,
        transient: TransientBase,
        coord: SkyCoord,
        redshift: float,
        t_explosion: Time,
        seed: int,
        *,
        luminosity_distance: Quantity | None = None,
        ebv: float | None = None,
        transient_type: str | None = None,
        photometry_pre_window: Quantity | None = 0 * u.day,
        photometry_post_window: Quantity | None = None,
    ):
        """
        Construct an `Event` and query `schedule` for its covering observations.

        Parameters
        ----------
        event_id : int
            This event's unique id, as assigned by `EventCatalog`.
        schedule : ~uvex_transients.surveys.base.SurveySchedule
            The survey schedule to check this event's visibility against.
        transient : ~uvex_transients.transients.base.TransientBase
            The transient type this event is a realization of -- supplies `sed` and
            `duration_limit`.
        coord : ~astropy.coordinates.SkyCoord
            Scalar sky position of the event.
        redshift : float
            Cosmological redshift.
        t_explosion : ~astropy.time.Time
            Scalar time of explosion.
        seed : int
            This event's stored parameter seed; deterministically regenerates its
            physical SED parameters (see :meth:`sample_parameters`).
        luminosity_distance : ~astropy.units.Quantity, optional
            Luminosity distance, if already known (e.g. cached by `EventCatalog` at
            generation time). If not given, it's derived from `redshift` via
            `transient.cosmology` -- a real cosmology call, so passing it in when
            it's already on hand is the whole point of caching it.
        ebv : float, optional
            Milky Way foreground E(B-V) at `coord`, if already known (cached by
            `EventCatalog`). If not given, reddening is treated as zero.
        transient_type : str, optional
            This event's transient-type name, for `__repr__` only.
        photometry_pre_window : ~astropy.units.Quantity, optional
            The duration (with units of time) prior to the true explosion time to include in the
            synthetic photometry. By default, this is ``0 * u.day``, meaning that the detections
            all occur after the explosion epoch. This setting should be used to generate pre-explosion
            reference data equivalent to drawing forced photometry from a pre-explosion template image.
        photometry_post_window : ~astropy.units.Quantity, optional
            The duration (with units of time) following the explosion time to include in the synthetic
            photometry. By default, ``photometry_post_window=transient_type.duration_limit``; however,
            this can be modified as needed by the user.
        """
        if not isinstance(schedule, SurveySchedule):
            raise TypeError(f"'schedule' must be a SurveySchedule, got {type(schedule)}.")
        if not isinstance(transient, TransientBase):
            raise TypeError(f"'transient' must be a TransientBase, got {type(transient)}.")

        self._event_id = int(event_id)
        self._schedule = schedule
        self._transient = transient
        self._transient_type = transient_type
        self._coord = coord
        self._redshift = float(redshift)
        self._t_explosion = t_explosion
        self._seed = int(seed)
        self._luminosity_distance = (
            luminosity_distance
            if luminosity_distance is not None
            else transient.cosmology.luminosity_distance(self._redshift)
        )
        self._ebv = 0.0 if ebv is None else float(ebv)

        # The one query this class exists to make: which scheduled observations actually
        # covered this event, while it was active. `get_observations_of`'s own window can
        # include a row that *started* slightly before `t_explosion` but overlaps into it
        # (e.g. a long downlink); that row is dropped here since only observations that
        # began at or after the explosion are ever useful for this event's lightcurve.
        if photometry_post_window is None:
            photometry_post_window = transient.duration_limit

        if not isinstance(photometry_post_window, u.Quantity):
            raise TypeError(f"'photometry_post_window' must be a Quantity, got {type(photometry_post_window)}.")
        if not isinstance(photometry_pre_window, u.Quantity):
            raise TypeError(f"'photometry_pre_window' must be a Quantity, got {type(photometry_pre_window)}.")

        try:
            _ = photometry_pre_window.to(u.day).value
        except u.UnitConversionError as err:
            raise u.UnitConversionError(
                f"'photometry_pre_window' must be convertible to time units, got {photometry_pre_window.unit}."
            ) from err

        try:
            _ = photometry_post_window.to(u.day).value
        except u.UnitConversionError as err:
            raise u.UnitConversionError(
                f"'photometry_post_window' must be convertible to time units, got {photometry_post_window.unit}."
            ) from err

        self._photometry_post_window = photometry_post_window
        self._photometry_pre_window = photometry_pre_window
        candidate = schedule.get_observations_of(
            coord, t_explosion - photometry_pre_window, t_explosion + photometry_post_window
        )

        # Determine the set of candidates that are actually observed.
        self._window_observations = candidate
        in_model = np.logical_and(
            candidate["start_time"] >= t_explosion,
            candidate["start_time"] + candidate["duration"] <= t_explosion + photometry_post_window,
        )
        self._observations = candidate[in_model]

        # Everything else in the query window -- pre-explosion rows from
        # `photometry_pre_window`, plus any row whose exposure runs past
        # `photometry_post_window` -- falls outside the transient SED's valid
        # domain (`t >= 0`). These are never evaluated against `transient.sed`
        # (see `simulate_photometry`): some light curve shapes are only smoothly
        # *wrong* there, but others (e.g. an early-time `1/t` singularity) diverge
        # outright, so this window is masked out rather than merely deprioritized.
        self._background_observations = candidate[~in_model]

    def __repr__(self) -> str:
        """
        Return a one-line summary showing the event's id, type, redshift, and observation count.

        Returns
        -------
        str
            ``<Event id=... type=... z=... n_observations=...>``.
        """
        return (
            f"<Event id={self._event_id} type={self._transient_type!r} "
            f"z={self._redshift:.4g} n_observations={len(self._observations)}>"
        )

    # ------------------------------ #
    # Properties                     #
    # ------------------------------ #
    @property
    def event_id(self) -> int:
        """int: This event's unique id."""
        return self._event_id

    @property
    def transient(self) -> TransientBase:
        """TransientBase: The transient type this event is a realization of."""
        return self._transient

    @property
    def transient_type(self) -> str | None:
        """Str or None: This event's transient-type name."""
        return self._transient_type

    @property
    def coord(self) -> SkyCoord:
        """~astropy.coordinates.SkyCoord: Sky position of the event."""
        return self._coord

    @property
    def redshift(self) -> float:
        """float: Cosmological redshift."""
        return self._redshift

    @property
    def luminosity_distance(self) -> Quantity:
        """~astropy.units.Quantity: Luminosity distance to the event."""
        return self._luminosity_distance

    @property
    def ebv(self) -> float:
        """float: Milky Way foreground E(B-V) at `coord` (0 if never supplied)."""
        return self._ebv

    @property
    def t_explosion(self) -> Time:
        """~astropy.time.Time: Time of explosion."""
        return self._t_explosion

    @property
    def seed(self) -> int:
        """int: This event's stored parameter seed."""
        return self._seed

    @property
    def observations(self) -> QTable:
        """
        QTable: The schedule's ``"observe"`` rows that covered this event while active.

        One row per candidate observation, chronological, with the same columns as
        `SurveySchedule.table` (``start_time``, ``duration``, ``observer_location``
        -- i.e. where the spacecraft was -- ``target_coord``, ``roll``, ...). Empty if
        the survey never observed this event's position during its active window.
        """
        return self._observations

    @property
    def window_observations(self) -> QTable:
        """
        QTable: The schedule's ``"observe"`` rows covering the full query window.

        `observations` plus `background_observations`, one row per candidate
        observation, chronological, over
        ``[t_explosion - photometry_pre_window, t_explosion + photometry_post_window]`` --
        i.e. `observations` before it's narrowed to rows the transient's SED is actually
        valid for. See `observations`/`background_observations` for the split.
        """
        return self._window_observations

    @property
    def background_observations(self) -> QTable:
        """
        QTable: The schedule's ``"observe"`` rows in `window_observations` outside `observations`.

        Pre-explosion rows (from `photometry_pre_window`) plus any row whose exposure
        runs past `photometry_post_window` -- i.e. every candidate observation
        `simulate_photometry` gives background-only (non-detection) photometry rather
        than evaluating `transient.sed` for, since `t < 0` isn't in the SED's valid
        domain. Empty whenever `photometry_pre_window` is the default ``0 * u.day`` and
        no candidate observation overruns `photometry_post_window`.
        """
        return self._background_observations

    @property
    def n_observations(self) -> int:
        """int: Number of candidate observations (``len(observations)``)."""
        return len(self._observations)

    # ------------------------------ #
    # Parameters                     #
    # ------------------------------ #
    def sample_parameters(self) -> dict:
        """
        Return this event's physical SED parameters, deterministically regenerated from `seed`.

        A pure, idempotent "peek" -- every call reseeds a fresh generator from `seed`
        rather than sharing state with `simulate_photometry`, so this always returns
        the same values regardless of how many times it (or `simulate_photometry`)
        has already been called.

        Returns
        -------
        dict
            ``{name: value}`` for each of this event's transient type's SED parameters.
        """
        rng = get_rng(self._seed)
        return {name: value[0] for name, value in self._transient.sed.sample_parameters(size=1, rng=rng).items()}

    # ------------------------------ #
    # Theoretical Photometry         #
    # ------------------------------ #
    def _resolve_bandpass(self, mission: Mission, band: Hashable | None):
        """
        Return the `~synphot.SpectralElement` named `band` in `mission.detector.bandpasses`.

        `band` may be omitted only if the detector has exactly one bandpass -- mirrors
        `~m4opt.synphot.Detector`'s own bandpass-resolution rule (see
        `~m4opt.synphot.Detector.get_snr`).

        Parameters
        ----------
        mission : ~m4opt.missions.Mission
            Supplies the `~m4opt.synphot.Detector` `band` selects a bandpass from.
        band : Hashable, optional
            Which of `mission.detector`'s bandpasses to return. Required unless the
            detector has exactly one.

        Returns
        -------
        ~synphot.SpectralElement
            The resolved bandpass.

        Raises
        ------
        ValueError
            If `mission` has no detector configured, `band` is required but not
            given, or `band` isn't one of the detector's bandpasses.
        """
        detector = mission.detector
        if detector is None:
            raise ValueError(f"Mission {mission.name!r} has no detector configured.")
        if band is not None:
            return detector.bandpasses[band]
        if len(detector.bandpasses) == 1:
            (bandpass,) = detector.bandpasses.values()
            return bandpass
        raise ValueError(
            f"Mission {mission.name!r} has more than one bandpass. Please specify "
            f"one of them: {list(detector.bandpasses)}."
        )

    def _pivot_nu_and_log_attenuation(self, bandpass) -> tuple[Quantity, np.ndarray]:
        """
        Return `bandpass`'s pivot frequency and this event's own dust attenuation there.

        The same single-wavelength approximation `simulate_photometry` uses for its
        noiseless flux (`SpectralElement.pivot`), not a full bandpass-throughput
        integral -- see :meth:`mag`'s docstring for why that distinction matters here.

        Parameters
        ----------
        bandpass : ~synphot.SpectralElement
            The bandpass to evaluate.

        Returns
        -------
        tuple of (~astropy.units.Quantity, numpy.ndarray)
            `bandpass`'s pivot frequency, and the natural log of this event's
            dust attenuation there (see :func:`~uvex_transients.dust.log_attenuation`).
        """
        nu = bandpass.pivot().to(u.Hz, equivalencies=u.spectral())
        return nu, log_attenuation(nu, self._ebv)

    def mag(self, t: Quantity, mission: Mission, band: Hashable | None = None) -> Quantity:
        """
        Theoretical (noiseless) apparent AB magnitude of this event's SED at time(s) `t`.

        This is the literal noiseless curve :meth:`simulate_photometry`'s noisy points
        scatter around, not merely a physically similar but numerically different
        quantity: like that method, the flux is evaluated at `band`'s pivot
        wavelength (`~synphot.SpectralElement.pivot`), *not* integrated across the
        full bandpass throughput, and this event's own foreground dust (`ebv`) is
        applied, in addition to its `redshift`/`luminosity_distance`/SED parameters
        (see :meth:`sample_parameters`) -- so only a time grid and a mission/band are
        needed. Useful for plotting a theory curve alongside real observations, not
        for simulating a detection.

        Parameters
        ----------
        t : ~astropy.units.Quantity
            Observed time(s) since explosion, any shape.
        mission : ~m4opt.missions.Mission
            Supplies the `~m4opt.synphot.Detector` `band` selects a bandpass from.
        band : Hashable, optional
            Which of `mission.detector`'s bandpasses to evaluate. Required unless the
            detector has exactly one.

        Returns
        -------
        ~astropy.units.Quantity
            The apparent AB magnitude, as an :attr:`~astropy.units.ABmag` Quantity,
            with `t`'s shape.
        """
        bandpass = self._resolve_bandpass(mission, band)
        nu, attenuation = self._pivot_nu_and_log_attenuation(bandpass)
        return self._transient.sed.mag(
            nu,
            t,
            redshift=self._redshift,
            luminosity_distance=self._luminosity_distance,
            log_attenuation=attenuation,
            **self.sample_parameters(),
        )

    def flux(self, t: Quantity, mission: Mission, band: Hashable | None = None) -> Quantity:
        """
        Theoretical (noiseless) observed flux density of this event's SED, at `band`'s pivot wavelength.

        Same inputs, and the same pivot-wavelength-plus-dust definition matching
        :meth:`simulate_photometry` exactly, as :meth:`mag` -- see its docstring.

        Parameters
        ----------
        t : ~astropy.units.Quantity
            Observed time(s) since explosion, any shape.
        mission : ~m4opt.missions.Mission
            Supplies the `~m4opt.synphot.Detector` `band` selects a bandpass from.
        band : Hashable, optional
            Which of `mission.detector`'s bandpasses to evaluate. Required unless the
            detector has exactly one.

        Returns
        -------
        ~astropy.units.Quantity
            The observed flux density, with `t`'s shape.
        """
        bandpass = self._resolve_bandpass(mission, band)
        nu, attenuation = self._pivot_nu_and_log_attenuation(bandpass)
        return self._transient.sed.flux(
            nu,
            t,
            redshift=self._redshift,
            luminosity_distance=self._luminosity_distance,
            log_attenuation=attenuation,
            **self.sample_parameters(),
        )

    def luminosity(self, t: Quantity) -> Quantity:
        r"""
        Bolometric luminosity of this event's SED at time(s) `t` since explosion.

        Intrinsic (rest-frame, distance-independent) -- unlike :meth:`mag`/:meth:`flux`,
        no `mission`/`band` is involved. This event's own SED parameters (see
        :meth:`sample_parameters`) are supplied automatically. Wraps
        `~uvex_transients.models.core.base.SpectralModel.eval_bolometric`.

        Parameters
        ----------
        t : ~astropy.units.Quantity
            Time(s) since explosion, any shape.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`L_\\mathrm{bol}(t)`, in erg/s, with `t`'s shape.
        """
        return self._transient.sed.eval_bolometric(t, **self.sample_parameters())

    # ------------------------------ #
    # Photometry                     #
    # ------------------------------ #
    @staticmethod
    def _empty_photometry_table() -> QTable:
        """
        Return a zero-row `QTable` with `simulate_photometry`'s columns and dtypes.

        Returned as-is by `simulate_photometry` when there are no observations to
        simulate, so a caller can always rely on the result having the right columns.

        Returns
        -------
        ~astropy.table.QTable
            An empty table with `simulate_photometry`'s schema.
        """
        table = QTable()
        table["event_id"] = np.array([], dtype=np.int64)
        table["obs_time"] = Time([], format="jd")
        table["rel_time"] = u.Quantity([], u.day)
        table["exptime"] = u.Quantity([], u.s)
        # A wide, explicit itemsize -- `dtype=str` alone infers itemsize 1 from
        # an empty array (silently truncating any band name to one character
        # once populated rows are ever `vstack`'d onto this one).
        table["band"] = np.array([], dtype="<U32")
        table["snr"] = np.array([], dtype=np.float64)
        table["flux"] = u.Quantity([], u.Jy)
        table["flux_err"] = u.Quantity([], u.Jy)
        table["flux_upper"] = u.Quantity([], u.Jy)
        table["flux_lower"] = u.Quantity([], u.Jy)
        table["ab_mag"] = np.array([], dtype=np.float64)
        table["mag_err"] = np.array([], dtype=np.float64)
        table["mag_upper"] = np.array([], dtype=np.float64)
        table["mag_lower"] = np.array([], dtype=np.float64)
        table["in_model"] = np.array([], dtype=bool)
        return table

    def simulate_photometry(
        self,
        mission: Mission,
        bands: list | None = None,
        n_sigma: float | None = None,
    ) -> QTable:
        """
        Evaluate this event's detectability at every observation in `window_observations`.

        Builds one `~synphot.SourceSpectrum` for `observations` -- batched over every
        candidate observation's own time since explosion, via
        `~uvex_transients.models.core.base.SpectralModel.as_source_spectrum`
        (dust folded in through its ``log_attenuation``, from this event's own cached
        `ebv`) -- and reuses it, unmodified, for every requested band:
        a `~synphot.SourceSpectrum` is purely a function of wavelength/time, so
        "band" only enters once it's integrated against a bandpass, via
        `~m4opt.synphot.Detector.get_snr`. That turns what used to be a Python loop
        over every (observation, band) pair, each doing its own scalar
        `SpectralModel.flux`/`astropy.stats.signal_to_noise_oir_ccd` call, into one
        vectorized `get_snr` call per band, regardless of how many observations
        there are.

        `background_observations` -- every candidate observation outside `observations`
        (pre-explosion, from `photometry_pre_window`, plus any exposure running past
        `photometry_post_window`) -- never reaches `transient.sed` at all: `t < 0` isn't
        in a `SpectralModel`'s valid domain (some light curve shapes are only smoothly
        *wrong* there, but others, e.g. an early-time ``1/t`` singularity, diverge
        outright), so these rows instead get
        `~uvex_transients.models._utils.simulate_flat_photometry`'s pure
        background/non-detection photometry -- the same detector noise model
        (`~m4opt.synphot.Detector.get_snr`, real observing geometry included), just
        against a true source flux of zero rather than an evaluated SED. The two sets
        are simulated separately, then combined into one table -- see the ``in_model``
        column below.

        The reported flux/magnitude at each (observation, band) is then a Gaussian
        realization of the true (noiseless) flux -- evaluated from that same
        `SourceSpectrum` at the band's pivot wavelength -- at that SNR's implied
        uncertainty; a synthetic measurement, not the ground truth. Physical SED
        parameters are sampled once, from `seed`; every band's noise draws (`observations`
        first, then `background_observations`) are one vectorized call over all
        observations at once, in `bands` order, on the same stream that draw consumed --
        so the whole event still replays identically given the same `seed`, but *not*
        row-for-row identically to an older, unbatched implementation, since the draws
        are now grouped per band across every observation rather than interleaved
        observation-by-observation.

        ``mag_err`` is the usual linearized (first-order) propagation of ``flux_err``
        through the magnitude log transform -- a good description of the uncertainty
        only while it's small relative to ``flux``, i.e. at high SNR. It is *not* a
        substitute for a real confidence interval: because magnitude is a nonlinear
        (logarithmic) function of flux, a symmetric interval in flux is an asymmetric
        one in magnitude, and that asymmetry grows as SNR drops -- comparing against
        ``ab_mag``/``mag_err`` as if they were a plain Gaussian pull systematically
        reads as biased at low SNR even when the underlying flux draw has no bias at
        all. ``flux_upper``/``flux_lower`` and ``mag_upper``/``mag_lower`` are the
        actual ``n_sigma`` interval, built the correct way around: bound `flux`
        symmetrically first (where the noise is actually Gaussian), then transform
        each bound to magnitude separately, rather than propagating one linearized
        width through the transform. For a `background_observations` row this interval
        is exactly the pure-noise upper limit expected of a non-detection: ``mag_upper``
        is ``nan`` (no faint bound -- the "source" is, by construction, never securely
        distinguished from zero flux) and ``mag_lower`` is the ``n_sigma`` detection
        depth reached by that observation's own real exposure/background.

        Parameters
        ----------
        mission : m4opt.missions.Mission
            Supplies the `~m4opt.synphot.Detector` (bandpasses, background, ...) evaluated
            against.
        bands : list of str, optional
            Which of `mission.detector`'s bandpasses to evaluate. Defaults to every
            bandpass the detector has.
        n_sigma : float, optional
            Width, in multiples of ``flux_err``, of the ``flux_upper``/``flux_lower``/
            ``mag_upper``/``mag_lower`` interval. If `None` (the default), uses
            ``config["simulation.detection_n_sigma"]`` (5 out of the box).

        Returns
        -------
        astropy.table.QTable
            One row per (observation, band), sorted by ``obs_time`` then ``band``,
            with columns ``event_id``, ``obs_time``, ``rel_time`` (``obs_time -
            t_explosion``, in days -- negative for a `background_observations` row),
            ``exptime``, ``band``, ``snr``, ``flux``/``flux_err`` (Jy),
            ``flux_upper``/``flux_lower`` (Jy, ``flux ±
            n_sigma*flux_err``), ``ab_mag``/``mag_err``, ``mag_upper``/``mag_lower``
            -- the ``n_sigma`` interval transformed to magnitude, brighter bound first:
            ``mag_lower`` (from ``flux_upper``) is always finite when ``flux_upper>0``;
            ``mag_upper`` (from ``flux_lower``) is ``nan`` whenever ``flux_lower<=0``,
            i.e. whenever the source isn't securely distinguished from zero flux at
            ``n_sigma`` -- the correct behavior is a one-sided (no faint bound) result
            there, not a spuriously finite one -- and ``in_model``, ``True`` for a row
            from `observations` (a real evaluation of `transient.sed`) and ``False``
            for one from `background_observations` (pure background/non-detection,
            `transient.sed` never evaluated). ``flux``/``flux_err``/``ab_mag``/
            ``mag_err`` are ``nan`` wherever ``snr`` is non-positive or non-finite
            (``ab_mag``/``mag_err`` are also ``nan`` wherever the noisy ``flux``
            realization itself came out non-positive). Empty (but correctly typed) if
            `window_observations` is empty.
        """
        detector = mission.detector
        if detector is None:
            raise ValueError(f"Mission {mission.name!r} has no detector configured.")

        # Validated up front (rather than left to `SpectralModel.simulate_photometry`/
        # `simulate_flat_photometry`) so an unknown band raises even when both sets of
        # observations below are empty.
        band_names = list(detector.bandpasses) if bands is None else list(bands)
        unknown = [band for band in band_names if band not in detector.bandpasses]
        if unknown:
            raise ValueError(f"Unknown bandpass(es) {unknown}; available: {list(detector.bandpasses)}.")

        n_obs = len(self._observations)
        if n_obs == 0 and len(self._background_observations) == 0:
            return self._empty_photometry_table()

        # One RNG, seeded from this event's own stored `seed`, drives the parameter
        # draw and every noise realization below (`observations` first, then
        # `background_observations`, in that order) -- so the whole event replays
        # identically from `seed` alone. Deliberately *not* `self.sample_parameters()`
        # (which reseeds fresh every call): the noise draws must continue on the same
        # stream the parameter draw already consumed.
        rng = get_rng(self._seed)

        if n_obs > 0:
            sed_params = {
                name: value[0] for name, value in self._transient.sed.sample_parameters(size=1, rng=rng).items()
            }
            t_obs = (self._observations["start_time"] - self._t_explosion).to(u.day)

            # The actual noise simulation -- batched `as_source_spectrum` plus `get_snr`,
            # the Gaussian flux realization, and the `n_sigma` bound math -- lives on
            # `SpectralModel.simulate_photometry` now, schedule-independent, so it's shared
            # with callers that have no `SurveySchedule`/`Event` at all (e.g. a target of
            # opportunity). This event's own `observer_location`/`start_time` are passed
            # through as real `observer_location`/`obstime`, so `detector`'s own
            # `background` (left unmodified -- `background` isn't overridden here) sees the
            # same real observing geometry it always did.
            phot_model = self._transient.sed.simulate_photometry(
                t_obs,
                self._observations["duration"],
                detector,
                self._coord,
                bands=band_names,
                observer_location=self._observations["observer_location"],
                obstime=self._observations["start_time"],
                redshift=self._redshift,
                luminosity_distance=self._luminosity_distance,
                ebv=self._ebv,
                n_sigma=n_sigma,
                rng=rng,
                **sed_params,
            )
            phot_model["obs_time"] = self._t_explosion + phot_model["t"]
            phot_model["rel_time"] = phot_model["t"]
            del phot_model["t"]
            phot_model["in_model"] = np.ones(len(phot_model), dtype=bool)
        else:
            phot_model = self._empty_photometry_table()

        # `simulate_flat_photometry` shares `simulate_detector_photometry` with
        # `SpectralModel.simulate_photometry` above -- same detector noise model, same
        # `n_sigma` bound math, same real observing geometry -- but against a flat,
        # zero-flux placeholder spectrum instead of `transient.sed`, so `t < 0` never
        # reaches it. Continues the same `rng` stream `sed_params` (if sampled) already
        # advanced, for the same reason.
        t_bg = (self._background_observations["start_time"] - self._t_explosion).to(u.day)
        phot_bg = simulate_flat_photometry(
            t_bg,
            self._background_observations["duration"],
            detector,
            self._coord,
            bands=band_names,
            observer_location=self._background_observations["observer_location"],
            obstime=self._background_observations["start_time"],
            n_sigma=n_sigma,
            rng=rng,
        )
        phot_bg["obs_time"] = self._t_explosion + phot_bg["t"]
        phot_bg["rel_time"] = phot_bg["t"]
        del phot_bg["t"]
        phot_bg["in_model"] = np.zeros(len(phot_bg), dtype=bool)

        phot = vstack([phot_model, phot_bg])
        phot["event_id"] = np.full(len(phot), self._event_id, dtype=np.int64)
        order = np.lexsort((phot["band"], phot["obs_time"].jd))
        phot = phot[order]

        return phot[
            [
                "event_id",
                "obs_time",
                "rel_time",
                "exptime",
                "band",
                "snr",
                "flux",
                "flux_err",
                "flux_upper",
                "flux_lower",
                "ab_mag",
                "mag_err",
                "mag_upper",
                "mag_lower",
                "in_model",
            ]
        ]
