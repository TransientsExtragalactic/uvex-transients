"""
A Monte Carlo realization of sampled transient events, produced by `SurveySimulator`.

`EventCatalog` is a pure data table -- it holds no live references to the
`~uvex_transients.surveys.base.SurveySchedule` or transient-type instances it was
generated against (see :meth:`EventCatalog.get_events`, which takes those in
explicitly rather than storing them), so it round-trips to/from disk cleanly and
stays trivially picklable/shareable on its own.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable, vstack
from astropy.time import Time
from astropy.units import Quantity
from m4opt.missions import Mission
from scipy.stats import beta as _beta_dist
from tqdm.auto import tqdm

from uvex_transients.utils import logger

from ..surveys.base import SurveySchedule
from ..transients.base import ExtragalacticTransient, TransientBase
from .event import Event
from .exposure_catalog import ExposureCatalog
from .yield_table import YieldTable

_SeedType = Union[np.random.SeedSequence, int, None]


def _seed_to_meta(seed: _SeedType) -> int | None:
    """Reduce a root seed to something ECSV-header-serializable, for `EventCatalog.to_disk`."""
    if seed is None or isinstance(seed, (int, np.integer)):
        return None if seed is None else int(seed)

    if isinstance(seed, np.random.SeedSequence):
        entropy = seed.entropy
        return int(entropy) if isinstance(entropy, int) else None

    return None


def _clopper_pearson_interval(k: int, n: int, confidence: float) -> tuple[float, float]:
    r"""
    Central Clopper-Pearson binomial confidence interval on :math:`k/n`.

    Implements :ref:`yield-statistics`'s "Confidence bounds from the simulated
    catalog" section exactly, including its boundary conventions -- ``k=0``, ``k=n``,
    and ``n=0`` are each handled as an explicit special case rather than left to the
    general Beta-quantile formula, which is singular at those points:

    .. math::

        \epsilon_{\mathrm L}=
        \begin{cases}
          0, & k=0,\\
          Q_{\rm B}(\alpha/2;k,n-k+1), & k>0,
        \end{cases}
        \qquad
        \epsilon_{\mathrm U}=
        \begin{cases}
          1, & k=n,\\
          Q_{\rm B}(1-\alpha/2;k+1,n-k), & k<n,
        \end{cases}

    where :math:`Q_{\rm B}` is the Beta-distribution quantile function. For an empty
    catalog (:math:`n=0`), the efficiency is unidentified; per the doc, this returns
    ``(0.0, 1.0)`` -- the widest possible interval, not a degenerate point.

    Parameters
    ----------
    k : int
        Number of "successes" (detections), ``0 <= k <= n``.
    n : int
        Number of trials (feasible Monte Carlo draws).
    confidence : float
        Confidence level :math:`C=1-\alpha`, in ``(0, 1)``.

    Returns
    -------
    tuple of float
        ``(lower, upper)`` bounds on the true binomial proportion.
    """
    if n == 0:
        return (0.0, 1.0)

    alpha = 1.0 - confidence
    lower = 0.0 if k == 0 else _beta_dist.ppf(alpha / 2, k, n - k + 1)
    upper = 1.0 if k == n else _beta_dist.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lower), float(upper))


@dataclass
class EventCatalog:
    """
    A Monte Carlo realization of sampled transient events, tagged by type and time bin.

    Produced by :meth:`~uvex_transients.simulation.core.SurveySimulator.generate_events`.
    Each event carries a ``parameter_seed`` rather than sampled physical SED parameters
    (see :meth:`~uvex_transients.transients.base.ExtragalacticTransient.sample_events_on_healpix_grid`)
    -- those are regenerated lazily, per event, by :meth:`Event.sample_parameters`/
    :meth:`Event.simulate_photometry` once an event is reconstructed via :meth:`get_events`.
    """

    table: QTable
    """QTable: One row per sampled event.

    Columns are those of ``sample_events_on_healpix_grid``'s own event table (``healpix_id``,
    ``healpix_dx``, ``healpix_dy``, ``coord``, ``redshift``, ``t_explosion``,
    ``parameter_seed``), plus ``transient_type`` (str), ``time_bin`` (the index into
    :attr:`time_bins` during which the event exploded), a unique ``event_id``, and two
    columns computed once at generation time so they never need re-deriving per event:
    ``luminosity_distance`` (from the event's own transient type's cached
    :attr:`~uvex_transients.transients.base.ExtragalacticTransient.luminosity_distance_grid`)
    and ``ebv`` (Milky Way foreground E(B-V) at the event's exact position).
    """

    nside: int
    """int: HEALPix resolution used both to query the observed footprint and to sample events."""

    order: str
    """str: HEALPix pixel ordering scheme (``"nested"`` or ``"ring"``)."""

    time_bins: Time
    """~astropy.time.Time: The ``n + 1`` bin edges events were sampled within."""

    seed: _SeedType = None
    """numpy.random.SeedSequence, int, or None: The root seed this catalog was generated from."""

    downsample: int | dict[str, int] | None = None
    """int, dict of str to int, or None: The downsample factor(s) generation was run with.

    Either a single factor applied to every transient type, a ``{transient key: factor}``
    mapping giving a per-type factor (a type missing from the mapping wasn't downsampled), or
    `None` if generation wasn't downsampled at all -- see
    `~uvex_transients.simulation.core.SurveySimulator.generate_events`.
    """

    # ----------------------------------------- #
    # Dunder Methods                            #
    # ----------------------------------------- #
    def __len__(self) -> int:
        return len(self.table)

    # ----------------------------------------- #
    # Array Export                              #
    # ----------------------------------------- #
    def column(self, name: str) -> np.ndarray:
        """
        Return a table column as a plain array, rather than an `astropy.table.Column`/`Quantity` view.

        Parameters
        ----------
        name : str
            Column name; must be one of `table.colnames`.

        Returns
        -------
        numpy.ndarray or ~astropy.units.Quantity
            `numpy.asarray` of the column, preserving units if the column is a
            `Quantity` column (e.g. ``luminosity_distance``).
        """
        if name not in self.table.colnames:
            raise KeyError(f"No column {name!r} in this catalog; available: {self.table.colnames}.")

        col = self.table[name]
        return col if isinstance(col, (Quantity, Time, SkyCoord)) else np.asarray(col)

    @property
    def event_id(self) -> np.ndarray:
        """numpy.ndarray: Every event's unique id, shape ``(n_events,)``."""
        return self.column("event_id")

    @property
    def healpix_id(self) -> np.ndarray:
        """numpy.ndarray: Every event's HEALPix pixel index, at this catalog's own :attr:`nside`/:attr:`order`."""
        return self.column("healpix_id")

    @property
    def transient_type(self) -> np.ndarray:
        """numpy.ndarray of str: Every event's transient-type name."""
        return np.asarray(self.table["transient_type"]).astype(str)

    @property
    def time_bin(self) -> np.ndarray:
        """numpy.ndarray: The time-bin index each event was sampled within."""
        return self.column("time_bin")

    @property
    def coord(self) -> SkyCoord:
        """~astropy.coordinates.SkyCoord: Every event's sky position."""
        return self.column("coord")

    @property
    def redshift(self) -> np.ndarray:
        """numpy.ndarray: Every event's redshift."""
        return self.column("redshift")

    @property
    def luminosity_distance(self) -> Quantity:
        """~astropy.units.Quantity: Every event's luminosity distance (cached at generation time)."""
        return self.column("luminosity_distance")

    @property
    def ebv(self) -> np.ndarray:
        """numpy.ndarray: Every event's Milky Way foreground E(B-V) (cached at generation time)."""
        return self.column("ebv")

    @property
    def t_explosion(self) -> Time:
        """~astropy.time.Time: Every event's explosion time."""
        return self.column("t_explosion")

    @property
    def parameter_seed(self) -> np.ndarray:
        """numpy.ndarray: Every event's stored SED parameter seed."""
        return self.column("parameter_seed")

    # ----------------------------------------- #
    # Event Reconstruction                      #
    # ----------------------------------------- #
    def get_events(
        self,
        ids: int | np.ndarray | list,
        transients: dict[str, TransientBase],
        schedule: SurveySchedule,
    ) -> Event | list[Event]:
        """
        Reconstruct one or more `Event` objects from their `event_id`.

        `EventCatalog` itself holds no live reference to `transients`/`schedule` (see
        the module docstring), so both must be supplied here -- typically the same
        `dict`/`SurveySchedule` the catalog was generated from.

        Parameters
        ----------
        ids : int or array_like of int
            One event id, or several. A scalar `ids` returns a single `Event`; anything
            array-like returns a `list` of `Event`, in the order given.
        transients : dict[str, TransientBase]
            Transient-type instances, keyed by the same names used in the ``transient_type``
            column (e.g. `SurveySimulator.transient_collection`).
        schedule : ~uvex_transients.surveys.base.SurveySchedule
            The survey schedule to check each reconstructed event's visibility against.

        Returns
        -------
        Event or list[Event]
            Reconstructed event or events corresponding to the requested event ids.

        Raises
        ------
        KeyError
            If an id isn't present in this catalog, or its ``transient_type`` isn't a key
            of `transients`.
        """
        scalar = np.ndim(ids) == 0
        id_array = np.atleast_1d(np.asarray(ids, dtype=np.int64))

        table = self.table
        event_id_col = np.asarray(table["event_id"])
        has_distance = "luminosity_distance" in table.colnames
        has_ebv = "ebv" in table.colnames

        events: list[Event] = []
        for eid in id_array:
            matches = np.flatnonzero(event_id_col == eid)
            if matches.size == 0:
                raise KeyError(f"No event with id {int(eid)!r} in this catalog.")

            row = table[int(matches[0])]
            name = str(row["transient_type"])
            if name not in transients:
                raise KeyError(f"No transient type {name!r} in 'transients'; available: {list(transients)}.")

            events.append(
                Event(
                    event_id=int(row["event_id"]),
                    schedule=schedule,
                    transient=transients[name],
                    coord=row["coord"],
                    redshift=float(row["redshift"]),
                    t_explosion=row["t_explosion"],
                    seed=int(row["parameter_seed"]),
                    luminosity_distance=row["luminosity_distance"] if has_distance else None,
                    ebv=float(row["ebv"]) if has_ebv else None,
                    transient_type=name,
                )
            )

        return events[0] if scalar else events

    def simulate_photometry(
        self,
        mission: Mission,
        transients: dict[str, TransientBase],
        schedule: SurveySchedule,
        bands: list[str] | None = None,
        n_sigma: float | None = None,
    ) -> QTable:
        """
        Run `~uvex_transients.simulation.event.Event.simulate_photometry` over every event in this catalog.

        Reconstructs every row as an `Event` (via `get_events`), runs its own
        `Event.simulate_photometry` (one `~synphot.SourceSpectrum` per event, batched over
        that event's own observations), and stacks every event's table together -- there is
        no cross-event batching here, just a loop; the expensive vectorization already
        happens per event, inside `Event.simulate_photometry` itself.

        Parameters
        ----------
        mission : m4opt.missions.Mission
            Supplies the `~m4opt.synphot.Detector` (bandpasses, background, ...) evaluated
            against.
        transients : dict[str, TransientBase]
            Transient-type instances, keyed by the same names used in this catalog's
            ``transient_type`` column -- forwarded to `get_events`.
        schedule : ~uvex_transients.surveys.base.SurveySchedule
            The survey schedule to check each event's visibility against -- forwarded to
            `get_events`.
        bands : list of str, optional
            Which of `mission.detector`'s bandpasses to evaluate. Defaults to every
            bandpass the detector has.
        n_sigma : float, optional
            Forwarded to `Event.simulate_photometry`; see its own docstring.

        Returns
        -------
        astropy.table.QTable
            The `astropy.table.vstack` of every event's own photometry table (see
            `Event.simulate_photometry`'s docstring for the column schema) -- one row per
            (event, observation, band). Empty (but correctly typed) if this catalog itself
            is empty.
        """
        if len(self) == 0:
            return Event._empty_photometry_table()

        events = self.get_events(self.event_id, transients, schedule)
        tables = [
            event.simulate_photometry(mission, bands=bands, n_sigma=n_sigma)
            for event in tqdm(events, desc="Simulating photometry", unit="event")
        ]
        return vstack(tables, metadata_conflicts="silent")

    def compute_detection_efficiency(
        self,
        detected: "EventCatalog",
        confidence: float = 0.9,
    ) -> dict[str, dict[str, float]]:
        r"""
        Estimate each transient type's detection efficiency :math:`\hat\epsilon = k/n`.

        ``n`` is this catalog's own row count for a type -- the number of *feasible*
        Monte Carlo events actually drawn for it (within the survey's footprint and
        the transient's redshift limit, before any detection cut) -- and ``k`` is
        `detected`'s row count for that same type, after whatever cut(s) produced it
        (see `~uvex_transients.simulation.core.SurveySimulator.run_cut`). This is
        exactly :ref:`yield-statistics`'s "Estimating the expected yield" and
        "Confidence bounds from the simulated catalog" sections; see
        `_clopper_pearson_interval` for the binomial bounds themselves.

        Parameters
        ----------
        detected : EventCatalog
            The subset of `self` that satisfied the detection criterion -- typically
            one or more `SurveySimulator.run_cut` calls applied to `self`.
        confidence : float, optional
            Confidence level for the Clopper-Pearson interval. The default is ``0.9``.

        Returns
        -------
        dict[str, dict[str, float]]
            ``{transient type: {"n", "k", "efficiency", "efficiency_lower",
            "efficiency_upper"}}``, one entry per distinct `transient_type` present in
            `self`. ``efficiency`` is `numpy.nan` when ``n == 0`` (unidentified; see
            :ref:`yield-statistics` -- this is deliberately not read as zero).
        """
        types = self.transient_type
        detected_types = detected.transient_type

        result = {}
        for name in np.unique(types):
            n = int(np.sum(types == name))
            k = int(np.sum(detected_types == name))
            lower, upper = _clopper_pearson_interval(k, n, confidence)
            result[name] = {
                "n": n,
                "k": k,
                "efficiency": (k / n) if n > 0 else np.nan,
                "efficiency_lower": lower,
                "efficiency_upper": upper,
            }
        return result

    def compute_yield_summary(
        self,
        detected: "EventCatalog",
        exposure: ExposureCatalog,
        transients: dict[str, ExtragalacticTransient],
        confidence: float = 0.9,
    ) -> YieldTable:
        r"""
        Build a per-transient-type yield summary, combining this catalog, `detected`, and `exposure`.

        One row per transient type in `transients`, with:

        - ``total_exposure``/``total_exposure_fraction``:
          `~uvex_transients.simulation.exposure_catalog.ExposureCatalog.total_effective_exposure`/
          `~uvex_transients.simulation.exposure_catalog.ExposureCatalog.coverage_fraction`.
        - ``integrated_rate``: `~uvex_transients.transients.base.ExtragalacticTransient.integrated_rate`
          (the per-steradian, per-year rate integrated over redshift).
        - ``all_sky_rate``: that same rate restored to the full :math:`4\pi` sky
          (`~uvex_transients.transients.base.ExtragalacticTransient.all_sky_rate`), with no survey
          footprint applied.
        - ``uvex_intrinsic_rate``/``uvex_intrinsic_events``: the footprint-aware analogues of the
          previous two, derived from `exposure` rather than the full sky -- ``uvex_intrinsic_events``
          is exactly `~uvex_transients.simulation.exposure_catalog.ExposureCatalog.total_expected_events`
          (:math:`\mu_0` in :ref:`yield-statistics`), and ``uvex_intrinsic_rate`` is that same count
          divided by `~uvex_transients.simulation.exposure_catalog.ExposureCatalog.total_duration`.
        - ``detected_events``: :math:`k`, from `compute_detection_efficiency`.
        - ``detection_probability``: :math:`\hat\epsilon=k/n` (`compute_detection_efficiency`).
        - ``expected_detections``: :math:`\hat\lambda=\mu_0\hat\epsilon`, :ref:`yield-statistics`'s
          boxed yield estimator.

        Every rate-derived quantity (``integrated_rate``, ``all_sky_rate``,
        ``uvex_intrinsic_rate``, ``uvex_intrinsic_events``) carries the rate-only bounds implied by
        each transient's own ``RATE_CI`` as ``..._lower``/``..._upper`` columns -- these collapse to
        the point estimate when ``RATE_CI`` is unset, exactly like
        `~uvex_transients.transients.base.ExtragalacticTransient.rate_ci` itself.

        ``detection_probability`` and ``expected_detections`` each carry *two* separate two-sided
        intervals rather than one combined box (:ref:`yield-statistics`'s "simulation-only" vs.
        "rate-only" bounds, kept apart so either source of uncertainty stays inspectable on its
        own): ``..._binom_lower``/``..._binom_upper`` (Clopper-Pearson, propagated through
        :math:`\hat\lambda=\mu_0\hat\epsilon` for `expected_detections`) and ``..._rate_lower``/
        ``..._rate_upper`` (``RATE_CI``, holding :math:`\hat\epsilon` fixed). `detection_probability`
        itself doesn't depend on the rate normalization at all -- it's a ratio of Monte Carlo counts
        -- so its ``..._rate_lower``/``..._rate_upper`` columns always equal its own point estimate;
        they're included only so every row shares one column schema.

        Parameters
        ----------
        detected : EventCatalog
            Forwarded to `compute_detection_efficiency`.
        exposure : ExposureCatalog
            Supplies every footprint-aware quantity above; see the column list.
        transients : dict[str, ExtragalacticTransient]
            Transient-type instances, keyed the same way as `self.transient_type` and
            `exposure.transient_type`. One output row per key, sorted by name.
        confidence : float, optional
            Confidence level for the Clopper-Pearson binomial bounds. The default is ``0.9``.

        Returns
        -------
        YieldTable
            One row per transient type, sorted by name; see the column list above.

        Raises
        ------
        KeyError
            If `exposure` has no tabulated exposure for a type named in `transients`.
        """
        efficiencies = self.compute_detection_efficiency(detected, confidence=confidence)
        total_exposure = exposure.total_effective_exposure
        total_events = exposure.total_expected_events
        coverage = exposure.coverage_fraction
        total_duration = exposure.total_duration

        names = sorted(transients)
        missing = [name for name in names if name not in total_exposure]
        if missing:
            raise KeyError(
                f"No exposure tabulated for transient type(s) {missing}; available: {sorted(total_exposure)}."
            )

        columns = (
            "transient_type",
            "total_exposure",
            "total_exposure_fraction",
            "integrated_rate",
            "integrated_rate_lower",
            "integrated_rate_upper",
            "all_sky_rate",
            "all_sky_rate_lower",
            "all_sky_rate_upper",
            "uvex_intrinsic_rate",
            "uvex_intrinsic_rate_lower",
            "uvex_intrinsic_rate_upper",
            "uvex_intrinsic_events",
            "uvex_intrinsic_events_lower",
            "uvex_intrinsic_events_upper",
            "detected_events",
            "detection_probability",
            "detection_probability_binom_lower",
            "detection_probability_binom_upper",
            "detection_probability_rate_lower",
            "detection_probability_rate_upper",
            "expected_detections",
            "expected_detections_binom_lower",
            "expected_detections_binom_upper",
            "expected_detections_rate_lower",
            "expected_detections_rate_upper",
        )
        rows = {column: [] for column in columns}

        default_efficiency = {"n": 0, "k": 0, "efficiency": np.nan, "efficiency_lower": 0.0, "efficiency_upper": 1.0}

        for name in names:
            transient = transients[name]
            lower_factor, upper_factor = transient.RATE_CI if transient.RATE_CI is not None else (1.0, 1.0)

            mu0 = total_events[name]
            intrinsic_rate = (mu0 / total_duration).to(u.yr**-1)

            eff = efficiencies.get(name, default_efficiency)
            eps = eff["efficiency"]
            eps_lower, eps_upper = eff["efficiency_lower"], eff["efficiency_upper"]

            lambda_hat = mu0 * eps

            rows["transient_type"].append(name)
            rows["total_exposure"].append(total_exposure[name])
            rows["total_exposure_fraction"].append(coverage[name])
            rows["integrated_rate"].append(transient.integrated_rate)
            rows["integrated_rate_lower"].append(transient.integrated_rate_ci[0])
            rows["integrated_rate_upper"].append(transient.integrated_rate_ci[1])
            rows["all_sky_rate"].append(transient.all_sky_rate)
            rows["all_sky_rate_lower"].append(transient.all_sky_rate_ci[0])
            rows["all_sky_rate_upper"].append(transient.all_sky_rate_ci[1])
            rows["uvex_intrinsic_rate"].append(intrinsic_rate)
            rows["uvex_intrinsic_rate_lower"].append(intrinsic_rate * lower_factor)
            rows["uvex_intrinsic_rate_upper"].append(intrinsic_rate * upper_factor)
            rows["uvex_intrinsic_events"].append(mu0)
            rows["uvex_intrinsic_events_lower"].append(mu0 * lower_factor)
            rows["uvex_intrinsic_events_upper"].append(mu0 * upper_factor)
            rows["detected_events"].append(eff["k"])
            rows["detection_probability"].append(eps)
            rows["detection_probability_binom_lower"].append(eps_lower)
            rows["detection_probability_binom_upper"].append(eps_upper)
            rows["detection_probability_rate_lower"].append(eps)
            rows["detection_probability_rate_upper"].append(eps)
            rows["expected_detections"].append(lambda_hat)
            rows["expected_detections_binom_lower"].append(mu0 * eps_lower)
            rows["expected_detections_binom_upper"].append(mu0 * eps_upper)
            rows["expected_detections_rate_lower"].append(lambda_hat * lower_factor)
            rows["expected_detections_rate_upper"].append(lambda_hat * upper_factor)

        table = QTable()
        table["transient_type"] = np.asarray(rows["transient_type"])
        table["total_exposure"] = u.Quantity(rows["total_exposure"])
        table["total_exposure_fraction"] = np.asarray(rows["total_exposure_fraction"], dtype=np.float64)
        for column in (
            "integrated_rate",
            "integrated_rate_lower",
            "integrated_rate_upper",
            "all_sky_rate",
            "all_sky_rate_lower",
            "all_sky_rate_upper",
            "uvex_intrinsic_rate",
            "uvex_intrinsic_rate_lower",
            "uvex_intrinsic_rate_upper",
        ):
            table[column] = u.Quantity(rows[column])
        table["uvex_intrinsic_events"] = np.asarray(rows["uvex_intrinsic_events"], dtype=np.float64)
        table["uvex_intrinsic_events_lower"] = np.asarray(rows["uvex_intrinsic_events_lower"], dtype=np.float64)
        table["uvex_intrinsic_events_upper"] = np.asarray(rows["uvex_intrinsic_events_upper"], dtype=np.float64)
        table["detected_events"] = np.asarray(rows["detected_events"], dtype=np.int64)
        for column in (
            "detection_probability",
            "detection_probability_binom_lower",
            "detection_probability_binom_upper",
            "detection_probability_rate_lower",
            "detection_probability_rate_upper",
            "expected_detections",
            "expected_detections_binom_lower",
            "expected_detections_binom_upper",
            "expected_detections_rate_lower",
            "expected_detections_rate_upper",
        ):
            table[column] = np.asarray(rows[column], dtype=np.float64)

        return YieldTable(table=table, confidence=confidence)

    # ----------------------------------------- #
    # IO Methods                                #
    # ----------------------------------------- #
    def to_disk(self, path: str | Path, table_format: str | None = None, overwrite: bool = False) -> None:
        """
        Write this catalog's event table to disk as ECSV, with its provenance in the header.

        Parameters
        ----------
        path : str or ~pathlib.Path
            Destination path.
        table_format : str, optional
            Passed through to :meth:`~astropy.table.QTable.write`; if `None`, inferred from
            ``path``'s suffix.
        overwrite : bool
            Whether to overwrite an existing file at ``path``.
        """
        table = self.table.copy()
        table.meta.update(
            {
                "nside": int(self.nside),
                "order": self.order,
                "time_bins": self.time_bins,
                "seed": _seed_to_meta(self.seed),
                "downsample": self.downsample,
            }
        )
        table.write(Path(path), format=table_format, overwrite=overwrite)
        logger.info("Wrote event catalog (%d events) to %s.", len(table), path)

    @classmethod
    def from_disk(cls, path: str | Path, table_format: str | None = None) -> "EventCatalog":
        """
        Read an event catalog back from disk, as written by :meth:`to_disk`.

        Parameters
        ----------
        path : str or ~pathlib.Path
            Path to the event table, as written by :meth:`to_disk`.
        table_format : str, optional
            Passed through to :meth:`~astropy.table.QTable.read`; if `None`, inferred from
            ``path``'s suffix.

        Returns
        -------
        EventCatalog
            Event catalog reconstructed from the serialized table and metadata.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        table = QTable.read(path, format=table_format)
        meta = dict(table.meta)
        table.meta.clear()

        logger.info("Read event catalog (%d events) from %s.", len(table), path)
        return cls(
            table=table,
            nside=int(meta.pop("nside")),
            order=meta.pop("order"),
            time_bins=meta.pop("time_bins"),
            seed=meta.pop("seed", None),
            downsample=meta.pop("downsample", None),
        )
