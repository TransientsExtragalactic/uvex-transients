"""
Synthetic photometry of a full event catalog, produced by `EventCatalog.compute_photometry_catalog`.

`PhotometryCatalog` is `EventCatalog`'s per-observation counterpart: rather than one row
per sampled event, it holds one row per ``(event, observation, band)`` synthetic
measurement -- exactly `Event.simulate_photometry`'s own table schema, `vstack`'d across
every event in a catalog (see `EventCatalog.simulate_photometry`). Like `EventCatalog` and
`ExposureCatalog`, it holds no live reference back to the schedule, transient instances, or
event catalog it was computed against, so it round-trips to/from disk and pickles cleanly.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.table import QTable
from astropy.time import Time
from astropy.units import Quantity

from uvex_transients.utils import logger

from ..transients.base import ExtragalacticTransient
from ._stats import clopper_pearson_interval
from .exposure_catalog import ExposureCatalog


@dataclass
class PhotometryCatalog:
    """
    Synthetic photometry for a whole `~uvex_transients.simulation.event_catalog.EventCatalog`.

    Produced by :meth:`~uvex_transients.simulation.event_catalog.EventCatalog.compute_photometry_catalog`.
    """

    table: QTable
    """QTable: One row per ``(event, observation, band)``.

    Columns are exactly `~uvex_transients.simulation.event.Event.simulate_photometry`'s own
    schema: ``event_id``, ``obs_time``, ``exptime``, ``band``, ``snr``, ``flux``/``flux_err``
    (Jy), ``flux_upper``/``flux_lower`` (Jy), ``ab_mag``/``mag_err``, and
    ``mag_upper``/``mag_lower``.
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
        numpy.ndarray or ~astropy.units.Quantity or ~astropy.time.Time
            `numpy.asarray` of the column, preserving units/time if the column is a
            `Quantity`/`Time` column.
        """
        if name not in self.table.colnames:
            raise KeyError(f"No column {name!r} in this catalog; available: {self.table.colnames}.")

        col = self.table[name]
        return col if isinstance(col, (Quantity, Time)) else np.asarray(col)

    @property
    def event_id(self) -> np.ndarray:
        """numpy.ndarray: Each row's event id."""
        return self.column("event_id")

    @property
    def obs_time(self) -> Time:
        """~astropy.time.Time: Each row's observation start time."""
        return self.column("obs_time")

    @property
    def exptime(self) -> Quantity:
        """~astropy.units.Quantity: Each row's exposure duration."""
        return self.column("exptime")

    @property
    def band(self) -> np.ndarray:
        """numpy.ndarray of str: Each row's bandpass name."""
        return np.asarray(self.table["band"]).astype(str)

    @property
    def snr(self) -> np.ndarray:
        """numpy.ndarray: Each row's signal-to-noise ratio."""
        return self.column("snr")

    @property
    def flux(self) -> Quantity:
        """~astropy.units.Quantity: Each row's noisy flux realization."""
        return self.column("flux")

    @property
    def flux_err(self) -> Quantity:
        """~astropy.units.Quantity: Each row's flux uncertainty."""
        return self.column("flux_err")

    @property
    def flux_upper(self) -> Quantity:
        """~astropy.units.Quantity: Each row's bright-side flux bound."""
        return self.column("flux_upper")

    @property
    def flux_lower(self) -> Quantity:
        """~astropy.units.Quantity: Each row's faint-side flux bound."""
        return self.column("flux_lower")

    @property
    def ab_mag(self) -> np.ndarray:
        """numpy.ndarray: Each row's noisy AB magnitude realization."""
        return self.column("ab_mag")

    @property
    def mag_err(self) -> np.ndarray:
        """numpy.ndarray: Each row's linearized magnitude uncertainty."""
        return self.column("mag_err")

    @property
    def mag_upper(self) -> np.ndarray:
        """numpy.ndarray: Each row's faint-side magnitude bound (from `flux_lower`)."""
        return self.column("mag_upper")

    @property
    def mag_lower(self) -> np.ndarray:
        """numpy.ndarray: Each row's bright-side magnitude bound (from `flux_upper`)."""
        return self.column("mag_lower")

    # ----------------------------------------- #
    # Row Access                                #
    # ----------------------------------------- #
    def get_event(self, event_id: int) -> QTable:
        """
        Return this catalog's photometry rows for a single event.

        Just a mask on `table`'s own ``event_id`` column -- unlike
        `~uvex_transients.simulation.event_catalog.EventCatalog.get_events`, this doesn't
        reconstruct an `~uvex_transients.simulation.event.Event` (this catalog holds
        neither the transient instances nor the schedule that would take); it hands back
        the raw synthetic-photometry rows already computed for that event.

        Parameters
        ----------
        event_id : int
            The event id to select, as it appears in `table`'s own ``event_id`` column.

        Returns
        -------
        astropy.table.QTable
            This event's own rows, in `table`'s existing order (one row per
            ``(observation, band)``).

        Raises
        ------
        KeyError
            If `event_id` isn't present in this catalog.
        """
        mask = np.asarray(self.table["event_id"]) == event_id
        if not np.any(mask):
            raise KeyError(f"No event with id {event_id!r} in this catalog.")
        return self.table[mask]

    # ----------------------------------------- #
    # Detection-Count Statistics                #
    # ----------------------------------------- #
    def compute_detection_count_table(
        self,
        event_catalog,
        exposure: ExposureCatalog,
        transients: dict[str, ExtragalacticTransient],
        snr_threshold: float,
        confidence: float = 0.9,
    ) -> QTable:
        r"""
        Per-transient-type estimator of how many events would show :math:`N_{\rm det}\ge k` detected epochs.

        A "detection" here is per-epoch, not per-band: an event's row at a given
        ``obs_time`` counts as one detected epoch if *any* of its bands has
        ``snr > snr_threshold`` there -- the same one-band-suffices convention
        `~uvex_transients.simulation.core.SurveySimulator.filter_by_snr` uses to collapse
        bands before counting visits. Each event's own count of detected epochs,
        :math:`N_{\rm det}`, is histogrammed within each transient type, exactly as
        `~uvex_transients.simulation.event_catalog.EventCatalog.compute_detection_efficiency`
        histograms its own single ``k`` (its detected/undetected split is this table's
        :math:`N_{\rm det}\ge 1` row).

        Every event in `event_catalog` is accounted for, including one with zero rows in
        this catalog at all (e.g. a survey never observed it) -- it contributes
        :math:`N_{\rm det}=0`, so `n_total` always equals `event_catalog`'s own per-type
        row count, not just however many events happen to appear in this catalog's table.

        This is the full :ref:`yield-statistics` treatment, one threshold :math:`k` at a
        time, not just a raw Monte Carlo histogram: `fraction` :math:`=n_{\ge k}/n` is
        `~uvex_transients.simulation.event_catalog.EventCatalog.compute_detection_efficiency`'s
        own :math:`\hat\epsilon`, generalized from "detected at all" to "detected in
        :math:`\ge k` epochs", with the same Clopper-Pearson bounds
        (`~uvex_transients.simulation._stats.clopper_pearson_interval`). `expected_events`
        :math:`=\mu_0\hat\epsilon` is
        `~uvex_transients.simulation.event_catalog.EventCatalog.compute_yield_summary`'s own
        ``expected_detections`` estimator, generalized the same way -- :math:`\mu_0` (`exposure`'s
        `~uvex_transients.simulation.exposure_catalog.ExposureCatalog.total_expected_events`) is
        the intrinsic, footprint-aware expected event count, independent of `event_catalog`'s own
        (possibly downsampled) size, so `expected_events` is a real expected number of UVEX events
        -- not an artifact of how many Monte Carlo draws happened to be sampled or how heavily
        `event_catalog` was downsampled. Both `expected_events` uncertainty sources are kept
        separate exactly as `compute_yield_summary` keeps them: ``..._binom_lower``/``_upper``
        (Clopper-Pearson, propagated through :math:`\mu_0\hat\epsilon`) and ``..._rate_lower``/
        ``_upper`` (`~uvex_transients.transients.base.ExtragalacticTransient.RATE_CI`, holding
        :math:`\hat\epsilon` fixed).

        Parameters
        ----------
        event_catalog : ~uvex_transients.simulation.event_catalog.EventCatalog
            Supplies the full per-type event list this catalog's photometry was computed
            over (via its own ``event_id``/``transient_type`` columns), so that events
            with zero qualifying epochs are still represented at :math:`N_{\rm det}=0`, and
            `n_total`/`n` for the Clopper-Pearson bounds.
        exposure : ExposureCatalog
            Supplies :math:`\mu_0`, via
            `~uvex_transients.simulation.exposure_catalog.ExposureCatalog.total_expected_events`.
        transients : dict[str, ExtragalacticTransient]
            Transient-type instances, keyed the same way as `event_catalog.transient_type` and
            `exposure.transient_type`; supplies each type's `RATE_CI`.
        snr_threshold : float
            An epoch counts as detected if at least one band's ``snr`` exceeds this value.
        confidence : float, optional
            Confidence level for the Clopper-Pearson binomial bounds. The default is ``0.9``.

        Returns
        -------
        astropy.table.QTable
            One row per ``(transient_type, n_detections)`` pair present for that type
            (``n_detections`` running ``0..max`` for each type), with columns
            ``transient_type``, ``n_detections`` (:math:`N_{\rm det}=k`), ``n_events``
            (number of that type's events with exactly :math:`k` detected epochs), ``n_total``
            (that type's total event count, repeated on every row), ``n_at_least`` (number of
            that type's events with :math:`N_{\rm det}\ge k`, i.e. the reverse cumulative sum of
            `n_events`), ``fraction``/``fraction_lower``/``fraction_upper`` (:math:`\hat\epsilon`
            and its Clopper-Pearson bounds), and ``expected_events``/``expected_events_binom_lower``/
            ``expected_events_binom_upper``/``expected_events_rate_lower``/``expected_events_rate_upper``
            (:math:`\mu_0\hat\epsilon` and its two uncertainty sources). ``n_at_least``/``fraction``
            at ``n_detections == 1`` reproduce
            `~uvex_transients.simulation.event_catalog.EventCatalog.compute_detection_efficiency`'s
            own ``k``/``efficiency`` for the same `snr_threshold` and ``n_visits=1``.

        Raises
        ------
        KeyError
            If `exposure` has no tabulated exposure, or `transients` no instance, for a
            transient type present in `event_catalog`.
        """
        all_ids = event_catalog.event_id
        all_types = event_catalog.transient_type

        names = sorted(np.unique(all_types))
        missing_exposure = [name for name in names if name not in exposure.total_expected_events]
        if missing_exposure:
            raise KeyError(
                f"No exposure tabulated for transient type(s) {missing_exposure}; "
                f"available: {sorted(exposure.total_expected_events)}."
            )
        missing_transients = [name for name in names if name not in transients]
        if missing_transients:
            raise KeyError(f"No transient instance for type(s) {missing_transients}; available: {sorted(transients)}.")

        n_detections = np.zeros(len(all_ids), dtype=np.int64)
        id_to_index = {int(eid): i for i, eid in enumerate(all_ids)}

        phot = self.table
        if len(phot) > 0:
            event_ids = np.asarray(phot["event_id"], dtype=np.int64)
            obs_jd = np.asarray(phot["obs_time"].jd, dtype=np.float64)
            detected = np.asarray(phot["snr"]) > snr_threshold

            # Collapse bands: an (event, obs_time) "epoch" is detected if any of its bands are.
            visit_key = np.stack([event_ids.astype(np.float64), obs_jd], axis=1)
            _, visit_index, visit_inverse = np.unique(visit_key, axis=0, return_index=True, return_inverse=True)
            visit_detected = np.zeros(len(visit_index), dtype=bool)
            np.logical_or.at(visit_detected, visit_inverse, detected)

            detected_event_ids = event_ids[visit_index][visit_detected]
            uniq_eids, counts = np.unique(detected_event_ids, return_counts=True)
            for eid, count in zip(uniq_eids, counts):
                idx = id_to_index.get(int(eid))
                if idx is not None:
                    n_detections[idx] = count

        total_expected_events = exposure.total_expected_events

        columns = (
            "transient_type",
            "n_detections",
            "n_events",
            "n_total",
            "n_at_least",
            "fraction",
            "fraction_lower",
            "fraction_upper",
            "expected_events",
            "expected_events_binom_lower",
            "expected_events_binom_upper",
            "expected_events_rate_lower",
            "expected_events_rate_upper",
        )
        rows = {column: [] for column in columns}

        for name in names:
            transient = transients[name]
            lower_factor, upper_factor = transient.RATE_CI if transient.RATE_CI is not None else (1.0, 1.0)
            mu0 = total_expected_events[name]

            type_counts = n_detections[all_types == name]
            n_total = int(type_counts.size)
            max_k = int(type_counts.max()) if n_total > 0 else 0
            hist = np.bincount(type_counts, minlength=max_k + 1)
            at_least = np.cumsum(hist[::-1])[::-1]

            for k in range(max_k + 1):
                n_at_least = int(at_least[k])
                fraction_lower, fraction_upper = clopper_pearson_interval(n_at_least, n_total, confidence)
                fraction = (n_at_least / n_total) if n_total > 0 else np.nan
                expected = mu0 * fraction

                rows["transient_type"].append(name)
                rows["n_detections"].append(k)
                rows["n_events"].append(int(hist[k]))
                rows["n_total"].append(n_total)
                rows["n_at_least"].append(n_at_least)
                rows["fraction"].append(fraction)
                rows["fraction_lower"].append(fraction_lower)
                rows["fraction_upper"].append(fraction_upper)
                rows["expected_events"].append(expected)
                rows["expected_events_binom_lower"].append(mu0 * fraction_lower)
                rows["expected_events_binom_upper"].append(mu0 * fraction_upper)
                rows["expected_events_rate_lower"].append(expected * lower_factor)
                rows["expected_events_rate_upper"].append(expected * upper_factor)

        table = QTable()
        table["transient_type"] = np.asarray(rows["transient_type"])
        table["n_detections"] = np.asarray(rows["n_detections"], dtype=np.int64)
        table["n_events"] = np.asarray(rows["n_events"], dtype=np.int64)
        table["n_total"] = np.asarray(rows["n_total"], dtype=np.int64)
        table["n_at_least"] = np.asarray(rows["n_at_least"], dtype=np.int64)
        for column in (
            "fraction",
            "fraction_lower",
            "fraction_upper",
            "expected_events",
            "expected_events_binom_lower",
            "expected_events_binom_upper",
            "expected_events_rate_lower",
            "expected_events_rate_upper",
        ):
            table[column] = np.asarray(rows[column], dtype=np.float64)
        return table

    # ----------------------------------------- #
    # IO Methods                                #
    # ----------------------------------------- #
    def to_disk(self, path: str | Path, table_format: str | None = None, overwrite: bool = False) -> None:
        """
        Write this catalog's photometry table to disk as ECSV.

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
        self.table.write(Path(path), format=table_format, overwrite=overwrite)
        logger.info("Wrote photometry catalog (%d rows) to %s.", len(self.table), path)

    @classmethod
    def from_disk(cls, path: str | Path, table_format: str | None = None) -> "PhotometryCatalog":
        """
        Read a photometry catalog back from disk, as written by :meth:`to_disk`.

        Parameters
        ----------
        path : str or ~pathlib.Path
            Path to the photometry table, as written by :meth:`to_disk`.
        table_format : str, optional
            Passed through to :meth:`~astropy.table.QTable.read`; if `None`, inferred from
            ``path``'s suffix.

        Returns
        -------
        PhotometryCatalog
            Photometry catalog reconstructed from the serialized table.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        table = QTable.read(path, format=table_format)
        logger.info("Read photometry catalog (%d rows) from %s.", len(table), path)
        return cls(table=table)
