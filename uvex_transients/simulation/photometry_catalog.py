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


@dataclass
class PhotometryCatalog:
    """
    Synthetic photometry for a whole `~uvex_transients.simulation.event_catalog.EventCatalog`.

    Produced by :meth:`~uvex_transients.simulation.event_catalog.EventCatalog.compute_photometry_catalog`.
    """

    table: QTable
    """QTable: One row per ``(event, observation, band)``.

    Columns are exactly `~uvex_transients.simulation.event.Event.simulate_photometry`'s own
    schema: ``event_id``, ``obs_time``, ``rel_time``, ``exptime``, ``band``, ``snr`` (the
    measured SNR), ``flux``/``flux_err`` (Jy), ``flux_upper``/``flux_lower`` (Jy),
    ``ab_mag``/``mag_err``, ``mag_upper``/``mag_lower``, and ``in_model``.
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
        """numpy.ndarray: Each row's measured signal-to-noise ratio, ``flux / flux_err``."""
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

        logger.info("Reading photometry catalog from %s.", path)
        table = QTable.read(path, format=table_format)
        logger.info("Read photometry catalog (%d rows) from %s.", len(table), path)
        return cls(table=table)
