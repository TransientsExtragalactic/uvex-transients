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
from astropy.coordinates import SkyCoord
from astropy.table import QTable, vstack
from astropy.time import Time
from astropy.units import Quantity
from m4opt.missions import Mission
from tqdm.auto import tqdm

from uvex_transients.utils import logger

from ..surveys.base import SurveySchedule
from ..surveys.footprints import SurveyFootprint, default_registry
from ..transients.base import TransientBase
from .event import Event
from .photometry_catalog import PhotometryCatalog

_SeedType = Union[np.random.SeedSequence, int, None]


def _counts_from_meta(counts) -> dict[str, int] | None:
    """Coerce a ``pre_cut_counts`` header entry (as read back from ECSV) to a plain ``{str: int}`` dict, or `None`."""
    return None if counts is None else {str(name): int(count) for name, count in dict(counts).items()}


def _seed_to_meta(seed: _SeedType) -> int | None:
    """Reduce a root seed to something ECSV-header-serializable, for `EventCatalog.to_disk`."""
    if seed is None or isinstance(seed, (int, np.integer)):
        return None if seed is None else int(seed)

    if isinstance(seed, np.random.SeedSequence):
        entropy = seed.entropy
        return int(entropy) if isinstance(entropy, int) else None

    return None


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

    pre_cut_counts: dict[str, int] | None = None
    """dict of str to int, or None: The number of events of each transient type when generated.

    Counted once, at the end of `~uvex_transients.simulation.core.SurveySimulator.generate_events`
    (after any downsampling), and never changed by a cut, a set operation, or any other row
    selection. It is the binomial denominator of every later detection fraction, so a selection
    of rows can be turned into an efficiency, and its Monte Carlo error, without the original
    catalog. `None` for a catalog built by hand or read from a file written before this field
    existed.
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
    def in_footprint(self, footprint: SurveyFootprint | str) -> np.ndarray:
        """
        Mask of events whose sky position lies inside a footprint.

        One vectorized MOC lookup over every event, so no per-event loop is needed.

        Parameters
        ----------
        footprint : SurveyFootprint or str
            A footprint or its registered name (e.g. ``"lsst:combined"``).

        Returns
        -------
        numpy.ndarray
            Boolean mask, shape ``(n_events,)``, aligned with the catalog rows. Use it to
            index :attr:`table` or any per-event array.

        Raises
        ------
        KeyError
            If `footprint` is a name that is not registered.
        """
        return default_registry.resolve(footprint).contains_skycoord(self.coord)

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

            transient = transients[name]
            events.append(
                Event(
                    event_id=int(row["event_id"]),
                    schedule=schedule,
                    transient=transient,
                    coord=row["coord"],
                    redshift=float(row["redshift"]),
                    t_explosion=row["t_explosion"],
                    seed=int(row["parameter_seed"]),
                    luminosity_distance=row["luminosity_distance"] if has_distance else None,
                    ebv=float(row["ebv"]) if has_ebv else None,
                    transient_type=name,
                    # Each transient type's own settings (see
                    # `TransientBase.photometry_pre_window`/`photometry_post_window`) --
                    # not a fixed default -- so a config's per-class override actually
                    # reaches `Event.simulate_photometry`.
                    photometry_pre_window=transient.photometry_pre_window,
                    photometry_post_window=transient.photometry_post_window,
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

    def compute_photometry_catalog(
        self,
        mission: Mission,
        transients: dict[str, TransientBase],
        schedule: SurveySchedule,
        bands: list[str] | None = None,
        n_sigma: float | None = None,
    ) -> PhotometryCatalog:
        """
        `simulate_photometry`, wrapped as a `PhotometryCatalog` rather than a bare `QTable`.

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
        PhotometryCatalog
            This catalog's full synthetic photometry.
        """
        table = self.simulate_photometry(mission, transients, schedule, bands=bands, n_sigma=n_sigma)
        return PhotometryCatalog(table=table)

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
                "pre_cut_counts": self.pre_cut_counts,
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
            pre_cut_counts=_counts_from_meta(meta.pop("pre_cut_counts", None)),
        )


def _default_example_catalog_path() -> Path:
    import uvex_transients

    return Path(uvex_transients.__file__).parent.parent / "test_data" / "simulation" / "example_event_catalog.ecsv"


def get_example_event_catalog(path: Union[str, Path, None] = None) -> EventCatalog:
    """
    Load the packaged example `EventCatalog`, so docs/examples don't have to resample one.

    The packaged catalog (``test_data/simulation/example_event_catalog.ecsv``) is a raw
    (pre-cut) catalog of 598 events -- 344 `~uvex_transients.transients.supernovae.MagnetarSLSNe`
    ("slsn") and 254 `~uvex_transients.transients.TDEs.TidalDisruptionEvent` ("tde") -- generated
    once against the default schedule (see `~uvex_transients.surveys.get_schedule`) with
    ``simulation_seed=42``, ``time_bins=10``, ``nside=32``, ``downsample=500``. Sphinx gallery
    examples that only need a representative catalog to demonstrate a cut/action on can load this
    instead of re-running `SurveySimulator.generate_events` -- a nontrivial Monte Carlo draw --
    on every doc build.

    Parameters
    ----------
    path : str or ~pathlib.Path, optional
        Path to the catalog file. If `None` (the default), uses the packaged
        ``test_data/simulation/example_event_catalog.ecsv``.

    Returns
    -------
    EventCatalog
        The packaged example catalog.

    See Also
    --------
    EventCatalog.from_disk : The general-purpose loader this delegates to.
    """
    return EventCatalog.from_disk(path if path is not None else _default_example_catalog_path())
