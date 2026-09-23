"""
Exposure catalog support module.

`ExposureCatalog` is `EventCatalog`'s footprint-only counterpart: rather than a Monte
Carlo realization of events, it tabulates how much sky each registered transient type
was actually exposed to, per time bin. This is the same footprint query
`~uvex_transients.simulation.core.SurveySimulator.generate_events` restricts its own
sampling to, reduced to a solid angle instead of a drawn population. Like
`EventCatalog`, it holds no live reference back to the schedule or transient instances
it was computed against, so it round-trips to/from disk and pickles cleanly.
"""

from dataclasses import dataclass
from pathlib import Path

import astropy_healpix as ah
import numpy as np
from astropy import units as u
from astropy.table import QTable
from astropy.time import Time
from astropy.units import Quantity

from uvex_transients.utils import logger


@dataclass
class ExposureCatalog:
    """
    Per-(transient type, time bin) effective exposure, tagged by type and time bin.

    Produced by :meth:`~uvex_transients.simulation.core.SurveySimulator.compute_effective_exposure`.
    """

    table: QTable
    """QTable: One row per ``(transient type, time bin)``.

    Columns: ``transient_type`` (str), ``time_bin`` (int), ``t_start``/``t_end``
    (the bin's own edges), ``n_pixels_visited``, ``solid_angle`` (the visited
    footprint, in steradians), ``duration`` (the bin width, i.e. ``t_end - t_start``,
    *not* padded by any transient's ``duration_limit``), ``effective_exposure``
    (``solid_angle * duration``), and ``expected_events`` (``effective_exposure *``
    that row's transient type's own
    :attr:`~uvex_transients.transients.base.ExtragalacticTransient.integrated_rate`).
    """

    nside: int
    """int: HEALPix resolution the visited footprint was queried at."""

    order: str
    """str: HEALPix pixel ordering scheme (``"nested"`` or ``"ring"``)."""

    time_bins: Time
    """~astropy.time.Time: The ``n + 1`` bin edges exposure was tabulated within."""

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
            `Quantity`/`Time` column.
        """
        if name not in self.table.colnames:
            raise KeyError(f"No column {name!r} in this catalog; available: {self.table.colnames}.")

        col = self.table[name]
        return col if isinstance(col, (Quantity, Time)) else np.asarray(col)

    @property
    def transient_type(self) -> np.ndarray:
        """numpy.ndarray of str: Each row's transient-type name."""
        return np.asarray(self.table["transient_type"]).astype(str)

    @property
    def time_bin(self) -> np.ndarray:
        """numpy.ndarray: Each row's time-bin index."""
        return self.column("time_bin")

    @property
    def t_start(self) -> Time:
        """~astropy.time.Time: Each row's bin start time."""
        return self.column("t_start")

    @property
    def t_end(self) -> Time:
        """~astropy.time.Time: Each row's bin end time."""
        return self.column("t_end")

    @property
    def n_pixels_visited(self) -> np.ndarray:
        """numpy.ndarray: Each row's number of visited HEALPix pixels."""
        return self.column("n_pixels_visited")

    @property
    def solid_angle(self) -> Quantity:
        """~astropy.units.Quantity: Each row's visited footprint solid angle."""
        return self.column("solid_angle")

    @property
    def duration(self) -> Quantity:
        """~astropy.units.Quantity: Each row's bin width."""
        return self.column("duration")

    @property
    def effective_exposure(self) -> Quantity:
        """~astropy.units.Quantity: Each row's effective exposure (``solid_angle * duration``)."""
        return self.column("effective_exposure")

    @property
    def expected_events(self) -> np.ndarray:
        """numpy.ndarray: Each row's expected event count (``effective_exposure * integrated_rate``)."""
        return self.column("expected_events")

    # ----------------------------------------- #
    # Per-Type Summaries                        #
    # ----------------------------------------- #
    def _sum_by_type(self, column: str) -> dict[str, Quantity | float]:
        """
        Sum a column across every time bin, grouped by ``transient_type``.

        Parameters
        ----------
        column : str
            Name of the column to sum; must be one of `table.colnames`.

        Returns
        -------
        dict[str, ~astropy.units.Quantity or float]
            ``{transient type: summed value}``, one entry per distinct
            ``transient_type`` present in this catalog.
        """
        values = self.column(column)
        types = self.transient_type
        return {name: values[types == name].sum() for name in np.unique(types)}

    @property
    def total_effective_exposure(self) -> dict[str, Quantity]:
        """
        Sum `effective_exposure` across every time bin, grouped by transient type.

        This is the quantity an expected count is derived from --
        ``rate.integrated_rate * total_effective_exposure[name]`` reproduces
        ``total_expected_events[name]`` for that same transient type.

        Returns
        -------
        dict[str, ~astropy.units.Quantity]
            ``{transient type: total effective exposure}``.
        """
        return self._sum_by_type("effective_exposure")

    @property
    def total_expected_events(self) -> dict[str, float]:
        r"""
        Sum `expected_events` across every time bin, grouped by transient type.

        Unlike `~uvex_transients.transients.base.ExtragalacticTransient.compute_all_sky_yield`,
        this accounts for the survey's actual footprint per bin rather than assuming
        every bin observes the full :math:`4\pi` sky.

        Returns
        -------
        dict[str, float]
            ``{transient type: total expected event count}``.
        """
        return self._sum_by_type("expected_events")

    @property
    def total_duration(self) -> Quantity:
        """~astropy.units.Quantity: The full window `time_bins` spans (last edge minus first)."""
        return (self.time_bins[-1] - self.time_bins[0]).to(u.day)

    @property
    def coverage_fraction(self) -> dict[str, float]:
        r"""
        dict[str, float]: Each transient type's time- and sky-averaged coverage fraction.

        .. math::

            f_{\rm cov} = \frac{\mathcal E}{4\pi\,{\rm sr}\times T},

        where :math:`\mathcal E` is `total_effective_exposure` and :math:`T` is
        `total_duration`. This is the fraction of the full :math:`4\pi` sky-time
        volume the survey's actual footprint swept out -- ``1.0`` only for a survey
        that observes the whole sky continuously over `time_bins`' entire span.
        """
        full_sky_exposure = 4 * np.pi * u.sr * self.total_duration
        return {
            name: (exposure / full_sky_exposure).to_value(u.dimensionless_unscaled)
            for name, exposure in self.total_effective_exposure.items()
        }

    # ----------------------------------------- #
    # Windowed Queries                          #
    # ----------------------------------------- #
    def _bin_overlap_weights(self, t_start: Time, t_end: Time) -> np.ndarray:
        """
        Fraction of each row's own bin duration that overlaps ``[t_start, t_end)``.

        Assumes exposure accrues uniformly across a bin's own duration -- the same
        assumption `compute_effective_exposure` itself already makes by treating the
        visited footprint as constant over a whole bin, so this introduces no further
        approximation beyond what's already baked into the catalog.

        Parameters
        ----------
        t_start, t_end : ~astropy.time.Time
            The query window.

        Returns
        -------
        numpy.ndarray
            Overlap fraction per row, in ``[0, 1]``.

        Raises
        ------
        ValueError
            If `t_end` is not after `t_start`.
        """
        if t_end <= t_start:
            raise ValueError(f"`t_end` ({t_end.iso}) must be after `t_start` ({t_start.iso}).")

        row_start, row_end = self.t_start.mjd, self.t_end.mjd
        overlap = np.clip(np.minimum(row_end, t_end.mjd) - np.maximum(row_start, t_start.mjd), 0.0, None)
        return overlap / (row_end - row_start)

    def get_exposure_between(
        self,
        t_start: Time,
        t_end: Time,
        transient_type: str | None = None,
    ) -> Quantity | dict[str, Quantity]:
        """
        Effective exposure accrued within ``[t_start, t_end)``, grouped by transient type.

        Partially overlapping bins are weighted by the fraction of their own duration
        that falls inside the window; see `_bin_overlap_weights`.

        Parameters
        ----------
        t_start, t_end : ~astropy.time.Time
            The query window.
        transient_type : str, optional
            If given, return only that type's exposure (a scalar `Quantity`) rather
            than a `dict` of every type present.

        Returns
        -------
        ~astropy.units.Quantity or dict[str, ~astropy.units.Quantity]
            The exposure between the two time limits.
        """
        weighted = self.effective_exposure * self._bin_overlap_weights(t_start, t_end)
        types = self.transient_type
        if transient_type is not None:
            return weighted[types == transient_type].sum()
        return {name: weighted[types == name].sum() for name in np.unique(types)}

    def get_expected_events_between(
        self,
        t_start: Time,
        t_end: Time,
        transient_type: str | None = None,
    ) -> float | dict[str, float]:
        """
        Compute the expected event count within ``[t_start, t_end)``, grouped by transient type.

        Uses the same bin-overlap weighting as `get_exposure_between` (see
        `_bin_overlap_weights`); valid because `expected_events` is `effective_exposure`
        scaled by a per-type constant (`integrated_rate`), so the two scale identically.

        Parameters
        ----------
        t_start, t_end : ~astropy.time.Time
            The query window.
        transient_type : str, optional
            If given, return only that type's expected count (a scalar `float`)
            rather than a `dict` of every type present.

        Returns
        -------
        float or dict[str, float]
            Expected number of events between the two times.
        """
        weighted = self.expected_events * self._bin_overlap_weights(t_start, t_end)
        types = self.transient_type
        if transient_type is not None:
            return float(weighted[types == transient_type].sum())
        return {name: float(weighted[types == name].sum()) for name in np.unique(types)}

    # ----------------------------------------- #
    # Rebinning                                 #
    # ----------------------------------------- #
    @staticmethod
    def _resolve_rebin_edges(time_bins: Time | int, span: Time) -> Time:
        """
        Resolve `rebin`'s `time_bins` argument down to a concrete array of edges.

        Parameters
        ----------
        time_bins : ~astropy.time.Time or int
            Either explicit new bin edges, or a positive number of equal-width bins
            spanning `span`'s own full range.
        span : ~astropy.time.Time
            This catalog's own `time_bins` -- used to derive evenly-spaced edges when
            `time_bins` is given as an int; ignored if `time_bins` is already a `Time`
            array.

        Returns
        -------
        ~astropy.time.Time
            The concrete array of new bin edges.

        Raises
        ------
        TypeError
            If `time_bins` is neither a `~astropy.time.Time` array nor an int.
        ValueError
            If `time_bins` is a `Time` array with fewer than 2 edges, or a
            non-positive int.
        """
        if isinstance(time_bins, Time):
            if time_bins.isscalar or time_bins.size < 2:
                raise ValueError("`time_bins`, given as a Time array, must contain at least 2 edges.")
            return time_bins

        if isinstance(time_bins, bool) or not isinstance(time_bins, (int, np.integer)):
            raise TypeError(
                f"`time_bins` must be an astropy Time array of bin edges or a positive int, got {type(time_bins)!r}."
            )

        if time_bins < 1:
            raise ValueError(f"`time_bins`, given as an int, must be a positive number of bins, got {time_bins!r}.")

        return span[0] + np.linspace(0.0, 1.0, time_bins + 1) * (span[-1] - span[0])

    def rebin(self, time_bins: Time | int) -> "ExposureCatalog":
        r"""
        Rebin this catalog onto an arbitrary new time binning.

        Unlike `~uvex_transients.simulation.core.SurveySimulator.compute_effective_exposure`,
        this never re-queries the schedule's footprint -- it redistributes each
        original bin's own `effective_exposure`/`expected_events` across whichever new
        bin(s) it overlaps, weighted by the fraction of the *original* bin's own
        duration that falls inside each new bin (see `_bin_overlap_weights`, applied
        once per new bin here). This is exact regardless of how the new edges align
        with the old ones -- including new bins narrower than, wider than, or
        straddling several original bins -- because both quantities are additive
        across disjoint time intervals and this class already treats a bin's own
        visited footprint as constant across its own duration (the same assumption
        `compute_effective_exposure` made when tabulating it in the first place).

        The new `solid_angle` is recovered as a duration-weighted average,
        ``effective_exposure / duration``, over whatever original-bin overlap
        contributed to each new bin -- exact only if the actually-visited footprint
        happens to be identical across every original bin contributing to a given new
        bin; otherwise it's the *equivalent* constant footprint that reproduces the
        same `effective_exposure`. `n_pixels_visited` is derived from that same
        `solid_angle`, converted back through `nside`'s own pixel area and rounded to
        the nearest integer -- not a re-union of pixel ids, which isn't retained per
        row. A new bin whose window doesn't overlap any original bin (e.g. it extends
        past this catalog's own `time_bins` range) gets zeros throughout, including a
        `duration` short of the new bin's own width.

        Parameters
        ----------
        time_bins : ~astropy.time.Time or int
            Either an explicit, monotonically increasing `Time` array of ``n + 1`` new
            bin edges, or a positive int giving the number of evenly-spaced bins to
            divide this catalog's own `time_bins` span into.

        Returns
        -------
        ExposureCatalog
            A new catalog, one row per ``(transient type, new time bin)``, with
            `time_bins` set to the resolved new edges.

        Raises
        ------
        TypeError
            If `time_bins` is neither a `~astropy.time.Time` array nor an int.
        ValueError
            If `time_bins` is a `Time` array with fewer than 2 edges, or a
            non-positive int.
        """
        new_edges = self._resolve_rebin_edges(time_bins, self.time_bins)
        n_new = len(new_edges) - 1
        types = self.transient_type
        pixel_area = ah.nside_to_pixel_area(self.nside)

        rows = {
            "transient_type": [],
            "time_bin": [],
            "t_start": [],
            "t_end": [],
            "n_pixels_visited": [],
            "solid_angle": [],
            "duration": [],
            "effective_exposure": [],
            "expected_events": [],
        }

        for k in range(n_new):
            a, b = new_edges[k], new_edges[k + 1]
            weights = self._bin_overlap_weights(a, b)

            for name in np.unique(types):
                mask = types == name
                duration_sum = (self.duration[mask] * weights[mask]).sum()
                exposure_sum = (self.effective_exposure[mask] * weights[mask]).sum()
                events_sum = (self.expected_events[mask] * weights[mask]).sum()
                solid_angle_avg = exposure_sum / duration_sum if duration_sum > 0 else 0.0 * u.sr

                rows["transient_type"].append(name)
                rows["time_bin"].append(k)
                rows["t_start"].append(a)
                rows["t_end"].append(b)
                rows["solid_angle"].append(solid_angle_avg)
                n_pixels = (solid_angle_avg / pixel_area).to_value(u.dimensionless_unscaled)
                rows["n_pixels_visited"].append(int(round(n_pixels)))
                rows["duration"].append(duration_sum)
                rows["effective_exposure"].append(exposure_sum)
                rows["expected_events"].append(events_sum)

        table = QTable()
        table["transient_type"] = np.asarray(rows["transient_type"])
        table["time_bin"] = np.asarray(rows["time_bin"], dtype=np.int64)
        table["t_start"] = Time(rows["t_start"])
        table["t_end"] = Time(rows["t_end"])
        table["n_pixels_visited"] = np.asarray(rows["n_pixels_visited"], dtype=np.int64)
        table["solid_angle"] = u.Quantity(rows["solid_angle"])
        table["duration"] = u.Quantity(rows["duration"])
        table["effective_exposure"] = u.Quantity(rows["effective_exposure"])
        table["expected_events"] = np.asarray(rows["expected_events"], dtype=np.float64)

        return ExposureCatalog(table=table, nside=self.nside, order=self.order, time_bins=new_edges)

    # ----------------------------------------- #
    # IO Methods                                #
    # ----------------------------------------- #
    def to_disk(self, path: str | Path, table_format: str | None = None, overwrite: bool = False) -> None:
        """
        Write this catalog's table to disk as ECSV, with its provenance in the header.

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
            }
        )
        table.write(Path(path), format=table_format, overwrite=overwrite)
        logger.info("Wrote exposure catalog (%d rows) to %s.", len(table), path)

    @classmethod
    def from_disk(cls, path: str | Path, table_format: str | None = None) -> "ExposureCatalog":
        """
        Read an exposure catalog back from disk, as written by :meth:`to_disk`.

        Parameters
        ----------
        path : str or ~pathlib.Path
            Path to the exposure table, as written by :meth:`to_disk`.
        table_format : str, optional
            Passed through to :meth:`~astropy.table.QTable.read`; if `None`, inferred from
            ``path``'s suffix.

        Returns
        -------
        ExposureCatalog
            Exposure catalog reconstructed from the serialized table and metadata.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        table = QTable.read(path, format=table_format)
        meta = dict(table.meta)
        table.meta.clear()

        logger.info("Read exposure catalog (%d rows) from %s.", len(table), path)
        return cls(
            table=table,
            nside=int(meta.pop("nside")),
            order=meta.pop("order"),
            time_bins=meta.pop("time_bins"),
        )
