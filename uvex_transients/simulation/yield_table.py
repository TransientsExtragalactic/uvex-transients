"""
Per-transient-type yield summary, produced by `EventCatalog.compute_yield_summary`.

`YieldTable` is `EventCatalog`/`ExposureCatalog`'s summary-statistics counterpart: one
row per transient type, combining a raw (feasible) `EventCatalog`, a detected
`EventCatalog`, and an `~uvex_transients.simulation.exposure_catalog.ExposureCatalog`
into the rate/efficiency/yield estimators derived in :ref:`yield-statistics`. Like the
other two, it holds no live reference back to whatever produced it, so it round-trips
to/from disk and pickles cleanly.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.table import QTable
from astropy.units import Quantity

from uvex_transients.utils import logger


def _fmt(value: float, precision: int) -> str:
    """Format a plain float to `precision` significant figures, or ``"--"`` for `numpy.nan`."""
    return "--" if math.isnan(value) else f"{value:.{precision}g}"


def _value(x: Quantity | float) -> float:
    """Strip a `~astropy.units.Quantity`'s unit, or pass a plain float through unchanged."""
    return x.value if isinstance(x, Quantity) else float(x)


def _single_uncertainty_cell(
    value: Quantity | float, lower: Quantity | float, upper: Quantity | float, precision: int
) -> str:
    """Format ``value`` with one asymmetric uncertainty as ``$v^{+u}_{-l}$``."""
    v, lo, up = _value(value), _value(lower), _value(upper)
    if math.isnan(v):
        return "--"
    return f"${_fmt(v, precision)}^{{+{_fmt(up - v, precision)}}}_{{-{_fmt(v - lo, precision)}}}$"


def _double_uncertainty_tex(
    value: Quantity | float,
    binom_lower: Quantity | float,
    binom_upper: Quantity | float,
    rate_lower: Quantity | float,
    rate_upper: Quantity | float,
    precision: int,
) -> str:
    """Format ``value`` with two stacked asymmetric uncertainties as ``v^{+u_b}_{-l_b}{}^{+u_r}_{-l_r}`` (no ``$``s)."""
    v = _value(value)
    if math.isnan(v):
        return "--"
    bl, bu, rl, ru = _value(binom_lower), _value(binom_upper), _value(rate_lower), _value(rate_upper)
    return (
        f"{_fmt(v, precision)}"
        f"^{{+{_fmt(bu - v, precision)}}}_{{-{_fmt(v - bl, precision)}}}"
        f"\\,{{}}^{{+{_fmt(ru - v, precision)}}}_{{-{_fmt(v - rl, precision)}}}"
    )


def _double_uncertainty_cell(
    value: Quantity | float,
    binom_lower: Quantity | float,
    binom_upper: Quantity | float,
    rate_lower: Quantity | float,
    rate_upper: Quantity | float,
    precision: int,
) -> str:
    """Format ``value`` with two stacked asymmetric uncertainties as ``$v^{+u_b}_{-l_b}{}^{+u_r}_{-l_r}$``."""
    body = _double_uncertainty_tex(value, binom_lower, binom_upper, rate_lower, rate_upper, precision)
    return body if body == "--" else f"${body}$"


@dataclass
class YieldTable:
    """
    A per-transient-type yield summary, tagged by transient type.

    Produced by :meth:`~uvex_transients.simulation.event_catalog.EventCatalog.compute_yield_summary`;
    see that method's own docstring for the full column list and the estimators each one implements.
    """

    table: QTable
    """QTable: One row per transient type; see `EventCatalog.compute_yield_summary` for the columns."""

    confidence: float = 0.9
    """float: Confidence level the Clopper-Pearson binomial columns were computed at."""

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
            `numpy.asarray` of the column, preserving units if the column is a `Quantity` column.
        """
        if name not in self.table.colnames:
            raise KeyError(f"No column {name!r} in this catalog; available: {self.table.colnames}.")

        col = self.table[name]
        return col if isinstance(col, Quantity) else np.asarray(col)

    @property
    def transient_type(self) -> np.ndarray:
        """numpy.ndarray of str: Each row's transient-type name."""
        return np.asarray(self.table["transient_type"]).astype(str)

    # ----------------------------------------- #
    # IO Methods                                #
    # ----------------------------------------- #
    def to_disk(self, path: str | Path, table_format: str | None = None, overwrite: bool = False) -> None:
        """
        Write this table to disk as ECSV, with its provenance in the header.

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
        table.meta.update({"confidence": float(self.confidence)})
        table.write(Path(path), format=table_format, overwrite=overwrite)
        logger.info("Wrote yield table (%d transient type(s)) to %s.", len(table), path)

    @classmethod
    def from_disk(cls, path: str | Path, table_format: str | None = None) -> "YieldTable":
        """
        Read a yield table back from disk, as written by :meth:`to_disk`.

        Parameters
        ----------
        path : str or ~pathlib.Path
            Path to the table, as written by :meth:`to_disk`.
        table_format : str, optional
            Passed through to :meth:`~astropy.table.QTable.read`; if `None`, inferred from
            ``path``'s suffix.

        Returns
        -------
        YieldTable
            Yield table reconstructed from the serialized table and metadata.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        table = QTable.read(path, format=table_format)
        meta = dict(table.meta)
        table.meta.clear()

        logger.info("Read yield table (%d transient type(s)) from %s.", len(table), path)
        return cls(table=table, confidence=float(meta.pop("confidence", 0.9)))

    def to_ascii(self, path: str | Path, overwrite: bool = False) -> None:
        """
        Write this table to a human-readable, fixed-width ASCII file.

        Uses `astropy.io.ascii`'s ``fixed_width_two_line`` format -- plain, aligned
        columns with a header separator line, readable directly in a text editor or
        terminal -- unlike :meth:`to_disk`'s ECSV (YAML-in-comments, optimized for a
        clean round trip back into a `QTable`, not for reading by eye).

        Parameters
        ----------
        path : str or ~pathlib.Path
            Destination path.
        overwrite : bool, optional
            Whether to overwrite an existing file at `path`.
        """
        path = Path(path)
        self.table.write(path, format="ascii.fixed_width_two_line", overwrite=overwrite)
        logger.info("Wrote yield table (%d transient type(s)) to %s.", len(self.table), path)

    # ----------------------------------------- #
    # LaTeX Export                              #
    # ----------------------------------------- #
    _LATEX_COLUMNS = (
        ("Transient Type", "transient_type"),
        ("Total Exposure", "total_exposure"),
        ("Exposure Fraction", "total_exposure_fraction"),
        ("Integrated Rate", "integrated_rate"),
        ("All-Sky Rate", "all_sky_rate"),
        ("UVEX Intrinsic Rate", "uvex_intrinsic_rate"),
        ("UVEX Intrinsic Events", "uvex_intrinsic_events"),
        ("Detected Events", "detected_events"),
        ("Detection Probability", "detection_probability"),
        ("Expected Detections", "expected_detections"),
    )
    """tuple[tuple[str, str]]: ``(header label, base column name)`` pairs, in output order."""

    _SINGLE_UNCERTAINTY = frozenset({"integrated_rate", "all_sky_rate", "uvex_intrinsic_rate", "uvex_intrinsic_events"})
    _DOUBLE_UNCERTAINTY = frozenset({"detection_probability", "expected_detections"})

    def to_latex(
        self,
        path: str | Path | None = None,
        precision: int = 3,
        caption: str | None = None,
        label: str | None = None,
        overwrite: bool = False,
    ) -> str:
        r"""
        Render this table as a LaTeX ``tabular``, with rate/binomial uncertainties as stacked superscripts.

        A column tabulated with a single ``..._lower``/``..._upper`` pair (`integrated_rate`,
        `all_sky_rate`, `uvex_intrinsic_rate`, `uvex_intrinsic_events`) is rendered as
        ``$v^{+u}_{-l}$``. A column tabulated with *both* Clopper-Pearson binomial bounds and
        `~uvex_transients.transients.base.ExtragalacticTransient.RATE_CI`-derived rate bounds
        (`detection_probability`, `expected_detections`; see
        `~uvex_transients.simulation.event_catalog.EventCatalog.compute_yield_summary`) is rendered
        with both, stacked, as ``$v^{+u_b}_{-l_b}{}^{+u_r}_{-l_r}$`` -- binomial uncertainty first,
        rate uncertainty second. `numpy.nan` (an unidentified `detection_probability`/
        `expected_detections`, when a transient type had zero feasible Monte Carlo draws) renders
        as ``--``.

        Parameters
        ----------
        path : str or ~pathlib.Path, optional
            If given, write the rendered LaTeX here as well as returning it.
        precision : int, optional
            Significant figures for every formatted number. The default is ``3``.
        caption, label : str, optional
            Forwarded into a wrapping ``table`` environment's ``\caption``/``\label``, if given.
        overwrite : bool, optional
            Whether to overwrite an existing file at `path`.

        Returns
        -------
        str
            The rendered LaTeX source.
        """
        lines = []
        if caption is not None or label is not None:
            lines.append("\\begin{table}")
            if caption is not None:
                lines.append(f"\\caption{{{caption}}}")
            if label is not None:
                lines.append(f"\\label{{{label}}}")

        colspec = "l" + "c" * (len(self._LATEX_COLUMNS) - 1)
        lines.append(f"\\begin{{tabular}}{{{colspec}}}")
        lines.append("\\hline")
        lines.append(" & ".join(header for header, _ in self._LATEX_COLUMNS) + " \\\\")
        lines.append("\\hline")

        for row in self.table:
            cells = []
            for _, column in self._LATEX_COLUMNS:
                if column == "transient_type":
                    cells.append(str(row[column]))
                elif column == "detected_events":
                    cells.append(str(int(row[column])))
                elif column == "total_exposure":
                    quantity = row[column]
                    unit = quantity.unit.to_string("latex_inline").strip("$")
                    cells.append(f"${_fmt(quantity.value, precision)}\\,{unit}$")
                elif column == "total_exposure_fraction":
                    cells.append(_fmt(float(row[column]), precision))
                elif column in self._SINGLE_UNCERTAINTY:
                    cells.append(
                        _single_uncertainty_cell(row[column], row[f"{column}_lower"], row[f"{column}_upper"], precision)
                    )
                elif column in self._DOUBLE_UNCERTAINTY:
                    cells.append(
                        _double_uncertainty_cell(
                            row[column],
                            row[f"{column}_binom_lower"],
                            row[f"{column}_binom_upper"],
                            row[f"{column}_rate_lower"],
                            row[f"{column}_rate_upper"],
                            precision,
                        )
                    )
                else:  # pragma: no cover - guards against a future column added to _LATEX_COLUMNS
                    raise ValueError(f"No LaTeX formatting rule registered for column {column!r}.")
            lines.append(" & ".join(cells) + " \\\\")

        lines.append("\\hline")
        lines.append("\\end{tabular}")
        if caption is not None or label is not None:
            lines.append("\\end{table}")

        latex = "\n".join(lines)

        if path is not None:
            path = Path(path)
            if path.exists() and not overwrite:
                raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it.")
            path.write_text(latex)
            logger.info("Wrote LaTeX yield table (%d transient type(s)) to %s.", len(self.table), path)

        return latex

    def expected_detections_latex(self, precision: int = 3) -> str:
        r"""
        Render each transient type's expected detections, with both uncertainty sources, as LaTeX.

        Uses the same double-stacked-uncertainty notation as `to_latex`'s ``expected_detections``
        column (``v^{+u_b}_{-l_b}{}^{+u_r}_{-l_r}``, binomial uncertainty first, rate uncertainty
        second), but on its own -- a LaTeX ``align*`` block, one line per transient type -- rather
        than a full table.

        Parameters
        ----------
        precision : int, optional
            Significant figures for every formatted number. The default is ``3``.

        Returns
        -------
        str
            The rendered LaTeX source, e.g. for `IPython.display.Latex`; see `display_expected_detections`.
        """
        lines = ["\\begin{align*}"]
        for row in self.table:
            name = str(row["transient_type"]).replace("_", "\\_")
            body = _double_uncertainty_tex(
                row["expected_detections"],
                row["expected_detections_binom_lower"],
                row["expected_detections_binom_upper"],
                row["expected_detections_rate_lower"],
                row["expected_detections_rate_upper"],
                precision,
            )
            lines.append(f"\\text{{{name}}} &: {body} \\\\")
        lines.append("\\end{align*}")
        return "\n".join(lines)

    def display_expected_detections(self, precision: int = 3) -> None:
        """
        Display each transient type's expected detections, rendered if possible.

        In a Jupyter/IPython context, renders `expected_detections_latex` via
        `IPython.display.Latex`. Outside one (`IPython` not installed, or no active display),
        falls back to printing the raw LaTeX source.

        Parameters
        ----------
        precision : int, optional
            Forwarded to `expected_detections_latex`.
        """
        latex = self.expected_detections_latex(precision=precision)
        try:
            from IPython.display import Latex, display
        except ImportError:
            print(latex)
            return
        display(Latex(latex))
