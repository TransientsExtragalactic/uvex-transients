"""Access to the packaged literature light-curve archive (``test_data/transients/lightcurves.h5``).

The archive groups every calibration light curve used by the Sphinx transient-type galleries
(``docs/source/transients/*.rst``) under ``/<transient_type>/<transient>/<field>``, where
``<transient_type>`` matches one of the `uvex_transients.transients` modules (``kilonovae``,
``lfbots``, ``supernovae``, ``tdes``), ``<transient>`` is ``<designation>_<citekey>`` (e.g.
``gw170817_waxman``), and ``<field>`` is an observable such as ``L_bol`` or ``T_phot``. Each leaf is
an `~astropy.table.QTable` with ``time`` and ``<field>`` columns as unit-aware `~astropy.units.Quantity`
columns, plus a ``reference`` entry in ``.meta`` recording the source.

Supernovae are additionally split by spectroscopic classification, so their ``<transient_type>`` is
``supernovae/<class>`` with ``<class>`` one of ``II``, ``IIP``, ``IIb``, ``Ib``, ``Ic`` or ``Ic-BL``
(e.g. ``archive.table("supernovae/IIb", "2011dh_lyman2016", "L_bol")``). :meth:`LightcurveArchive.types`
lists only top-level groups, so use :meth:`LightcurveArchive.events` on ``"supernovae"`` to list the classes.
"""

from pathlib import Path

import h5py
from astropy.table import QTable

import uvex_transients

__all__ = ["LightcurveArchive"]


def _default_archive_path() -> Path:
    return Path(uvex_transients.__file__).parent.parent / "test_data" / "transients" / "lightcurves.h5"


class LightcurveArchive:
    """
    Read-only access to the packaged literature light-curve archive.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        Path to the archive HDF5 file. Defaults to the packaged
        ``test_data/transients/lightcurves.h5``.

    Examples
    --------
    >>> archive = LightcurveArchive()
    >>> archive.types()  # doctest: +SKIP
    ['kilonovae', 'lfbots', 'supernovae', 'tdes']
    >>> lbol = archive.table(
    ...     "kilonovae", "gw170817_waxman", "L_bol"
    ... )  # doctest: +SKIP
    >>> lbol["time"], lbol["L_bol"]  # doctest: +SKIP
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else _default_archive_path()

    def types(self) -> list[str]:
        """Return the sorted list of transient types (top-level groups) in the archive."""
        with h5py.File(self.path, "r") as f:
            return sorted(f.keys())

    def events(self, transient_type: str) -> list[str]:
        """Return the sorted list of transients (``<designation>_<citekey>``) under ``transient_type``."""
        with h5py.File(self.path, "r") as f:
            return sorted(f[transient_type].keys())

    def fields(self, transient_type: str, event: str) -> list[str]:
        """Return the sorted list of observable fields (e.g. ``L_bol``, ``T_phot``) for one transient."""
        with h5py.File(self.path, "r") as f:
            keys = f[f"{transient_type}/{event}"].keys()
            return sorted(k for k in keys if not k.endswith("__table_column_meta__"))

    def table(self, transient_type: str, event: str, field: str) -> QTable:
        """
        Return the ``time``/``<field>`` light curve for one transient as a `~astropy.table.QTable`.

        Parameters
        ----------
        transient_type : str
            One of :meth:`types`, e.g. ``"supernovae"``.
        event : str
            One of :meth:`events`, e.g. ``"1999em_bersten2009"``.
        field : str
            One of :meth:`fields`, e.g. ``"L_bol"`` or ``"T_phot"``.

        Returns
        -------
        astropy.table.QTable
            Columns ``time`` and ``<field>``, both unit-aware. ``.meta["reference"]`` records the
            literature source.
        """
        return QTable.read(self.path, path=f"{transient_type}/{event}/{field}")

    def __repr__(self) -> str:
        try:
            types = self.types()
            n_events = sum(len(self.events(t)) for t in types)
        except OSError:
            return f"<LightcurveArchive path={self.path!s} (not found)>"
        return f"<LightcurveArchive path={self.path!s} types={types} n_events={n_events}>"
