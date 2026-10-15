"""Access to the packaged Swift/UVOT supernova UV light-curve archive (``swift_sne_uv_lightcurves.h5``).

The archive holds detection-only UVW1/UVM2/UVW2 luminosity light curves from SOUSA under
``/<object>/<band>/{t, L, L_err, mjd}``. Per-object metadata (``type_standard``, ``redshift``, ``ra_deg``, ...)
lives in the attributes of ``/<object>``, and per-band metadata (``lambda_eff_AA``, ``A_band_mw``, ...) in the
attributes of ``/<object>/<band>``.
"""

from pathlib import Path
from typing import Any

import h5py
from astropy import units as u
from astropy.table import QTable

import uvex_transients

__all__ = ["SwiftSNeArchive"]


def _default_archive_path() -> Path:
    return Path(uvex_transients.__file__).parent.parent / "test_data" / "transients" / "swift_sne_uv_lightcurves.h5"


class SwiftSNeArchive:
    """
    Read-only access to the packaged Swift/UVOT supernova UV light-curve archive.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        Path to the archive HDF5 file. Defaults to the packaged
        ``test_data/transients/swift_sne_uv_lightcurves.h5``.

    Examples
    --------
    >>> archive = SwiftSNeArchive()
    >>> archive.objects(sn_type="IIP")  # doctest: +SKIP
    >>> lc = archive.lightcurve(
    ...     "Gaia16apd", "UVW1"
    ... )  # doctest: +SKIP
    >>> plt.errorbar(
    ...     lc["t"], lc["L"], lc["L_err"], fmt="o"
    ... )  # doctest: +SKIP
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else _default_archive_path()

    def objects(self, sn_type: str | None = None) -> list[str]:
        """
        Return the supernovae in the archive.

        Parameters
        ----------
        sn_type : str, optional
            Keep only objects whose ``type_standard`` attribute equals this (e.g. ``"IIP"``, ``"SLSN-I"``).

        Returns
        -------
        list of str
            Sorted object names.
        """
        with h5py.File(self.path, "r") as f:
            if sn_type is None:
                return sorted(f.keys())
            return sorted(k for k, g in f.items() if g.attrs.get("type_standard") == sn_type)

    def bands(self, obj: str) -> list[str]:
        """
        Return the UVOT bands observed for one object.

        Parameters
        ----------
        obj : str
            One of :meth:`objects`.

        Returns
        -------
        list of str
            Sorted band names (``UVW1``, ``UVM2``, ``UVW2``).
        """
        with h5py.File(self.path, "r") as f:
            return sorted(f[obj].keys())

    def meta(self, obj: str, band: str | None = None) -> dict[str, Any]:
        """
        Return the HDF5 attributes of one object, or of one of its bands if ``band`` is given.

        Parameters
        ----------
        obj : str
            One of :meth:`objects`.
        band : str, optional
            One of :meth:`bands`. If omitted, the object-level metadata is returned.

        Returns
        -------
        dict
            Attribute name to value.
        """
        with h5py.File(self.path, "r") as f:
            return dict((f[obj] if band is None else f[obj][band]).attrs)

    def lightcurve(self, obj: str, band: str) -> QTable:
        """
        Return one band's detections as a `~astropy.table.QTable`.

        Parameters
        ----------
        obj : str
            One of :meth:`objects`, e.g. ``"Gaia16apd"``.
        band : str
            One of :meth:`bands`, e.g. ``"UVW1"``.

        Returns
        -------
        astropy.table.QTable
            Columns ``t`` (observer-frame days since ``t0``), ``L`` and ``L_err`` (erg/s, ``nu L_nu``, photometric
            error only) and ``mjd``. ``.meta`` holds the band-level attributes.
        """
        with h5py.File(self.path, "r") as f:
            g = f[obj][band]
            table = QTable(
                {
                    "t": g["t"][()] * u.day,
                    "L": g["L"][()] * (u.erg / u.s),
                    "L_err": g["L_err"][()] * (u.erg / u.s),
                    "mjd": g["mjd"][()] * u.day,
                }
            )
            table.meta.update(g.attrs)
        return table

    def __repr__(self) -> str:
        try:
            return f"<SwiftSNeArchive path={self.path!s} n_objects={len(self.objects())}>"
        except OSError:
            return f"<SwiftSNeArchive path={self.path!s} (not found)>"
