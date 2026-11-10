"""Access to the packaged archive of observed luminosity functions (``test_data/transients/luminosity_functions.h5``).

The archive holds each curve under ``/<transient_type>/luminosity_function_<band>/<source>/{M, dAndM}``, with
``<transient_type>`` named as in the light-curve archive (`~uvex_transients.utils.lightcurve_archive`), for example
``supernovae/Ia``. ``M`` is the peak absolute magnitude in the band and ``dAndM`` the volumetric rate per magnitude,
stored as given by the source. The rates have an overall normalization that is not ours, so the curves compare to a
model by shape: normalize both over the same range of ``M``.
"""

from pathlib import Path

import h5py
from astropy import units as u
from astropy.table import QTable

import uvex_transients

__all__ = ["LuminosityFunctionArchive"]


def _default_archive_path() -> Path:
    return Path(uvex_transients.__file__).parent.parent / "test_data" / "transients" / "luminosity_functions.h5"


class LuminosityFunctionArchive:
    """
    Read-only access to the packaged archive of observed luminosity functions.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        Path to the archive HDF5 file. Defaults to the packaged ``test_data/transients/luminosity_functions.h5``.

    Examples
    --------
    >>> archive = LuminosityFunctionArchive()
    >>> archive.types()  # doctest: +SKIP
    ['supernovae/II', 'supernovae/Ia', 'supernovae/SESNe', 'supernovae/ccSNe']
    >>> lf = archive.table(
    ...     "supernovae/Ia", "ASAS-SN"
    ... )  # doctest: +SKIP
    >>> lf["M"], lf["dAndM"]  # doctest: +SKIP
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else _default_archive_path()

    def types(self, band: str = "V") -> list[str]:
        """
        Return the transient types that have a luminosity function in a band.

        Parameters
        ----------
        band : str, optional
            The band, such as ``"V"``.

        Returns
        -------
        list of str
            Sorted group names, such as ``"supernovae/Ia"``.
        """
        found = []

        def _visit(name: str, obj) -> None:
            if isinstance(obj, h5py.Group) and name.endswith(f"/luminosity_function_{band}"):
                found.append(name.rsplit("/", 1)[0])

        with h5py.File(self.path, "r") as f:
            f.visititems(_visit)
        return sorted(found)

    def sources(self, transient_type: str, band: str = "V") -> list[str]:
        """
        Return the sources of the luminosity function of a transient type.

        Parameters
        ----------
        transient_type : str
            One of :meth:`types`, such as ``"supernovae/Ia"``.
        band : str, optional
            The band, such as ``"V"``.

        Returns
        -------
        list of str
            Sorted source names, such as ``"ASAS-SN"``.
        """
        with h5py.File(self.path, "r") as f:
            return sorted(f[f"{transient_type}/luminosity_function_{band}"].keys())

    def table(self, transient_type: str, source: str | None = None, band: str = "V") -> QTable:
        """
        Return one luminosity function as a `~astropy.table.QTable`.

        Parameters
        ----------
        transient_type : str
            One of :meth:`types`, such as ``"supernovae/Ia"``.
        source : str, optional
            One of :meth:`sources`. It may be left out when the type has only one.
        band : str, optional
            The band, such as ``"V"``.

        Returns
        -------
        astropy.table.QTable
            Columns ``M`` (peak absolute magnitude, ascending, in `~astropy.units.mag`) and ``dAndM`` (rate per
            magnitude, in Mpc^-3 yr^-1 mag^-1). ``.meta`` holds the attributes of the source group: the label it has
            in the source file and the header that says what the axes are.

        Raises
        ------
        ValueError
            If `source` is left out and the type has more than one.
        """
        sources = self.sources(transient_type, band)
        if source is None:
            if len(sources) != 1:
                raise ValueError(f"{transient_type!r} has several sources, {sources}; say which.")
            source = sources[0]
        with h5py.File(self.path, "r") as f:
            group = f[f"{transient_type}/luminosity_function_{band}/{source}"]
            table = QTable({"M": group["M"][()] * u.mag, "dAndM": group["dAndM"][()] * u.Mpc**-3 / u.yr / u.mag})
            table.meta.update(group.attrs)
        return table

    def __repr__(self) -> str:
        try:
            return f"<LuminosityFunctionArchive path={self.path!s} types={self.types()}>"
        except OSError:
            return f"<LuminosityFunctionArchive path={self.path!s} (not found)>"
