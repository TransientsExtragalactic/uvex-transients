"""Utilities for fetching filters from SVO.

A thin, astroquery-free client for the `SVO Filter Profile Service <https://svo2.cab.inta-csic.es/theory/fps/>`_.
Queries use the same parameters as ``astroquery.svo_fps`` and return the same tables, but every response is stored
in the astropy download cache (see `astropy.config.get_cache_dir`), so each filter is only fetched once. Use
``cache="update"`` to refresh an entry and `clear_svo_cache` to drop everything astropy has cached.
"""

import io
from typing import Literal
from urllib.parse import urlencode

import numpy as np
from astropy import units as u
from astropy.io.votable import parse_single_table
from astropy.table import Table
from astropy.utils.data import clear_download_cache, download_file
from synphot import Empirical1D, SpectralElement

__all__ = ["SVO_URL", "clear_svo_cache", "get_bandpass", "get_filter_list", "get_transmission_data", "query_svo"]

SVO_URL = "https://svo2.cab.inta-csic.es/theory/fps/fps.php"
"""str: Endpoint of the SVO Filter Profile Service."""

_RANGE_PARAMETERS = {
    "WavelengthRef",
    "WavelengthMean",
    "WavelengthEff",
    "WavelengthMin",
    "WavelengthMax",
    "WidthEff",
    "FWHM",
}
_QUERY_PARAMETERS = (
    _RANGE_PARAMETERS
    | {p + s for p in _RANGE_PARAMETERS for s in ("_min", "_max")}
    | {"Instrument", "Facility", "PhotSystem", "ID", "PhotCalID", "FORMAT", "VERB"}
)


def query_svo(
    query: dict,
    *,
    cache: bool | Literal["update"] = True,
    timeout: float = 60,
    error_msg: str = "No data found for requested query",
) -> Table:
    """
    Send a query to the SVO Filter Profile Service and return the resulting table.

    Parameters
    ----------
    query : dict
        Search parameters, as described at https://svo2.cab.inta-csic.es/theory/fps/index.php?mode=voservice.
        Entries whose value is `None` are dropped.
    cache : bool or "update", optional
        Whether to read from and write to the astropy download cache. ``"update"`` forces a fresh download and
        overwrites the cached copy.
    timeout : float, optional
        HTTP timeout in seconds.
    error_msg : str, optional
        Message of the `IndexError` raised if the response contains no table.

    Returns
    -------
    astropy.table.Table
        The table returned by SVO.

    Raises
    ------
    ValueError
        If ``query`` contains a parameter SVO does not recognize.
    IndexError
        If the response contains no table.
    """
    bad = sorted(set(query) - _QUERY_PARAMETERS)
    if bad:
        raise ValueError(f"Invalid SVO query parameter(s): {', '.join(bad)}.")

    params = {k: v for k, v in query.items() if v is not None}
    path = download_file(f"{SVO_URL}?{urlencode(params)}", cache=cache, timeout=timeout, show_progress=False)
    with open(path, "rb") as f:
        content = f.read()
    try:
        return parse_single_table(io.BytesIO(content)).to_table()
    except IndexError:
        raise IndexError(error_msg) from None


def get_filter_list(facility: str, instrument: str | None = None, **kwargs) -> Table:
    """
    List the filters SVO has for a facility, optionally restricted to one instrument.

    Parameters
    ----------
    facility : str
        Facility name as registered in SVO, e.g. ``"Swift"``.
    instrument : str, optional
        Instrument name, e.g. ``"UVOT"``. Leave `None` to get every instrument of the facility.
    **kwargs
        Passed to `query_svo`, e.g. ``cache`` and ``timeout``.

    Returns
    -------
    astropy.table.Table
        Filter metadata, with columns such as ``filterID``, ``WavelengthEff``, ``FWHM`` and ``ZeroPoint``.
    """
    return query_svo(
        {"Facility": facility, "Instrument": instrument},
        error_msg="No filter found for requested Facility (and Instrument)",
        **kwargs,
    )


def get_transmission_data(filter_id: str, **kwargs) -> Table:
    """
    Fetch the transmission curve of one filter.

    Parameters
    ----------
    filter_id : str
        SVO filter ID in ``facility/instrument.band`` form, e.g. ``"Swift/UVOT.UVW1"``. These are the ``filterID``
        entries of `get_filter_list`.
    **kwargs
        Passed to `query_svo`, e.g. ``cache`` and ``timeout``.

    Returns
    -------
    astropy.table.Table
        Table with a ``Wavelength`` column (Angstrom) and a ``Transmission`` column.
    """
    table = query_svo({"ID": filter_id}, error_msg=f"No filter found for ID {filter_id!r}", **kwargs)
    if table["Wavelength"].unit is None:
        table["Wavelength"].unit = u.AA
    return table


def get_bandpass(filter_id: str, **kwargs) -> SpectralElement:
    """
    Fetch one filter as a `~synphot.SpectralElement`.

    The result can be passed straight to
    :meth:`~uvex_transients.models.core.base.SpectralModel.mag_bandpass`.

    Parameters
    ----------
    filter_id : str
        SVO filter ID in ``facility/instrument.band`` form, e.g. ``"Swift/UVOT.UVW1"``.
    **kwargs
        Passed to `get_transmission_data`, e.g. ``cache`` and ``timeout``.

    Returns
    -------
    synphot.SpectralElement
        The transmission curve, with non-finite or negative values set to zero.
    """
    table = get_transmission_data(filter_id, **kwargs)
    throughput = np.asarray(table["Transmission"], dtype=float)
    throughput = np.where(np.isfinite(throughput) & (throughput >= 0.0), throughput, 0.0)
    return SpectralElement(Empirical1D, points=table["Wavelength"].quantity.to(u.AA), lookup_table=throughput)


def clear_svo_cache() -> None:
    """
    Empty the astropy download cache.

    This removes every URL astropy has downloaded, not only SVO responses.
    """
    clear_download_cache()
