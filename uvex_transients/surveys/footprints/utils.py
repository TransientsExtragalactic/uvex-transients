"""
Building blocks for the `~mocpy.MOC` generators behind `~uvex_transients.surveys.footprints.base.SurveyFootprint`.

Every generator here follows the registry contract ``generator(*, max_order, **params) -> MOC``
(see `~uvex_transients.surveys.footprints.base.SurveyFootprint.generator`): a download helper
(`download_file_cached`), DS9 region files (`fetch_and_generate_MOC_from_URL`), and simple
analytic shapes (`all_sky_MOC`, `dec_band_MOC`, `union_of_galactic_boxes_MOC`).
"""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.utils.data import download_file
from mocpy import MOC
from regions import CircleSkyRegion, PointSkyRegion, PolygonSkyRegion, Regions

__all__ = [
    "download_file_cached",
    "convert_region_to_MOC",
    "fetch_and_generate_MOC_from_URL",
    "all_sky_MOC",
    "dec_band_MOC",
    "galactic_box_MOC",
    "union_of_galactic_boxes_MOC",
]


# ====================================== #
# Downloading                            #
# ====================================== #
def download_file_cached(url: str, cache: bool = True, show_progress: bool = True) -> Path:
    """
    Download a file, reusing astropy's on-disk cache when possible.

    This is the same cache `uvex_transients.dust.dust_map` uses. The cached path
    does not keep the URL's filename or extension.

    Parameters
    ----------
    url : str
        HTTP/HTTPS URL of the file.
    cache : bool, optional
        Whether to read from and write to astropy's download cache. Default `True`.
    show_progress : bool, optional
        Whether to show a progress bar while downloading. Default `True`.

    Returns
    -------
    pathlib.Path
        Local path to the (cached) file.
    """
    return Path(download_file(url, cache=cache, show_progress=show_progress))


# ====================================== #
# Shared helpers                         #
# ====================================== #
def _union(mocs: list[MOC], what: str) -> MOC:
    """Union of `mocs`, raising `ValueError` if there are none (`what` names them in the message)."""
    if not mocs:
        raise ValueError(f"`{what}` is empty: the footprint would cover no sky.")
    return mocs[0] if len(mocs) == 1 else MOC.union(*mocs)


# ====================================== #
# Region -> MOC conversion               #
# ====================================== #
def convert_region_to_MOC(regions: Regions, max_order: int, point_region_size: float | None = None) -> MOC:
    """
    Convert parsed DS9 regions into a single `~mocpy.MOC`.

    Circles and polygons convert directly. A `~regions.PointSkyRegion` is a pointing,
    not an area, so it is first expanded into a circle of radius `point_region_size`.

    Parameters
    ----------
    regions : regions.Regions
        Parsed region file contents, e.g. ``Regions.read(path, format="ds9")``.
    max_order : int
        MOC resolution (HEALPix order; order 10 is ~3.4 arcmin cells).
    point_region_size : float, optional
        Radius, in degrees, used to expand point regions. Required if `regions`
        contains any.

    Returns
    -------
    mocpy.MOC
        The union of every region's MOC.

    Raises
    ------
    ValueError
        If `regions` is empty, or has a point region and `point_region_size` is `None`.
    TypeError
        If a region is not a circle, polygon, or point.
    """
    mocs = []
    for region in regions:
        if isinstance(region, PointSkyRegion):
            if point_region_size is None:
                raise ValueError("`regions` contains a point region; `point_region_size` is required.")
            region = CircleSkyRegion(center=region.center, radius=point_region_size * u.deg)
        elif not isinstance(region, (CircleSkyRegion, PolygonSkyRegion)):
            raise TypeError(f"Unsupported region type: {type(region).__name__}.")
        mocs.append(MOC.from_astropy_regions(region, max_depth=max_order))

    return _union(mocs, "regions")


# ====================================== #
# Registry-contract generators           #
# ====================================== #
def fetch_and_generate_MOC_from_URL(*, max_order: int, url: str, point_region_size: float | None = None) -> MOC:
    """
    Download a DS9 region file and convert it to a `~mocpy.MOC`.

    Parameters
    ----------
    max_order : int
        Forwarded to `convert_region_to_MOC`.
    url : str
        URL of the region file (see `download_file_cached`).
    point_region_size : float, optional
        Forwarded to `convert_region_to_MOC`.

    Returns
    -------
    mocpy.MOC
        The footprint's MOC.
    """
    path = download_file_cached(url)
    regions = Regions.read(str(path), format="ds9")
    return convert_region_to_MOC(regions, max_order=max_order, point_region_size=point_region_size)


def all_sky_MOC(*, max_order: int) -> MOC:
    """
    Registry-contract generator for an unrestricted, all-sky footprint.

    Parameters
    ----------
    max_order : int
        MOC resolution (HEALPix order).

    Returns
    -------
    mocpy.MOC
        A MOC covering the entire sky.
    """
    return MOC.from_zone(SkyCoord([[0, -90], [360, 90]], unit="deg"), max_depth=max_order)


def dec_band_MOC(*, max_order: int, min_dec: float, max_dec: float = 90.0) -> MOC:
    """
    Registry-contract generator for a declination band covering every right ascension.

    Parameters
    ----------
    max_order : int
        MOC resolution (HEALPix order).
    min_dec, max_dec : float
        Declination limits, in degrees. `max_dec` defaults to the north celestial pole.

    Returns
    -------
    mocpy.MOC
        The band between `min_dec` and `max_dec`.
    """
    return MOC.from_zone(SkyCoord([[0, min_dec], [360, max_dec]], unit="deg"), max_depth=max_order)


def galactic_box_MOC(
    *, max_order: int, l_min: float, l_max: float, b_min: float, b_max: float, n_edge: int = 180
) -> MOC:
    """
    Build a box bounded by constant Galactic longitude and latitude.

    `~mocpy.MOC.from_zone` always draws ICRS edges, so it cannot bound a Galactic box.
    This builds a polygon instead, sampling each constant-latitude edge (a small circle)
    with `n_edge` points, enough to keep the chord error far below one cell at `max_order`.

    Parameters
    ----------
    max_order : int
        MOC resolution (HEALPix order).
    l_min, l_max : float
        Galactic longitude range, in degrees.
    b_min, b_max : float
        Galactic latitude range, in degrees.
    n_edge : int, optional
        Points sampled along each constant-latitude edge. Default 180.

    Returns
    -------
    mocpy.MOC
        The box.
    """
    lon = np.linspace(l_min, l_max, n_edge)
    top = np.column_stack([lon, np.full_like(lon, b_max)])
    bottom = np.column_stack([lon[::-1], np.full_like(lon, b_min)])
    vertices = np.vstack([top, bottom])
    coords = SkyCoord(l=vertices[:, 0] * u.deg, b=vertices[:, 1] * u.deg, frame="galactic")
    return MOC.from_polygon_skycoord(coords, max_depth=max_order)


def union_of_galactic_boxes_MOC(*, max_order: int, boxes: Iterable[dict]) -> MOC:
    """
    Registry-contract generator for the union of several `galactic_box_MOC` boxes.

    Parameters
    ----------
    max_order : int
        Forwarded to `galactic_box_MOC`.
    boxes : Iterable of dict
        One dict of `galactic_box_MOC` keyword arguments (``l_min``, ``l_max``,
        ``b_min``, ``b_max``, and optionally ``n_edge``) per box.

    Returns
    -------
    mocpy.MOC
        The union of every box.

    Raises
    ------
    ValueError
        If `boxes` is empty.
    """
    return _union([galactic_box_MOC(max_order=max_order, **box) for box in boxes], "boxes")
