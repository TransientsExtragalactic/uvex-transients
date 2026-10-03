"""
Rubin Observatory LSST survey footprints, built with the optional ``rubin_scheduler`` package.

The regions come from the scheduler's own labeled sky-area map
(`rubin_scheduler.scheduler.utils.SkyAreaGenerator`) and the deep drilling fields from
`rubin_scheduler.utils.ddf_locations`. Region labels, map resolution, and the DDF radius
live in ``config["observatories.lsst"]``. The footprints register themselves in
`~uvex_transients.surveys.footprints.base.default_registry` under ``lsst:*``.

``rubin_scheduler`` is only needed to *generate* a MOC. The scheduler data is downloaded
once through astropy's cache, and each finished MOC is cached by
`~uvex_transients.surveys.footprints.base.SurveyFootprint`, so later runs need neither.
Install it with ``pip install "uvex-transients[rubin]"``.
"""

import tarfile
from functools import cache, reduce
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy_healpix.core import ring_to_nested
from mocpy import MOC

from uvex_transients.utils import cache_dir, config, logger

from .base import SurveyFootprint, combine_footprints
from .utils import download_file_cached

__all__ = ["lsst_regions_MOC", "lsst_ddf_MOC", "lsst_footprints", "lsst_ddf_footprint", "lsst_combined_footprint"]

_cfg = config["observatories.lsst"]

_INSTALL_HINT = 'Generating the LSST footprints requires `rubin_scheduler`: `pip install "uvex-transients[rubin]"`.'


def _require_rubin_scheduler() -> None:
    """Raise an `ImportError` with install instructions if ``rubin_scheduler`` is missing."""
    try:
        import rubin_scheduler  # noqa: F401
    except ImportError as err:
        raise ImportError(_INSTALL_HINT) from err


def _dust_map_path(nside: int) -> Path:
    """
    Path to the scheduler's E(B-V) map at `nside`, downloading it on first use.

    The map is a ~1 MB member of the ~240 MB ``scheduler`` data archive. The archive is
    fetched through astropy's download cache, and the one member is extracted into this
    package's cache, so neither step repeats.
    """
    path = cache_dir / "lsst" / f"dust_nside_{nside}.npz"
    if path.is_file():
        return path

    from rubin_scheduler.data import DEFAULT_DATA_URL, data_dict

    logger.info("Loading the Rubin scheduler data (~240 MB, downloaded once and cached by astropy).")
    archive = download_file_cached(DEFAULT_DATA_URL + data_dict()["scheduler"])

    wanted = f"dust_nside_{nside}.npz"
    with tarfile.open(archive, "r:gz") as tar:
        member = next((m for m in tar if m.isfile() and Path(m.name).name == wanted), None)
        if member is None:
            raise FileNotFoundError(f"{wanted} is not in the Rubin scheduler data archive.")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(tar.extractfile(member).read())
        tmp.replace(path)
    return path


@cache
def _pixel_labels(nside: int) -> np.ndarray:
    """Region label of every HEALPix pixel (RING order) in the scheduler's sky-area map, ``""`` if unobserved."""
    _require_rubin_scheduler()
    from rubin_scheduler.scheduler.utils import SkyAreaGenerator

    dust_path = _dust_map_path(nside)

    class _CachedDustSkyArea(SkyAreaGenerator):
        def read_dustmap(self, dustmap_file=None):
            """
            Load the dust map from this package's cache, so no scheduler data directory is needed.

            Parameters
            ----------
            dustmap_file : None
                Unused; kept for compatibility with the parent signature.
            """
            with np.load(dust_path) as data:
                self.dustmap = data["ebvMap"]

    return np.asarray(_CachedDustSkyArea(nside=nside).return_maps()[1])


def lsst_regions_MOC(*, max_order: int, labels: list[str], nside: int) -> MOC:
    """
    Registry-contract generator for the union of LSST sky-area regions.

    Parameters
    ----------
    max_order : int
        Maximum order of the MOC. It is degraded to this if `nside` is finer.
    labels : list of str
        Scheduler region labels to include (e.g. ``["lowdust"]`` for the WFD area).
        If empty, every observed pixel is included.
    nside : int
        HEALPix resolution of the scheduler's sky-area map.

    Returns
    -------
    mocpy.MOC
        The union of the selected pixels.
    """
    pixel_labels = _pixel_labels(nside)
    selected = np.isin(pixel_labels, labels) if labels else pixel_labels != ""

    order = int(np.log2(nside))
    ipix = ring_to_nested(np.flatnonzero(selected), nside).astype(np.uint64)
    moc = MOC.from_healpix_cells(ipix, np.full(ipix.shape, order, dtype=np.uint8), order)
    return moc.degrade_to_order(max_order) if order > max_order else moc


def lsst_ddf_MOC(*, max_order: int, radius: float, fields: list[str] | None = None) -> MOC:
    """
    Registry-contract generator for the LSST deep drilling fields.

    Parameters
    ----------
    max_order : int
        MOC resolution (HEALPix order).
    radius : float
        Radius, in degrees, of the circle placed on each field center.
    fields : list of str, optional
        Names from `rubin_scheduler.utils.ddf_locations` (e.g. ``["COSMOS"]``).
        Default: all fields.

    Returns
    -------
    mocpy.MOC
        The union of the fields' circles.

    Raises
    ------
    KeyError
        If a name in `fields` is not a known field.
    """
    _require_rubin_scheduler()
    from rubin_scheduler.utils import ddf_locations

    locations = ddf_locations()
    names = list(fields) if fields else list(locations)
    if unknown := [n for n in names if n not in locations]:
        raise KeyError(f"Unknown LSST DDF(s) {unknown}. Known: {', '.join(locations)}")

    mocs = [
        MOC.from_cone(
            lon=locations[n][0] * u.deg, lat=locations[n][1] * u.deg, radius=radius * u.deg, max_depth=max_order
        )
        for n in names
    ]
    return reduce(MOC.union, mocs)


def _region_footprint(entry: dict) -> SurveyFootprint:
    return SurveyFootprint(
        name=entry["name"],
        generator=lsst_regions_MOC,
        description=entry["description"],
        params={"labels": entry["labels"], "nside": _cfg["nside"]},
        version=_cfg["version"],
        persist=True,
    )


lsst_footprints = {key: _region_footprint(entry) for key, entry in _cfg["regions"].items()}
"""dict: One footprint per ``observatories.lsst.regions`` entry, keyed by its config key."""

lsst_ddf_footprint = SurveyFootprint(
    name=_cfg["ddf"]["name"],
    generator=lsst_ddf_MOC,
    description=_cfg["ddf"]["description"],
    params={"radius": _cfg["ddf"]["radius"]},
    version=_cfg["version"],
    persist=True,
)

lsst_combined_footprint = combine_footprints(
    name=_cfg["combined"]["name"],
    operation="union",
    footprints=[lsst_footprints["main"], lsst_ddf_footprint],
    description=_cfg["combined"]["description"],
    version=_cfg["version"],
)
"""SurveyFootprint: Everything LSST observes: the main survey plus the deep drilling fields."""
