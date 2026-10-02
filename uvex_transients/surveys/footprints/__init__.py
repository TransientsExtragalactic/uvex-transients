"""Named survey sky footprints; see `~uvex_transients.surveys.footprints.base.SurveyFootprint`."""

from .base import FootprintRegistry, SurveyFootprint, combine_footprints, default_registry
from .ls4 import ls4_sole_footprint
from .lsst import lsst_combined_footprint, lsst_ddf_footprint, lsst_footprints
from .uvex import (
    uvex_allsky_footprint,
    uvex_lmlz_deep_footprint,
    uvex_lmlz_wide_footprint,
    uvex_mc_footprint,
)
from .ztf import ztf_footprint

__all__ = [
    "SurveyFootprint",
    "FootprintRegistry",
    "default_registry",
    "combine_footprints",
    "uvex_allsky_footprint",
    "uvex_lmlz_wide_footprint",
    "uvex_lmlz_deep_footprint",
    "uvex_mc_footprint",
    "ztf_footprint",
    "ls4_sole_footprint",
    "lsst_footprints",
    "lsst_ddf_footprint",
    "lsst_combined_footprint",
]
