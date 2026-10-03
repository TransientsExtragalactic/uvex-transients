"""
The ZTF survey footprint.

ZTF has no published region file. Its coverage is the sky visible from Palomar, which a
declination cut (``config["observatories.ztf.min_dec"]``) describes well; see
`~uvex_transients.surveys.footprints.utils.dec_band_MOC`. The footprint registers itself
in `~uvex_transients.surveys.footprints.base.default_registry` as ``"ztf:main"``.
"""

from uvex_transients.utils import config

from .base import SurveyFootprint
from .utils import dec_band_MOC

ztf_footprint = SurveyFootprint(
    name="ztf:main",
    generator=dec_band_MOC,
    description="ZTF's visible sky from Palomar Observatory (a declination cut, not a region file).",
    params={"min_dec": config["observatories.ztf.min_dec"]},
)
