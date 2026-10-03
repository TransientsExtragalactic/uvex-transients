"""
UVEX survey footprints, from DS9 region files in the uvex-scheduler repo.

The files live at https://github.com/m4opt/uvex-scheduler/tree/main/survey-footprints.
URLs and per-footprint settings live in ``config["observatories.uvex"]``. The footprints
register themselves in `~uvex_transients.surveys.footprints.base.default_registry`,
e.g. ``default_registry["uvex:lmlz:wide"]``.
"""

from uvex_transients.utils import config

from .base import SurveyFootprint
from .utils import all_sky_MOC, fetch_and_generate_MOC_from_URL

_cfg = config["observatories.uvex"]


def _region_footprint(key: str) -> SurveyFootprint:
    entry = _cfg["footprints"][key]
    return SurveyFootprint(
        name=entry["name"],
        generator=fetch_and_generate_MOC_from_URL,
        description=entry["description"],
        params={"url": f"{_cfg['repo_url']}/{entry['file']}", "point_region_size": _cfg["point_region_size"]},
        persist=True,
    )


uvex_allsky_footprint = SurveyFootprint(
    name="uvex:allsky",
    generator=all_sky_MOC,
    description="UVEX all-sky survey (full sky).",
)
uvex_lmlz_wide_footprint = _region_footprint("lmlz_wide")
uvex_lmlz_deep_footprint = _region_footprint("lmlz_deep")
uvex_mc_footprint = _region_footprint("magellanic_clouds")
