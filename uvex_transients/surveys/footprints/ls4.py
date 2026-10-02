"""
The LS4 SOLE survey footprint.

LS4 (Bellm et al. 2025, arXiv:2503.14579) has no published footprint file. Only the SOLE
(Stellar Oscillations, Lensing, and Eruptions) survey is included, since Section 3.1.3 of
the paper gives its Galactic plane and bulge region explicitly as two boxes in Galactic
coordinates (``config["observatories.ls4.sole"]``; see
`~uvex_transients.surveys.footprints.utils.union_of_galactic_boxes_MOC`). The LEG survey
has no numeric declination cutoff in the paper, and FOOT follows LSST's evolving
low-season field list, so neither is included.

The footprint registers itself in `~uvex_transients.surveys.footprints.base.default_registry`.
"""

from uvex_transients.utils import config

from .base import SurveyFootprint
from .utils import union_of_galactic_boxes_MOC

_sole_cfg = config["observatories.ls4.sole"]

ls4_sole_footprint = SurveyFootprint(
    name=_sole_cfg["name"],
    generator=union_of_galactic_boxes_MOC,
    description=_sole_cfg["description"],
    params={"boxes": _sole_cfg["boxes"]},
    persist=True,
)
