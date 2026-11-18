"""Composite supernova SED models."""

__all__ = [
    "ArnettMagnetarSpindownSED",
    "LLGRBSED",
    "TypeIaSED",
    "TypeIIPExcessSED",
    "TypeIIPSED",
    "TypeIIbSED",
    "TypeIbSED",
    "TypeIcSED",
    "TypeIcBLSED",
    "VillarCoolingBlackbodySED",
]

from ..arnett import ArnettMagnetarSpindownSED
from .Ia import TypeIaSED
from .Ibc import TypeIbSED, TypeIcSED
from .IcBL import TypeIcBLSED
from .IIb import TypeIIbSED
from .IIp import TypeIIPExcessSED, TypeIIPSED
from .llgrb import LLGRBSED
from .villar import VillarCoolingBlackbodySED
