"""Composite supernova SED models."""

__all__ = [
    "ArnettMagnetarSpindownSED",
    "MoragShockCoolingBlackbodySED",
    "MoragShockCoolingSED",
    "TypeIaSED",
    "TypeIIPExcessSED",
    "TypeIIPSED",
    "TypeIIbSED",
    "TypeIbSED",
    "TypeIcSED",
    "VillarCoolingBlackbodySED",
]

from ..arnett import ArnettMagnetarSpindownSED
from .Ia import TypeIaSED
from .Ibc import TypeIbSED, TypeIcSED
from .IIb import MoragShockCoolingBlackbodySED, MoragShockCoolingSED, TypeIIbSED
from .IIp import TypeIIPExcessSED, TypeIIPSED
from .villar import VillarCoolingBlackbodySED
