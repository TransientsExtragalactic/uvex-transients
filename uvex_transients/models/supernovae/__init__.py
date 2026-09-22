"""Composite supernova SED models."""

__all__ = [
    "ArnettMagnetarSpindownSED",
    "MoragShockCoolingBlackbodySED",
    "MoragShockCoolingSED",
    "TypeIIPExcessSED",
    "TypeIIPSED",
    "TypeIIbSED",
    "TypeIbSED",
    "TypeIcSED",
    "VillarCoolingBlackbodySED",
]

from .Ibc import TypeIbSED, TypeIcSED
from .IIb import MoragShockCoolingBlackbodySED, MoragShockCoolingSED, TypeIIbSED
from .IIp import TypeIIPExcessSED, TypeIIPSED
from .magnetar import ArnettMagnetarSpindownSED
from .villar import VillarCoolingBlackbodySED
