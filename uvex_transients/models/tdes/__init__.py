"""Composite tidal disruption event SED models."""

__all__ = [
    "AlushStoneTDESED",
    "VanVelzenTDESED",
]

from .alush_stone import AlushStoneTDESED
from .van_velzen import VanVelzenTDESED
