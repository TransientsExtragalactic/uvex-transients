"""Composite tidal disruption event SED models."""

__all__ = [
    "AlushStoneTDESED",
    "VanVelzenTDESED",
]

from ._alush_stone import AlushStoneTDESED
from ._van_velzen import VanVelzenTDESED
