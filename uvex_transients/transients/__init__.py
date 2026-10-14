"""Transient population base classes composing a TransientSED with rate/metadata."""

from .base import ExtragalacticTransient, TransientBase
from .horizon import HorizonCurve

__all__ = [
    "ExtragalacticTransient",
    "HorizonCurve",
    "TransientBase",
]

from . import kilonovae
from .kilonovae import *

__all__ += kilonovae.__all__

from . import supernovae
from .supernovae import *

__all__ += supernovae.__all__

from . import LFBOTs
from .LFBOTs import *

__all__ += LFBOTs.__all__

from . import TDEs
from .TDEs import *

__all__ += TDEs.__all__
