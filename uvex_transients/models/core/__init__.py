"""Core SED framework (models, parameters, priors) shared by every transient class."""

__all__ = [
    "ComposedSpectralModel",
    "Lightcurve",
    "Parameter",
    "SpectralModel",
    "Spectrum",
    "priors",
]

from . import priors
from .priors import *

__all__.extend(priors.__all__)

from .base import ComposedSpectralModel, Lightcurve, SpectralModel, Spectrum
from .parameters import Parameter
