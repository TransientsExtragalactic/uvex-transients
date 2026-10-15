"""Frequency-dependent spectral shape models."""

__all__ = [
    "BlackbodySpectrum",
    "BrokenPowerLawSpectrum",
    "ModifiedBlackbodySpectrum",
    "PowerLawSpectrum",
]

from .powerlaw import BrokenPowerLawSpectrum, PowerLawSpectrum
from .thermal import BlackbodySpectrum, ModifiedBlackbodySpectrum
