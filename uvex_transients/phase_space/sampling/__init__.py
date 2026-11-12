"""Samplers: how many detections in n tries at a point of a phase-space map, and how many tries to make."""

from uvex_transients.phase_space.sampling.base import (
    Estimate,
    Outcome,
    Product,
    Sampler,
    Scale,
    StoppingRule,
    Weighted,
    available_samplers,
    get_sampler,
)
from uvex_transients.phase_space.sampling.breakout import BreakoutSampler
from uvex_transients.phase_space.sampling.canonical import CanonicalSampler, RedshiftLimit
from uvex_transients.phase_space.sampling.core import ParameterAxis, SurveySampler, check_detection
from uvex_transients.phase_space.sampling.logistic import LogisticSampler

__all__ = [
    "BreakoutSampler",
    "CanonicalSampler",
    "Estimate",
    "LogisticSampler",
    "Outcome",
    "ParameterAxis",
    "Product",
    "RedshiftLimit",
    "Sampler",
    "Scale",
    "StoppingRule",
    "SurveySampler",
    "Weighted",
    "available_samplers",
    "check_detection",
    "get_sampler",
]
