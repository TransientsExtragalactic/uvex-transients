"""Samplers: how many detections in n tries at a point of a phase-space map, and how many tries to make."""

from uvex_transients.phase_space.samplers.base import Outcome, Sampler, available_samplers, get_sampler
from uvex_transients.phase_space.samplers.products import Estimate, Product, Scale, Weighted
from uvex_transients.phase_space.samplers.stopping import StoppingRule
from uvex_transients.phase_space.samplers.toy import LogisticSampler

__all__ = [
    "Estimate",
    "LogisticSampler",
    "Outcome",
    "Product",
    "Sampler",
    "Scale",
    "StoppingRule",
    "Weighted",
    "available_samplers",
    "get_sampler",
]
