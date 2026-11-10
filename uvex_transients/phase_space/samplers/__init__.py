"""Samplers: how many detections in n tries at a point of a phase-space map, and how many tries to make."""

from uvex_transients.phase_space.samplers.base import Outcome, Product, Sampler, available_samplers, get_sampler
from uvex_transients.phase_space.samplers.stopping import StoppingRule
from uvex_transients.phase_space.samplers.toy import LogisticSampler

__all__ = [
    "LogisticSampler",
    "Outcome",
    "Product",
    "Sampler",
    "StoppingRule",
    "available_samplers",
    "get_sampler",
]
