"""Samplers: how many detections in n tries at a point of a phase-space map, and how many tries to make."""

from uvex_transients.phase_space.samplers.base import Outcome, Sampler, available_samplers, get_sampler
from uvex_transients.phase_space.samplers.products import Estimate, Product, Scale, Weighted
from uvex_transients.phase_space.samplers.stopping import StoppingRule
from uvex_transients.phase_space.samplers.toy import LogisticSampler

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
    "get_sampler",
]


#: The survey-backed samplers, which are imported when they are first asked for, since they need the survey simulation.
_LAZY = {
    "BreakoutSampler": "uvex_transients.phase_space.breakout",
    "CanonicalSampler": "uvex_transients.phase_space.samplers.canonical",
    "RedshiftLimit": "uvex_transients.phase_space.samplers.canonical",
    "ParameterAxis": "uvex_transients.phase_space.samplers.survey",
    "SurveySampler": "uvex_transients.phase_space.samplers.survey",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
