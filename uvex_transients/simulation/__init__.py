"""Monte Carlo simulation of transient detectability against a survey schedule."""

from .core import SurveySimulator
from .event import Event
from .event_catalog import EventCatalog, get_example_event_catalog
from .exposure_catalog import ExposureCatalog
from .photometry_catalog import PhotometryCatalog
from .rates import estimate_yield

__all__ = [
    "Event",
    "EventCatalog",
    "ExposureCatalog",
    "PhotometryCatalog",
    "SurveySimulator",
    "estimate_yield",
    "get_example_event_catalog",
]
