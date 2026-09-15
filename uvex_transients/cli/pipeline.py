"""Plain functions implementing each CLI stage, kept free of `click` so they're directly testable.

`uvex_transients.cli.main`'s subcommands are thin wrappers around these -- each one parses
arguments, calls one of these, and reports a short summary.
"""

from astropy.table import QTable

from ..simulation.event_catalog import EventCatalog
from .config import RunConfig


def run_generate(config: RunConfig) -> EventCatalog:
    """Sample a Monte Carlo `EventCatalog` per the config's ``generate:`` section."""
    settings = config.generate
    return config.simulator.generate_events(
        time_bins=settings.time_bins,
        nside=settings.nside,
        order=settings.order,
        downsample=settings.downsample,
    )


def run_cuts(config: RunConfig, catalog: EventCatalog, names: list[str] | None = None) -> EventCatalog:
    """
    Run one or more of the config's declared ``cuts:`` against `catalog`, chained in order.

    Parameters
    ----------
    config : RunConfig
    catalog : EventCatalog
    names : list of str, optional
        Which of the config's `cuts:` keys to run, and in what order. If `None` (the
        default), runs every declared cut, in the order it was declared in the config.

    Returns
    -------
    EventCatalog
    """
    cuts = config.cuts

    if names is None:
        selected = list(cuts)
    else:
        unknown = [name for name in names if name not in cuts]
        if unknown:
            raise ValueError(f"Unknown cut key(s) {unknown}; available: {list(cuts)}.")
        selected = list(names)

    for key in selected:
        spec = cuts[key]
        catalog = config.simulator.run_cut(spec.type, catalog, config.mission, **spec.params)

    return catalog


def run_photometry(config: RunConfig, catalog: EventCatalog) -> QTable:
    """Run synthetic photometry over every event in `catalog` per the config's ``photometry:`` section."""
    settings = config.photometry
    return catalog.simulate_photometry(
        config.mission,
        config.transients,
        config.schedule,
        bands=settings.bands,
        n_sigma=settings.n_sigma,
    )
