"""
Place the shipped transient classes on a recovery map, by measuring the two quantities the map is made of.

A map is over the peak luminosity in a band and the rest-frame half-decline time of a canonical transient (see
`~uvex_transients.phase_space.transient._CanonicalTransient`). `measure_class` draws parameters from a class's SED
priors and measures the same two quantities on each draw, in the band of the map, which gives the region of the map the
class occupies.

The overlay shows where a class sits, not how many of it the survey recovers: two events with the same peak and
half-decline time can differ in rise, tail and colour evolution, which the map holds fixed. A class's yield at its
place on the map should be checked against a full simulation of it.
"""

import importlib
from collections.abc import Sequence

import numpy as np
from astropy import units as u
from astropy.table import QTable, Table

from uvex_transients.models.core.lightcurve_metrics import decline_time, peak_luminosity
from uvex_transients.transients.base import ExtragalacticTransient, TransientBase
from uvex_transients.utils import config

__all__ = ["measure_class", "shipped_classes", "split_measurements", "stack_measurements"]

# ============================================ #
# Classes                                      #
# ============================================ #
#: The modules of shipped transient classes, imported so that the classes register.
_SHIPPED_MODULES = ("TDEs", "LFBOTs", "kilonovae", "supernovae")

#: Classes left off the maps. ``ShockCoolingIIb`` is the bespoke shock-cooling phase of a Type IIb alone, with an SED
#: that is zero outside its regime of validity, so its half-decline time is set by where that window ends. The Type IIb
#: population is on the maps as ``TypeIIbSNe``.
_NOT_MAPPED = frozenset({"ShockCoolingIIb"})


def shipped_classes() -> dict[str, type[ExtragalacticTransient]]:
    """
    Return the extragalactic transient classes that belong on the maps, by name.

    Returns
    -------
    dict
        ``{class name: class}``, for every class in the transient modules that is not private, apart from the bespoke
        ones that are not mapped.
    """
    # Importing the modules is what registers the classes
    for module in _SHIPPED_MODULES:
        importlib.import_module(f"uvex_transients.transients.{module}")
    return {
        name: cls
        for name, cls in sorted(TransientBase.registry().items())
        if not name.startswith("_") and name not in _NOT_MAPPED and issubclass(cls, ExtragalacticTransient)
    }


# ============================================ #
# Measurement                                  #
# ============================================ #
def measure_class(
    transient_class: type[ExtragalacticTransient],
    bandpass,
    *,
    n_draws: int | None = None,
    t_min: u.Quantity | None = None,
    t_max: u.Quantity | None = None,
    n_time: int | None = None,
    rng=0,
) -> QTable:
    r"""
    Measure the peak band luminosity and the half-decline time of draws from a class's SED priors.

    Parameters
    ----------
    transient_class : type
        A transient class; its SED, with the priors it gives its parameters, is sampled.
    bandpass : synphot.SpectralElement
        The band of the map, for example ``mission.detector.bandpasses["FUV"]``.
    n_draws : int, optional
        Number of parameter draws. If `None` (the default), uses ``config["phase_space.overlay.n_draws"]``.
    t_min, t_max : ~astropy.units.Quantity, optional
        Ends of the logarithmic grid of rest-frame times the light curve is evaluated on. By default
        ``config["phase_space.overlay.t_min"]`` days, and ``config["phase_space.overlay.t_max_factor"]`` times the
        class's ``DEFAULT_DURATION``.
    n_time : int, optional
        Number of grid points. If `None` (the default), uses ``config["phase_space.overlay.n_time"]``.
    rng : int or numpy.random.Generator, optional
        Random-number source for the draws, as for
        `~uvex_transients.models.core.base.SpectralModel.sample_parameters`.

    Returns
    -------
    ~astropy.table.QTable
        One row per draw: ``log_luminosity`` (log10 of the peak band luminosity in erg/s), ``log_half_time`` (log10 of
        the half-decline time in days), and ``measured``, false where the draw has no peak or its luminosity never
        halves within the grid. ``meta`` holds the class name.
    """
    # Anything not given comes from the configuration
    n_draws = config["phase_space.overlay.n_draws"] if n_draws is None else n_draws
    n_time = config["phase_space.overlay.n_time"] if n_time is None else n_time
    if t_min is None:
        t_min = config["phase_space.overlay.t_min"] * u.day
    if t_max is None:
        t_max = config["phase_space.overlay.t_max_factor"] * transient_class.DEFAULT_DURATION

    # The light curve of every draw on a shared log-spaced grid of rest-frame times, in the band of the map
    sed = transient_class().sed
    t = np.geomspace(t_min.to_value(u.day), t_max.to_value(u.day), n_time) * u.day
    draws = sed.sample_parameters(size=n_draws, rng=rng)
    luminosity = sed.eval_band_luminosity(bandpass, t, **draws)

    # The two numbers the map is made of
    peak, _ = peak_luminosity(t, luminosity)
    half_time = decline_time(t, luminosity, 0.5)
    peak_value, half_value = peak.to_value(u.erg / u.s), half_time.to_value(u.day)

    # A draw is measured only if it has a peak and its light halves within the grid
    measured = np.isfinite(peak_value) & (peak_value > 0) & np.isfinite(half_value) & (half_value > 0)
    table = QTable()
    with np.errstate(divide="ignore", invalid="ignore"):
        table["log_luminosity"] = np.log10(peak_value)
        table["log_half_time"] = np.log10(half_value)
    table["measured"] = measured
    table.meta["class"] = transient_class.__name__
    return table


def stack_measurements(measurements: Sequence[Table]) -> Table:
    """
    Stack the measurements of several classes into one table, for a single file.

    Parameters
    ----------
    measurements : sequence of astropy.table.Table
        Tables from `measure_class`.

    Returns
    -------
    ~astropy.table.Table
        Their rows one after another, with the class name of each in a ``class`` column.
    """
    names = ("log_luminosity", "log_half_time", "measured")
    return Table(
        {
            "class": np.concatenate([[table.meta["class"]] * len(table) for table in measurements]),
            **{name: np.concatenate([np.asarray(table[name]) for table in measurements]) for name in names},
        }
    )


def split_measurements(stacked: Table) -> list[Table]:
    """
    Split a table from `stack_measurements` back into one table per class.

    Parameters
    ----------
    stacked : astropy.table.Table
        A stacked table, for example a map's ``overlay.ecsv``.

    Returns
    -------
    list of astropy.table.Table
        One table per class, in order of first appearance, each with the class name in ``meta["class"]`` as
        `measure_class` leaves it (so each can be passed to ``plot_map(overlays=...)``).
    """
    classes = np.asarray(stacked["class"]).astype(str)
    tables = []
    for name in dict.fromkeys(classes):
        table = Table(stacked[classes == name]["log_luminosity", "log_half_time", "measured"])
        table.meta["class"] = str(name)
        tables.append(table)
    return tables
