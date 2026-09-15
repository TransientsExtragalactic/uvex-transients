"""Tests for the `@cut`/registry mechanism on `SurveySimulator`."""

import numpy as np
import pytest
from astropy.table import QTable
from astropy.time import Time
from m4opt.missions._uvex import uvex

from uvex_transients.simulation.core import SurveySimulator, cut
from uvex_transients.simulation.event_catalog import EventCatalog
from uvex_transients.transients.TDEs import TidalDisruptionEvent


def _empty_catalog() -> EventCatalog:
    """Build an empty catalog, cheap to pass through either built-in cut (both short-circuit on it)."""
    table = QTable()
    for name in (
        "healpix_id",
        "healpix_dx",
        "healpix_dy",
        "redshift",
        "parameter_seed",
        "transient_type",
        "time_bin",
        "event_id",
        "ebv",
    ):
        table[name] = np.array([])
    table["luminosity_distance"] = np.array([])
    return EventCatalog(
        table=table,
        nside=64,
        order="nested",
        time_bins=Time(["2025-01-01", "2025-06-01"]),
    )


def test_available_cuts_includes_builtins():
    """`SurveySimulator.available_cuts` lists the two built-in filter methods by their `@cut` name."""
    assert SurveySimulator.available_cuts() == ("limiting_magnitude", "snr")


def test_run_cut_dispatches_to_the_registered_method(make_schedule):
    """`run_cut("limiting_magnitude", ...)` produces the same result as calling the method directly."""
    simulator = SurveySimulator(make_schedule(), transients={"tde": TidalDisruptionEvent()})
    catalog = _empty_catalog()

    direct = simulator.filter_by_limiting_magnitude(catalog, uvex, mag_limit=25.0)
    via_registry = simulator.run_cut("limiting_magnitude", catalog, uvex, mag_limit=25.0)

    assert len(direct) == len(via_registry) == 0


def test_run_cut_unknown_name_raises(make_schedule):
    """An unregistered cut name raises a `ValueError` naming the ones that do exist."""
    simulator = SurveySimulator(make_schedule(), transients={"tde": TidalDisruptionEvent()})
    with pytest.raises(ValueError, match="Unknown cut 'bogus'"):
        simulator.run_cut("bogus", _empty_catalog(), uvex)


def test_subclass_extends_the_registry():
    """A subclass adding a new `@cut`-decorated method registers it without touching the base class."""

    class _ExtraCutSimulator(SurveySimulator):
        @cut("always_keep")
        def filter_by_nothing(self, catalog, mission, **kwargs):
            return catalog

    assert "always_keep" in _ExtraCutSimulator.available_cuts()
    assert "always_keep" not in SurveySimulator.available_cuts()
