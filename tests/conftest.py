"""Shared pytest fixtures for the ``uvex_transients`` test suite."""

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable
from astropy.time import Time
from m4opt.missions._uvex import uvex
from regions import CircleSkyRegion

from uvex_transients.surveys.base import SurveySchedule


@pytest.fixture
def fov() -> CircleSkyRegion:
    """
    A generous, boresight-centered (RA=0/Dec=0) circular FOV.

    `SurveySchedule` only supports Rectangle/Circle regions (see
    `SurveySchedule._bounding_radius`), unlike `uvex.fov`'s real chip-gapped polygon
    footprint -- the tests using this fixture only care about scheduling/photometry
    logic, not exact FOV containment.
    """
    return CircleSkyRegion(center=SkyCoord(0 * u.deg, 0 * u.deg), radius=1 * u.deg)


@pytest.fixture
def hot_spot() -> SkyCoord:
    """A fixed sky position that every `make_schedule`-built schedule below points at."""
    return SkyCoord(ra=150 * u.deg, dec=20 * u.deg)


@pytest.fixture
def make_schedule(fov, hot_spot):
    """Factory for a synthetic `SurveySchedule` of ``n_sched`` observations, all pointed at `hot_spot`."""

    def _make(n_sched=20, seed=0) -> SurveySchedule:
        rng = np.random.default_rng(seed)
        times = Time("2025-01-01T00:00:00") + np.sort(rng.uniform(0, 150, n_sched)) * u.day
        table = QTable()
        table["start_time"] = times
        table["duration"] = np.full(n_sched, 900.0) * u.s
        table["observer_location"] = uvex.observer_location(times)
        table["action"] = np.full(n_sched, "observe")
        table["target_coord"] = SkyCoord(
            np.full(n_sched, hot_spot.ra.value) * u.deg,
            np.full(n_sched, hot_spot.dec.value) * u.deg,
        )
        table["roll"] = np.zeros(n_sched) * u.deg
        table["field_id"] = np.arange(n_sched)
        table["block_id"] = np.zeros(n_sched, dtype=int)
        return SurveySchedule(table, fov)

    return _make


@pytest.fixture
def make_schedule_from_pointings(fov):
    """Factory for a synthetic `SurveySchedule` at explicit, per-observation ``coords``/``times``."""

    def _make(coords: SkyCoord, times: Time) -> SurveySchedule:
        n = len(times)
        table = QTable()
        table["start_time"] = times
        table["duration"] = np.full(n, 900.0) * u.s
        table["observer_location"] = uvex.observer_location(times)
        table["action"] = np.full(n, "observe")
        table["target_coord"] = coords
        table["roll"] = np.zeros(n) * u.deg
        table["field_id"] = np.arange(n)
        table["block_id"] = np.zeros(n, dtype=int)
        return SurveySchedule(table, fov)

    return _make
