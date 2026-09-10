"""Tests for `SurveySchedule.get_healpix_coverage_index`/`get_observation_indices_of`."""

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable
from astropy.time import Time
from m4opt.missions._uvex import uvex
from regions import CircleSkyRegion

from uvex_transients.surveys.base import SurveySchedule

_FOV = CircleSkyRegion(center=SkyCoord(0 * u.deg, 0 * u.deg), radius=1 * u.deg)
_HOT_SPOT = SkyCoord(ra=150 * u.deg, dec=20 * u.deg)


def _make_schedule(n_sched=20, seed=0):
    rng = np.random.default_rng(seed)
    times = Time("2025-01-01T00:00:00") + np.sort(rng.uniform(0, 150, n_sched)) * u.day
    table = QTable()
    table["start_time"] = times
    table["duration"] = np.full(n_sched, 900.0) * u.s
    table["observer_location"] = uvex.observer_location(times)
    table["action"] = np.full(n_sched, "observe")
    table["target_coord"] = SkyCoord(
        np.full(n_sched, _HOT_SPOT.ra.value) * u.deg,
        np.full(n_sched, _HOT_SPOT.dec.value) * u.deg,
    )
    table["roll"] = np.zeros(n_sched) * u.deg
    table["field_id"] = np.arange(n_sched)
    table["block_id"] = np.zeros(n_sched, dtype=int)
    return SurveySchedule(table, _FOV)


def test_get_observation_indices_of_matches_exact_well_inside_fov():
    """
    For a query position solidly inside the FOV (not near a pixel/footprint edge),
    the fast HEALPix-index path and the exact `get_observations_of` agree exactly.
    """
    schedule = _make_schedule()
    nside, order = 128, "nested"

    # `_HOT_SPOT` itself is exactly the pointing center -- as far from any FOV edge
    # as a position can be, so this comparison isn't sensitive to the known
    # pixel-center-vs-exact-boundary approximation (see `get_healpix_coverage_index`'s
    # own docstring for that tradeoff).
    t_start = Time("2025-01-01T00:00:00")
    t_end = Time("2025-06-01T00:00:00")

    exact = schedule.get_observations_of(_HOT_SPOT, t_start, t_end)
    _, row_index = schedule.get_observation_indices_of(
        _HOT_SPOT, nside=nside, order=order, start_time=t_start, end_time=t_end
    )
    fast = schedule.observe_rows[row_index]

    assert set(np.asarray(exact["field_id"])) == set(np.asarray(fast["field_id"]))
    assert len(exact) > 0  # a meaningless pass if nothing was scheduled there at all


def test_get_healpix_coverage_index_caches_index_per_resolution():
    schedule = _make_schedule()

    assert schedule._HPX_MAP_CACHE == {}

    schedule.get_healpix_coverage_index(nside=128, order="nested")
    assert list(schedule._HPX_MAP_CACHE) == [(128, "nested")]

    # A second call at the *same* resolution reuses the cached index, not rebuilds it.
    cached_offsets, cached_rows = schedule._HPX_MAP_CACHE[(128, "nested")]
    schedule.get_healpix_coverage_index(nside=128, order="nested")
    offsets_again, rows_again = schedule._HPX_MAP_CACHE[(128, "nested")]
    assert offsets_again is cached_offsets
    assert rows_again is cached_rows

    # A *different* resolution gets its own, separately cached index.
    schedule.get_healpix_coverage_index(nside=64, order="nested")
    assert set(schedule._HPX_MAP_CACHE) == {(128, "nested"), (64, "nested")}


def test_get_observation_indices_of_empty_for_unobserved_pixel():
    schedule = _make_schedule()
    far_away = SkyCoord(ra=10 * u.deg, dec=-60 * u.deg)

    query_index, row_index = schedule.get_observation_indices_of(far_away, nside=128, order="nested")
    assert len(query_index) == 0
    assert len(row_index) == 0


def test_get_observation_indices_of_requires_both_time_bounds():
    schedule = _make_schedule()

    with pytest.raises(ValueError, match="must be given together"):
        schedule.get_observation_indices_of(_HOT_SPOT, nside=128, order="nested", start_time=Time("2025-01-01"))
