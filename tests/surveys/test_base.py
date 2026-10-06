"""Tests for `SurveySchedule.get_healpix_coverage_index`/`get_observation_indices_of` and cadence diagnostics."""

import astropy_healpix as ah
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable, vstack
from astropy.time import Time
from m4opt.missions._uvex import uvex

from uvex_transients.surveys.base import SurveySchedule


def test_get_observation_indices_of_matches_exact_well_inside_fov(make_schedule, hot_spot):
    """
    For a query position solidly inside the FOV (not near a pixel/footprint edge),
    the fast HEALPix-index path and the exact `get_observations_of` agree exactly.
    """
    schedule = make_schedule()
    nside, order = 128, "nested"

    # `hot_spot` itself is exactly the pointing center, as far from any FOV edge as a position can
    # be. Positions near the edge are covered by the test below.
    t_start = Time("2025-01-01T00:00:00")
    t_end = Time("2025-06-01T00:00:00")

    exact = schedule.get_observations_of(hot_spot, t_start, t_end)
    _, row_index = schedule.get_observation_indices_of(
        hot_spot, nside=nside, order=order, start_time=t_start, end_time=t_end
    )
    fast = schedule.observe_rows[row_index]

    assert set(np.asarray(exact["field_id"])) == set(np.asarray(fast["field_id"]))
    assert len(exact) > 0  # a meaningless pass if nothing was scheduled there at all


def _edge_points(hot_spot, n=300, seed=0):
    """Query positions scattered just inside and just outside the 1 degree circular FOV around `hot_spot`."""
    rng = np.random.default_rng(seed)
    separation = rng.uniform(0.85, 1.15, n) * u.deg
    position_angle = rng.uniform(0, 360, n) * u.deg
    return hot_spot.directional_offset_by(position_angle, separation)


@pytest.mark.parametrize("nside", [64, 128])
def test_get_observation_indices_of_matches_exact_at_the_footprint_edge(make_schedule, hot_spot, nside):
    """
    A point inside the footprint is found even when its own pixel's center is outside it.

    The index registers an observation under a pixel only if the pixel center is inside the
    footprint, so a lookup of the point's own pixel alone misses edge points. Neighbouring
    pixels are searched too, and every candidate is then confirmed with the exact test, so the
    result matches `get_observations_of` for points on both sides of the edge.
    """
    schedule = make_schedule(n_sched=6)
    points = _edge_points(hot_spot)
    t_start, t_end = Time("2025-01-01T00:00:00"), Time("2026-01-01T00:00:00")

    query_index, row_index = schedule.get_observation_indices_of(
        points, nside=nside, order="nested", start_time=t_start, end_time=t_end
    )

    found = np.bincount(query_index, minlength=len(points))
    exact = np.array([len(schedule.get_observations_of(point, t_start, t_end)) for point in points])
    np.testing.assert_array_equal(found, exact)
    assert 0 < np.count_nonzero(exact) < len(points)  # points fall on both sides of the edge


def test_get_observation_indices_of_applies_a_separate_window_per_query(make_schedule, hot_spot):
    """Per-query windows select the same observations as one exact query per position."""
    schedule = make_schedule(n_sched=30)
    rng = np.random.default_rng(3)
    n = 40
    points = hot_spot.directional_offset_by(rng.uniform(0, 360, n) * u.deg, rng.uniform(0, 0.9, n) * u.deg)
    starts = Time("2025-01-01T00:00:00") + rng.uniform(0, 100, n) * u.day
    ends = starts + rng.uniform(1, 60, n) * u.day

    query_index, row_index = schedule.get_observation_indices_of(
        points, nside=128, order="nested", start_time=starts, end_time=ends
    )

    jd = schedule.observe_rows["start_time"].jd
    for i in range(n):
        exact = schedule.get_observations_of(points[i], starts[i], ends[i])
        np.testing.assert_allclose(np.sort(jd[row_index[query_index == i]]), np.sort(exact["start_time"].jd))


def test_get_observation_indices_of_without_a_window_returns_every_covering_row(make_schedule, hot_spot):
    schedule = make_schedule(n_sched=7)
    query_index, row_index = schedule.get_observation_indices_of(hot_spot, nside=128, order="nested")
    assert len(row_index) == 7
    assert list(row_index) == sorted(row_index)  # chronological, and no duplicates from neighbouring pixels


def test_get_healpix_coverage_index_caches_index_per_resolution(make_schedule):
    schedule = make_schedule()

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


def test_get_observed_healpix_ids_reuses_coverage_index_cache(make_schedule):
    """
    `get_observed_healpix_ids` must derive its answer from `get_healpix_coverage_index`'s
    cached CSR structure rather than re-rasterizing footprints itself, so the same
    `(nside, order)` cache entry -- not a fresh rebuild -- backs every call.
    """
    schedule = make_schedule()
    nside, order = 128, "nested"

    schedule.get_observed_healpix_ids(Time("2025-01-01"), Time("2025-06-01"), nside=nside, order=order)
    assert (nside, order) in schedule._HPX_MAP_CACHE
    cached_offsets, cached_rows = schedule._HPX_MAP_CACHE[(nside, order)]

    # A second call, at the same resolution but a different window, must reuse the
    # exact same cached arrays rather than rebuilding the index.
    schedule.get_observed_healpix_ids(Time("2025-02-01"), Time("2025-03-01"), nside=nside, order=order)
    offsets_again, rows_again = schedule._HPX_MAP_CACHE[(nside, order)]
    assert offsets_again is cached_offsets
    assert rows_again is cached_rows


def test_get_observed_healpix_ids_matches_direct_rasterization(make_schedule_from_pointings, hot_spot):
    """
    Cross-check the cached-index-based implementation against directly rasterizing
    only the rows that actually fall in the query window -- the brute-force
    definition of "pixels covered by observations in this interval".
    """
    from m4opt.fov import footprint_healpix

    offsets_days = [0, 10, 20, 45, 90]
    schedule = _single_pixel_schedule(make_schedule_from_pointings, hot_spot, offsets_days)
    t_start = Time("2025-01-01T00:00:00") + 8 * u.day
    t_end = Time("2025-01-01T00:00:00") + 46 * u.day

    fast = schedule.get_observed_healpix_ids(t_start, t_end, nside=128, order="nested")

    subtable = schedule.get_rows_between_times(t_start, t_end)
    subtable = subtable[subtable["action"] == "observe"]
    hpx = ah.HEALPix(nside=128, order="nested", frame=subtable["target_coord"].frame)
    pixel_arrays = footprint_healpix(hpx, schedule._instrument_fov, subtable["target_coord"], subtable["roll"])
    expected = np.unique(np.concatenate(pixel_arrays)) if len(pixel_arrays) else np.array([], dtype=np.int64)

    np.testing.assert_array_equal(fast, expected)
    assert len(fast) > 0  # a meaningless pass if the window caught nothing


def test_get_observed_healpix_ids_empty_outside_any_visit_window(make_schedule_from_pointings, hot_spot):
    schedule = _single_pixel_schedule(make_schedule_from_pointings, hot_spot, [50])

    pixel_ids = schedule.get_observed_healpix_ids(Time("2025-01-01"), Time("2025-01-10"), nside=128, order="nested")
    assert len(pixel_ids) == 0


def test_get_observation_indices_of_empty_for_unobserved_pixel(make_schedule):
    schedule = make_schedule()
    far_away = SkyCoord(ra=10 * u.deg, dec=-60 * u.deg)

    query_index, row_index = schedule.get_observation_indices_of(far_away, nside=128, order="nested")
    assert len(query_index) == 0
    assert len(row_index) == 0


def test_get_observation_indices_of_requires_both_time_bounds(make_schedule, hot_spot):
    schedule = make_schedule()

    with pytest.raises(ValueError, match="must be given together"):
        schedule.get_observation_indices_of(hot_spot, nside=128, order="nested", start_time=Time("2025-01-01"))


def _single_pixel_schedule(make_schedule_from_pointings, hot_spot, offsets_days):
    """A schedule of visits to one fixed sky position, at explicit elapsed-day offsets."""
    times = Time("2025-01-01T00:00:00") + np.asarray(offsets_days) * u.day
    coords = SkyCoord(
        ra=np.full(len(offsets_days), hot_spot.ra.value) * u.deg,
        dec=np.full(len(offsets_days), hot_spot.dec.value) * u.deg,
    )
    return make_schedule_from_pointings(coords, times)


def test_compute_cadence_time_differences_consecutive_matches_diff(make_schedule_from_pointings, hot_spot):
    """`pairs='consecutive'` reduces to the successive-visit gaps `t_{i+1} - t_i`."""
    schedule = _single_pixel_schedule(make_schedule_from_pointings, hot_spot, [0, 1, 3, 6])

    differences, offsets = schedule.compute_cadence_time_differences(pairs="consecutive")
    pixel = ah.HEALPix(nside=128, order="nested", frame=hot_spot.frame).skycoord_to_healpix(hot_spot)

    values = np.sort(differences[offsets[pixel] : offsets[pixel + 1]].to_value(u.day))
    np.testing.assert_allclose(values, [1, 2, 3])


def test_compute_cadence_time_differences_all_includes_every_unique_pair(make_schedule_from_pointings, hot_spot):
    """`pairs='all'` gives every unique pairwise separation, not just consecutive ones."""
    schedule = _single_pixel_schedule(make_schedule_from_pointings, hot_spot, [0, 1, 3, 6])

    differences, offsets = schedule.compute_cadence_time_differences(pairs="all")
    pixel = ah.HEALPix(nside=128, order="nested", frame=hot_spot.frame).skycoord_to_healpix(hot_spot)

    values = np.sort(differences[offsets[pixel] : offsets[pixel + 1]].to_value(u.day))
    np.testing.assert_allclose(values, [1, 2, 3, 3, 5, 6])


def test_compute_cadence_time_differences_rejects_bad_pairs_mode(make_schedule):
    schedule = make_schedule()

    with pytest.raises(ValueError, match="'pairs'"):
        schedule.compute_cadence_time_differences(pairs="bogus")


def test_compute_max_gap_is_worst_case_successive_gap(make_schedule_from_pointings, hot_spot):
    schedule = _single_pixel_schedule(make_schedule_from_pointings, hot_spot, [0, 1, 3, 6])

    max_gap = schedule.compute_max_gap()
    pixel = ah.HEALPix(nside=128, order="nested", frame=hot_spot.frame).skycoord_to_healpix(hot_spot)

    assert max_gap[pixel].to_value(u.day) == pytest.approx(3.0)


def test_compute_max_gap_nan_for_pixels_observed_fewer_than_twice(make_schedule_from_pointings, hot_spot):
    schedule = _single_pixel_schedule(make_schedule_from_pointings, hot_spot, [0])

    max_gap = schedule.compute_max_gap()
    pixel = ah.HEALPix(nside=128, order="nested", frame=hot_spot.frame).skycoord_to_healpix(hot_spot)

    assert np.isnan(max_gap[pixel].to_value(u.day))


def test_compute_pair_counts_consecutive_vs_all_differ(make_schedule_from_pointings, hot_spot):
    """
    Visits at [0, 1, 3, 6] days give consecutive gaps {1, 2, 3} and all-pair
    separations {1, 2, 3, 3, 5, 6}. A window of [2.5, 3.5] days brackets both
    "3"s among all pairs ((0, 2) and (2, 3)) but only one of them is
    consecutive ((2, 3)), so the two modes must disagree here.
    """
    schedule = _single_pixel_schedule(make_schedule_from_pointings, hot_spot, [0, 1, 3, 6])
    pixel = ah.HEALPix(nside=128, order="nested", frame=hot_spot.frame).skycoord_to_healpix(hot_spot)

    timescale = 3 * u.day
    minimum_factor = 2.5 / 3
    maximum_factor = 3.5 / 3

    all_counts, _ = schedule.compute_pair_counts(
        timescale, minimum_factor=minimum_factor, maximum_factor=maximum_factor, pairs="all"
    )
    consecutive_counts, _ = schedule.compute_pair_counts(
        timescale, minimum_factor=minimum_factor, maximum_factor=maximum_factor, pairs="consecutive"
    )

    assert all_counts[pixel] == 2
    assert consecutive_counts[pixel] == 1


def test_compute_pair_counts_rejects_bad_pairs_mode(make_schedule):
    schedule = make_schedule()

    with pytest.raises(ValueError, match="'pairs'"):
        schedule.compute_pair_counts(1 * u.day, pairs="bogus")


def test_compute_pair_count_curve_consecutive_vs_all_differ(make_schedule_from_pointings, hot_spot):
    schedule = _single_pixel_schedule(make_schedule_from_pointings, hot_spot, [0, 1, 3, 6])

    timescale = 3 * u.day
    minimum_factor = 2.5 / 3
    maximum_factor = 3.5 / 3

    all_area = schedule.compute_pair_count_curve(
        [timescale], minimum_factor=minimum_factor, maximum_factor=maximum_factor, pairs="all"
    )
    consecutive_area = schedule.compute_pair_count_curve(
        [timescale], minimum_factor=minimum_factor, maximum_factor=maximum_factor, pairs="consecutive"
    )

    # Both modes find the one pixel sensitive at this timescale/window (see
    # `test_compute_pair_counts_consecutive_vs_all_differ`), so the sensitive
    # *area* -- an existence-only statistic -- agrees even though the
    # underlying pair counts do not.
    assert all_area == consecutive_area
    assert all_area.to_value(u.sr) > 0


# --------------------------------------------------------------------------- #
# next_action_time                                                            #
# --------------------------------------------------------------------------- #
def _add_downlinks(schedule, downlink_starts: Time, duration: u.Quantity):
    """Append one ``"downlink"`` row per `downlink_starts` entry to `schedule`."""
    table = schedule.table.copy()

    rows = QTable()
    rows["start_time"] = downlink_starts
    rows["duration"] = u.Quantity(np.full(len(downlink_starts), duration.to_value(u.s)), u.s)
    rows["observer_location"] = uvex.observer_location(downlink_starts)
    rows["action"] = np.full(len(downlink_starts), "downlink")
    rows["target_coord"] = table["target_coord"][: len(downlink_starts)]
    rows["roll"] = table["roll"][: len(downlink_starts)]
    rows["field_id"] = np.full(len(downlink_starts), -1, dtype=table["field_id"].dtype)
    rows["block_id"] = table["block_id"][: len(downlink_starts)]

    return SurveySchedule(vstack([table, rows], metadata_conflicts="silent"), schedule.fov)


def test_next_action_time_picks_earliest_at_or_after_query(make_schedule):
    """The earliest downlink whose start is >= each query time is returned, exactly at the boundary too."""
    base = make_schedule(n_sched=5)
    downlink_starts = base.start_time + np.array([1, 5, 10]) * u.hour
    schedule = _add_downlinks(base, downlink_starts, duration=10 * u.min)

    query = base.start_time + np.array([0, 1, 6, 20]) * u.hour  # before all, exactly on one, between two, after all
    result = schedule.next_action_time(query, "downlink")

    assert list(result.mask) == [False, False, False, True]
    assert abs((result[0] - downlink_starts[0]).sec) < 1e-3
    assert abs((result[1] - downlink_starts[0]).sec) < 1e-3  # exact match at the boundary counts
    assert abs((result[2] - downlink_starts[2]).sec) < 1e-3


def test_next_action_time_completion_adds_duration(make_schedule):
    base = make_schedule(n_sched=5)
    downlink_starts = base.start_time + np.array([1]) * u.hour
    duration = 15 * u.min
    schedule = _add_downlinks(base, downlink_starts, duration=duration)

    query = base.start_time
    start = schedule.next_action_time(query, "downlink", completion=False)
    completion = schedule.next_action_time(query, "downlink", completion=True)

    assert abs((completion - start - duration).sec) < 1e-3


def test_next_action_time_fully_masked_when_action_never_scheduled(make_schedule):
    """`make_schedule` never schedules a `"downlink"`, so every query comes back masked."""
    schedule = make_schedule(n_sched=5)
    query = schedule.start_time + np.array([0, 1, 2]) * u.hour

    result = schedule.next_action_time(query, "downlink")

    assert np.all(result.mask)


def test_next_action_time_accepts_scalar_time(make_schedule):
    base = make_schedule(n_sched=5)
    downlink_starts = base.start_time + np.array([1]) * u.hour
    schedule = _add_downlinks(base, downlink_starts, duration=10 * u.min)

    result = schedule.next_action_time(base.start_time, "downlink")

    assert result.isscalar
    assert not result.mask
    assert abs((result - downlink_starts[0]).sec) < 1e-3


def test_next_action_time_rejects_unknown_action(make_schedule):
    schedule = make_schedule(n_sched=5)

    with pytest.raises(ValueError, match="Unknown action"):
        schedule.next_action_time(schedule.start_time, "bogus")


# --------------------------------------------------------------------------- #
# first_visit_mask                                                            #
# --------------------------------------------------------------------------- #
def test_first_visit_mask_all_true_when_fields_never_repeat(make_schedule):
    """`make_schedule` gives every row its own `field_id`, so every row is its field's first visit."""
    schedule = make_schedule(n_sched=5)

    assert np.all(schedule.first_visit_mask)


def test_first_visit_mask_flags_only_each_fields_earliest_row(make_schedule):
    """A revisited field is flagged only on its earliest (not any later) chronological row."""
    base = make_schedule(n_sched=6)
    table = base.table.copy()
    table["field_id"] = [0, 1, 0, 1, 2, 0]  # `make_schedule`'s rows are already time-sorted
    schedule = SurveySchedule(table, base.fov)

    assert list(schedule.first_visit_mask) == [True, True, False, False, True, False]


def test_first_visit_mask_empty_when_no_observe_rows(make_schedule):
    """A schedule with no `"observe"` rows has an empty mask, not an error."""
    base = make_schedule(n_sched=3)
    table = base.table.copy()
    table["action"] = np.full(len(table), "slew")
    schedule = SurveySchedule(table, base.fov)

    assert schedule.first_visit_mask.shape == (0,)
