"""Tests for `SurveySchedule.get_healpix_coverage_index`/`get_observation_indices_of` and cadence diagnostics."""

import astropy_healpix as ah
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time


def test_get_observation_indices_of_matches_exact_well_inside_fov(make_schedule, hot_spot):
    """
    For a query position solidly inside the FOV (not near a pixel/footprint edge),
    the fast HEALPix-index path and the exact `get_observations_of` agree exactly.
    """
    schedule = make_schedule()
    nside, order = 128, "nested"

    # `hot_spot` itself is exactly the pointing center -- as far from any FOV edge
    # as a position can be, so this comparison isn't sensitive to the known
    # pixel-center-vs-exact-boundary approximation (see `get_healpix_coverage_index`'s
    # own docstring for that tradeoff).
    t_start = Time("2025-01-01T00:00:00")
    t_end = Time("2025-06-01T00:00:00")

    exact = schedule.get_observations_of(hot_spot, t_start, t_end)
    _, row_index = schedule.get_observation_indices_of(
        hot_spot, nside=nside, order=order, start_time=t_start, end_time=t_end
    )
    fast = schedule.observe_rows[row_index]

    assert set(np.asarray(exact["field_id"])) == set(np.asarray(fast["field_id"]))
    assert len(exact) > 0  # a meaningless pass if nothing was scheduled there at all


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
