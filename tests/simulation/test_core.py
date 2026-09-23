"""Tests for `SurveySimulator.filter_by_snr`."""

import astropy_healpix as ah
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable
from astropy.time import Time
from m4opt.missions._uvex import uvex
from m4opt.synphot import observing

from uvex_transients.simulation.core import (
    SurveySimulator,
    _sample_parameters_from_seeds,
)
from uvex_transients.simulation.event_catalog import EventCatalog
from uvex_transients.transients.TDEs import TidalDisruptionEvent

CATALOG_NSIDE = 128
CATALOG_ORDER = "nested"


def _make_catalog(transient, hot_spot, n_events=30, seed=1):
    rng = np.random.default_rng(seed)
    t_explosion = Time("2025-01-01T00:00:00") + rng.uniform(0, 100, n_events) * u.day
    redshift = rng.uniform(0.01, 0.08, n_events)
    in_footprint = rng.random(n_events) < 0.7
    coord = SkyCoord(
        np.where(in_footprint, hot_spot.ra.value, rng.uniform(0, 360, n_events)) * u.deg,
        np.where(in_footprint, hot_spot.dec.value, rng.uniform(-89, 89, n_events)) * u.deg,
    )
    # `FLRW.luminosity_distance` chokes on a size-0 array (its `np.vectorize`
    # internals require `otypes` to be set for empty input) -- side-step it
    # entirely for the n_events=0 case rather than triggering that unrelated bug.
    luminosity_distance = transient.cosmology.luminosity_distance(redshift) if n_events > 0 else u.Quantity([], u.Mpc)

    # `filter_by_snr` now uses this column directly (see its own docstring on why
    # that's exact, not approximate) -- so, unlike before, it must actually match
    # `coord` at (CATALOG_NSIDE, CATALOG_ORDER), not just be arbitrary placeholder
    # values.
    hpx = ah.HEALPix(nside=CATALOG_NSIDE, order=CATALOG_ORDER, frame=coord.frame)
    healpix_id = hpx.skycoord_to_healpix(coord) if n_events > 0 else np.array([], dtype=np.int64)

    table = QTable()
    table["healpix_id"] = healpix_id
    table["healpix_dx"] = np.full(n_events, 0.5)
    table["healpix_dy"] = np.full(n_events, 0.5)
    table["coord"] = coord
    table["redshift"] = redshift
    table["t_explosion"] = t_explosion
    table["parameter_seed"] = np.arange(n_events, dtype=np.uint64) + 100
    # `np.full` (not a bare scalar assignment on a fixed-width column) so the
    # dtype is sized from this actual fill value -- see
    # `test_filter_by_snr_unknown_transient_type_raises`, which later needs to
    # replace it with a longer string without silent truncation.
    table["transient_type"] = np.full(n_events, "tde")
    table["time_bin"] = 0
    table["event_id"] = np.arange(n_events, dtype=np.int64)
    table["luminosity_distance"] = luminosity_distance
    table["ebv"] = np.full(n_events, 0.05)

    return (
        EventCatalog(
            table=table,
            nside=CATALOG_NSIDE,
            order=CATALOG_ORDER,
            time_bins=Time(["2025-01-01", "2025-06-01"]),
        ),
        coord,
        t_explosion,
        redshift,
    )


def test_filter_by_snr_matches_independent_unbatched_computation(make_schedule, hot_spot):
    """
    `filter_by_snr`'s flatten/batch/reduce result matches an independent, unbatched
    computation using the same primitives it uses internally
    (`get_observation_indices_of`, queried at `catalog`'s own `nside`/`order` --
    *not* the exact `get_observations_of`; see `filter_by_snr`'s own docstring for
    why these are nonetheless exactly equivalent at that shared resolution --
    `as_source_spectrum`, `get_snr`) and the same per-event `parameter_seed` draw
    `filter_by_snr` itself makes internally, event by event (see
    `_sample_parameters_from_seeds`).
    """
    transient = TidalDisruptionEvent()
    schedule = make_schedule(n_sched=40)
    catalog, coord, t_explosion, redshift = _make_catalog(transient, hot_spot)
    n_events = len(catalog)

    sim = SurveySimulator(schedule, transients={"tde": transient}, simulation_seed=7)

    snr_threshold = 5.0
    filtered = sim.filter_by_snr(
        catalog,
        uvex,
        snr_threshold=snr_threshold,
        chunk_size=7,
    )
    kept_ids = set(np.asarray(filtered.table["event_id"]))

    # Ground truth: each event's own `parameter_seed` (exactly matching
    # `_sample_parameters_from_seeds`, which `filter_by_snr` uses internally), then
    # per event, per observation, per band -- all unbatched.
    sed_params_all = _sample_parameters_from_seeds(transient.sed, np.asarray(catalog.table["parameter_seed"]))

    expected_kept = set()
    for i in range(n_events):
        _, row_index_i = schedule.get_observation_indices_of(
            coord[i],
            nside=catalog.nside,
            order=catalog.order,
            start_time=t_explosion[i],
            end_time=t_explosion[i] + transient.duration_limit,
        )
        obs_i = schedule.observe_rows[row_index_i]
        obs_i = obs_i[np.argsort(obs_i["start_time"])]
        if len(obs_i) == 0:
            continue

        sed_params_i = {name: value[i] for name, value in sed_params_all.items()}
        best_snr = None
        for j in range(len(obs_i)):
            t_obs = (obs_i["start_time"][j] - t_explosion[i]).to(u.day)
            spectrum = transient.sed.as_source_spectrum(
                t_obs,
                redshift=redshift[i],
                luminosity_distance=catalog.table["luminosity_distance"][i],
                ebv=0.05,
                **sed_params_i,
            )
            with observing(obs_i["observer_location"][j], coord[i], obs_i["start_time"][j]):
                snr = max(
                    uvex.detector.get_snr(obs_i["duration"][j], spectrum, band) for band in uvex.detector.bandpasses
                )
            best_snr = snr if best_snr is None else max(best_snr, snr)

        if best_snr is not None and best_snr > snr_threshold:
            expected_kept.add(i)

    assert kept_ids == expected_kept


def test_filter_by_snr_discards_never_observed_events(make_schedule):
    """Events far outside the schedule's footprint are dropped without error."""
    transient = TidalDisruptionEvent()
    schedule = make_schedule(n_sched=5)

    coord = SkyCoord([200] * u.deg, [10] * u.deg)  # far from the schedule
    hpx = ah.HEALPix(nside=CATALOG_NSIDE, order=CATALOG_ORDER, frame=coord.frame)

    table = QTable()
    table["healpix_id"] = hpx.skycoord_to_healpix(coord)
    table["healpix_dx"] = [0.5]
    table["healpix_dy"] = [0.5]
    table["coord"] = coord
    table["redshift"] = [0.05]
    table["t_explosion"] = Time("2025-01-01T00:00:00") + [1] * u.day
    table["parameter_seed"] = np.array([1], dtype=np.uint64)
    table["transient_type"] = "tde"
    table["time_bin"] = [0]
    table["event_id"] = np.array([0], dtype=np.int64)
    table["luminosity_distance"] = transient.cosmology.luminosity_distance([0.05])
    table["ebv"] = [0.05]
    catalog = EventCatalog(
        table=table,
        nside=CATALOG_NSIDE,
        order=CATALOG_ORDER,
        time_bins=Time(["2025-01-01", "2025-06-01"]),
    )

    sim = SurveySimulator(schedule, transients={"tde": transient}, simulation_seed=1)
    filtered = sim.filter_by_snr(catalog, uvex, snr_threshold=5.0)
    assert len(filtered) == 0


def test_filter_by_snr_empty_catalog(make_schedule, hot_spot):
    transient = TidalDisruptionEvent()
    schedule = make_schedule(n_sched=5)
    catalog, *_ = _make_catalog(transient, hot_spot, n_events=0)

    sim = SurveySimulator(schedule, transients={"tde": transient}, simulation_seed=1)
    filtered = sim.filter_by_snr(catalog, uvex, snr_threshold=5.0)
    assert len(filtered) == 0


def test_filter_by_snr_and_filter_by_limiting_magnitude_carry_downsample_through(make_schedule, hot_spot):
    """A cut's output catalog keeps the input catalog's `downsample` unchanged, empty or not."""
    transient = TidalDisruptionEvent()
    schedule = make_schedule(n_sched=5)
    sim = SurveySimulator(schedule, transients={"tde": transient}, simulation_seed=1)

    catalog, *_ = _make_catalog(transient, hot_spot, n_events=0)
    catalog.downsample = {"tde": 5}
    assert sim.filter_by_snr(catalog, uvex, snr_threshold=5.0).downsample == {"tde": 5}
    assert sim.filter_by_limiting_magnitude(catalog, uvex, mag_limit=25.0).downsample == {"tde": 5}

    catalog, *_ = _make_catalog(transient, hot_spot, n_events=3, seed=2)
    catalog.downsample = 20
    assert sim.filter_by_snr(catalog, uvex, snr_threshold=5.0).downsample == 20
    assert sim.filter_by_limiting_magnitude(catalog, uvex, mag_limit=25.0).downsample == 20


def test_filter_by_snr_unknown_transient_type_raises(make_schedule, hot_spot):
    transient = TidalDisruptionEvent()
    schedule = make_schedule(n_sched=5)
    catalog, *_ = _make_catalog(transient, hot_spot, n_events=1)
    # Replaces the column outright (rather than a scalar in-place assignment,
    # which would silently truncate to the existing "tde"-sized <U3 dtype).
    catalog.table["transient_type"] = np.full(len(catalog.table), "not-a-real-type")

    sim = SurveySimulator(schedule, transients={"tde": transient}, simulation_seed=1)
    with pytest.raises(ValueError, match="transient type"):
        sim.filter_by_snr(catalog, uvex, snr_threshold=5.0)


def test_filter_by_snr_unknown_band_raises(make_schedule, hot_spot):
    transient = TidalDisruptionEvent()
    schedule = make_schedule(n_sched=5)
    catalog, *_ = _make_catalog(transient, hot_spot, n_events=1)

    sim = SurveySimulator(schedule, transients={"tde": transient}, simulation_seed=1)
    with pytest.raises(ValueError, match="Unknown bandpass"):
        sim.filter_by_snr(catalog, uvex, snr_threshold=5.0, bands=["not-a-real-band"])


# --------------------------------------------------------------------------- #
# generate_events: per-type downsample                                       #
# --------------------------------------------------------------------------- #
def _stub_sample_events_on_healpix_grid(n_events):
    """Build a monkeypatch replacement for `sample_events_on_healpix_grid` returning exactly `n_events` rows.

    Sidesteps the real volumetric-rate sampling (whose event counts, for a tiny test
    schedule, are too small/random to assert precise downsample ratios against) while
    keeping every column `SurveySimulator.generate_events` actually reads.
    """

    def _stub(
        self,
        nside,
        *,
        t_start,
        t_end=None,
        duration=None,
        pixel_mask=None,
        pixel_ids=None,
        order="nested",
        jitter=True,
        seed=None,
    ):
        rng = np.random.default_rng(seed)
        table = QTable()
        table["healpix_id"] = np.zeros(n_events, dtype=np.int64)
        table["healpix_dx"] = np.full(n_events, 0.5)
        table["healpix_dy"] = np.full(n_events, 0.5)
        table["coord"] = SkyCoord(np.full(n_events, 150.0) * u.deg, np.full(n_events, 20.0) * u.deg)
        table["redshift"] = np.full(n_events, 0.01)
        table["t_explosion"] = t_start + rng.uniform(0, 1, n_events) * u.day
        table["parameter_seed"] = np.arange(n_events, dtype=np.uint64)
        return table

    return _stub


def test_generate_events_downsample_int_applies_to_every_type(monkeypatch, make_schedule):
    """A single `downsample` int applies uniformly across every registered transient type."""
    monkeypatch.setattr(TidalDisruptionEvent, "sample_events_on_healpix_grid", _stub_sample_events_on_healpix_grid(40))

    schedule = make_schedule(n_sched=20)
    sim = SurveySimulator(
        schedule, transients={"tde_a": TidalDisruptionEvent(), "tde_b": TidalDisruptionEvent()}, simulation_seed=1
    )
    catalog = sim.generate_events(time_bins=1, nside=16, downsample=5)

    counts = np.unique(catalog.table["transient_type"], return_counts=True)
    assert dict(zip(*counts)) == {"tde_a": 8, "tde_b": 8}  # ceil(40 / 5)
    assert catalog.downsample == 5


def test_generate_events_downsample_mapping_applies_per_type(monkeypatch, make_schedule):
    """A `{type key: factor}` `downsample` mapping downsamples only the named type(s)."""
    monkeypatch.setattr(TidalDisruptionEvent, "sample_events_on_healpix_grid", _stub_sample_events_on_healpix_grid(40))

    schedule = make_schedule(n_sched=20)
    sim = SurveySimulator(
        schedule, transients={"tde_a": TidalDisruptionEvent(), "tde_b": TidalDisruptionEvent()}, simulation_seed=1
    )
    catalog = sim.generate_events(time_bins=1, nside=16, downsample={"tde_a": 5})

    counts = np.unique(catalog.table["transient_type"], return_counts=True)
    # `tde_a` is downsampled (ceil(40 / 5) == 8); `tde_b`, left out of the mapping, keeps every event.
    assert dict(zip(*counts)) == {"tde_a": 8, "tde_b": 40}
    assert catalog.downsample == {"tde_a": 5}


def test_generate_events_downsample_mapping_unknown_key_raises(make_schedule):
    """A `downsample` mapping naming a key not in `transients:` raises."""
    schedule = make_schedule(n_sched=5)
    sim = SurveySimulator(schedule, transients={"tde": TidalDisruptionEvent()}, simulation_seed=1)

    with pytest.raises(ValueError, match="unknown transient key"):
        sim.generate_events(time_bins=1, nside=16, downsample={"not-a-real-key": 5})
