"""Tests for `Event.simulate_photometry`."""

from functools import partial

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from m4opt.missions._uvex import uvex
from m4opt.synphot import observing

from uvex_transients.dust import log_attenuation
from uvex_transients.simulation.event import Event
from uvex_transients.transients.TDEs import TidalDisruptionEvent


def test_simulate_photometry_batches_observations_and_bands(make_schedule_from_pointings):
    """
    One event, many observations: `simulate_photometry`'s batched result matches
    an independent, unbatched `as_source_spectrum`/`get_snr` computation exactly,
    for every (observation, band) pair -- not just the first or a spot check.
    """
    event_coord = SkyCoord(ra=150 * u.deg, dec=20 * u.deg)
    t_explosion = Time("2025-01-01T00:00:00", scale="utc")

    n_obs = 6
    obs_times = t_explosion + np.sort(np.random.default_rng(0).uniform(1, 150, n_obs)) * u.day
    coords = SkyCoord(
        np.full(n_obs, event_coord.ra.to_value(u.deg)) * u.deg,
        np.full(n_obs, event_coord.dec.to_value(u.deg)) * u.deg,
    )
    schedule = make_schedule_from_pointings(coords, obs_times)

    transient = TidalDisruptionEvent()
    event = Event(
        event_id=1,
        schedule=schedule,
        transient=transient,
        coord=event_coord,
        redshift=0.05,
        t_explosion=t_explosion,
        seed=42,
        ebv=0.1,
    )
    assert event.n_observations == n_obs

    phot = event.simulate_photometry(uvex)
    band_names = list(uvex.detector.bandpasses)
    assert len(phot) == n_obs * len(band_names)
    assert set(phot["band"]) == set(band_names)
    assert np.all(np.isfinite(phot["snr"]))

    # Every (observation, band) cell, cross-checked against an independent,
    # unbatched computation using the same underlying primitives.
    sed_params = event.sample_parameters()
    for obs_row in event.observations:
        t_single = (obs_row["start_time"] - t_explosion).to(u.day)
        spectrum_single = transient.sed.as_source_spectrum(
            t_single,
            redshift=event.redshift,
            luminosity_distance=event.luminosity_distance,
            log_attenuation=partial(log_attenuation, Ebv=event.ebv),
            **sed_params,
        )
        with observing(obs_row["observer_location"], event_coord, obs_row["start_time"]):
            for band in band_names:
                expected_snr = uvex.detector.get_snr(obs_row["duration"], spectrum_single, band)
                row = phot[(phot["band"] == band) & (phot["obs_time"] == obs_row["start_time"])]
                assert len(row) == 1
                np.testing.assert_allclose(row["snr"][0], expected_snr, rtol=1e-6)


def test_simulate_photometry_preserves_full_band_names(make_schedule_from_pointings):
    """
    Regression test: an earlier `np.full(n, band, dtype=str)` silently truncated
    every band name to one character (`dtype=str` alone infers itemsize 1 from
    an empty fill), so this specifically checks the full band names survive.
    """
    event_coord = SkyCoord(ra=10 * u.deg, dec=-5 * u.deg)
    t_explosion = Time("2025-01-01T00:00:00", scale="utc")
    obs_times = t_explosion + [10] * u.day
    coords = SkyCoord(
        np.full(1, event_coord.ra.to_value(u.deg)) * u.deg,
        np.full(1, event_coord.dec.to_value(u.deg)) * u.deg,
    )
    schedule = make_schedule_from_pointings(coords, obs_times)

    event = Event(
        event_id=1,
        schedule=schedule,
        transient=TidalDisruptionEvent(),
        coord=event_coord,
        redshift=0.05,
        t_explosion=t_explosion,
        seed=0,
    )
    phot = event.simulate_photometry(uvex)

    band_names = list(uvex.detector.bandpasses)
    assert max(len(b) for b in band_names) > 1, "test needs a multi-character band name"
    assert set(phot["band"]) == set(band_names)


def test_simulate_photometry_empty_observations(make_schedule_from_pointings):
    """No observations covered this event -> an empty, but correctly typed, table."""
    event_coord = SkyCoord(ra=10 * u.deg, dec=-10 * u.deg)
    t_explosion = Time("2025-01-01T00:00:00", scale="utc")
    obs_times = t_explosion + [1] * u.day
    # Far from `event_coord`, so nothing actually covers it.
    coords = SkyCoord(np.full(1, 200.0) * u.deg, np.full(1, 10.0) * u.deg)
    schedule = make_schedule_from_pointings(coords, obs_times)

    event = Event(
        event_id=2,
        schedule=schedule,
        transient=TidalDisruptionEvent(),
        coord=event_coord,
        redshift=0.05,
        t_explosion=t_explosion,
        seed=1,
    )
    assert event.n_observations == 0

    phot = event.simulate_photometry(uvex)
    assert len(phot) == 0
    assert phot.colnames == [
        "event_id",
        "obs_time",
        "exptime",
        "band",
        "snr",
        "flux",
        "flux_err",
        "flux_upper",
        "flux_lower",
        "ab_mag",
        "mag_err",
        "mag_upper",
        "mag_lower",
    ]


def test_simulate_photometry_unknown_band_raises(make_schedule_from_pointings):
    event_coord = SkyCoord(ra=10 * u.deg, dec=-10 * u.deg)
    t_explosion = Time("2025-01-01T00:00:00", scale="utc")
    obs_times = t_explosion + [1] * u.day
    coords = SkyCoord(np.full(1, 200.0) * u.deg, np.full(1, 10.0) * u.deg)
    schedule = make_schedule_from_pointings(coords, obs_times)

    event = Event(
        event_id=3,
        schedule=schedule,
        transient=TidalDisruptionEvent(),
        coord=event_coord,
        redshift=0.05,
        t_explosion=t_explosion,
        seed=2,
    )
    with pytest.raises(ValueError, match="Unknown bandpass"):
        event.simulate_photometry(uvex, bands=["not-a-real-band"])
