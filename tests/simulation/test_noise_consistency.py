"""
End-to-end checks that the detection cuts and the synthetic photometry share one noise realization.

Both draw each measurement's noise from `keyed_standard_normal`, keyed on the event's
``parameter_seed``, the observation's start time, and the band. These tests verify that a cut
and `Event.simulate_photometry` therefore see the same measured flux for the same measurement,
and that neither depends on which other observations, bands, or events are evaluated with it.
"""

from collections import Counter

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import vstack
from astropy.time import Time
from m4opt.missions._uvex import uvex

from uvex_transients.models._utils import simulate_flat_photometry
from uvex_transients.simulation.core import SurveySimulator, _sample_parameters_from_seeds
from uvex_transients.simulation.event import Event
from uvex_transients.transients import LFBOTs, TDEs, kilonovae, supernovae  # noqa: F401  (register every transient type)
from uvex_transients.transients.base import TransientBase
from uvex_transients.transients.TDEs import TidalDisruptionEvent
from uvex_transients.utils.keyed_noise import keyed_standard_normal, time_key

from .test_core import _make_catalog

BAND_NAMES = list(uvex.detector.bandpasses)


@pytest.fixture
def setup(make_schedule, hot_spot):
    """A TDE catalog, its schedule, and a simulator over them."""
    transient = TidalDisruptionEvent()
    schedule = make_schedule(n_sched=40)
    catalog, *_ = _make_catalog(transient, hot_spot)
    sim = SurveySimulator(schedule, transients={"tde": transient}, simulation_seed=7)
    return transient, schedule, catalog, sim


def _epochs(sim, catalog, mission=uvex, **kwargs):
    chunks = list(sim.iter_epochs(catalog, mission, progress=False, **kwargs))
    return vstack(chunks)


# --------------------------------------------------------------------------- #
# Cuts and photometry share one state                                         #
# --------------------------------------------------------------------------- #
def test_cut_iterator_and_event_photometry_see_the_same_measured_flux(setup):
    """For every epoch the cuts judge, `Event.simulate_photometry` reports the identical flux and error."""
    transient, schedule, catalog, sim = setup
    epochs = _epochs(sim, catalog)

    event_ids = np.unique(np.asarray(epochs["event_id"]))
    events = dict(zip(event_ids, catalog.get_events(event_ids, {"tde": transient}, schedule)))

    n_matched = 0
    for event_id, event in events.items():
        phot = event.simulate_photometry(uvex)
        phot = phot[phot["in_model"]]
        rows = epochs[np.asarray(epochs["event_id"]) == event_id]
        for row in rows:
            dt = np.abs((phot["obs_time"] - row["t_obs"]).to_value(u.s))
            match = phot[(dt < 1e-3) & (phot["band"] == row["band"])]
            if len(match) == 0:
                continue  # an exposure the event treats as outside its SED's valid window
            assert len(match) == 1
            n_matched += 1

            sigma = row["flux_err"].to_value(u.Jy)
            assert match["flux_err"][0].to_value(u.Jy) == pytest.approx(sigma, rel=1e-6)
            # The noise draw is identical, so any difference is float roundoff in the true flux.
            assert abs(match["flux"][0].to_value(u.Jy) - row["flux"].to_value(u.Jy)) < 1e-6 * sigma
            assert match["snr"][0] == pytest.approx(row["snr"], abs=1e-5)

    assert n_matched > 20  # guard against a vacuous all-skipped comparison


@pytest.mark.parametrize("threshold", [30.0, 60.0])
def test_detected_epochs_agree_between_cuts_and_photometry(setup, threshold):
    """Per event, the cuts and the photometry table count exactly the same detected epochs."""
    transient, schedule, catalog, sim = setup

    epochs = _epochs(sim, catalog)
    detected = (np.asarray(epochs["snr"]) > threshold) & ~np.asarray(epochs["pre_explosion"])
    from_cuts = Counter(np.asarray(epochs["event_id"])[detected].tolist())

    # Photometry has one row per band: an epoch is detected if any band is.
    phot = catalog.simulate_photometry(uvex, {"tde": transient}, schedule)
    phot = phot[np.asarray(phot["in_model"])]
    epoch_detected = {}
    for event_id, jd, value in zip(phot["event_id"].tolist(), phot["obs_time"].jd.tolist(), np.asarray(phot["snr"])):
        epoch_detected[(event_id, jd)] = epoch_detected.get((event_id, jd), False) or bool(value > threshold)
    from_photometry = Counter(event_id for (event_id, _), hit in epoch_detected.items() if hit)

    assert sum(from_cuts.values()) > 100  # guard against a vacuous comparison
    assert from_cuts == from_photometry


def test_cuts_and_event_regenerate_the_same_sed_parameters(setup):
    """The cuts' per-seed parameter draw is the one the reconstructed `Event` uses."""
    transient, schedule, catalog, _ = setup
    events = catalog.get_events(np.asarray(catalog.table["event_id"]), {"tde": transient}, schedule)
    from_seeds = _sample_parameters_from_seeds(transient.sed, np.asarray(catalog.table["parameter_seed"]))

    assert all(event.seed == seed for event, seed in zip(events, np.asarray(catalog.table["parameter_seed"])))
    for i, event in enumerate(events):
        for name, value in event.sample_parameters().items():
            assert u.allclose(u.Quantity(value), u.Quantity(from_seeds[name][i]), rtol=1e-12)


# --------------------------------------------------------------------------- #
# Independence from what else is evaluated                                    #
# --------------------------------------------------------------------------- #
def test_iterator_noise_is_independent_of_chunking_and_event_subset(setup):
    """An event's realized epochs are identical whether it is evaluated alone or in a big chunk."""
    _, _, catalog, sim = setup
    everything = _epochs(sim, catalog, chunk_size=1000)

    event_ids = np.unique(np.asarray(everything["event_id"]))
    lone_id = event_ids[len(event_ids) // 2]
    mask = np.asarray(catalog.table["event_id"]) == lone_id
    alone = _epochs(sim, catalog, chunk_size=1, mask=mask)

    expected = everything[np.asarray(everything["event_id"]) == lone_id]
    assert len(alone) == len(expected) > 0
    np.testing.assert_array_equal(alone["snr"], expected["snr"])
    np.testing.assert_array_equal(alone["flux"], expected["flux"])


@pytest.fixture
def event_pair(make_schedule_from_pointings):
    """Two `Event`s of one source that differ only in how far before explosion they look."""
    coord = SkyCoord(ra=150 * u.deg, dec=20 * u.deg)
    t_explosion = Time("2025-01-01T00:00:00", scale="utc")
    n_obs = 12
    times = t_explosion + np.sort(np.random.default_rng(0).uniform(-40, 150, n_obs)) * u.day
    pointings = SkyCoord(np.full(n_obs, 150.0) * u.deg, np.full(n_obs, 20.0) * u.deg)
    schedule = make_schedule_from_pointings(pointings, times)

    def make(pre_window):
        return Event(
            event_id=1,
            schedule=schedule,
            transient=TidalDisruptionEvent(),
            coord=coord,
            redshift=0.05,
            t_explosion=t_explosion,
            seed=42,
            ebv=0.1,
            photometry_pre_window=pre_window,
        )

    return make(0 * u.day), make(60 * u.day)


def test_event_noise_is_independent_of_the_observation_window(event_pair):
    """Widening the pre-explosion window adds rows but leaves every shared row's flux untouched."""
    narrow, wide = event_pair
    phot_narrow = narrow.simulate_photometry(uvex)
    phot_wide = wide.simulate_photometry(uvex)

    assert len(phot_wide) > len(phot_narrow)
    assert (~phot_wide["in_model"]).any()  # the wide window really did add background-only rows

    for row in phot_narrow:
        dt = np.abs((phot_wide["obs_time"] - row["obs_time"]).to_value(u.s))
        match = phot_wide[(dt < 1e-3) & (phot_wide["band"] == row["band"])]
        assert len(match) == 1
        assert match["flux"][0].to_value(u.Jy) == row["flux"].to_value(u.Jy)


def test_event_noise_is_independent_of_which_bands_are_requested(event_pair):
    """Simulating one band gives exactly that band's rows from the all-band call."""
    _, wide = event_pair
    full = wide.simulate_photometry(uvex)
    single = wide.simulate_photometry(uvex, bands=[BAND_NAMES[-1]])

    expected = full[full["band"] == BAND_NAMES[-1]]
    assert len(single) == len(expected) > 0
    np.testing.assert_array_equal(single["flux"], expected["flux"])


def test_event_photometry_is_reproducible(event_pair):
    """Repeated calls give identical fluxes."""
    _, wide = event_pair
    first = wide.simulate_photometry(uvex)
    second = wide.simulate_photometry(uvex)
    np.testing.assert_array_equal(first["flux"], second["flux"])


# --------------------------------------------------------------------------- #
# What the cuts threshold                                                     #
# --------------------------------------------------------------------------- #
def test_filter_by_snr_thresholds_the_measured_snr(setup):
    """`filter_by_snr` keeps exactly the events with an epoch whose measured SNR clears the threshold."""
    _, _, catalog, sim = setup

    threshold = 3.0
    epochs = _epochs(sim, catalog)
    detected = (np.asarray(epochs["snr"]) > threshold) & ~np.asarray(epochs["pre_explosion"])
    expected = set(np.asarray(epochs["event_id"])[detected].tolist())

    kept = sim.filter_by_snr(catalog, uvex, snr_threshold=threshold, exclude_first_visit_detections=False)
    assert set(np.asarray(kept.table["event_id"])) == expected
    assert len(expected) > 0


def test_iterator_include_snr_selects_columns(setup):
    """`include_snr` adds or removes exactly the measurement columns."""
    _, _, catalog, sim = setup

    measured = _epochs(sim, catalog)
    assert {"snr", "band", "flux", "flux_err"} <= set(measured.colnames)

    geometry = _epochs(sim, catalog, mission=None, include_snr=False)
    assert not {"snr", "band", "flux", "flux_err"} & set(geometry.colnames)


# --------------------------------------------------------------------------- #
# Keyed noise in the photometry primitives                                    #
# --------------------------------------------------------------------------- #
def test_keyed_photometry_requires_matching_keys():
    """`noise_seed` without a correctly shaped `noise_observation_keys` raises."""
    coord = SkyCoord(ra=150 * u.deg, dec=20 * u.deg)
    t = np.arange(3) * u.day
    with pytest.raises(ValueError, match="noise_observation_keys"):
        simulate_flat_photometry(t, 900 * u.s, uvex.detector, coord, noise_seed=1)
    with pytest.raises(ValueError, match="noise_observation_keys"):
        simulate_flat_photometry(
            t, 900 * u.s, uvex.detector, coord, noise_seed=1, noise_observation_keys=np.arange(2, dtype=np.uint64)
        )


def test_keyed_flat_photometry_depends_only_on_its_key():
    """Dropping other observations from a call does not change the survivors' noise."""
    coord = SkyCoord(ra=150 * u.deg, dec=20 * u.deg)
    t = np.arange(6) * u.day
    keys = np.arange(6, dtype=np.uint64) + 1000
    full = simulate_flat_photometry(t, 900 * u.s, uvex.detector, coord, noise_seed=5, noise_observation_keys=keys)

    pick = [4, 1]
    subset = simulate_flat_photometry(
        t[pick], 900 * u.s, uvex.detector, coord, noise_seed=5, noise_observation_keys=keys[pick]
    )
    for i in pick:
        for band in BAND_NAMES:
            a = full[(full["t"] == t[i]) & (full["band"] == band)]["flux"][0]
            b = subset[(subset["t"] == t[i]) & (subset["band"] == band)]["flux"][0]
            assert a == b


# --------------------------------------------------------------------------- #
# Every transient type                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("transient_name", sorted(TransientBase.registry()))
def test_batched_measurement_equals_per_event_measurement_for_every_transient(transient_name):
    """
    For every registered transient type, one call over several events equals one call per event.

    The survey iterator measures many events at once, each with its own position, redshift,
    distance, dust column, SED parameters and noise seed, so every SED class has to accept all
    of those as per-epoch arrays and give each epoch the same answer it would get alone.
    """
    sed = TransientBase.registry()[transient_name]().sed
    n_events = 3
    params = sed.sample_parameters(n_events, rng=0)
    coords = SkyCoord(ra=[150, 30, 250] * u.deg, dec=[20, 5, -40] * u.deg)
    redshift = np.array([0.03, 0.06, 0.09])
    distance = np.array([130, 280, 410]) * u.Mpc
    ebv = np.array([0.02, 0.1, 0.3])
    seeds = np.array([5, 6, 7], dtype=np.uint64)

    # Some epochs are before the explosion, so every event mixes modelled and background-only epochs.
    event = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    t = np.array([1, -2, 4, 8, -16, 32, 64, 128]) * u.day
    obstime = Time("2025-01-01T00:00:00", scale="utc") + t
    location = uvex.observer_location(obstime)
    keys = time_key(obstime)

    batched = sed.measure_photometry(
        t,
        900 * u.s,
        uvex.detector,
        coords[event],
        observer_location=location,
        obstime=obstime,
        redshift=redshift[event],
        luminosity_distance=distance[event],
        ebv=ebv[event],
        noise_seed=seeds[event],
        noise_observation_keys=keys,
        **{name: value[event] for name, value in params.items()},
    )

    for i in range(n_events):
        rows = np.flatnonzero(event == i)
        alone = sed.measure_photometry(
            t[rows],
            900 * u.s,
            uvex.detector,
            coords[i],
            observer_location=location[rows],
            obstime=obstime[rows],
            redshift=redshift[i],
            luminosity_distance=distance[i],
            ebv=ebv[i],
            noise_seed=int(seeds[i]),
            noise_observation_keys=keys[rows],
            **{name: value[i] for name, value in params.items()},
        )
        for batched_values, alone_values in zip(batched, alone):
            np.testing.assert_allclose(batched_values[:, rows], alone_values, rtol=1e-9)
