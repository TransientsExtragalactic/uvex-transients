"""
Tests for transients that are measured over the whole exposure (``AVERAGE_OVER_EXPOSURE``).

The survey-wide cuts (`SurveySimulator.iter_epochs`) and the per-event photometry (`Event.simulate_photometry`) must
agree, as they do for every transient, and an exposure that starts before the explosion and ends after it must be a
measurement of the source. The default, which no shipped transient changes, must be untouched.
"""

import astropy_healpix as ah
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable, vstack
from astropy.time import Time
from m4opt.missions._uvex import uvex

from uvex_transients.phase_space.transient import _CanonicalTransient
from uvex_transients.simulation.core import SurveySimulator
from uvex_transients.simulation.event_catalog import EventCatalog
from uvex_transients.transients import LFBOTs, TDEs, kilonovae, supernovae  # noqa: F401  (register every transient type)
from uvex_transients.transients.base import TransientBase

NSIDE = 128
ORDER = "nested"
DWELL = 900.0 * u.s
FIRST = Time("2025-03-01T00:00:00")
REDSHIFT = 0.02

# When each event explodes, in seconds after the start of the first exposure, and what that makes of the exposure
EXPLOSIONS = {
    0: 300.0,  # the exposure started 300 s before the explosion and ends 600 s after it: a straddler
    1: -100.0,  # the exposure starts 100 s after the explosion and runs past the end of the event's 208 s window
    2: 1000.0,  # the exposure ended 100 s before the explosion
    3: -10.0,  # the exposure starts 10 s after the explosion, at the rise of the pulse
}


@pytest.fixture(scope="module")
def sky() -> SkyCoord:
    """Return the position of every event and of every exposure."""
    return SkyCoord(150 * u.deg, 20 * u.deg)


@pytest.fixture
def schedule(make_schedule_from_pointings, sky):
    """Return three exposures of the position, three days apart."""
    times = FIRST + np.array([0.0, 3.0, 6.0]) * u.day
    return make_schedule_from_pointings(
        SkyCoord(np.full(3, sky.ra.deg) * u.deg, np.full(3, sky.dec.deg) * u.deg), times
    )


def _transient(average: bool) -> _CanonicalTransient:
    """Return a bright canonical transient with a 30 s half-decline time, measured over the exposure or not."""
    transient = _CanonicalTransient()
    transient.set_point(1e27 * u.erg / u.s / u.Hz, 30 * u.s, redshift_limit=0.05)
    transient.AVERAGE_OVER_EXPOSURE = average
    return transient


@pytest.fixture
def catalog(sky) -> EventCatalog:
    """Return one event per entry of `EXPLOSIONS`, all at the position of the exposures."""
    n = len(EXPLOSIONS)
    coord = SkyCoord(np.full(n, sky.ra.deg) * u.deg, np.full(n, sky.dec.deg) * u.deg)
    transient = _transient(False)

    table = QTable()
    table["healpix_id"] = ah.HEALPix(nside=NSIDE, order=ORDER, frame=coord.frame).skycoord_to_healpix(coord)
    table["healpix_dx"] = np.full(n, 0.5)
    table["healpix_dy"] = np.full(n, 0.5)
    table["coord"] = coord
    table["redshift"] = np.full(n, REDSHIFT)
    table["t_explosion"] = FIRST + np.array(list(EXPLOSIONS.values())) * u.s
    table["parameter_seed"] = np.arange(n, dtype=np.uint64) + 100
    table["transient_type"] = np.full(n, "canonical")
    table["time_bin"] = 0
    table["event_id"] = np.array(list(EXPLOSIONS), dtype=np.int64)
    table["luminosity_distance"] = transient.cosmology.luminosity_distance(np.full(n, REDSHIFT))
    table["ebv"] = np.full(n, 0.05)
    return EventCatalog(table=table, nside=NSIDE, order=ORDER, time_bins=Time(["2025-01-01", "2025-06-01"]))


def _epochs(schedule, catalog, average, **kwargs) -> QTable:
    """Return every epoch of the catalog, kept whatever its SNR."""
    simulator = SurveySimulator(schedule, {"canonical": _transient(average)}, simulation_seed=7)
    chunks = list(simulator.iter_epochs(catalog, uvex, progress=False, detection_floor=-np.inf, **kwargs))
    return vstack(chunks) if chunks else QTable()


def _rows(epochs: QTable, event_id: int) -> QTable:
    return epochs[np.asarray(epochs["event_id"]) == event_id]


# --------------------------------------------------------------------------- #
# The default                                                                 #
# --------------------------------------------------------------------------- #
def test_default_is_off_for_every_shipped_class():
    assert TransientBase.AVERAGE_OVER_EXPOSURE is False
    for name, cls in TransientBase.registry().items():
        if cls.__module__.startswith("uvex_transients.transients"):
            assert cls.AVERAGE_OVER_EXPOSURE is False, name


def test_straddling_exposure_is_not_an_epoch_by_default(schedule, catalog):
    """Documents the existing behavior: an exposure that started before the explosion is not an epoch."""
    epochs = _epochs(schedule, catalog, average=False)
    assert len(_rows(epochs, 0)) == 0


# --------------------------------------------------------------------------- #
# Averaging                                                                   #
# --------------------------------------------------------------------------- #
def test_straddling_exposure_is_a_measurement_of_the_source(schedule, catalog):
    epochs = _epochs(schedule, catalog, average=True)

    rows = _rows(epochs, 0)
    assert len(rows) == 1
    assert rows["t_since_explosion"][0].to_value(u.s) == pytest.approx(-300.0, abs=1e-3)
    assert not rows["pre_explosion"][0]

    # The pulse is bright, and over within the 600 s the exposure sees of it
    assert rows["snr"][0] > 5.0


def test_exposure_after_the_explosion_is_unchanged_in_kind(schedule, catalog):
    epochs = _epochs(schedule, catalog, average=True)
    for event_id in (1, 3):
        rows = _rows(epochs, event_id)
        assert len(rows) == 1
        assert not rows["pre_explosion"][0]
        assert rows["t_since_explosion"][0] > 0 * u.s


def test_exposure_that_ended_before_the_explosion_is_not_a_measurement(schedule, catalog):
    """Not an epoch by default, and background with a `lookback`, however the transient is measured."""
    assert len(_rows(_epochs(schedule, catalog, average=True), 2)) == 0

    for average in (False, True):
        rows = _rows(_epochs(schedule, catalog, average=average, lookback=1 * u.day), 2)
        assert len(rows) == 1
        assert rows["pre_explosion"][0]


def test_pre_explosion_flag_follows_the_end_of_the_exposure_when_averaging(schedule, catalog):
    """With a `lookback` the straddler is kept, and is a source only if the exposure is averaged over."""
    averaged = _rows(_epochs(schedule, catalog, average=True, lookback=1 * u.day), 0)
    assert len(averaged) == 1
    assert not averaged["pre_explosion"][0]

    # Not averaging, the same exposure is by its start a pre-explosion one, as before
    instantaneous = _rows(_epochs(schedule, catalog, average=False, lookback=1 * u.day), 0)
    assert len(instantaneous) == 1
    assert instantaneous["pre_explosion"][0]


def test_averaging_does_not_change_which_other_epochs_are_found(schedule, catalog):
    """The extra reach only adds exposures that overlap the window, so everything else is the same epochs."""
    averaged = _epochs(schedule, catalog, average=True)
    default = _epochs(schedule, catalog, average=False)
    expected = {(int(row["event_id"]), int(row["observation_index"])) for row in default}
    found = {(int(row["event_id"]), int(row["observation_index"])) for row in averaged}
    assert expected <= found
    assert found - expected == {(0, 0)}


# --------------------------------------------------------------------------- #
# The cuts and the photometry agree                                           #
# --------------------------------------------------------------------------- #
def test_event_photometry_matches_the_epochs_when_averaging(schedule, catalog):
    """`Event.simulate_photometry` measures what `iter_epochs` did, including the straddler and the overrun."""
    transient = _transient(True)
    epochs = _epochs(schedule, catalog, average=True)

    event_ids = np.unique(np.asarray(epochs["event_id"]))
    events = dict(zip(event_ids, catalog.get_events(event_ids, {"canonical": transient}, schedule)))
    assert set(events) == {0, 1, 3}

    n_matched = 0
    for event_id, event in events.items():
        photometry = event.simulate_photometry(uvex)

        # Every observation the event has is a source, however it sits in the window: that is what the cuts did
        assert np.all(photometry["in_model"])
        assert len(event.background_observations) == 0

        for row in _rows(epochs, event_id):
            dt = np.abs((photometry["obs_time"] - row["t_obs"]).to_value(u.s))
            match = photometry[(dt < 1e-3) & (photometry["band"] == row["band"])]
            assert len(match) == 1
            n_matched += 1

            sigma = row["flux_err"].to_value(u.Jy)
            assert match["flux_err"][0].to_value(u.Jy) == pytest.approx(sigma, rel=1e-6)
            assert abs(match["flux"][0].to_value(u.Jy) - row["flux"].to_value(u.Jy)) < 1e-6 * sigma
            assert match["snr"][0] == pytest.approx(row["snr"], abs=1e-5)
    assert n_matched == 3


def test_event_without_averaging_treats_the_overrun_as_background(schedule, catalog):
    """The default is untouched: an exposure running past the window is background, as it was."""
    transient = _transient(False)
    (event,) = catalog.get_events([1], {"canonical": transient}, schedule)
    assert not np.any(event.simulate_photometry(uvex)["in_model"])
