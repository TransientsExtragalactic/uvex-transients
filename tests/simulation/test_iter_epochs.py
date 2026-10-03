"""Tests for `SurveySimulator.iter_epochs`: lookback, pre-explosion rows, geometry-only mode, grouping helpers."""

import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable, vstack
from m4opt.missions._uvex import uvex

from uvex_transients.simulation.core import SurveySimulator, event_boundaries
from uvex_transients.simulation.event import Event
from uvex_transients.transients.TDEs import TidalDisruptionEvent
from uvex_transients.utils.keyed_noise import keyed_standard_normal, time_key

from .test_core import _make_catalog

BAND_NAMES = list(uvex.detector.bandpasses)
EVERYTHING = -np.inf


@pytest.fixture
def setup(make_schedule, hot_spot):
    """A TDE catalog (events explode 0 to 100 days into a 150 day schedule) and a simulator over it."""
    transient = TidalDisruptionEvent()
    schedule = make_schedule(n_sched=40)
    catalog, *_ = _make_catalog(transient, hot_spot)
    sim = SurveySimulator(schedule, transients={"tde": transient}, simulation_seed=7)
    return transient, schedule, catalog, sim


def _epochs(sim, catalog, mission=uvex, **kwargs):
    kwargs.setdefault("detection_floor", EVERYTHING)
    return vstack(list(sim.iter_epochs(catalog, mission, progress=False, **kwargs)))


def _keys(epochs):
    return set(zip(np.asarray(epochs["event_id"]).tolist(), np.asarray(epochs["observation_index"]).tolist()))


# --------------------------------------------------------------------------- #
# Grouping helpers                                                            #
# --------------------------------------------------------------------------- #
def test_event_boundaries():
    """Contiguous blocks of equal ids are located, in order of appearance."""
    ids = np.array([4, 4, 4, 9, 2, 2])
    starts, stops = event_boundaries(ids)
    assert starts.tolist() == [0, 3, 4]
    assert stops.tolist() == [3, 4, 6]

    # Usable directly for a vectorized per-event reduction.
    values = np.array([5.0, 1.0, 3.0, 7.0, 2.0, 8.0])
    assert np.minimum.reduceat(values, starts).tolist() == [1.0, 7.0, 2.0]


def test_event_boundaries_empty():
    """An empty chunk has no events."""
    starts, stops = event_boundaries(np.array([], dtype=int))
    assert len(starts) == len(stops) == 0


def test_chunks_are_grouped_by_event(setup):
    """Every yielded chunk has each event's rows contiguous and time-ordered."""
    _, _, catalog, sim = setup
    for chunk in sim.iter_epochs(
        catalog, uvex, progress=False, chunk_size=5, detection_floor=EVERYTHING, lookback=None
    ):
        starts, stops = event_boundaries(chunk["event_id"])
        assert len(np.unique(chunk["event_id"])) == len(starts)
        for a, b in zip(starts, stops):
            assert np.all(np.diff(chunk["t_obs"][a:b].jd) > 0)


# --------------------------------------------------------------------------- #
# Geometry-only mode                                                          #
# --------------------------------------------------------------------------- #
def test_geometry_only_needs_no_mission_and_matches_snr_rows(setup):
    """With both SNR flags off no mission is needed, and the rows are those the SNR path yields at no floor."""
    _, _, catalog, sim = setup
    geometry = _epochs(sim, catalog, mission=None, include_snr=False)
    with_snr = _epochs(sim, catalog)

    assert set(geometry.colnames) == {"event_id", "observation_index", "t_obs", "t_since_explosion", "pre_explosion"}
    assert len(geometry) > 0
    assert _keys(geometry) == _keys(with_snr)


def test_snr_requires_a_mission(setup):
    """Asking for an SNR without a mission is a TypeError, raised eagerly."""
    _, _, catalog, sim = setup
    with pytest.raises(TypeError, match="Mission"):
        sim.iter_epochs(catalog, None)


@pytest.mark.parametrize("bad", [-1 * u.day, 5 * u.m, np.inf * u.day, [1, 2] * u.day, "soon"])
def test_lookback_is_validated(setup, bad):
    """A lookback must be a finite, non-negative scalar duration or None."""
    _, _, catalog, sim = setup
    with pytest.raises(ValueError, match="lookback"):
        sim.iter_epochs(catalog, uvex, lookback=bad)


# --------------------------------------------------------------------------- #
# Lookback                                                                    #
# --------------------------------------------------------------------------- #
def test_lookback_only_adds_pre_explosion_rows(setup):
    """Widening the lookback adds earlier observations and leaves every existing row's values untouched."""
    _, _, catalog, sim = setup
    default = _epochs(sim, catalog)
    wide = _epochs(sim, catalog, lookback=80 * u.day)
    everything = _epochs(sim, catalog, lookback=None)

    assert _keys(default) <= _keys(wide) <= _keys(everything)
    assert len(everything) > len(wide) > len(default)

    added = wide[[k not in _keys(default) for k in zip(wide["event_id"], wide["observation_index"])]]
    assert len(added) > 0
    assert np.all(added["pre_explosion"])
    assert np.all(added["t_since_explosion"] >= -80 * u.day)

    # Rows present in both are bit-identical: noise is keyed, not streamed.
    shared = {k: i for i, k in enumerate(zip(wide["event_id"], wide["observation_index"]))}
    for row in default:
        match = wide[shared[(row["event_id"], row["observation_index"])]]
        assert match["snr"] == row["snr"]
        assert match["flux"] == row["flux"]


def test_default_lookback_has_no_real_pre_explosion_epochs(setup):
    """With no lookback, the only negative-time rows are exposures that straddle the explosion itself."""
    _, _, catalog, sim = setup
    default = _epochs(sim, catalog)
    pre = default[default["pre_explosion"]]
    assert np.all(pre["t_since_explosion"] > -900 * u.s)


def test_pre_explosion_rows_are_pure_background(setup):
    """Before the explosion the measured SNR is exactly the keyed background draw."""
    _, _, catalog, sim = setup
    band = BAND_NAMES[0]
    epochs = _epochs(sim, catalog, bands=[band], lookback=None)
    pre = epochs[epochs["pre_explosion"]]

    assert len(pre) > 100

    seed_of = dict(zip(np.asarray(catalog.table["event_id"]), np.asarray(catalog.table["parameter_seed"])))
    seeds = np.array([seed_of[e] for e in np.asarray(pre["event_id"])], dtype=np.uint64)
    draw = keyed_standard_normal(seeds, time_key(pre["t_obs"]), BAND_NAMES.index(band))
    np.testing.assert_allclose(pre["snr"], draw, atol=1e-6)

    z = np.asarray(pre["snr"])
    assert abs(z.mean()) < 5 / np.sqrt(len(z))
    assert abs(z.std() - 1) < 5 / np.sqrt(2 * len(z))


def test_pre_explosion_rows_match_event_background_photometry(setup):
    """The iterator's pre-explosion measurements are exactly the ones `Event` reports as background rows."""
    transient, schedule, catalog, sim = setup
    epochs = _epochs(sim, catalog, lookback=None)
    pre = epochs[epochs["pre_explosion"]]
    assert len(pre) > 100

    table = catalog.table
    n_matched = 0
    for event_id in np.unique(np.asarray(pre["event_id"]))[:6]:
        row = table[int(np.flatnonzero(np.asarray(table["event_id"]) == event_id)[0])]
        event = Event(
            event_id=int(event_id),
            schedule=schedule,
            transient=transient,
            coord=row["coord"],
            redshift=float(row["redshift"]),
            t_explosion=row["t_explosion"],
            seed=int(row["parameter_seed"]),
            luminosity_distance=row["luminosity_distance"],
            ebv=float(row["ebv"]),
            photometry_pre_window=250 * u.day,
        )
        phot = event.simulate_photometry(uvex)
        phot = phot[~phot["in_model"]]

        for epoch in pre[np.asarray(pre["event_id"]) == event_id]:
            dt = np.abs((phot["obs_time"] - epoch["t_obs"]).to_value(u.s))
            match = phot[(dt < 1e-3) & (phot["band"] == epoch["band"])]
            assert len(match) == 1
            sigma = epoch["flux_err"].to_value(u.Jy)
            assert match["flux_err"][0].to_value(u.Jy) == pytest.approx(sigma, rel=1e-6)
            assert abs(match["flux"][0].to_value(u.Jy) - epoch["flux"].to_value(u.Jy)) < 1e-6 * sigma
            n_matched += 1

    assert n_matched > 20


# --------------------------------------------------------------------------- #
# Cuts ignore pre-explosion rows                                              #
# --------------------------------------------------------------------------- #
def test_detection_epochs_ignore_pre_explosion_rows(setup, monkeypatch):
    """A high SNR before the explosion is a noise fluctuation: never a detection, and not a non-detection."""
    _, _, catalog, sim = setup
    chunk = QTable(
        {
            "snr": [9.0, 9.0, 1.0, 1.0],
            "band": ["NUV"] * 4,
            "pre_explosion": [False, True, False, True],
        }
    )
    monkeypatch.setattr(sim, "iter_epochs", lambda *args, **kwargs: iter([chunk]))

    (out,) = sim._iter_detection_epochs(catalog, uvex, 5.0, None, None, keep_all=True)

    assert out["detected"].tolist() == [True, False, False, False]
    assert out["non_detected"].tolist() == [False, False, True, True]


def test_cuts_do_not_change_with_lookback_available(setup):
    """`filter_by_snr` ignores pre-explosion rows, so it matches a hand reduction of post-explosion rows only."""
    transient, schedule, catalog, _ = setup
    sim = SurveySimulator(schedule, transients={"tde": transient}, simulation_seed=7)

    epochs = _epochs(sim, catalog, lookback=None)
    post = epochs[~epochs["pre_explosion"]]
    expected = set(np.asarray(post[np.asarray(post["snr"]) > 3.0]["event_id"]))

    kept = sim.filter_by_snr(catalog, uvex, snr_threshold=3.0, exclude_first_visit_detections=False)
    assert set(np.asarray(kept.table["event_id"])) == expected


def test_unmeasurable_band_never_wins(setup, monkeypatch):
    """A band that could not be measured (NaN) is never chosen over one that was."""
    transient, _, catalog, sim = setup
    real = transient.sed.measure_photometry

    def first_band_fails(*args, **kwargs):
        measurements = [values.copy() for values in real(*args, **kwargs)]
        for values in measurements:
            values[0] = np.nan
        return tuple(measurements)

    monkeypatch.setattr(transient.sed, "measure_photometry", first_band_fails)
    epochs = _epochs(sim, catalog)

    assert len(epochs) > 0
    assert set(np.asarray(epochs["band"])) == {BAND_NAMES[1]}
    assert np.all(np.isfinite(epochs["snr"]))
