"""Tests of what counts as a detection: the `detection` cuts of a survey sampler, and the `detect` it runs."""

import dataclasses

import numpy as np
import pytest

from uvex_transients.cli.phase_space import PhaseSpaceConfig
from uvex_transients.phase_space.sampling import CanonicalSampler, check_detection
from uvex_transients.phase_space.stages import describe_detection, map_title
from uvex_transients.simulation.core import SurveySimulator

NSIDE = 32
TIME_BINS = 3
BW = 14.55  # the FUV equivalent bandwidth is about 10**14.55 Hz
BRIGHT_AND_LONG = [30.0 + BW, 2.0]
N = 120

SNR = {"cut": "snr"}
BRACKET = {"cut": "time_since_last_nondetection", "params": {"max_delay": 1.0}}


@pytest.fixture(scope="module")
def schedule(make_schedule):
    return make_schedule(n_sched=20)


def make_sampler(schedule, **kwargs):
    defaults = {"mission": "uvex_fast", "nside": NSIDE, "order": "nested", "time_bins": TIME_BINS, "n_visits": 1}
    defaults.update(kwargs)
    return CanonicalSampler(schedule, **defaults)


def count(sampler, seed=3):
    return sampler.draw(BRIGHT_AND_LONG, N, np.random.SeedSequence(seed))


# ---------------------------------------------------------------------------- #
# The definition                                                               #
# ---------------------------------------------------------------------------- #
def test_no_definition_is_the_default():
    assert check_detection(None) is None


def test_a_value_of_a_subclass_of_a_builtin_type_is_made_a_builtin_one():
    """A header cannot hold what a YAML loader makes of a number, so it is converted."""

    class Float(float):
        pass

    class Mapping(dict):
        pass

    (step,) = check_detection([Mapping(cut="time_since_last_nondetection", params=Mapping(max_delay=Float(1.0)))])
    assert type(step["params"]) is dict and type(step["params"]["max_delay"]) is float


def test_a_definition_is_returned_as_plain_dicts_with_empty_params_left_out():
    steps = check_detection([SNR, {"cut": "snr", "params": {"n_visits": 2, "bands": ("FUV",)}}, BRACKET])
    assert steps == (
        {"cut": "snr"},
        {"cut": "snr", "params": {"n_visits": 2, "bands": ["FUV"]}},
        {"cut": "time_since_last_nondetection", "params": {"max_delay": 1.0}},
    )


@pytest.mark.parametrize(
    "detection, message",
    [
        ([], "at least one cut"),
        ("snr", "list of cuts"),
        ({"cut": "snr"}, "list of cuts"),
        ([{"params": {}}], "'cut' name"),
        (["snr"], "'cut' name"),
        ([{"cut": "snr", "extra": 1}], r"unknown key\(s\) \['extra'\]"),
        ([{"cut": "bogus"}], "Unknown cut 'bogus'"),
        ([{"cut": "snr", "params": {"n_visit": 2}}], r"takes no parameter\(s\) \['n_visit'\]"),
        ([{"cut": "snr", "params": [1]}], "'params' must be a mapping"),
        ([{"cut": "snr", "params": {"bands": [object()]}}], "not a plain value"),
    ],
)
def test_a_bad_definition_is_refused_with_the_entry_that_is_wrong(detection, message):
    with pytest.raises(ValueError, match=message):
        check_detection(detection)


def test_the_entry_that_is_wrong_is_named():
    with pytest.raises(ValueError, match=r"detection\[1\]"):
        check_detection([SNR, {"cut": "bogus"}])


# ---------------------------------------------------------------------------- #
# What it does                                                                 #
# ---------------------------------------------------------------------------- #
def test_the_default_is_the_snr_cut_with_the_settings_of_the_sampler(schedule):
    """`detection=None` is `[{cut: snr}]`, which is the single `filter_by_snr` call the sampler used to make."""
    default = make_sampler(schedule)
    assert count(default) == count(make_sampler(schedule, detection=[SNR])) > 0

    # The same draw, made by hand
    default._place(default._points(BRIGHT_AND_LONG))
    simulator = SurveySimulator(
        default.schedule, {default.TRANSIENT_KEY: default._transient}, simulation_seed=np.random.SeedSequence(3)
    )
    catalog = simulator.sample_N_events(N, default.time_bins, nside=default.nside, order=default.order)
    by_hand = simulator.filter_by_snr(
        catalog,
        default.mission,
        default.snr_threshold,
        bands=[default.band],
        n_visits=default.n_visits,
        exclude_first_visit_detections=default.exclude_first_visit_detections,
    )
    assert len(by_hand) == count(default)


def test_a_cut_takes_what_its_entry_leaves_out_from_the_sampler_and_an_entry_overrides_it(schedule):
    sampler = make_sampler(schedule, n_visits=2)
    explicit = make_sampler(
        schedule,
        detection=[{"cut": "snr", "params": {"snr_threshold": 5.0, "n_visits": 2, "bands": ["FUV"]}}],
    )
    assert count(sampler) == count(explicit)

    # An entry that gives n_visits wins over the sampler's
    assert count(make_sampler(schedule, n_visits=2, detection=[{"cut": "snr", "params": {"n_visits": 1}}])) == count(
        make_sampler(schedule, n_visits=1)
    )


def test_more_required_detections_recover_no_more_events(schedule):
    counts = [count(make_sampler(schedule, n_visits=n)) for n in (1, 2, 4, 1000)]
    assert counts == sorted(counts, reverse=True)

    # More visits than the schedule has makes a recovery impossible
    assert counts[0] > 0 and counts[-1] == 0


def test_a_cut_after_the_first_narrows_what_the_first_kept(schedule):
    base = make_sampler(schedule)
    bracketed = make_sampler(schedule, detection=[SNR, BRACKET])
    assert 0 <= count(bracketed) <= count(base)

    # A bracket that asks nothing of the delay, and keeps an event that was never seen before it was detected, keeps
    # every event the first cut kept
    lenient = make_sampler(
        schedule,
        detection=[
            SNR,
            {"cut": "time_since_last_nondetection", "params": {"max_delay": 1e9, "keep_if_no_nondetection": True}},
        ],
    )
    assert count(lenient) == count(base)


def test_a_rule_that_keeps_nothing_stops_early_and_recovers_nothing(schedule):
    impossible = make_sampler(schedule, detection=[{"cut": "snr", "params": {"snr_threshold": 1e9}}, BRACKET])
    assert count(impossible) == 0


def test_detect_is_what_a_draw_counts_and_a_subclass_can_replace_it(schedule):
    class EveryEvent(CanonicalSampler):
        def detect(self, simulator, catalog):
            return dataclasses.replace(catalog, table=catalog.table[: len(catalog) // 2])

    sampler = EveryEvent(
        schedule, "uvex_fast", nside=NSIDE, order="nested", time_bins=TIME_BINS, n_visits=1, detection=None
    )
    assert sampler.draw(BRIGHT_AND_LONG, N, np.random.SeedSequence(3)) == N // 2


# ---------------------------------------------------------------------------- #
# Recording                                                                    #
# ---------------------------------------------------------------------------- #
def test_the_default_records_no_detection_so_that_earlier_files_are_unchanged(schedule):
    assert "detection" not in make_sampler(schedule).settings


def test_a_definition_is_recorded_and_rebuilds_the_same_sampler(schedule):
    sampler = make_sampler(schedule, detection=[SNR, BRACKET])
    settings = sampler.settings
    assert settings["detection"] == [SNR, BRACKET]

    rebuilt = CanonicalSampler.from_settings(settings, schedule=schedule, mission="uvex_fast")
    assert rebuilt.detection == sampler.detection
    assert count(rebuilt) == count(sampler)


def test_the_recorded_definition_survives_an_experiment_file(schedule, tmp_path):
    pytest.importorskip("sklearn")
    from uvex_transients.phase_space.experiment import Experiment

    sampler = make_sampler(schedule, detection=[SNR, BRACKET])
    path = tmp_path / "points.ecsv"
    Experiment.create(sampler).save(path)
    loaded = Experiment.load(path, schedule=schedule, mission="uvex_fast")
    assert loaded.sampler.detection == sampler.detection


# ---------------------------------------------------------------------------- #
# Describing it                                                                #
# ---------------------------------------------------------------------------- #
def test_the_default_is_described_by_its_threshold_and_visits():
    assert describe_detection({"snr_threshold": 5.0, "n_visits": 2}) == "SNR >= 5 on 2 visit(s)"


def test_a_definition_is_described_by_its_cuts():
    settings = {"snr_threshold": 5.0, "n_visits": 2, "detection": [SNR, BRACKET]}
    assert describe_detection(settings) == "SNR >= 5: snr + time_since_last_nondetection(max_delay=1.0)"
    title = map_title("canonical", {"band": "FUV", **settings}, 10, 1000)
    assert title.startswith("FUV: SNR >= 5: snr + time_since_last_nondetection(max_delay=1.0), 10 points")


# ---------------------------------------------------------------------------- #
# The run-config                                                               #
# ---------------------------------------------------------------------------- #
def _config(recovery, kind="canonical"):
    return PhaseSpaceConfig(
        {"schedule": {"name": "uvex_v0.6.0"}, "maps": [{"id": "m", "kind": kind, "recovery": recovery}]}
    )


def test_a_run_config_can_define_a_detection_for_a_map():
    spec = _config({"band": "FUV", "detection": [SNR, BRACKET]}).maps[0]
    assert spec.recovery["detection"] == [SNR, BRACKET]


def test_a_run_config_with_a_bad_detection_says_which_map_and_what_is_wrong():
    with pytest.raises(ValueError, match=r"map 'm'.*recovery: detection:.*max_dely"):
        _config({"detection": [{"cut": "time_since_last_nondetection", "params": {"max_dely": 1.0}}]})


def test_a_breakout_map_takes_a_detection_too():
    spec = _config({"band": "FUV", "detection": [{"cut": "snr", "params": {"n_visits": 2}}]}, kind="breakout").maps[0]
    assert spec.recovery["detection"] == [{"cut": "snr", "params": {"n_visits": 2}}]
