"""Tests for `SurveySampler`, `CanonicalSampler`, `ParameterAxis` and the `RecoveryEvaluator` that sits over them."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

from uvex_transients.phase_space.evaluator import RecoveryEvaluator, RedshiftLimit
from uvex_transients.phase_space.samplers import (
    CanonicalSampler,
    Estimate,
    ParameterAxis,
    Sampler,
    StoppingRule,
    SurveySampler,
    available_samplers,
    get_sampler,
)
from uvex_transients.phase_space.samplers import survey as survey_module
from uvex_transients.phase_space.space import Axis, Space
from uvex_transients.utils import config

NSIDE = 32
TIME_BINS = 3
MISSION = "uvex_fast"
BW = 14.55  # the FUV equivalent bandwidth is about 10**14.55 Hz

BRIGHT_AND_LONG = [30.0 + BW, 2.0]
HOPELESS = [24.0 + BW, -1.0]


@pytest.fixture(scope="module")
def schedule(make_schedule):
    return make_schedule(n_sched=20)


def make_sampler(schedule, **kwargs):
    defaults = {"mission": MISSION, "nside": NSIDE, "order": "nested", "time_bins": TIME_BINS, "n_visits": 1}
    defaults.update(kwargs)
    return CanonicalSampler(schedule, **defaults)


def make_evaluator(schedule, **kwargs):
    defaults = {
        "mission": MISSION,
        "nside": NSIDE,
        "order": "nested",
        "time_bins": TIME_BINS,
        "chunk_size": 200,
        "n_max": 600,
        "k_target": 5,
        "n_visits": 1,
        "seed": 3,
    }
    defaults.update(kwargs)
    return RecoveryEvaluator(schedule, **defaults)


@pytest.fixture(scope="module")
def sampler(schedule):
    return make_sampler(schedule)


# ---------------------------------------------------------------------------- #
# The classes                                                                  #
# ---------------------------------------------------------------------------- #
def test_the_survey_sampler_is_abstract_and_leaves_the_four_pieces_to_a_subclass():
    assert SurveySampler.__abstractmethods__ == {
        "_log_window",
        "_place",
        "from_settings",
        "make_transient",
        "redshift_limit",
    }
    with pytest.raises(TypeError, match="abstract"):
        SurveySampler(None)


def test_the_canonical_sampler_is_a_complete_registered_sampler(sampler):
    assert not CanonicalSampler.__abstractmethods__
    assert isinstance(sampler, SurveySampler) and isinstance(sampler, Sampler)
    assert CanonicalSampler.kind == "canonical"
    assert get_sampler("canonical") is CanonicalSampler
    assert "canonical" in available_samplers()


def test_the_survey_samplers_are_imported_only_when_a_built_in_kind_is_asked_for():
    """Importing the sampler package does not import the survey samplers; asking for the kind does."""
    code = (
        "import sys; import uvex_transients.phase_space.samplers as s; "
        "survey = ('uvex_transients.phase_space.samplers.canonical', 'uvex_transients.phase_space.samplers.survey', "
        "'uvex_transients.phase_space.transient'); "
        "assert not any(m in sys.modules for m in survey), [m for m in survey if m in sys.modules]; "
        "assert 'canonical' in s.available_samplers(); "
        "assert not any(m in sys.modules for m in survey); "
        "s.get_sampler('canonical'); "
        "assert all(m in sys.modules for m in survey)"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False, env=env)
    assert result.returncode == 0, result.stderr[-600:]


# ---------------------------------------------------------------------------- #
# The space                                                                    #
# ---------------------------------------------------------------------------- #
def test_the_space_has_the_two_canonical_axes_with_the_configured_ranges(sampler):
    space = sampler.space
    assert space.names == ("log_luminosity", "log_half_time")
    np.testing.assert_array_equal(
        space.bounds, [config["phase_space.map.log_luminosity"], config["phase_space.map.log_half_time"]]
    )
    assert [axis.unit for axis in space] == ["erg/s", "day"]
    assert all(axis.scale == "log10" for axis in space)


def test_the_luminosity_axis_is_labelled_with_the_band(schedule):
    assert "FUV" in make_sampler(schedule, band="FUV").space["log_luminosity"].label
    assert "NUV" in make_sampler(schedule, band="NUV").space["log_luminosity"].label


# ---------------------------------------------------------------------------- #
# Recording and rebuilding                                                     #
# ---------------------------------------------------------------------------- #
def test_the_settings_are_the_physics_and_not_the_rule_or_the_seed(sampler):
    settings = sampler.settings
    assert {
        "band",
        "spectral_index",
        "rise_ratio",
        "fade_mag",
        "snr_threshold",
        "n_visits",
        "mag_limit",
        "time_bins",
    } <= set(settings)
    assert not {"chunk_size", "k_target", "n_max", "seed"} & set(settings)
    assert settings["nside"] == NSIDE and settings["order"] == "nested"


def test_a_sampler_is_rebuilt_from_its_settings_directly_and_through_the_registry(schedule, sampler):
    settings = sampler.settings
    for rebuilt in (
        CanonicalSampler.from_settings(settings, schedule=schedule, mission=MISSION),
        get_sampler("canonical").from_settings(settings, schedule=schedule, mission=MISSION),
    ):
        assert rebuilt.settings == settings
        assert rebuilt.space == sampler.space
        assert rebuilt.bandwidth == sampler.bandwidth


def test_settings_of_a_top_hat_map_are_refused(schedule, sampler):
    settings = sampler.settings
    del settings["rise_ratio"]
    with pytest.raises(ValueError, match="top-hat"):
        CanonicalSampler.from_settings(settings, schedule=schedule, mission=MISSION)


def test_a_settings_without_a_schedule_or_a_mission_say_so(sampler):
    settings = sampler.settings
    with pytest.raises(ValueError, match="no schedule"):
        CanonicalSampler.from_settings({**settings, "schedule_name": None}, mission=MISSION)


def test_the_table_meta_labels_the_axis_and_leaves_the_rule_to_the_evaluator(schedule, sampler):
    meta = sampler.table_meta
    assert meta["band"] == "FUV" and meta["luminosity_unit"] == "erg/s"
    assert "n_max" not in meta
    assert "n_max" in make_evaluator(schedule).table_meta


# ---------------------------------------------------------------------------- #
# The tries                                                                    #
# ---------------------------------------------------------------------------- #
def test_a_draw_is_a_count_between_zero_and_the_tries(sampler):
    k = sampler.draw(BRIGHT_AND_LONG, 200, np.random.SeedSequence(1))
    assert isinstance(k, int) and 0 < k <= 200
    assert sampler.draw(HOPELESS, 200, np.random.SeedSequence(1)) == 0


def test_a_draw_is_reproducible_from_its_seed(sampler):
    first = sampler.draw(BRIGHT_AND_LONG, 200, np.random.SeedSequence(5))
    assert first == sampler.draw(BRIGHT_AND_LONG, 200, np.random.SeedSequence(5))


def test_a_draw_needs_a_point_of_the_space(sampler):
    with pytest.raises(ValueError, match="2 coordinates"):
        sampler.draw([44.0, 1.0, 0.0], 10, np.random.SeedSequence(0))


def test_a_draw_refuses_a_simulation_that_returns_another_number_of_events(schedule, monkeypatch):
    """The count of successes means something else if the number of tries is not the one the rule counted."""
    sampler = make_sampler(schedule)

    class _Short:
        def __init__(self, *args, **kwargs):
            pass

        def sample_N_events(self, n, *args, **kwargs):
            return list(range(n - 1))

    monkeypatch.setattr(survey_module, "SurveySimulator", _Short)
    with pytest.raises(RuntimeError, match="got 199"):
        sampler.draw(BRIGHT_AND_LONG, 200, np.random.SeedSequence(0))


@pytest.mark.parametrize("point", [BRIGHT_AND_LONG, HOPELESS, [28.0 + BW, 0.5]])
def test_the_rule_on_the_sampler_is_what_the_evaluator_does(schedule, sampler, point):
    """`RecoveryEvaluator` is a `StoppingRule` run on a `CanonicalSampler`: the same counts, from the same seed."""
    rule = StoppingRule(chunk_size=200, k_target=5, n_max=600)
    outcome = rule.run(sampler, point, np.random.SeedSequence(3))
    result = make_evaluator(schedule, seed=3)(point)
    assert (outcome.k, outcome.n, outcome.extras["n_chunks"]) == (result["k"], result["n"], result["n_chunks"])


# ---------------------------------------------------------------------------- #
# The scales and the products                                                  #
# ---------------------------------------------------------------------------- #
def test_the_scales_of_a_batch_of_points_are_those_of_each_point(sampler):
    x = np.array([[43.0, 0.0], [44.0, 1.0], [45.0, 2.0]])
    for scale in (sampler.redshift_limit, sampler.volume_time, sampler.coverage, sampler.all_sky_volume_time):
        batch = scale(x)
        assert batch.shape == (3,)
        np.testing.assert_array_equal(batch, [float(scale(point)) for point in x])
    assert sampler.volume_time(x.reshape(3, 1, 2)).shape == (3, 1)


def test_the_volume_time_is_the_coverage_times_the_all_sky_volume_time(sampler):
    x = np.array([[43.0, 0.0], [44.0, 1.0], [45.0, 2.0]])
    np.testing.assert_allclose(sampler.volume_time(x), sampler.coverage(x) * sampler.all_sky_volume_time(x), rtol=1e-10)


def test_the_redshift_limit_depends_on_the_luminosity_alone(sampler):
    z = sampler.redshift_limit(np.array([[44.0, -1.0], [44.0, 2.0]]))
    assert z[0] == z[1]
    assert sampler.redshift_limit([45.0, 0.0]) > sampler.redshift_limit([43.0, 0.0])


def test_the_products_are_the_scales_and_the_quantities_made_from_them(sampler):
    products = {product.name: product for product in sampler.products}
    assert set(products) == {
        "z_limit",
        "coverage",
        "volume_time",
        "all_sky_volume_time",
        "efficiency",
        "recovered_volume_time",
    }
    x = np.array([[43.0, 0.0], [44.0, 1.0], [45.0, 2.0]])
    p = np.array([0.1, 0.4, 0.8])
    estimate = Estimate(p, p - 0.05, p + 0.05)

    np.testing.assert_array_equal(products["z_limit"](x, estimate)["z_limit"], sampler.redshift_limit(x))
    np.testing.assert_array_equal(products["volume_time"](x, estimate)["volume_time"], sampler.volume_time(x))
    np.testing.assert_allclose(products["efficiency"](x, estimate)["efficiency"], p * sampler.coverage(x))
    recovered = products["recovered_volume_time"](x, estimate)
    np.testing.assert_allclose(recovered["recovered_volume_time"], p * sampler.volume_time(x))
    np.testing.assert_allclose(recovered["recovered_volume_time_hi"], (p + 0.05) * sampler.volume_time(x))
    assert products["volume_time"].unit == "Mpc3 yr"


def test_the_redshift_limit_table_is_the_one_that_moved_with_it():
    assert RedshiftLimit(24.5)(43.0) > 0
    from uvex_transients.phase_space.samplers.canonical import RedshiftLimit as moved

    assert moved is RedshiftLimit


# ---------------------------------------------------------------------------- #
# Parameter axes                                                               #
# ---------------------------------------------------------------------------- #
def test_a_parameter_axis_reads_a_coordinate_on_its_scale():
    axis = ParameterAxis("log_t0", "t_scale", u.s)
    assert axis.value(2.0, "log10").to_value(u.s) == pytest.approx(100.0)
    assert axis.value(2.0, "linear").to_value(u.s) == pytest.approx(2.0)
    assert axis.value(2.0, "log10").unit == u.s
    assert ParameterAxis("a", "b", "day").unit == u.day


def test_a_parameter_axis_needs_a_unit():
    with pytest.raises(ValueError):
        ParameterAxis("a", "b", "not_a_unit")


@pytest.fixture
def bound(schedule):
    """Return a canonical sampler whose two axes are bound to the amplitude and decay time of its SED."""
    sampler = make_sampler(schedule)
    sampler.parameter_axes = (
        ParameterAxis("log_luminosity", "amplitude", u.erg / u.s),
        ParameterAxis("log_half_time", "decay", u.s),
    )
    sampler._check_parameter_axes()
    return sampler


def test_bound_axes_become_the_parameters_they_name(bound):
    values = bound.parameters_at([44.0, 1.0])
    assert set(values) == {"amplitude", "decay"}
    assert values["amplitude"].to_value(u.erg / u.s) == pytest.approx(1e44)
    assert values["decay"].to_value(u.s) == pytest.approx(10.0)


def test_a_linear_axis_is_read_as_the_value_itself(bound):
    bound._space = Space(
        [Axis("log_luminosity", 1.0, 100.0, scale="linear"), Axis("log_half_time", 1.0, 100.0, scale="log10")]
    )
    values = bound.parameters_at([5.0, 1.0])
    assert values["amplitude"].to_value(u.erg / u.s) == pytest.approx(5.0)
    assert values["decay"].to_value(u.s) == pytest.approx(10.0)


def test_fixing_the_bound_parameters_leaves_the_others_to_their_priors(bound):
    sed = bound._transient.sed
    sed["amplitude"].unfix()
    sed["decay"].unfix()
    was_fixed = {name for name in sed if sed[name].is_fixed}
    bound._fix_parameters([44.0, 1.0])
    assert sed["amplitude"].fixed_value.to_value(u.erg / u.s) == pytest.approx(1e44)
    assert sed["decay"].fixed_value.to_value(u.s) == pytest.approx(10.0)
    # Nothing else was touched
    assert {name for name in sed if sed[name].is_fixed} == was_fixed | {"amplitude", "decay"}


def test_a_binding_to_an_unknown_axis_or_parameter_is_refused_by_name(schedule):
    sampler = make_sampler(schedule)
    sampler.parameter_axes = (ParameterAxis("no_such_axis", "amplitude", u.erg / u.s),)
    with pytest.raises(ValueError, match="no_such_axis"):
        sampler._check_parameter_axes()
    sampler.parameter_axes = (ParameterAxis("log_luminosity", "no_such_parameter", u.erg / u.s),)
    with pytest.raises(ValueError, match="no_such_parameter"):
        sampler._check_parameter_axes()


def test_a_binding_in_the_wrong_unit_is_refused(schedule):
    sampler = make_sampler(schedule)
    sampler.parameter_axes = (ParameterAxis("log_luminosity", "amplitude", u.s),)
    with pytest.raises(ValueError, match="not a unit of 'amplitude'"):
        sampler._check_parameter_axes()


def test_two_bindings_cannot_share_an_axis_or_a_parameter(schedule):
    sampler = make_sampler(schedule)
    sampler.parameter_axes = (
        ParameterAxis("log_luminosity", "amplitude", u.erg / u.s),
        ParameterAxis("log_luminosity", "decay", u.s),
    )
    with pytest.raises(ValueError, match="repeats"):
        sampler._check_parameter_axes()


def test_a_sampler_with_no_bindings_sets_no_parameters(sampler):
    assert sampler.parameter_axes == ()
    assert sampler.parameters_at([44.0, 1.0]) == {}


# ---------------------------------------------------------------------------- #
# The evaluator over them                                                      #
# ---------------------------------------------------------------------------- #
def test_the_evaluator_keeps_the_rule_and_hands_the_physics_to_the_sampler(schedule):
    evaluator = make_evaluator(schedule)
    assert (evaluator.chunk_size, evaluator.k_target, evaluator.n_max) == (200, 5, 600)
    assert isinstance(evaluator._sampler, CanonicalSampler)
    assert evaluator.band == "FUV" and evaluator.mission is evaluator._sampler.mission
    assert evaluator.redshift_limit is evaluator._sampler._limit
    assert not hasattr(evaluator, "no_such_attribute")


def test_the_evaluator_settings_are_the_sampler_settings_with_the_rule_and_the_seed(schedule):
    evaluator = make_evaluator(schedule, seed=7)
    settings = evaluator.settings
    assert {key: settings[key] for key in ("chunk_size", "k_target", "n_max", "seed")} == {
        "chunk_size": 200,
        "k_target": 5,
        "n_max": 600,
        "seed": 7,
    }
    sampler_settings = evaluator._sampler.settings
    assert {key: settings[key] for key in sampler_settings} == sampler_settings

    # The keys come in the order the header has always had them
    assert list(settings) == [
        "schedule_name",
        "mission",
        "band",
        "spectral_index",
        "rise_ratio",
        "fade_mag",
        "snr_threshold",
        "n_visits",
        "exclude_first_visit_detections",
        "mag_limit",
        "time_bins",
        "nside",
        "order",
        "chunk_size",
        "k_target",
        "n_max",
        "seed",
        "cosmology",
    ]


def test_the_evaluator_rule_is_validated_before_anything_else(schedule):
    with pytest.raises(ValueError, match="Need chunk_size >= 1"):
        make_evaluator(schedule, chunk_size=0)


def test_the_shape_of_the_transient_reaches_the_sampler(schedule):
    sampler = make_sampler(schedule, rise_ratio=0.2, fade_mag=8.0)
    assert sampler.settings["rise_ratio"] == 0.2 and sampler.settings["fade_mag"] == 8.0
    assert sampler._transient.rise_ratio == 0.2 and sampler._transient.fade_mag == 8.0
