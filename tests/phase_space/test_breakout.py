"""Tests for `BreakoutSampler`: the window, the redshift limit, the placement of a point, and a run end to end."""

import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table

from uvex_transients.models._utils import to_cgs_value
from uvex_transients.models.core.priors import LogNormalPrior
from uvex_transients.models.lightcurves import InverseGammaBrokenPowerLawLightcurve
from uvex_transients.models.shock_cooling import InverseGammaBlackbodySED
from uvex_transients.phase_space.breakout import BreakoutSampler, half_decline_time
from uvex_transients.phase_space.experiment import CampaignSettings, Experiment
from uvex_transients.phase_space.plotting import plot_grid
from uvex_transients.phase_space.samplers import Sampler, StoppingRule, available_samplers, get_sampler
from uvex_transients.utils.cosmology import resolve_cosmological_distances
from uvex_transients.utils.phot_utils import bandpass_table

NSIDE = 32
MISSION = "uvex_fast"


@pytest.fixture(scope="module")
def schedule(make_schedule):
    return make_schedule(n_sched=60)


def make_sampler(schedule, **kwargs):
    defaults = {"mission": MISSION, "nside": NSIDE, "order": "nested", "time_bins": 50, "mag_limit": 25.0}
    defaults.update(kwargs)
    return BreakoutSampler(schedule, **defaults)


@pytest.fixture(scope="module")
def sampler(schedule):
    return make_sampler(schedule)


# ---------------------------------------------------------------------------- #
# The window                                                                   #
# ---------------------------------------------------------------------------- #
def test_the_half_decline_of_the_default_shape():
    x_peak, x_half = half_decline_time(4 / 3, 0.3, 10.0, 2.0)
    assert x_peak == pytest.approx(1.0067, abs=1e-4)
    assert x_half == pytest.approx(3.604, abs=1e-3)


def test_the_light_curve_is_half_its_peak_at_the_half_decline():
    parameters = {"p": 4 / 3, "q": 0.3, "x_break": 10.0, "smoothness": 2.0}
    x_peak, x_half = half_decline_time(**parameters)
    at = lambda x: float(  # noqa: E731
        InverseGammaBrokenPowerLawLightcurve.eval_cgs(np.array([x]), amplitude=1.0, t_scale=1.0, **parameters)[0]
    )
    assert at(x_peak) == pytest.approx(1.0, rel=1e-5)
    assert at(x_half) == pytest.approx(0.5, rel=1e-6)
    assert at(0.99 * x_half) > 0.5 > at(1.01 * x_half)


def test_a_light_curve_that_never_falls_to_half_has_no_half_decline():
    """With the break far inside the peak the light curve is the shallow tail from the start, and takes forever to halve."""
    with pytest.raises(ValueError, match="never falls to half"):
        half_decline_time(4 / 3, 1e-3, 0.01, 2.0)
    # The same slow tail behind the break of the default shape still halves, on the steep part before it
    assert half_decline_time(4 / 3, 1e-3, 10.0, 2.0)[1] == pytest.approx(3.6, abs=0.1)


# ---------------------------------------------------------------------------- #
# The sampler                                                                  #
# ---------------------------------------------------------------------------- #
def test_the_breakout_sampler_is_a_complete_registered_sampler(sampler):
    assert not BreakoutSampler.__abstractmethods__
    assert isinstance(sampler, Sampler) and BreakoutSampler.kind == "breakout"
    assert get_sampler("breakout") is BreakoutSampler
    assert "breakout" in available_samplers()


def test_the_axes_are_parameters_of_the_sed_and_need_no_placing_code(sampler):
    assert sampler.space.names == ("log_L0", "log_t0")
    assert [axis.unit for axis in sampler.space] == ["erg/s", "s"]
    assert [(b.axis, b.parameter, b.unit) for b in sampler.parameter_axes] == [
        ("log_L0", "amplitude", u.erg / u.s),
        ("log_t0", "t_scale", u.s),
    ]
    values = sampler.parameters_at([44.0, 2.0])
    assert values["amplitude"].to_value(u.erg / u.s) == pytest.approx(1e44)
    assert values["t_scale"].to_value(u.s) == pytest.approx(100.0)


def test_the_defaults_come_from_the_configuration_with_one_visit_and_many_bins(schedule):
    built = BreakoutSampler(schedule, mission=MISSION, nside=NSIDE, order="nested")
    settings = built.settings
    assert settings["n_visits"] == 1 and settings["time_bins"] == 400
    assert settings["log_L0"] == [42.0, 46.5] and settings["log_t0"] == [0.0, 4.5]
    assert built.exclude_first_visit_detections is True


def test_a_transient_of_the_sampler_is_measured_over_the_whole_exposure(sampler):
    assert sampler.make_transient().AVERAGE_OVER_EXPOSURE is True
    assert sampler._transient.rate.to_value(u.Mpc**-3 / u.yr) == 1.0


def test_the_shape_of_the_light_curve_has_to_be_fixed(schedule, monkeypatch):
    monkeypatch.setitem(
        InverseGammaBlackbodySED._DEFAULT_PARAMETERS,
        "p",
        InverseGammaBlackbodySED._DEFAULT_PARAMETERS["p"].__class__(
            prior=LogNormalPrior(mean=0.0, sigma=0.1), scale=1.0 * u.dimensionless_unscaled, description="p", latex="p"
        ),
    )
    with pytest.raises(ValueError, match="fixed shape"):
        make_sampler(schedule)


# ---------------------------------------------------------------------------- #
# Placing a point                                                              #
# ---------------------------------------------------------------------------- #
def test_placing_a_point_fixes_the_sed_and_sets_the_window(schedule):
    sampler = make_sampler(schedule)
    x = np.array([45.0, 3.0])
    sampler._place(x)
    sed = sampler._transient.sed
    assert sed["amplitude"].fixed_value.to_value(u.erg / u.s) == pytest.approx(1e45)
    assert sed["t_scale"].fixed_value.to_value(u.s) == pytest.approx(1e3)

    z = float(sampler.redshift_limit(x))
    assert sampler._transient.redshift_limit == pytest.approx(z)
    assert sampler._transient.duration_limit.to_value(u.s) == pytest.approx((1.0 + z) * sampler.x_half * 1e3)


def test_the_window_in_the_exposure_scale_is_the_duration_limit(sampler):
    x = np.array([[45.0, 3.0], [43.0, 1.0]])
    z = sampler.redshift_limit(x)
    expected_days = (1.0 + z) * sampler.x_half * 10.0 ** x[:, 1] / 86400.0
    np.testing.assert_allclose(10.0 ** sampler._log_window(z, x), expected_days, rtol=1e-12)


# ---------------------------------------------------------------------------- #
# The redshift limit                                                           #
# ---------------------------------------------------------------------------- #
def test_the_table_of_the_limit_is_built_on_first_use(schedule):
    fresh = make_sampler(schedule)
    assert fresh._log_redshift_table is None
    fresh.redshift_limit([44.0, 2.0])
    table = fresh._log_redshift_table
    assert table is not None
    fresh.redshift_limit([43.0, 1.0])
    assert fresh._log_redshift_table is table


def test_a_brighter_breakout_is_seen_farther(sampler):
    z = sampler.redshift_limit(np.column_stack([np.linspace(42.5, 46.0, 8), np.full(8, 2.5)]))
    assert np.all(np.diff(z) > 0)
    assert z[0] > 0 and z[-1] <= sampler.z_max * (1 + 1e-9)


def test_the_limit_has_the_shape_of_the_points_and_is_held_at_the_edge_of_the_table(sampler):
    x = np.random.default_rng(0).uniform([42.0, 0.0], [46.0, 4.0], size=(5, 3, 2))
    assert sampler.redshift_limit(x).shape == (5, 3)
    inside = sampler.redshift_limit([46.5, 4.5])
    np.testing.assert_allclose(sampler.redshift_limit([50.0, 9.0]), inside)


@pytest.mark.parametrize("point", [(44.0, 2.0), (45.0, 3.0), (43.0, 1.0)])
def test_at_the_limit_the_brightest_band_magnitude_of_the_window_is_the_depth(sampler, point):
    """The independent check: the SED's own band magnitude, at the limit, peaks at the depth within the window."""
    log_l, log_t = point
    z = float(sampler.redshift_limit(np.array(point)))
    t0 = 10.0**log_t
    nu, throughput = bandpass_table(sampler.mission.detector.bandpasses[sampler.band])
    d_l = float(
        resolve_cosmological_distances(redshift=z, cosmology=sampler.cosmology)["luminosity_distance"].cgs.value
    )

    parameters = {
        name: float(np.asarray(to_cgs_value(value)).ravel()[0])
        for name, value in InverseGammaBlackbodySED().sample_parameters(size=1, rng=0).items()
    }
    parameters.update(amplitude=10.0**log_l, t_scale=t0)
    x = np.geomspace(1e-2, sampler.x_half, 400)
    magnitudes = InverseGammaBlackbodySED.mag_band_cgs(nu, throughput, (1.0 + z) * x * t0, z, d_l, **parameters)
    assert np.nanmin(magnitudes) == pytest.approx(sampler.mag_limit, abs=0.15)


# ---------------------------------------------------------------------------- #
# The tries                                                                    #
# ---------------------------------------------------------------------------- #
def test_a_bright_and_long_breakout_is_recovered_and_a_faint_short_one_is_not(sampler):
    bright = sampler.draw(np.array([46.0, 4.0]), 1000, np.random.SeedSequence(1))
    assert 0 < bright <= 1000
    assert bright == sampler.draw(np.array([46.0, 4.0]), 1000, np.random.SeedSequence(1))
    assert sampler.draw(np.array([43.0, 1.0]), 500, np.random.SeedSequence(1)) == 0


def test_the_scales_of_a_breakout_are_those_of_the_survey_footprint(sampler):
    x = np.array([[45.0, 3.0], [44.0, 2.0]])
    np.testing.assert_allclose(sampler.volume_time(x), sampler.coverage(x) * sampler.all_sky_volume_time(x), rtol=1e-10)
    assert np.all(sampler.coverage(x) > 0) and np.all(sampler.coverage(x) <= 1)
    # A longer breakout is followed for longer, so more of the survey can see it
    assert sampler.coverage([45.0, 4.0]) > sampler.coverage([45.0, 1.0])


def test_the_settings_rebuild_the_sampler_through_the_registry(schedule, sampler):
    rebuilt = get_sampler("breakout").from_settings(sampler.settings, schedule=schedule, mission=MISSION)
    assert rebuilt.settings == sampler.settings
    assert rebuilt.space == sampler.space
    assert rebuilt.x_half == sampler.x_half
    np.testing.assert_allclose(rebuilt.redshift_limit([45.0, 3.0]), sampler.redshift_limit([45.0, 3.0]))


# ---------------------------------------------------------------------------- #
# End to end                                                                   #
# ---------------------------------------------------------------------------- #
def test_an_experiment_of_the_breakout_runs_saves_loads_and_plots(schedule, sampler, tmp_path):
    experiment = Experiment.create(
        sampler,
        bounds=[(45.0, 46.5), (3.0, 4.5)],
        rule=StoppingRule(chunk_size=300, k_target=3, n_max=900),
        campaign=CampaignSettings(n_initial=4, batch_size=2, max_points=6, tolerance=0.0, seed=1),
        seed=1,
        n_restarts=0,
        n_candidates=64,
    ).run()
    assert len(experiment.phase_map) == 6
    assert experiment.phase_map.table.colnames == ["log_L0", "log_t0", "k", "n", "n_chunks"]

    experiment.save(tmp_path / "points.ecsv")
    header = Table.read(tmp_path / "points.ecsv", format="ascii.ecsv").meta
    assert header["sampler"]["kind"] == "breakout" and "uvex_transients." not in str(header["sampler"])

    loaded = Experiment.load(tmp_path / "points.ecsv", schedule=schedule)
    assert isinstance(loaded.sampler, BreakoutSampler)
    np.testing.assert_array_equal(loaded.phase_map.points, experiment.phase_map.points)
    assert loaded.phase_map.space.names == ("log_L0", "log_t0")

    grid = loaded.grid_table(5, rate=10.0 / u.Gpc**3 / u.yr)
    assert {"f", "z_limit", "volume_time", "recovered_volume_time", "expected_recoveries"} <= set(grid.colnames)
    np.testing.assert_allclose(grid["expected_recoveries"], grid["f"] * 10.0e-9 * grid["volume_time"], rtol=1e-10)
    figure = plot_grid(grid, loaded.phase_map.table)
    labels = {ax.get_xlabel() for ax in figure.axes}
    assert r"time scale $t_0$ [s]" in labels
