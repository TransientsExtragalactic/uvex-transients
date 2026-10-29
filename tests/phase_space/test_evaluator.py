"""Tests for `RedshiftLimit` and `RecoveryEvaluator`."""

import numpy as np
import pytest
from astropy import units as u
from scipy.integrate import quad

from uvex_transients.phase_space.evaluator import RecoveryEvaluator, RedshiftLimit
from uvex_transients.phase_space.transient import FREQUENCY_MAX
from uvex_transients.simulation.core import SurveySimulator
from uvex_transients.utils.cosmology import get_cosmology

NSIDE = 32
TIME_BINS = 3
MISSION = "uvex_fast"

# The FUV equivalent bandwidth is about 10**14.55 Hz, so L_band is about 14.55 dex above L_nu.
BW = 14.55


def _magnitude(log_luminosity, redshift, spectral_index=0.0):
    """The AB magnitude of a source with L_nu proportional to nu^alpha, at the frequency its luminosity is quoted at."""
    d_l = get_cosmology(None).luminosity_distance(redshift).to_value(u.cm)
    return -2.5 * np.log10((1 + redshift) ** (1 + spectral_index) * 10.0**log_luminosity / (4 * np.pi * d_l**2)) - 48.6


# ---------------------------------------------------------------------------- #
# RedshiftLimit                                                                #
# ---------------------------------------------------------------------------- #
@pytest.mark.parametrize("spectral_index", [0.0, -1.0, -2.5, 0.5])
@pytest.mark.parametrize("mag_limit", [24.5, 26.0])
def test_the_source_reaches_the_depth_at_the_redshift_limit(mag_limit, spectral_index):
    limit = RedshiftLimit(mag_limit, spectral_index=spectral_index)
    log_luminosity = np.linspace(26.0, 28.0, 6)
    redshift = limit(log_luminosity)
    assert np.all(limit.z_min < redshift) and np.all(redshift < limit.z_max)
    np.testing.assert_allclose(_magnitude(log_luminosity, redshift, spectral_index), mag_limit, atol=0.01)


def test_a_redder_source_is_seen_less_far_at_the_same_luminosity():
    # The K-correction (1+z)^(1+alpha) brightens a source of alpha > -1 and dims one of alpha < -1.
    log_luminosity = 28.0
    steep, neutral, flat = (RedshiftLimit(26.0, spectral_index=alpha)(log_luminosity) for alpha in (-2.0, -1.0, 0.0))
    assert steep < neutral < flat


def test_a_spectrum_too_blue_to_have_a_redshift_limit_raises():
    with pytest.raises(ValueError, match="too blue"):
        RedshiftLimit(24.5, spectral_index=4.0)


def test_a_brighter_source_is_visible_farther():
    limit = RedshiftLimit(26.0)
    below_the_cap = limit(np.linspace(26.0, 29.0, 20))
    assert np.all(np.diff(below_the_cap) > 0)
    # Past the cap the limit stops growing, but never falls.
    assert np.all(np.diff(limit(np.linspace(26.0, 31.0, 40))) >= 0)


def test_a_deeper_limit_reaches_farther():
    log_luminosity = np.array([27.0, 28.0])
    assert np.all(RedshiftLimit(27.0)(log_luminosity) > RedshiftLimit(25.0)(log_luminosity))


def test_a_luminosity_beyond_the_table_is_clipped_and_flagged():
    limit = RedshiftLimit(26.0, z_max=5.0)
    assert limit(40.0) == pytest.approx(5.0)
    assert limit.is_clipped(40.0)
    assert limit.is_clipped(10.0)
    assert not limit.is_clipped(28.0)
    assert limit.log_luminosity_max > 28.0
    assert limit(limit.log_luminosity_max) == pytest.approx(5.0)


def test_a_bandwidth_moves_the_axis_from_l_nu_to_l_band():
    bandwidth = 3.5e14 * u.Hz
    spectral, banded = RedshiftLimit(24.5), RedshiftLimit(24.5, bandwidth=bandwidth)
    log_nu = np.linspace(27.0, 28.5, 5)
    np.testing.assert_allclose(banded(log_nu + np.log10(3.5e14)), spectral(log_nu))
    assert banded.log_luminosity_max == pytest.approx(spectral.log_luminosity_max + np.log10(3.5e14))


# ---------------------------------------------------------------------------- #
# RecoveryEvaluator                                                            #
# ---------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def schedule(make_schedule):
    return make_schedule(n_sched=20)


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


def test_a_point_reports_its_counts_and_scale(schedule):
    result = make_evaluator(schedule)([29.5 + BW, 1.5])
    assert set(result) == {"k", "n", "z_limit", "volume_time", "coverage", "n_chunks"}
    assert 0 <= result["k"] <= result["n"]
    assert result["n"] == 200 * result["n_chunks"]
    assert result["volume_time"] > 0
    assert 0.0 < result["coverage"] <= 1.0
    assert 0.001 <= result["z_limit"] <= 10.0


def test_a_bright_long_event_is_recovered_in_the_first_chunk(schedule):
    # Every event lands in the footprint of 20 observations, so one chunk reaches `k_target`.
    result = make_evaluator(schedule)([30.0 + BW, 2.0])
    assert result["k"] >= 5
    assert result["n_chunks"] == 1


def test_a_hopeless_event_is_topped_up_to_n_max(schedule):
    result = make_evaluator(schedule)([24.0 + BW, -1.0])
    assert result["k"] == 0
    assert result["n"] == 600
    assert result["n_chunks"] == 3


def test_the_same_seed_repeats_and_another_differs(schedule):
    first = make_evaluator(schedule, seed=7)([29.0 + BW, 1.0])
    again = make_evaluator(schedule, seed=7)([29.0 + BW, 1.0])
    other = make_evaluator(schedule, seed=8, n_max=2000, chunk_size=1000, k_target=10_000)([29.0 + BW, 1.0])
    assert first == again
    assert other["n"] == 2000


def test_each_band_has_its_own_luminosity_axis(schedule):
    fuv, nuv = make_evaluator(schedule, band="FUV"), make_evaluator(schedule, band="NUV")
    assert fuv.band == "FUV" and fuv.settings["band"] == "FUV"
    assert fuv.bandwidth > nuv.bandwidth > 1e14 * u.Hz
    # The same luminosity in a narrower band is a higher L_nu, so it is seen farther.
    assert nuv.redshift_limit(43.0) > fuv.redshift_limit(43.0)
    for evaluator in (fuv, nuv):
        result = evaluator([28.5 + BW, 1.5])
        assert 0 <= result["k"] <= result["n"]


def test_an_unknown_band_raises(schedule):
    with pytest.raises(ValueError, match="Unknown band"):
        make_evaluator(schedule, band="Q")


def test_the_table_meta_labels_the_axis(schedule):
    evaluator = make_evaluator(schedule)
    meta = evaluator.table_meta
    assert meta["band"] == "FUV" and meta["luminosity_unit"] == "erg/s"
    assert meta["bandwidth_hz"] == pytest.approx(evaluator.bandwidth.to_value(u.Hz))
    # M = Z - 2.5 log10 L_band, with Z set by the distance of 10 pc and the bandwidth.
    luminosity = 1e43 * u.erg / u.s
    spectral = luminosity / evaluator.bandwidth
    absolute = (
        -2.5 * np.log10(spectral.to_value(u.erg / u.s / u.Hz) / (4 * np.pi * (10 * u.pc).to_value(u.cm) ** 2)) - 48.6
    )
    assert meta["absolute_magnitude_zero_point"] - 2.5 * 43.0 == pytest.approx(absolute, abs=1e-6)


def test_the_volume_time_matches_an_independent_calculation(schedule):
    evaluator = make_evaluator(schedule)
    log_luminosity, log_half_time = 28.5 + BW, 1.0
    z_limit = float(evaluator.redshift_limit(log_luminosity))

    # The exposure straight from the simulator at the window length of this point, and the volume from quadrature.
    transient = evaluator.make_transient()
    transient.set_point(10.0**log_luminosity * u.erg / u.s / evaluator.bandwidth, 10.0**log_half_time * u.day, z_limit)
    simulator = SurveySimulator(schedule, {"canonical": transient})
    exposure = simulator.compute_effective_exposure(TIME_BINS, nside=NSIDE, order="nested").total_effective_exposure
    cosmology = evaluator.cosmology
    volume, _ = quad(
        lambda z: cosmology.differential_comoving_volume(z).to_value(u.Mpc**3 / u.sr) / (1 + z), 0.0, z_limit
    )
    expected = exposure["canonical"].to_value(u.sr * u.yr) * volume

    # The evaluator interpolates the exposure between windows 0.1 dex apart.
    assert float(evaluator.volume_time(log_luminosity, log_half_time)) == pytest.approx(expected, rel=0.05)


def test_the_volume_time_grows_with_luminosity_and_broadcasts(schedule):
    evaluator = make_evaluator(schedule)
    log_luminosity = np.linspace(26.5, 28.5, 5) + BW
    values = evaluator.volume_time(log_luminosity, 1.0)
    assert values.shape == (5,)
    assert np.all(np.diff(values) > 0)
    assert evaluator.volume_time(np.full((2, 3), 28.0 + BW), np.full((2, 3), 0.5)).shape == (2, 3)


@pytest.mark.parametrize("kwargs", [{"chunk_size": 0}, {"k_target": 0}, {"n_max": 10, "chunk_size": 100}])
def test_invalid_settings_raise(schedule, kwargs):
    with pytest.raises(ValueError):
        make_evaluator(schedule, **kwargs)


def test_coverage_is_the_exposure_over_the_whole_sky_and_survey(schedule):
    evaluator = make_evaluator(schedule)
    log_luminosity, log_half_time = 28.5 + BW, 1.0
    z_limit = float(evaluator.redshift_limit(log_luminosity))

    transient = evaluator.make_transient()
    transient.set_point(10.0**log_luminosity * u.erg / u.s / evaluator.bandwidth, 10.0**log_half_time * u.day, z_limit)
    simulator = SurveySimulator(schedule, {"canonical": transient})
    exposure = simulator.compute_effective_exposure(TIME_BINS, nside=NSIDE, order="nested").total_effective_exposure
    expected = exposure["canonical"].to_value(u.sr * u.day) / (4 * np.pi * schedule.duration.to_value(u.day))

    coverage = float(evaluator.coverage(log_luminosity, log_half_time))
    assert 0.0 < coverage <= 1.0
    assert coverage == pytest.approx(expected, rel=0.05)


def test_the_all_sky_volume_time_is_the_volume_time_over_the_coverage(schedule):
    evaluator = make_evaluator(schedule)
    log_luminosity = np.linspace(27.0, 29.5, 4) + BW
    log_half_time = np.full(4, 1.0)
    np.testing.assert_allclose(
        evaluator.all_sky_volume_time(log_luminosity),
        evaluator.volume_time(log_luminosity, log_half_time) / evaluator.coverage(log_luminosity, log_half_time),
        rtol=1e-12,
    )


def test_a_bright_event_cut_reaches_a_few_megaparsecs_without_being_clipped(schedule):
    # At 20 AB a source of L_band = 1e38 erg/s is seen only to about 2.5 Mpc, below where a coarser table would stop.
    evaluator = make_evaluator(schedule, mag_limit=20.0)
    log_band = 38.0
    assert not evaluator.redshift_limit.is_clipped(log_band)
    z_limit = float(evaluator.redshift_limit(log_band))
    assert z_limit < 1e-3
    log_nu = log_band - np.log10(evaluator.bandwidth.to_value(u.Hz))
    assert _magnitude(log_nu, z_limit, evaluator.spectral_index) == pytest.approx(20.0, abs=0.02)
    # The volume out to there is the cube law, not an interpolation across a coarse grid.
    cosmology = evaluator.cosmology
    volume, _ = quad(
        lambda z: cosmology.differential_comoving_volume(z).to_value(u.Mpc**3 / u.sr) / (1 + z), 0.0, z_limit
    )
    expected = 4 * np.pi * volume * schedule.duration.to_value(u.yr)
    assert float(evaluator.all_sky_volume_time(log_band)) == pytest.approx(expected, rel=2e-3)


def test_the_all_sky_volume_time_matches_the_comoving_volume(schedule):
    evaluator = make_evaluator(schedule)
    log_luminosity = 28.0 + BW
    z_limit = float(evaluator.redshift_limit(log_luminosity))
    cosmology = evaluator.cosmology
    volume, _ = quad(
        lambda z: cosmology.differential_comoving_volume(z).to_value(u.Mpc**3 / u.sr) / (1 + z), 0.0, z_limit
    )
    expected = 4 * np.pi * volume * schedule.duration.to_value(u.yr)
    assert float(evaluator.all_sky_volume_time(log_luminosity)) == pytest.approx(expected, rel=1e-3)


def test_the_efficiency_is_the_recovery_times_the_coverage_and_never_exceeds_it(schedule):
    evaluator = make_evaluator(schedule)
    log_luminosity, log_half_time = np.array([27.5, 28.5]) + BW, np.array([0.5, 1.5])
    recovery = np.array([0.2, 1.0])
    efficiency = evaluator.efficiency(recovery, log_luminosity, log_half_time)
    np.testing.assert_allclose(efficiency, recovery * evaluator.coverage(log_luminosity, log_half_time))
    assert efficiency[1] <= evaluator.coverage(log_luminosity[1], log_half_time[1])


def test_the_spectral_index_is_recorded_and_reaches_every_piece(schedule):
    evaluator = make_evaluator(schedule, spectral_index=-2.0)
    assert evaluator.settings["spectral_index"] == -2.0
    assert evaluator.table_meta["spectral_index"] == -2.0
    assert evaluator.redshift_limit.spectral_index == -2.0
    transient = evaluator.make_transient()
    assert transient.spectral_index == -2.0
    assert transient.reference_frequency == evaluator.reference_frequency
    # The default is not flat, and a different index changes the bandwidth that converts L_band to L_nu.
    assert make_evaluator(schedule).spectral_index == -2.0
    assert evaluator.bandwidth != make_evaluator(schedule, spectral_index=0.0).bandwidth


def test_the_redshift_table_ends_where_the_band_leaves_the_spectrum(schedule):
    for band in ("FUV", "NUV"):
        evaluator = make_evaluator(schedule, band=band)
        edge = FREQUENCY_MAX.to_value(u.Hz) / evaluator.reference_frequency.to_value(u.Hz) - 1.0
        assert evaluator.redshift_edge == pytest.approx(edge)
        assert evaluator.redshift_limit.z_max == evaluator.redshift_edge > 20.0
        assert "z_max" not in evaluator.settings
    # Fainter than anything the map reaches, the limit comes from the depth alone and is nowhere near the edge.
    evaluator = make_evaluator(schedule)
    assert not evaluator.redshift_limit.is_clipped(45.0)
    assert evaluator.redshift_limit(44.0) < 0.2 * evaluator.redshift_edge


def test_the_light_curve_shape_is_recorded_and_reaches_the_transient(schedule):
    evaluator = make_evaluator(schedule, rise_ratio=0.2, fade_mag=7.0)
    assert evaluator.settings["rise_ratio"] == 0.2 and evaluator.settings["fade_mag"] == 7.0
    transient = evaluator.make_transient()
    assert transient.rise_ratio == 0.2 and transient.fade_mag == 7.0
    assert evaluator._transient.window_factor == transient.window_factor
    # A longer fade means a longer window, so a larger footprint.
    assert make_evaluator(schedule, fade_mag=8.0)._transient.window_factor > evaluator._transient.window_factor


def test_settings_of_a_top_hat_map_are_refused(schedule):
    settings = make_evaluator(schedule).settings
    del settings["rise_ratio"]
    with pytest.raises(ValueError, match="top-hat"):
        RecoveryEvaluator.from_settings(settings, schedule=schedule, mission=MISSION)


def test_an_index_with_no_redshift_limit_is_refused_by_the_evaluator(schedule):
    with pytest.raises(ValueError, match="too blue"):
        make_evaluator(schedule, spectral_index=4.0)
