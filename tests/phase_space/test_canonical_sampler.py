"""Tests for `RedshiftLimit` and the physics of `CanonicalSampler`: the scales of a point and how a point is measured."""

import numpy as np
import pytest
from astropy import units as u
from scipy.integrate import quad

from uvex_transients.phase_space.sampling import CanonicalSampler, RedshiftLimit, StoppingRule
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
# CanonicalSampler                                                             #
# ---------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def schedule(make_schedule):
    return make_schedule(n_sched=20)


def make_sampler(schedule, **kwargs):
    defaults = {"mission": MISSION, "nside": NSIDE, "order": "nested", "time_bins": TIME_BINS, "n_visits": 1}
    defaults.update(kwargs)
    return CanonicalSampler(schedule, **defaults)


def measure(sampler, x, seed=3, **rule):
    """Measure a point with a small stopping rule, as a campaign does, and return the outcome."""
    rule = {"chunk_size": 200, "n_max": 600, "k_target": 5, **rule}
    return StoppingRule(**rule).run(sampler, x, np.random.SeedSequence(seed))


def points(log_luminosity, log_half_time):
    """Stack two coordinates, broadcast against each other, into points of the space."""
    return np.stack(np.broadcast_arrays(log_luminosity, log_half_time), axis=-1)


def test_a_point_reports_its_counts(schedule):
    outcome = measure(make_sampler(schedule), [29.5 + BW, 1.5])
    assert 0 <= outcome.k <= outcome.n
    assert outcome.n == 200 * outcome.extras["n_chunks"]


def test_a_bright_long_event_is_recovered_in_the_first_chunk(schedule):
    # Every event lands in the footprint of 20 observations, so one chunk reaches `k_target`.
    outcome = measure(make_sampler(schedule), [30.0 + BW, 2.0])
    assert outcome.k >= 5
    assert outcome.extras["n_chunks"] == 1


def test_a_hopeless_event_is_topped_up_to_n_max(schedule):
    outcome = measure(make_sampler(schedule), [24.0 + BW, -1.0])
    assert (outcome.k, outcome.n, outcome.extras["n_chunks"]) == (0, 600, 3)


def test_the_same_seed_repeats_and_another_differs(schedule):
    sampler = make_sampler(schedule)
    first, again = (measure(sampler, [29.0 + BW, 1.0], seed=7) for _ in range(2))
    assert first == again
    other = measure(sampler, [29.0 + BW, 1.0], seed=8, n_max=2000, chunk_size=1000, k_target=10_000)
    assert other.n == 2000


def test_each_band_has_its_own_luminosity_axis(schedule):
    fuv, nuv = make_sampler(schedule, band="FUV"), make_sampler(schedule, band="NUV")
    assert fuv.band == "FUV" and fuv.settings["band"] == "FUV"
    assert fuv.bandwidth > nuv.bandwidth > 1e14 * u.Hz
    # The same luminosity in a narrower band is a higher L_nu, so it is seen farther.
    assert nuv._limit(43.0) > fuv._limit(43.0)
    for sampler in (fuv, nuv):
        outcome = measure(sampler, [28.5 + BW, 1.5])
        assert 0 <= outcome.k <= outcome.n


def test_an_unknown_band_raises(schedule):
    with pytest.raises(ValueError, match="Unknown band"):
        make_sampler(schedule, band="Q")


def test_the_table_meta_labels_the_axis(schedule):
    sampler = make_sampler(schedule)
    meta = sampler.table_meta
    assert meta["band"] == "FUV" and meta["luminosity_unit"] == "erg/s"
    assert meta["bandwidth_hz"] == pytest.approx(sampler.bandwidth.to_value(u.Hz))
    # The shape of the canonical transient, which `difference_map` requires two maps to share
    assert meta["rise_ratio"] == sampler.settings["rise_ratio"] and meta["fade_mag"] == sampler.settings["fade_mag"]
    # M = Z - 2.5 log10 L_band, with Z set by the distance of 10 pc and the bandwidth.
    luminosity = 1e43 * u.erg / u.s
    spectral = luminosity / sampler.bandwidth
    absolute = (
        -2.5 * np.log10(spectral.to_value(u.erg / u.s / u.Hz) / (4 * np.pi * (10 * u.pc).to_value(u.cm) ** 2)) - 48.6
    )
    assert meta["absolute_magnitude_zero_point"] - 2.5 * 43.0 == pytest.approx(absolute, abs=1e-6)


def _exposure(sampler, schedule, log_luminosity, log_half_time):
    """The exposure straight from the simulator at the window length of a point, and the point's redshift limit."""
    z_limit = float(sampler._limit(log_luminosity))
    transient = sampler.make_transient()
    transient.set_point(10.0**log_luminosity * u.erg / u.s / sampler.bandwidth, 10.0**log_half_time * u.day, z_limit)
    simulator = SurveySimulator(schedule, {"canonical": transient})
    exposure = simulator.compute_effective_exposure(TIME_BINS, nside=NSIDE, order="nested").total_effective_exposure
    return exposure["canonical"], z_limit


def _volume(sampler, z_limit):
    """The comoving volume out to a redshift per steradian, from quadrature."""
    volume, _ = quad(
        lambda z: sampler.cosmology.differential_comoving_volume(z).to_value(u.Mpc**3 / u.sr) / (1 + z), 0.0, z_limit
    )
    return volume


def test_the_volume_time_matches_an_independent_calculation(schedule):
    sampler = make_sampler(schedule)
    log_luminosity, log_half_time = 28.5 + BW, 1.0
    exposure, z_limit = _exposure(sampler, schedule, log_luminosity, log_half_time)
    expected = exposure.to_value(u.sr * u.yr) * _volume(sampler, z_limit)

    # The sampler interpolates the exposure between windows 0.1 dex apart.
    assert float(sampler.volume_time(points(log_luminosity, log_half_time))) == pytest.approx(expected, rel=0.05)


def test_the_volume_time_grows_with_luminosity_and_broadcasts(schedule):
    sampler = make_sampler(schedule)
    values = sampler.volume_time(points(np.linspace(26.5, 28.5, 5) + BW, 1.0))
    assert values.shape == (5,)
    assert np.all(np.diff(values) > 0)
    assert sampler.volume_time(points(np.full((2, 3), 28.0 + BW), np.full((2, 3), 0.5))).shape == (2, 3)


@pytest.mark.parametrize("kwargs", [{"chunk_size": 0}, {"k_target": 0}, {"n_max": 10, "chunk_size": 100}])
def test_an_invalid_rule_raises(kwargs):
    with pytest.raises(ValueError):
        StoppingRule(**{"chunk_size": 200, "n_max": 600, "k_target": 5, **kwargs})


def test_coverage_is_the_exposure_over_the_whole_sky_and_survey(schedule):
    sampler = make_sampler(schedule)
    log_luminosity, log_half_time = 28.5 + BW, 1.0
    exposure, _ = _exposure(sampler, schedule, log_luminosity, log_half_time)
    # The exposure straight from the simulator, in sr yr, is the one the coverage is made from
    expected = exposure.to_value(u.sr * u.day) / (4 * np.pi * schedule.duration.to_value(u.day))

    coverage = float(sampler.coverage(points(log_luminosity, log_half_time)))
    assert 0.0 < coverage <= 1.0
    assert coverage == pytest.approx(expected, rel=0.05)


def test_the_all_sky_volume_time_is_the_volume_time_over_the_coverage(schedule):
    sampler = make_sampler(schedule)
    x = points(np.linspace(27.0, 29.5, 4) + BW, 1.0)
    np.testing.assert_allclose(sampler.all_sky_volume_time(x), sampler.volume_time(x) / sampler.coverage(x), rtol=1e-12)


def test_a_bright_event_cut_reaches_a_few_megaparsecs_without_being_clipped(schedule):
    # At 20 AB a source of L_band = 1e38 erg/s is seen only to about 2.5 Mpc, below where a coarser table would stop.
    sampler = make_sampler(schedule, mag_limit=20.0)
    log_band = 38.0
    assert not sampler._limit.is_clipped(log_band)
    z_limit = float(sampler._limit(log_band))
    assert z_limit < 1e-3
    log_nu = log_band - np.log10(sampler.bandwidth.to_value(u.Hz))
    assert _magnitude(log_nu, z_limit, sampler.spectral_index) == pytest.approx(20.0, abs=0.02)
    # The volume out to there is the cube law, not an interpolation across a coarse grid.
    expected = 4 * np.pi * _volume(sampler, z_limit) * schedule.duration.to_value(u.yr)
    assert float(sampler.all_sky_volume_time(points(log_band, 0.0))) == pytest.approx(expected, rel=2e-3)


def test_the_all_sky_volume_time_matches_the_comoving_volume(schedule):
    sampler = make_sampler(schedule)
    log_luminosity = 28.0 + BW
    z_limit = float(sampler._limit(log_luminosity))
    expected = 4 * np.pi * _volume(sampler, z_limit) * schedule.duration.to_value(u.yr)
    assert float(sampler.all_sky_volume_time(points(log_luminosity, 0.0))) == pytest.approx(expected, rel=1e-3)


def test_the_spectral_index_is_recorded_and_reaches_every_piece(schedule):
    sampler = make_sampler(schedule, spectral_index=-2.0)
    assert sampler.settings["spectral_index"] == -2.0
    assert sampler.table_meta["spectral_index"] == -2.0
    assert sampler._limit.spectral_index == -2.0
    transient = sampler.make_transient()
    assert transient.spectral_index == -2.0
    assert transient.reference_frequency == sampler.reference_frequency
    # The default is not flat, and a different index changes the bandwidth that converts L_band to L_nu.
    assert make_sampler(schedule).spectral_index == -2.0
    assert sampler.bandwidth != make_sampler(schedule, spectral_index=0.0).bandwidth


def test_the_redshift_table_ends_where_the_band_leaves_the_spectrum(schedule):
    for band in ("FUV", "NUV"):
        sampler = make_sampler(schedule, band=band)
        edge = FREQUENCY_MAX.to_value(u.Hz) / sampler.reference_frequency.to_value(u.Hz) - 1.0
        assert sampler.redshift_edge == pytest.approx(edge)
        assert sampler._limit.z_max == sampler.redshift_edge > 20.0
        assert "z_max" not in sampler.settings
    # Fainter than anything the map reaches, the limit comes from the depth alone and is nowhere near the edge.
    sampler = make_sampler(schedule)
    assert not sampler._limit.is_clipped(45.0)
    assert sampler._limit(44.0) < 0.2 * sampler.redshift_edge


def test_the_light_curve_shape_is_recorded_and_reaches_the_transient(schedule):
    sampler = make_sampler(schedule, rise_ratio=0.2, fade_mag=7.0)
    assert sampler.settings["rise_ratio"] == 0.2 and sampler.settings["fade_mag"] == 7.0
    transient = sampler.make_transient()
    assert transient.rise_ratio == 0.2 and transient.fade_mag == 7.0
    assert sampler._transient.window_factor == transient.window_factor
    # A longer fade means a longer window, so a larger footprint.
    assert make_sampler(schedule, fade_mag=8.0)._transient.window_factor > sampler._transient.window_factor


def test_an_index_with_no_redshift_limit_is_refused(schedule):
    with pytest.raises(ValueError, match="too blue"):
        make_sampler(schedule, spectral_index=4.0)
