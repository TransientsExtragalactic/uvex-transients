"""Tests for the Wilks tolerance-limit helpers and `SpectralModel.get_observability_curve`."""

import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable
from synphot import SpectralElement
from synphot.models import Box1D

from uvex_transients.models._utils import to_cgs_value
from uvex_transients.models.core._tolerance import (
    first_crossing,
    min_samples,
    reduce_bands,
    resolve_bandpasses,
    resolve_time_grid,
    tolerance_limit,
    wilks_rank,
)
from uvex_transients.models.kilonovae import KilonovaCoolingBlackbodySED
from uvex_transients.utils.cosmology import resolve_cosmological_distances


# =========================================================================== #
# Pure helpers                                                                #
# =========================================================================== #
class TestReduceBands:
    """`reduce_bands` combines per-band peaks under each detection criterion."""

    PEAKS = np.array([[20.0, 22.0, 21.0], [25.0, 19.0, 23.0]])

    def test_any_is_min(self):
        np.testing.assert_array_equal(reduce_bands(self.PEAKS, "any"), [20.0, 19.0])

    def test_all_is_max(self):
        np.testing.assert_array_equal(reduce_bands(self.PEAKS, "all"), [22.0, 25.0])

    def test_k_of_n_is_kth_smallest(self):
        np.testing.assert_array_equal(reduce_bands(self.PEAKS, "k_of_n", k=2), [21.0, 23.0])

    @pytest.mark.parametrize("k", [None, 0, 4])
    def test_k_of_n_validates_k(self, k):
        with pytest.raises(ValueError, match="'k'"):
            reduce_bands(self.PEAKS, "k_of_n", k=k)

    def test_unknown_criterion(self):
        with pytest.raises(ValueError, match="criterion"):
            reduce_bands(self.PEAKS, "most")


class TestWilks:
    """Rank selection and coverage of the distribution-free limit."""

    def test_min_samples_matches_textbook_value(self):
        assert min_samples(0.95, 0.01) == 299

    def test_maximum_is_valid_exactly_at_the_minimum(self):
        assert wilks_rank(299, 0.95, 0.01) == 1
        with pytest.raises(ValueError, match="at least 299"):
            wilks_rank(298, 0.95, 0.01)

    @pytest.mark.parametrize("confidence, tolerance", [(0.0, 0.1), (1.0, 0.1), (0.9, 0.0), (0.9, 1.0)])
    def test_probabilities_are_validated(self, confidence, tolerance):
        with pytest.raises(ValueError):
            wilks_rank(1000, confidence, tolerance)

    def test_rank_grows_with_samples(self):
        assert wilks_rank(1000, 0.95, 0.01) > wilks_rank(500, 0.95, 0.01) >= 1

    @pytest.mark.parametrize(
        "n_samples, confidence, tolerance", [(299, 0.95, 0.01), (1000, 0.95, 0.01), (200, 0.9, 0.05)]
    )
    def test_empirical_coverage_meets_confidence(self, n_samples, confidence, tolerance):
        """The r-th largest of M uniform draws leaves at most `tolerance` beyond it, `confidence` of the time."""
        rng = np.random.default_rng(1)
        rank = wilks_rank(n_samples, confidence, tolerance)
        draws = np.sort(rng.random((4000, n_samples)), axis=1)[:, n_samples - rank]
        coverage = np.mean(1.0 - draws <= tolerance)
        # Binomial standard error of 4000 trials is under 0.01.
        assert coverage >= confidence - 0.015

    def test_tolerance_limit_picks_the_rth_largest(self):
        z_det = np.random.default_rng(2).random(1000)
        z_limit, rank, n_censored = tolerance_limit(z_det, 0.95, 0.01)
        assert z_limit == np.sort(z_det)[-rank]
        assert n_censored == 0

    def test_censored_draws_are_counted_and_propagate_when_they_set_the_limit(self):
        z_det = np.random.default_rng(2).random(400)
        z_det[:5] = np.inf
        z_limit, rank, n_censored = tolerance_limit(z_det, 0.95, 0.01)
        assert rank == 1
        assert np.isinf(z_limit)
        assert n_censored == 5


class TestFirstCrossing:
    """`first_crossing` finds each draw's detection redshift on the grid."""

    Z = np.array([0.1, 0.2, 0.4, 0.8])

    def test_returns_upper_bracket(self):
        m = np.array([[20.0, 22.0, 24.0, 26.0]])
        assert first_crossing(m, self.Z, 23.0)[0] == 0.4

    def test_only_first_crossing_counts(self):
        """A K-correction wiggle that dips back below the limit is ignored."""
        m = np.array([[20.0, 24.0, 22.0, 26.0]])
        assert first_crossing(m, self.Z, 23.0)[0] == 0.2

    def test_nan_counts_as_not_detected(self):
        m = np.array([[20.0, np.nan, 22.0, 26.0], [np.nan, np.nan, np.nan, np.nan]])
        np.testing.assert_array_equal(first_crossing(m, self.Z, 23.0), [0.2, 0.1])

    def test_never_crossing_is_inf(self):
        m = np.array([[20.0, 21.0, 22.0, 22.5]])
        assert np.isinf(first_crossing(m, self.Z, 23.0)[0])

    def test_faint_at_first_grid_point_is_first_grid_point(self):
        m = np.array([[30.0, 31.0, 32.0, 33.0]])
        assert first_crossing(m, self.Z, 23.0)[0] == 0.1

    def test_interpolation_lies_between_the_bracket(self):
        m = np.array([[20.0, 22.0, 24.0, 26.0]])
        z = first_crossing(m, self.Z, 23.0, interpolate=True)[0]
        assert 0.2 < z < 0.4
        np.testing.assert_allclose(z, np.expm1(0.5 * (np.log1p(0.2) + np.log1p(0.4))))


class TestInputResolution:
    """`resolve_time_grid` and `resolve_bandpasses` normalise user-facing arguments."""

    def test_log_time_grid_in_seconds(self):
        t = resolve_time_grid(None, 1 * u.day, 100 * u.day, 3)
        np.testing.assert_allclose(t, np.array([1.0, 10.0, 100.0]) * 86400.0)

    def test_explicit_grid_ignores_the_other_arguments(self):
        t = resolve_time_grid(2 * u.day, 5 * u.day, 1 * u.day, 7)
        np.testing.assert_allclose(t, [2 * 86400.0])

    @pytest.mark.parametrize(
        "t_min, t_max", [(None, 1 * u.day), (1 * u.day, None), (2 * u.day, 1 * u.day), (0 * u.day, 1 * u.day)]
    )
    def test_bad_time_grid_is_rejected(self, t_min, t_max):
        with pytest.raises(ValueError):
            resolve_time_grid(None, t_min, t_max, 10)

    def test_sequence_bands_are_named_by_position(self):
        band = SpectralElement(Box1D, amplitude=1.0, x_0=2300 * u.AA, width=500 * u.AA)
        names, grids = resolve_bandpasses([band, band])
        assert names == ("band0", "band1")
        assert len(grids) == 2
        nu, throughput = grids[0]
        assert nu.shape == throughput.shape

    def test_empty_bands_are_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            resolve_bandpasses({})


# =========================================================================== #
# SpectralModel methods                                                       #
# =========================================================================== #
@pytest.fixture(scope="module")
def sed():
    """Build a kilonova SED with its default priors."""
    return KilonovaCoolingBlackbodySED()


@pytest.fixture(scope="module")
def bands():
    """Two simple top-hat bands (far- and near-UV-like)."""
    return {
        "FUV": SpectralElement(Box1D, amplitude=1.0, x_0=1550 * u.AA, width=300 * u.AA),
        "NUV": SpectralElement(Box1D, amplitude=1.0, x_0=2300 * u.AA, width=500 * u.AA),
    }


Z_GRID = np.geomspace(0.01, 0.5, 8)
TIME = {"t_min": 0.05 * u.day, "t_max": 30 * u.day, "n_time": 40}


@pytest.fixture(scope="module")
def grid(sed, bands):
    """Build a small shared effective-peak grid."""
    return sed.get_effective_peak_magnitudes(Z_GRID, bands, n_samples=40, rng=3, chunk_size=16, **TIME)


class TestEffectivePeakMagnitudes:
    """`SpectralModel.get_effective_peak_magnitudes`."""

    def test_shapes_and_metadata(self, grid):
        assert grid.m_eff.shape == (40, Z_GRID.size)
        assert grid.bands == ("FUV", "NUV")
        assert grid.n_samples == 40
        assert np.all(np.isfinite(grid.m_eff))

    def test_matches_direct_evaluation(self, sed, bands, grid):
        """A grid entry equals a hand-rolled min over time and band of `mag_band_cgs`."""
        i, j = 7, 3
        z = Z_GRID[j]
        d_l = resolve_cosmological_distances(redshift=z)["luminosity_distance"].cgs.value
        params = {name: to_cgs_value(value)[i] for name, value in grid.parameters.items()}
        peaks = []
        for bp in bands.values():
            wave = bp.waveset
            mags = sed.mag_band_cgs(
                wave.to_value(u.Hz, equivalencies=u.spectral()),
                bp(wave).to_value(u.dimensionless_unscaled),
                grid.t_rest * (1 + z),
                z,
                d_l,
                **params,
            )
            peaks.append(np.min(mags))
        np.testing.assert_allclose(grid.m_eff[i, j], min(peaks))

    def test_peak_gets_fainter_with_redshift_overall(self, grid):
        assert np.median(grid.m_eff[:, -1]) > np.median(grid.m_eff[:, 0])

    def test_is_invariant_to_chunk_size_and_reproducible(self, sed, bands, grid):
        other = sed.get_effective_peak_magnitudes(Z_GRID, bands, n_samples=40, rng=3, chunk_size=7, **TIME)
        np.testing.assert_allclose(other.m_eff, grid.m_eff)

    def test_band_criteria_are_ordered(self, sed, bands):
        kwargs = dict(n_samples=20, rng=3, **TIME)
        any_ = sed.get_effective_peak_magnitudes(Z_GRID, bands, criterion="any", **kwargs).m_eff
        all_ = sed.get_effective_peak_magnitudes(Z_GRID, bands, criterion="all", **kwargs).m_eff
        k2 = sed.get_effective_peak_magnitudes(Z_GRID, bands, criterion="k_of_n", k=2, **kwargs).m_eff
        assert np.all(any_ <= all_)
        np.testing.assert_allclose(k2, all_)

    @pytest.mark.parametrize("z_grid", [[0.0, 0.1], [0.2, 0.1], [[0.1, 0.2]]])
    def test_z_grid_must_be_positive_increasing_1d(self, sed, bands, z_grid):
        with pytest.raises(ValueError, match="z_grid"):
            sed.get_effective_peak_magnitudes(z_grid, bands, **TIME)

    def test_a_time_grid_is_required(self, sed, bands):
        with pytest.raises(ValueError, match="t_rest"):
            sed.get_effective_peak_magnitudes(Z_GRID, bands)

    def test_k_of_n_requires_k(self, sed, bands):
        with pytest.raises(ValueError, match="'k'"):
            sed.get_effective_peak_magnitudes(Z_GRID, bands, criterion="k_of_n", **TIME)

    def test_progress_flag_does_not_change_the_result(self, sed, bands, grid):
        quiet = sed.get_effective_peak_magnitudes(Z_GRID, bands, n_samples=40, rng=3, progress=False, **TIME)
        np.testing.assert_allclose(quiet.m_eff, grid.m_eff)

    def test_progress_bar_runs_over_every_chunk_and_redshift(self, sed, bands, capsys):
        sed.get_effective_peak_magnitudes(Z_GRID, bands, n_samples=10, chunk_size=5, rng=3, **TIME)
        # Two chunks of five draws, each stepped across every redshift.
        assert f"{2 * Z_GRID.size}/{2 * Z_GRID.size}" in capsys.readouterr().err

    def test_explicit_rest_frame_times(self, sed, bands):
        t_rest = np.array([0.5, 1.0, 2.0]) * u.day
        grid = sed.get_effective_peak_magnitudes(Z_GRID, bands, t_rest=t_rest, n_samples=5, rng=3)
        np.testing.assert_allclose(grid.t_rest, t_rest.to_value(u.s))


class TestObservabilityCurve:
    """`SpectralModel.get_observability_curve`."""

    def test_matches_brute_force_from_the_grid(self, sed, grid):
        limits = [22.0, 24.0, 26.0]
        table = sed.get_observability_curve(limits, grid=grid, confidence=0.5, tolerance=0.1)
        assert isinstance(table, QTable)
        for row, limit in zip(table, limits):
            z_det = np.array([Z_GRID[np.argmax(m > limit)] if np.any(m > limit) else np.inf for m in grid.m_eff])
            assert row["z_limit"] == np.sort(z_det)[-table.meta["rank"]]
            assert row["n_censored"] == np.isinf(z_det).sum()

    def test_horizon_grows_for_deeper_limits(self, sed, grid):
        table = sed.get_observability_curve([20.0, 23.0, 26.0, 29.0], grid=grid, confidence=0.5, tolerance=0.1)
        z_limit = np.asarray(table["z_limit"])
        assert np.all(z_limit[1:] >= z_limit[:-1])

    def test_interpolation_never_exceeds_the_conservative_default(self, sed, grid):
        kwargs = {"grid": grid, "confidence": 0.5, "tolerance": 0.1}
        rounded_up = sed.get_observability_curve([24.0], **kwargs)["z_limit"][0]
        interpolated = sed.get_observability_curve([24.0], interpolate=True, **kwargs)["z_limit"][0]
        assert interpolated <= rounded_up

    def test_short_grid_gives_inf_and_warns(self, sed, grid):
        table = sed.get_observability_curve([40.0], grid=grid, confidence=0.5, tolerance=0.1)
        assert np.isinf(table["z_limit"][0])
        assert table["n_censored"][0] > 0.8 * grid.n_samples

    def test_builds_its_own_grid(self, sed, bands):
        table = sed.get_observability_curve(
            25.0, z_grid=Z_GRID, bandpasses=bands, n_samples=30, rng=3, confidence=0.5, tolerance=0.1, **TIME
        )
        assert table.meta["n_samples"] == 30
        assert table.meta["bands"] == ("FUV", "NUV")

    def test_too_few_samples_fails_before_the_expensive_pass(self, sed, bands):
        with pytest.raises(ValueError, match="at least 299"):
            sed.get_observability_curve(25.0, z_grid=Z_GRID, bandpasses=bands, n_samples=100, **TIME)

    def test_precomputed_grid_rejects_grid_arguments(self, sed, grid):
        with pytest.raises(TypeError):
            sed.get_observability_curve(25.0, grid=grid, confidence=0.5, tolerance=0.1, n_samples=10)
