"""Tests for `uvex_transients.utils.keyed_noise`."""

import numpy as np
import pytest
from scipy import stats

from uvex_transients.utils.keyed_noise import keyed_standard_normal


class TestReproducibility:
    """A key's draw must not depend on which other keys are requested."""

    def test_same_key_same_value(self):
        a = keyed_standard_normal(12345, 7, 2)
        b = keyed_standard_normal(12345, 7, 2)
        assert a == b

    def test_subset_matches_full(self):
        obs = np.arange(1000)
        full = keyed_standard_normal(99, obs, 1)
        pick = np.array([5, 400, 999, 0])
        assert np.array_equal(keyed_standard_normal(99, pick, 1), full[pick])

    def test_order_independent(self):
        obs = np.arange(500)
        perm = np.random.default_rng(0).permutation(500)
        assert np.array_equal(keyed_standard_normal(3, obs[perm], 0), keyed_standard_normal(3, obs, 0)[perm])

    def test_batching_independent(self):
        obs = np.arange(600)
        whole = keyed_standard_normal(8, obs, 0)
        parts = np.concatenate([keyed_standard_normal(8, obs[i : i + 100], 0) for i in range(0, 600, 100)])
        assert np.array_equal(whole, parts)

    def test_broadcasts(self):
        seeds = np.arange(4)[:, np.newaxis]
        obs = np.arange(6)[np.newaxis, :]
        out = keyed_standard_normal(seeds, obs, 1)
        assert out.shape == (4, 6)
        assert out[2, 3] == keyed_standard_normal(2, 3, 1)

    def test_large_seed(self):
        big = 2**64 - 1
        assert np.isfinite(keyed_standard_normal(big, 0, 0))

    def test_negative_key_raises(self):
        with pytest.raises(OverflowError):
            keyed_standard_normal(-1, 0, 0)


class TestDistinctKeys:
    """Different keys, including swapped ones, must give different draws."""

    def test_swapped_seed_and_observation_differ(self):
        assert keyed_standard_normal(3, 5, 0) != keyed_standard_normal(5, 3, 0)

    def test_bands_differ(self):
        draws = keyed_standard_normal(1, 10, np.arange(8))
        assert len(np.unique(draws)) == 8

    def test_no_collisions_over_grid(self):
        seeds = np.arange(200)[:, np.newaxis, np.newaxis]
        obs = np.arange(200)[np.newaxis, :, np.newaxis]
        bands = np.arange(5)[np.newaxis, np.newaxis, :]
        out = keyed_standard_normal(seeds, obs, bands)
        assert len(np.unique(out)) == out.size


class TestStatistics:
    """Draws over many keys must look like independent standard normals."""

    N = 200_000

    def test_standard_normal(self):
        x = keyed_standard_normal(2024, np.arange(self.N), 0)
        assert abs(x.mean()) < 5 / np.sqrt(self.N)
        assert abs(x.std() - 1) < 5 / np.sqrt(2 * self.N)
        assert stats.kstest(x, "norm").pvalue > 1e-3

    def test_tails_are_finite(self):
        x = keyed_standard_normal(1, np.arange(self.N), 0)
        assert np.all(np.isfinite(x))

    def test_uncorrelated_across_neighbouring_observations(self):
        x = keyed_standard_normal(7, np.arange(self.N), 0)
        r = np.corrcoef(x[:-1], x[1:])[0, 1]
        assert abs(r) < 5 / np.sqrt(self.N)

    def test_uncorrelated_across_bands(self):
        obs = np.arange(self.N)
        a = keyed_standard_normal(7, obs, 0)
        b = keyed_standard_normal(7, obs, 1)
        assert abs(np.corrcoef(a, b)[0, 1]) < 5 / np.sqrt(self.N)

    def test_uncorrelated_across_neighbouring_seeds(self):
        obs = np.arange(self.N)
        a = keyed_standard_normal(100, obs, 0)
        b = keyed_standard_normal(101, obs, 0)
        assert abs(np.corrcoef(a, b)[0, 1]) < 5 / np.sqrt(self.N)

    def test_sequential_seeds_same_observation(self):
        x = keyed_standard_normal(np.arange(self.N), 0, 0)
        assert abs(x.mean()) < 5 / np.sqrt(self.N)
        assert stats.kstest(x, "norm").pvalue > 1e-3
