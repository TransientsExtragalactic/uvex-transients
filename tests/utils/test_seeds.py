"""Tests for `spawn_seeds`."""

import numpy as np
import pytest

from uvex_transients.utils import spawn_seeds


def _sequence(already_spawned=0):
    sequence = np.random.SeedSequence(1234)
    sequence.spawn(already_spawned)
    return sequence


@pytest.mark.parametrize("already_spawned", [0, 3])
def test_indices_match_the_same_positions_of_the_full_set(already_spawned):
    full = spawn_seeds(_sequence(already_spawned), 500)
    indices = np.array([0, 7, 8, 250, 499])
    np.testing.assert_array_equal(spawn_seeds(_sequence(already_spawned), 500, indices=indices), full[indices])


def test_empty_indices():
    assert spawn_seeds(_sequence(), 10, indices=[]).shape == (0,)


@pytest.mark.parametrize("indices", [[-1], [10]])
def test_out_of_range_indices_raise(indices):
    with pytest.raises(IndexError):
        spawn_seeds(_sequence(), 10, indices=indices)
