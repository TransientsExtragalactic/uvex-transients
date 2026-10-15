"""Tests for `ExtragalacticTransient.sample_events_on_healpix_grid`'s `downsample` option."""

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time

from uvex_transients.transients.TDEs import TidalDisruptionEvent

_KWARGS = {
    "t_start": Time("2030-01-01"),
    "duration": 30 * u.day,
    "pixel_ids": np.arange(0, 3000),
    "seed": 11,
}


@pytest.fixture(scope="module")
def transient():
    return TidalDisruptionEvent()


@pytest.fixture(scope="module")
def full(transient):
    table = transient.sample_events_on_healpix_grid(16, **_KWARGS)
    assert len(table) > 200  # enough events that the selection below is meaningful
    return table


@pytest.mark.parametrize("factor", [1, 3, 20])
def test_downsample_keeps_a_subset_of_the_full_sample(transient, full, factor):
    reduced = transient.sample_events_on_healpix_grid(16, downsample=factor, **_KWARGS)
    assert len(reduced) == -(-len(full) // factor)

    # Each kept event is the same event the full sample holds, seed and all, in the same order.
    positions = [int(np.flatnonzero(full["parameter_seed"] == seed)[0]) for seed in reduced["parameter_seed"]]
    assert positions == sorted(positions)
    kept = full[positions]
    for name in ("healpix_id", "healpix_dx", "healpix_dy", "redshift", "parameter_seed"):
        np.testing.assert_array_equal(np.asarray(reduced[name]), np.asarray(kept[name]))
    np.testing.assert_array_equal(reduced["coord"].ra.deg, kept["coord"].ra.deg)
    np.testing.assert_array_equal(reduced["t_explosion"].jd1, kept["t_explosion"].jd1)
    np.testing.assert_array_equal(reduced["t_explosion"].jd2, kept["t_explosion"].jd2)


def test_downsample_is_reproducible(transient):
    first = transient.sample_events_on_healpix_grid(16, downsample=5, **_KWARGS)
    second = transient.sample_events_on_healpix_grid(16, downsample=5, **_KWARGS)
    np.testing.assert_array_equal(first["parameter_seed"], second["parameter_seed"])


@pytest.mark.parametrize("bad", [0, -2, 1.5, True])
def test_bad_downsample_raises(transient, bad):
    with pytest.raises(ValueError, match="downsample"):
        transient.sample_events_on_healpix_grid(16, downsample=bad, **_KWARGS)
