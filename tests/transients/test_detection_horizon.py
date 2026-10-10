"""Tests for `ExtragalacticTransient.get_detection_horizon`."""

import numpy as np
import pytest
from astropy import units as u
from synphot import SpectralElement
from synphot.models import Box1D

from uvex_transients.models.kilonovae import KilonovaCoolingBlackbodySED
from uvex_transients.transients.base import ExtragalacticTransient

BANDS = {"NUV": SpectralElement(Box1D, amplitude=1.0, x_0=2300 * u.AA, width=500 * u.AA)}
SMALL = {"n_samples": 40, "n_z": 8, "n_time": 20, "confidence": 0.5, "tolerance": 0.1, "progress": False}


class _Kilonovalike(ExtragalacticTransient):
    """A minimal population over the kilonova SED, for testing."""

    DEFAULT_MODEL = KilonovaCoolingBlackbodySED
    DEFAULT_DURATION = 30 * u.day
    DEFAULT_Z_LIM = 0.2

    @property
    def rate(self):
        return 1e-6 / (u.Mpc**3 * u.yr)

    def rate_shape(self, z):
        z = np.asarray(z)
        shape = np.ones_like(z, dtype=np.float64)
        return shape if z.ndim > 0 else shape.item()


def test_horizon_is_monotone_in_depth_and_reproducible():
    """Deeper limits reach farther, and a fixed `rng` reproduces the result."""
    kwargs = {"mag_limits": [22.0, 25.0], "bandpasses": BANDS, "rng": 1, **SMALL}
    curve, grid = _Kilonovalike().get_detection_horizon(**kwargs)
    again, _ = _Kilonovalike().get_detection_horizon(**kwargs)
    assert curve["z_limit"][0] <= curve["z_limit"][1]
    np.testing.assert_array_equal(curve["z_limit"], again["z_limit"])
    assert grid.n_samples == 40


def test_time_window_comes_from_the_duration():
    """The rest-frame time grid ends at `duration_limit`."""
    _, grid = _Kilonovalike().get_detection_horizon(25.0, BANDS, rng=1, **SMALL)
    np.testing.assert_allclose(grid.t_rest[-1], (30 * u.day).to_value(u.s))


def test_redshift_grid_is_extended_until_the_limit_is_bracketed():
    """A starting range too short for a deep limit is doubled rather than reporting `inf`."""
    curve, grid = _Kilonovalike().get_detection_horizon(27.0, BANDS, rng=1, z_max=0.05, **SMALL)
    assert np.isfinite(curve["z_limit"][0])
    assert grid.z_grid[-1] > 0.05


def test_extending_the_grid_keeps_the_same_population():
    """The grid's parameter draws do not depend on how many extensions were needed."""
    _, short = _Kilonovalike().get_detection_horizon(21.0, BANDS, rng=1, **SMALL)
    _, extended = _Kilonovalike().get_detection_horizon(27.0, BANDS, rng=1, z_max=0.05, **SMALL)
    for name in short.parameters:
        np.testing.assert_array_equal(short.parameters[name], extended.parameters[name])


def test_too_few_samples_for_the_confidence_is_rejected():
    """The sample maximum needs enough draws for the requested confidence."""
    with pytest.raises(ValueError, match="at least"):
        _Kilonovalike().get_detection_horizon(
            25.0, BANDS, n_samples=20, n_z=6, n_time=10, confidence=0.95, tolerance=0.01, progress=False
        )
