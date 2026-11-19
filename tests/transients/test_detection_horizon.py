"""Tests for `ExtragalacticTransient.get_detection_horizon`."""

import numpy as np
import pytest
from astropy import units as u
from synphot import SpectralElement
from synphot.models import Box1D

from uvex_transients.missions import downsample_bandpass
from uvex_transients.models.kilonovae import KilonovaCoolingBlackbodySED
from uvex_transients.transients.base import ExtragalacticTransient

# A `Box1D` is tabulated on ~50,000 wavelengths, which every SED evaluation integrates over; downsampled it has ~60.
BANDS = {"NUV": downsample_bandpass(SpectralElement(Box1D, amplitude=1.0, x_0=2300 * u.AA, width=500 * u.AA))}
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


def test_the_horizon_does_not_depend_on_the_redshift_limit_it_is_used_to_set():
    """A class's `redshift_limit` is derived from its horizon, so the horizon cannot depend on it."""
    low, high = _Kilonovalike(), _Kilonovalike()
    low.redshift_limit, high.redshift_limit = 0.05, 3.0
    kwargs = {"mag_limits": [22.0, 25.0], "bandpasses": BANDS, "rng": 1, **SMALL}
    curve_low, grid_low = low.get_detection_horizon(**kwargs)
    curve_high, grid_high = high.get_detection_horizon(**kwargs)
    np.testing.assert_array_equal(curve_low["z_limit"], curve_high["z_limit"])
    np.testing.assert_array_equal(grid_low.z_grid, grid_high.z_grid)


def test_too_few_samples_for_the_confidence_is_rejected():
    """The sample maximum needs enough draws for the requested confidence."""
    with pytest.raises(ValueError, match="at least"):
        _Kilonovalike().get_detection_horizon(
            25.0, BANDS, n_samples=20, n_z=6, n_time=10, confidence=0.95, tolerance=0.01, progress=False
        )


# --------------------------------------------------------------------------- #
# The Lyman limit                                                             #
# --------------------------------------------------------------------------- #
def test_lyman_limit_redshift_is_set_by_the_blue_edge_of_the_reddest_band():
    """The Lyman limit reaches a band's blue edge at lambda_blue / 911.76 A - 1, and the reddest band is the last."""
    from uvex_transients.models.core.detection_horizon import lyman_limit_redshift

    box = lambda centre: SpectralElement(Box1D, amplitude=1.0, x_0=centre * u.AA, width=400 * u.AA)  # noqa: E731
    z_nuv = lyman_limit_redshift({"NUV": box(2300)})
    np.testing.assert_allclose(z_nuv, 2100.0 / 911.76 - 1.0, rtol=1e-3)
    assert lyman_limit_redshift({"FUV": box(1500), "NUV": box(2300)}) == pytest.approx(z_nuv)
    assert lyman_limit_redshift([box(1500)]) < z_nuv


def test_lyman_limit_redshift_rejects_a_bad_threshold():
    from uvex_transients.models.core.detection_horizon import lyman_limit_redshift

    with pytest.raises(ValueError, match="threshold"):
        lyman_limit_redshift(BANDS, threshold=1.5)


def test_the_horizon_is_capped_at_the_lyman_limit_by_default():
    """A population that stays detectable past the cap gets the cap, flagged, and the grid stops there."""
    from uvex_transients.models.core.detection_horizon import lyman_limit_redshift

    z_lyman = lyman_limit_redshift(BANDS)
    curve, grid = _Kilonovalike().get_detection_horizon([18.0, 40.0], BANDS, rng=1, z_max=0.5, **{**SMALL, "n_z": 12})
    assert grid.z_grid[-1] == pytest.approx(z_lyman)
    assert curve["z_limit"][1] == pytest.approx(z_lyman)  # nothing is too faint at 40 AB, so this one is capped
    assert bool(curve["capped"][1]) and not bool(curve["capped"][0])
    assert curve["z_limit"][0] < z_lyman
    assert curve.meta["z_lyman"] == pytest.approx(z_lyman)


def test_the_cap_can_be_switched_off():
    """`lyman_limit=False` restores the uncapped horizon, which can lie beyond the Lyman limit."""
    from uvex_transients.models.core.detection_horizon import lyman_limit_redshift

    z_lyman = lyman_limit_redshift(BANDS)
    curve, grid = _Kilonovalike().get_detection_horizon(
        40.0, BANDS, rng=1, z_max=0.5, lyman_limit=False, **{**SMALL, "n_z": 12}
    )
    assert "capped" not in curve.colnames and "z_lyman" not in curve.meta
    assert grid.z_grid[-1] > z_lyman or not np.isfinite(curve["z_limit"][0])
