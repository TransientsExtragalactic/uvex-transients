"""Tests for `HorizonCurve` and `ExtragalacticTransient.horizon_curve`."""

import numpy as np
import pytest
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM, Planck18
from astropy.table import QTable

import uvex_transients.transients  # noqa: F401  (registers every shipped class)
from uvex_transients.models.kilonovae import KilonovaCoolingBlackbodySED
from uvex_transients.transients import horizon
from uvex_transients.transients.base import ExtragalacticTransient, TransientBase
from uvex_transients.transients.horizon import HorizonCurve, curve_path

SMALL = {"n_samples": 40, "n_z": 8, "n_time": 20, "confidence": 0.5, "tolerance": 0.1}
MAGS = np.arange(22.0, 26.01, 1.0)


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


def _table(mag, z, **meta):
    return QTable(
        {"mag_limit": np.asarray(mag, dtype=float) * u.ABmag, "z_limit": np.asarray(z, dtype=float)}, meta=meta
    )


def _small_curve(transient=None):
    transient = _Kilonovalike() if transient is None else transient
    return HorizonCurve.generate(transient, mag_limits=MAGS, progress=False, **SMALL)


# =========================================================================== #
# Lookup                                                                      #
# =========================================================================== #
class TestLookup:
    """`HorizonCurve.z_limit` interpolation and range handling."""

    curve = HorizonCurve(_table([22.0, 23.0, 24.0], [0.1, 0.2, 0.4]))

    def test_tabulated_points_are_returned_exactly(self):
        assert self.curve.z_limit(23.0) == pytest.approx(0.2)

    def test_interpolates_in_log_z(self):
        """Halfway between two magnitudes is the geometric mean of their redshifts."""
        assert self.curve.z_limit(23.5) == pytest.approx(np.sqrt(0.2 * 0.4))

    def test_shallower_than_the_grid_returns_the_first_redshift(self):
        assert self.curve.z_limit(15.0) == pytest.approx(0.1)

    def test_deeper_than_the_grid_raises(self):
        with pytest.raises(ValueError, match="outside the curve"):
            self.curve.z_limit(24.5)

    def test_array_and_quantity_input(self):
        np.testing.assert_allclose(self.curve.z_limit([22.0, 24.0]), [0.1, 0.4])
        assert self.curve.z_limit(23.0 * u.ABmag) == pytest.approx(0.2)

    def test_scalar_in_scalar_out(self):
        assert isinstance(self.curve.z_limit(23.0), float)


class TestValidation:
    """`HorizonCurve` rejects tables that are not a usable curve."""

    @pytest.mark.parametrize(
        "mag, z",
        [
            ([22.0, 22.0], [0.1, 0.2]),  # magnitudes not strictly increasing
            ([22.0], [0.1]),  # too short
            ([22.0, 23.0], [0.2, 0.1]),  # redshifts decrease
            ([22.0, 23.0], [0.1, np.inf]),  # unbracketed
            ([22.0, 23.0], [0.0, 0.1]),  # non-positive
        ],
    )
    def test_bad_tables_are_rejected(self, mag, z):
        with pytest.raises(ValueError):
            HorizonCurve(_table(mag, z))

    def test_missing_columns_are_rejected(self):
        with pytest.raises(ValueError, match="needs columns"):
            HorizonCurve(QTable({"mag_limit": [22.0, 23.0] * u.ABmag}))


# =========================================================================== #
# Input and output                                                            #
# =========================================================================== #
class TestIO:
    """Writing and reading the ECSV form."""

    def test_round_trip_preserves_values_and_metadata(self, tmp_path):
        curve = _small_curve()
        curve.write(tmp_path / "curve.ecsv")
        again = HorizonCurve.read(tmp_path / "curve.ecsv")
        np.testing.assert_array_equal(again.mag_limits, curve.mag_limits)
        np.testing.assert_array_equal(again.z_limits, curve.z_limits)
        assert again.meta == curve.meta

    def test_a_round_tripped_curve_still_matches(self, tmp_path):
        transient = _Kilonovalike()
        curve = _small_curve(transient)
        curve.write(tmp_path / "curve.ecsv")
        assert HorizonCurve.read(tmp_path / "curve.ecsv").matches(transient)

    def test_reading_a_missing_file_returns_none(self, tmp_path):
        assert HorizonCurve.read(tmp_path / "nope.ecsv") is None

    def test_write_creates_the_directory(self, tmp_path):
        _small_curve().write(tmp_path / "a" / "b" / "curve.ecsv")
        assert (tmp_path / "a" / "b" / "curve.ecsv").exists()


# =========================================================================== #
# Generation                                                                  #
# =========================================================================== #
class TestGenerate:
    """`HorizonCurve.generate`."""

    def test_curve_is_monotone_and_recorded(self):
        curve = _small_curve()
        assert np.all(np.diff(curve.z_limits) >= 0)
        np.testing.assert_array_equal(curve.mag_limits, MAGS)
        meta = curve.meta
        assert meta["class"] == "_Kilonovalike"
        assert meta["duration_day"] == 30.0
        assert meta["cosmology"] == Planck18.name
        assert meta["n_samples"] == 40
        assert meta["confidence"] == 0.5

    def test_generation_is_reproducible(self):
        np.testing.assert_array_equal(_small_curve().z_limits, _small_curve().z_limits)

    def test_unknown_setting_is_rejected(self):
        with pytest.raises(TypeError, match="Unknown setting"):
            HorizonCurve.generate(_Kilonovalike(), bogus=1)

    def test_uses_the_transients_cosmology(self):
        far = FlatLambdaCDM(H0=40.0, Om0=0.3, name="slow")
        default = _small_curve()
        slow = _small_curve(_Kilonovalike(cosmology=far))
        # A smaller H0 puts the same redshift at a larger distance, so the same depth is reached at a lower z.
        assert np.all(slow.z_limits <= default.z_limits)
        assert slow.meta["cosmology"] == "slow"


# =========================================================================== #
# Validity                                                                    #
# =========================================================================== #
class TestMatches:
    """`HorizonCurve.matches` notices every change the curve depends on."""

    def test_matches_the_transient_it_was_made_for(self):
        transient = _Kilonovalike()
        assert _small_curve(transient).matches(transient)

    def test_a_pinned_parameter_invalidates(self):
        transient = _Kilonovalike()
        curve = _small_curve(transient)
        transient.sed["amplitude"].fix(1.0e42 * u.erg / u.s)
        assert not curve.matches(transient)

    def test_a_changed_prior_invalidates(self):
        transient = _Kilonovalike()
        curve = _small_curve(transient)
        transient.sed["amplitude"].prior = type(transient.sed["amplitude"].prior)(mean=42.5, sigma=0.3)
        assert not curve.matches(transient)

    def test_a_changed_duration_invalidates(self):
        transient = _Kilonovalike()
        curve = _small_curve(transient)
        transient.duration_limit = 10 * u.day
        assert not curve.matches(transient)

    def test_a_curve_without_provenance_never_matches(self):
        assert not HorizonCurve(_table([22.0, 23.0], [0.1, 0.2])).matches(_Kilonovalike())


# =========================================================================== #
# ExtragalacticTransient.horizon_curve                                        #
# =========================================================================== #
class TestHorizonCurveProperty:
    """The lazy, self-validating `ExtragalacticTransient.horizon_curve`."""

    @pytest.fixture
    def counted(self, monkeypatch, tmp_path):
        """Point curve storage at an empty directory and make `generate` cheap and counted."""
        monkeypatch.setattr(horizon, "CURVE_DIR", tmp_path)
        real = HorizonCurve.generate.__func__
        calls = []

        def cheap(cls, transient, *args, **kwargs):
            calls.append(type(transient).__name__)
            return real(cls, transient, *args, mag_limits=MAGS, progress=False, **SMALL)

        monkeypatch.setattr(HorizonCurve, "generate", classmethod(cheap))
        return calls

    def test_is_lazy_and_remembered(self, counted):
        transient = _Kilonovalike()
        assert transient._horizon_curve is None
        first = transient.horizon_curve
        assert transient.horizon_curve is first
        assert counted == ["_Kilonovalike"]

    def test_a_class_without_a_stored_curve_is_generated(self, counted):
        assert isinstance(_Kilonovalike().horizon_curve, HorizonCurve)
        assert counted == ["_Kilonovalike"]

    def test_a_matching_stored_curve_is_read_not_generated(self, counted, tmp_path):
        stored = HorizonCurve.generate(_Kilonovalike())
        stored.write(curve_path("_Kilonovalike"))
        counted.clear()
        curve = _Kilonovalike().horizon_curve
        assert counted == []
        np.testing.assert_array_equal(curve.z_limits, stored.z_limits)

    def test_a_stale_stored_curve_is_regenerated(self, counted):
        HorizonCurve.generate(_Kilonovalike()).write(curve_path("_Kilonovalike"))
        counted.clear()
        transient = _Kilonovalike()
        transient.sed["amplitude"].fix(1.0e42 * u.erg / u.s)
        _ = transient.horizon_curve
        assert counted == ["_Kilonovalike"]

    def test_changing_the_duration_afterwards_regenerates(self, counted):
        transient = _Kilonovalike()
        _ = transient.horizon_curve
        transient.duration_limit = 10 * u.day
        _ = transient.horizon_curve
        assert counted == ["_Kilonovalike", "_Kilonovalike"]
        assert transient.horizon_curve.meta["duration_day"] == 10.0

    def test_a_class_without_a_stored_curve_still_generates_one_on_demand(self, counted):
        """`HAS_STORED_HORIZON_CURVE = False` skips the stored file but does not stop anyone generating a curve."""

        class _Bespoke(_Kilonovalike):
            HAS_STORED_HORIZON_CURVE = False

        HorizonCurve.generate(_Bespoke()).write(curve_path("_Bespoke"))  # a file that would match if it were read
        counted.clear()
        assert isinstance(_Bespoke().horizon_curve, HorizonCurve)
        assert counted == ["_Bespoke"]

    def test_the_curve_can_set_the_redshift_limit(self, counted):
        transient = _Kilonovalike()
        transient.redshift_limit = transient.horizon_curve.z_limit(24.0)
        assert transient.redshift_limit == pytest.approx(transient.horizon_curve.z_limit(24.0))


# =========================================================================== #
# Shipped classes                                                             #
# =========================================================================== #
SHIPPED = sorted(
    name
    for name, cls in TransientBase.registry().items()
    if issubclass(cls, ExtragalacticTransient)
    and cls.HAS_STORED_HORIZON_CURVE
    and cls.__module__.startswith("uvex_transients.transients.")
)


def test_every_shipped_class_is_checked():
    """The shipped classes are found, so the parametrized test below is not vacuous."""
    assert len(SHIPPED) >= 10
    assert "ShockCoolingIIb" not in SHIPPED


def test_bespoke_classes_ship_no_stored_curve():
    """`ShockCoolingIIb` opts out of a stored curve, so it is neither built by default nor checked."""
    from uvex_transients.transients.supernovae import ShockCoolingIIb

    assert not ShockCoolingIIb.HAS_STORED_HORIZON_CURVE
    assert not curve_path("ShockCoolingIIb").exists()


@pytest.mark.parametrize("name", SHIPPED)
def test_stored_curves_match_their_classes(name):
    """
    Every shipped class has a stored curve that still describes it.

    If this fails, a shipped class's SED, priors or duration changed since its curve was built. Regenerate it
    with ``python scripts/build_horizon_curves.py <ClassName>`` (``make horizons``) and commit the file.
    """
    transient = TransientBase.registry()[name]()
    curve = HorizonCurve.read(curve_path(name))
    assert curve is not None, (
        f"No stored horizon curve for {name}; run `python scripts/build_horizon_curves.py {name}`."
    )
    assert curve.matches(transient), (
        f"The stored horizon curve of {name} is stale; run `python scripts/build_horizon_curves.py {name}`."
    )


@pytest.mark.parametrize("name", SHIPPED)
def test_stored_curves_were_built_with_the_current_settings(name):
    """A stored curve was built with the module's `SETTINGS` and `MAG_LIMITS`, which the fingerprint does not cover."""
    curve = HorizonCurve.read(curve_path(name))
    assert curve is not None
    assert {key: curve.meta[key] for key in horizon.SETTINGS} == horizon.SETTINGS, (
        f"The stored horizon curve of {name} was built with other settings; run `make horizons CLASSES={name}`."
    )
    np.testing.assert_allclose(curve.mag_limits, horizon.MAG_LIMITS)


@pytest.mark.parametrize("name", SHIPPED)
def test_stored_curves_cover_the_default_depth(name):
    """The 24.5 AB single-dwell depth is on every stored curve, and the curve is a usable table."""
    curve = HorizonCurve.read(curve_path(name))
    assert curve is not None
    assert curve.mag_limits[0] <= 24.5 <= curve.mag_limits[-1]
    assert curve.z_limit(24.5) > 0


def test_no_orphan_curve_files():
    """A stored curve whose class no longer exists is deleted, not left to rot."""
    stored = {path.stem for path in horizon.CURVE_DIR.glob("*.ecsv")}
    assert stored <= set(SHIPPED)
