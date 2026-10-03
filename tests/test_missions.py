"""
Tests for :mod:`uvex_transients.missions`: bandpass downsampling and the registered fast missions.

Accuracy is tested on a synthetic 11000-point bandpass that mimics the m4opt 3.1.0 tables (a
smooth core, soft edges, a faint red tail), so the results do not depend on which m4opt version
is installed. The real-mission tests check structure only, for the same reason.
"""

import dataclasses

import numpy as np
import pytest
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from m4opt.missions import uvex
from m4opt.synphot import observing
from synphot import SpectralElement
from synphot.models import Empirical1D

from uvex_transients import missions
from uvex_transients.missions import (
    FAST_MISSIONS,
    downsample_bandpass,
    downsample_mission,
    get_mission,
    list_missions,
)
from uvex_transients.models import (
    KilonovaCoolingBlackbodySED,
    LFBOTCoolingBlackbodySED,
    TypeIIPSED,
    VanVelzenTDESED,
)
from uvex_transients.utils.io_utils import config_yaml

T_MIN, T_MAX, N_SPEC, TOL = 1500.0, 50000.0, 12, 1.0e-3


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_dense_bandpass(low, high, peak, leak) -> SpectralElement:
    """A 1 Angstrom-sampled bandpass over 1000-12000 A: soft-edged core with a faint red tail."""
    wave = np.arange(1000.0, 12000.0, 1.0)
    core = peak * _sigmoid((wave - low) / 15.0) * _sigmoid((high - wave) / 25.0) * (1 + 0.05 * np.sin(wave / 37.0))
    tail = leak * (wave / high) ** 1.5 * np.exp(-(((wave - high) / 6000.0) ** 2)) * _sigmoid((wave - high) / 40.0)
    return SpectralElement(Empirical1D, points=wave * u.AA, lookup_table=core + tail)


@pytest.fixture(scope="module")
def fuv_like() -> SpectralElement:
    return make_dense_bandpass(low=1350.0, high=1750.0, peak=0.07, leak=1e-5)


@pytest.fixture(scope="module")
def nuv_like() -> SpectralElement:
    return make_dense_bandpass(low=2000.0, high=2800.0, peak=0.13, leak=3e-6)


@pytest.fixture(scope="module")
def fuv_downsampled(fuv_like) -> SpectralElement:
    return downsample_bandpass(fuv_like, t_min=T_MIN, t_max=T_MAX, n_spec=N_SPEC, tol_mag=TOL)


def _table(bandpass):
    """Wavelength (Angstrom) and throughput of a bandpass's own table."""
    wave = bandpass.waveset.to_value(u.AA)
    return wave, np.asarray(bandpass(wave * u.AA).to_value(u.dimensionless_unscaled))


def _mass(bandpass):
    """Integrated transmission in frequency, ``int T dnu``."""
    wave, throughput = _table(bandpass)
    nu = const.c.cgs.value / (wave * 1e-8)
    order = np.argsort(nu)
    return np.trapezoid(throughput[order], nu[order])


def _blackbody_band_integrals(bandpass, temperatures):
    """
    Independent reimplementation of the four band quantities for the tests.

    Rows: flux-weighted band average, photon-weighted band average, absolute flux integral,
    absolute photon integral.
    """
    h, c, k = const.h.cgs.value, const.c.cgs.value, const.k_B.cgs.value
    wave, throughput = _table(bandpass)
    nu = c / (wave * 1e-8)
    order = np.argsort(nu)
    nu, throughput = nu[order], throughput[order]
    out = []
    for temperature in temperatures:
        planck = 2 * h * nu**3 / c**2 / np.expm1(h * nu / (k * temperature))
        flux = np.trapezoid(planck * throughput, nu)
        photons = np.trapezoid(planck * throughput / nu, nu)
        out.append([flux / np.trapezoid(throughput, nu), photons / np.trapezoid(throughput / nu, nu), flux, photons])
    return np.array(out).T


def _worst_mag_error(full, reduced, temperatures):
    a = _blackbody_band_integrals(full, temperatures)
    b = _blackbody_band_integrals(reduced, temperatures)
    return np.max(np.abs(2.5 * np.log10(b / a)))


class TestDownsampleBandpass:
    def test_reduces_points_and_keeps_end_points(self, fuv_like, fuv_downsampled):
        wave, _ = _table(fuv_like)
        reduced_wave, reduced = _table(fuv_downsampled)
        assert reduced_wave.size < 0.1 * wave.size
        assert reduced_wave[0] == wave[0]
        assert reduced_wave[-1] == wave[-1]
        assert np.all(np.diff(reduced_wave) > 0)
        assert np.all(reduced >= 0)

    def test_integrated_transmission_is_preserved_by_rescale(self, fuv_like, fuv_downsampled):
        np.testing.assert_allclose(_mass(fuv_downsampled), _mass(fuv_like), rtol=1e-10)

    def test_without_rescale_mass_differs(self, fuv_like):
        raw = downsample_bandpass(fuv_like, t_min=T_MIN, t_max=T_MAX, n_spec=N_SPEC, tol_mag=TOL, rescale=False)
        assert raw.meta["downsampling"]["mass_rescale"] == 1.0
        assert abs(_mass(raw) / _mass(fuv_like) - 1) > 1e-12

    def test_meta_records_the_run(self, fuv_like, fuv_downsampled):
        record = fuv_downsampled.meta["downsampling"]
        assert record["n_original"] == fuv_like.waveset.size
        assert record["n_downsampled"] == fuv_downsampled.waveset.size
        assert record["achieved_mag_error"] <= TOL
        assert (record["t_min"], record["t_max"], record["n_spec"], record["tol_mag"]) == (T_MIN, T_MAX, N_SPEC, TOL)

    def test_pivot_wavelength_is_preserved(self, fuv_like, fuv_downsampled):
        np.testing.assert_allclose(fuv_downsampled.pivot().to_value(u.AA), fuv_like.pivot().to_value(u.AA), rtol=1e-3)

    @pytest.mark.parametrize("band", ["fuv_like", "nuv_like"])
    def test_held_out_temperatures_stay_within_tolerance(self, band, request):
        """
        Blackbodies at temperatures that are neither optimization nor verification temperatures.

        The temperatures are offset irrationally from the log-uniform grids the function uses, so
        this guards against a downsampling that is only accurate where it was tuned. The factor of
        two allows for variation between the verification points and anywhere in between.
        """
        dense = request.getfixturevalue(band)
        reduced = downsample_bandpass(dense, t_min=T_MIN, t_max=T_MAX, n_spec=N_SPEC, tol_mag=TOL)
        held_out = np.geomspace(T_MIN * 1.0137, T_MAX * 0.9871, 61) * (1 + 0.00311 * np.arange(61) / 61)
        assert _worst_mag_error(dense, reduced, held_out) <= 2 * TOL

    def test_tighter_tolerance_keeps_more_points(self, fuv_like):
        loose = downsample_bandpass(fuv_like, t_min=T_MIN, t_max=T_MAX, n_spec=N_SPEC, tol_mag=1e-2)
        tight = downsample_bandpass(fuv_like, t_min=T_MIN, t_max=T_MAX, n_spec=N_SPEC, tol_mag=1e-4)
        assert tight.waveset.size > loose.waveset.size

    @pytest.mark.parametrize(("t_min", "t_max"), [(1500.0, 50000.0), (6000.0, 50000.0), (3000.0, 12000.0)])
    def test_guarantee_covers_the_requested_range(self, fuv_like, t_min, t_max):
        """Whatever range is requested, blackbodies inside it meet the tolerance (not only the default range)."""
        reduced = downsample_bandpass(fuv_like, t_min=t_min, t_max=t_max, n_spec=N_SPEC, tol_mag=TOL)
        inside = np.geomspace(t_min * 1.0137, t_max * 0.9871, 41)
        assert _worst_mag_error(fuv_like, reduced, inside) <= 2 * TOL

    def test_table_that_cannot_be_reduced_is_returned_unchanged(self):
        wave = np.linspace(1000.0, 3000.0, 12)
        coarse = SpectralElement(Empirical1D, points=wave * u.AA, lookup_table=np.sin(wave / 300.0) ** 2 + 0.1)
        assert downsample_bandpass(coarse, t_min=T_MIN, t_max=T_MAX, n_spec=N_SPEC, tol_mag=1e-6) is coarse

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"t_min": 5000.0, "t_max": 5000.0},
            {"t_min": -1.0},
            {"t_min": 6000.0, "t_max": 3000.0},
            {"n_spec": 1},
            {"tol_mag": 0.0},
        ],
    )
    def test_invalid_arguments_raise(self, fuv_like, kwargs):
        with pytest.raises(ValueError):
            downsample_bandpass(
                fuv_like, **{"t_min": T_MIN, "t_max": T_MAX, "n_spec": N_SPEC, "tol_mag": TOL, **kwargs}
            )

    def test_defaults_come_from_the_configuration(self, fuv_like, monkeypatch):
        settings = {"enabled": False, "tol_mag": 1e-2, "t_min": 3000.0, "t_max": 30000.0, "n_spec": 6}
        monkeypatch.setattr(missions, "_downsample_settings", lambda: settings)
        from_config = downsample_bandpass(fuv_like)
        explicit = downsample_bandpass(fuv_like, t_min=3000.0, t_max=30000.0, n_spec=6, tol_mag=1e-2)
        np.testing.assert_array_equal(from_config.waveset.value, explicit.waveset.value)


# --- SED comparisons ---------------------------------------------------------------------------

SED_CASES = [
    pytest.param(KilonovaCoolingBlackbodySED, (1.0, 20.0), id="kilonova"),
    pytest.param(VanVelzenTDESED, (10.0, 200.0), id="tde"),
    pytest.param(LFBOTCoolingBlackbodySED, (1.0, 100.0), id="lfbot"),
    pytest.param(TypeIIPSED, (5.0, 150.0), id="type_iip"),
]


@pytest.mark.parametrize("band", ["fuv_like", "nuv_like"])
@pytest.mark.parametrize(("model_class", "days"), SED_CASES)
def test_sed_band_magnitudes_match(model_class, days, band, request):
    """Real SEDs integrated through the dense and downsampled bandpasses agree to within twice the tolerance."""
    dense = request.getfixturevalue(band)
    reduced = downsample_bandpass(dense, t_min=T_MIN, t_max=T_MAX, n_spec=N_SPEC, tol_mag=TOL)

    model = model_class()
    rng = np.random.default_rng(0)
    parameters = {name: value[:, np.newaxis] for name, value in model.sample_parameters(8, rng=rng).items()}
    t = np.geomspace(*days, 12) * u.day

    full = model.mag_bandpass(dense, t, redshift=0.05, **parameters).to_value(u.ABmag)
    fast = model.mag_bandpass(reduced, t, redshift=0.05, **parameters).to_value(u.ABmag)

    # Epochs fainter than mag 35 are unobservable, and an SED there can be steeper than the
    # thermal range the downsampling was verified over (an early Type IIP epoch at mag 51 is off
    # by 3e-3 mag, while every epoch down to mag 23 agrees to 1e-4), so only epochs that could
    # matter to a survey are compared.
    comparable = np.isfinite(full) & np.isfinite(fast) & (full < 35)
    assert comparable.sum() >= 20
    assert np.max(np.abs(full - fast)[comparable]) <= 2 * TOL


@pytest.mark.parametrize(("model_class", "days"), [SED_CASES[0], SED_CASES[1]])
def test_detector_snr_matches(model_class, days, fuv_like, nuv_like):
    """
    m4opt's own count-rate path (source and background) agrees between dense and downsampled bandpasses.

    The same detector, background and noise terms are used for both, with only the bandpasses
    swapped, so any difference is the downsampling.
    """
    bands = {"FUV": fuv_like, "NUV": nuv_like}
    reduced_bands = {
        name: downsample_bandpass(bp, t_min=T_MIN, t_max=T_MAX, n_spec=N_SPEC, tol_mag=TOL)
        for name, bp in bands.items()
    }
    dense_detector = dataclasses.replace(uvex.detector, bandpasses=bands)
    fast_detector = dataclasses.replace(uvex.detector, bandpasses=reduced_bands)

    model = model_class()
    rng = np.random.default_rng(1)
    n_events, n_times = 3, 3
    parameters = {
        name: value[:, np.newaxis, np.newaxis] for name, value in model.sample_parameters(n_events, rng=rng).items()
    }
    t_since_explosion = np.geomspace(*days, n_times) * u.day
    spectra = model.as_source_spectrum(t=t_since_explosion[:, np.newaxis], redshift=0.03, **parameters)

    obstimes = Time("2025-01-01T00:00:00", scale="utc") + t_since_explosion
    locations = uvex.observer_location(obstimes)
    coord = SkyCoord(150 * u.deg, 20 * u.deg)
    with observing(locations, coord, obstimes):
        for band in bands:
            full = u.Quantity(dense_detector.get_snr(900 * u.s, spectra, band)).to_value(u.dimensionless_unscaled)
            fast = u.Quantity(fast_detector.get_snr(900 * u.s, spectra, band)).to_value(u.dimensionless_unscaled)
            valid = np.isfinite(full) & np.isfinite(fast) & (full > 0)
            assert valid.any()
            np.testing.assert_allclose(fast[valid], full[valid], rtol=1e-2)


# --- Missions and registration -----------------------------------------------------------------


class TestDownsampleMission:
    def test_copy_has_fewer_bandpass_points_and_leaves_the_original_alone(self):
        before = {band: bp.waveset.size for band, bp in uvex.detector.bandpasses.items()}
        fast = downsample_mission(uvex)
        assert fast is not uvex
        assert fast.name == "uvex_fast"
        assert list(fast.detector.bandpasses) == list(uvex.detector.bandpasses)
        for band, bp in fast.detector.bandpasses.items():
            assert bp.waveset.size <= before[band]
            np.testing.assert_allclose(
                bp.pivot().to_value(u.AA), uvex.detector.bandpasses[band].pivot().to_value(u.AA), rtol=1e-3
            )
        assert {band: bp.waveset.size for band, bp in uvex.detector.bandpasses.items()} == before

    def test_everything_but_the_bandpasses_is_shared(self):
        fast = downsample_mission(uvex)
        assert fast.fov is uvex.fov
        assert fast.detector.background is uvex.detector.background
        assert fast.detector.area == uvex.detector.area

    def test_custom_name(self):
        assert downsample_mission(uvex, name="uvex_coarse").name == "uvex_coarse"

    def test_non_spectral_element_bandpass_raises(self):
        broken = dataclasses.replace(uvex, detector=dataclasses.replace(uvex.detector, bandpasses={"X": object()}))
        with pytest.raises(TypeError, match="not a SpectralElement"):
            downsample_mission(broken)


class TestGetMission:
    def test_registered_fast_mission(self):
        assert "uvex_fast" in FAST_MISSIONS
        assert "uvex_fast" in list_missions()
        fast = get_mission("uvex_fast")
        assert fast.name == "uvex_fast"
        assert get_mission("uvex_fast") is fast
        assert missions.uvex_fast is fast

    def test_fast_name_ignores_the_enabled_setting(self, monkeypatch):
        monkeypatch.setattr(
            missions, "_downsample_settings", lambda: {**missions._DEFAULT_DOWNSAMPLE_SETTINGS, "enabled": False}
        )
        assert get_mission("uvex_fast").name == "uvex_fast"

    def test_plain_name_is_the_original_by_default(self):
        assert get_mission("uvex") is uvex

    def test_enabled_setting_downsamples_plain_names(self, monkeypatch):
        monkeypatch.setattr(
            missions, "_downsample_settings", lambda: {**missions._DEFAULT_DOWNSAMPLE_SETTINGS, "enabled": True}
        )
        assert get_mission("uvex").name == "uvex_fast"
        assert get_mission("uvex", downsample=False) is uvex

    def test_explicit_downsample_overrides_a_disabled_setting(self):
        assert get_mission("uvex", downsample=True).name == "uvex_fast"

    def test_unknown_name_raises_listing_the_choices(self):
        with pytest.raises(ValueError, match="Unknown mission 'nope'.*uvex_fast"):
            get_mission("nope")

    def test_missing_configuration_keys_fall_back_to_defaults(self, monkeypatch):
        monkeypatch.setattr(missions, "config", {})
        assert missions._downsample_settings() == missions._DEFAULT_DOWNSAMPLE_SETTINGS

    def test_packaged_config_matches_the_code_defaults(self):
        path = missions.__file__.replace("missions.py", "config.yaml")
        with open(path) as f:
            packaged = config_yaml.load(f)["missions"]["downsample"]
        assert dict(packaged) == missions._DEFAULT_DOWNSAMPLE_SETTINGS
