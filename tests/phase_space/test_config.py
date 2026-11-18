"""Tests that the phase-space defaults come from the configuration, and that an argument overrides them."""

import sys

import numpy as np
import pytest
import synphot
from astropy import units as u

pytest.importorskip("sklearn")

from uvex_transients.phase_space.analysis.overlay import measure_class  # noqa: E402
from uvex_transients.phase_space.sampling import CanonicalSampler, StoppingRule  # noqa: E402
from uvex_transients.phase_space.sampling.canonical import _CanonicalTransient  # noqa: E402
from uvex_transients.transients.TDEs import TidalDisruptionEvent  # noqa: E402
from uvex_transients.utils import config  # noqa: E402

NSIDE = 32

# Every key of ``config["phase_space"]`` the code reads, with the type it is read as
KEYS = {
    "canonical.spectral_index": float,
    "canonical.rise_ratio": float,
    "canonical.fade_mag": float,
    "breakout.log_L0": list,
    "breakout.log_t0": list,
    "breakout.n_visits": int,
    "breakout.time_bins": int,
    "breakout.z_max": float,
    "breakout.table_points": int,
    "mission": str,
    "detection.band": str,
    "detection.snr_threshold": float,
    "detection.n_visits": int,
    "detection.exclude_first_visit_detections": bool,
    "simulation.mag_limit": float,
    "simulation.time_bins": int,
    "stopping.chunk_size": int,
    "stopping.k_target": int,
    "stopping.n_max": int,
    "numerics.exposure_spacing": float,
    "numerics.z_table_min": float,
    "numerics.z_table_points": int,
    "numerics.volume_z_min": float,
    "numerics.volume_points": int,
    "campaign.n_initial": int,
    "campaign.batch_size": int,
    "campaign.max_points": int,
    "campaign.tolerance": float,
    "campaign.acquisition": str,
    "campaign.n_candidates": int,
    "campaign.floor": float,
    "campaign.n_restarts": int,
    "canonical.log_luminosity": list,
    "canonical.log_half_time": list,
    "grid_size": int,
    "overlay.n_draws": int,
    "overlay.n_time": int,
    "overlay.t_min": float,
    "overlay.t_max_factor": float,
}


class _Config:
    """The real configuration, with some keys replaced."""

    def __init__(self, overrides):
        self._overrides = {f"phase_space.{key}": value for key, value in overrides.items()}

    def __getitem__(self, key):
        return self._overrides[key] if key in self._overrides else config[key]


@pytest.fixture(scope="module")
def schedule(make_schedule):
    return make_schedule(n_sched=20)


@pytest.fixture
def configured(monkeypatch):
    """Replace the configuration the phase-space modules read with the real one plus some changes."""

    def apply(**overrides):
        fake = _Config({key.replace("__", "."): value for key, value in overrides.items()})
        for name, module in list(sys.modules.items()):
            if name.startswith("uvex_transients.phase_space") and hasattr(module, "config"):
                monkeypatch.setattr(module, "config", fake)

    return apply


# ---------------------------------------------------------------------------- #
# The configuration                                                            #
# ---------------------------------------------------------------------------- #
@pytest.mark.parametrize("key, kind", KEYS.items())
def test_every_key_the_code_reads_is_in_the_configuration(key, kind):
    value = config[f"phase_space.{key}"]
    assert isinstance(value, kind) and not (kind is int and isinstance(value, bool))
    if kind is list:
        assert len(value) == 2 and value[0] < value[1]


# ---------------------------------------------------------------------------- #
# The transient                                                                #
# ---------------------------------------------------------------------------- #
def test_the_transient_takes_its_shape_from_the_configuration(configured):
    default = _CanonicalTransient()
    assert default.spectral_index == config["phase_space.canonical.spectral_index"]
    assert default.rise_ratio == config["phase_space.canonical.rise_ratio"]
    assert default.fade_mag == config["phase_space.canonical.fade_mag"]

    configured(canonical__spectral_index=-0.5, canonical__rise_ratio=0.3, canonical__fade_mag=4.0)
    changed = _CanonicalTransient()
    assert (changed.spectral_index, changed.rise_ratio, changed.fade_mag) == (-0.5, 0.3, 4.0)
    # An argument is not overridden by the configuration
    assert _CanonicalTransient(spectral_index=1.0).spectral_index == 1.0


# ---------------------------------------------------------------------------- #
# The sampler and the stopping rule                                            #
# ---------------------------------------------------------------------------- #
def test_a_sampler_given_nothing_takes_every_setting_from_the_configuration(schedule):
    settings = CanonicalSampler(schedule, nside=NSIDE, order="nested").settings
    assert settings["mission"] == config["phase_space.mission"]
    for key in ("band", "snr_threshold", "n_visits", "exclude_first_visit_detections"):
        assert settings[key] == config[f"phase_space.detection.{key}"], key
    for key in ("mag_limit", "time_bins"):
        assert settings[key] == config[f"phase_space.simulation.{key}"], key
    for key in ("spectral_index", "rise_ratio", "fade_mag"):
        assert settings[key] == config[f"phase_space.canonical.{key}"], key


def test_a_sampler_follows_a_changed_configuration_and_an_argument_overrides_it(schedule, configured):
    configured(
        detection__band="NUV",
        detection__snr_threshold=3.0,
        detection__n_visits=1,
        detection__exclude_first_visit_detections=False,
        canonical__spectral_index=-1.0,
    )
    sampler = CanonicalSampler(schedule, nside=NSIDE, order="nested")
    assert sampler.band == "NUV" and sampler.snr_threshold == 3.0 and sampler.n_visits == 1
    assert sampler.exclude_first_visit_detections is False
    assert sampler.spectral_index == -1.0 and sampler.settings["spectral_index"] == -1.0

    explicit = CanonicalSampler(schedule, band="FUV", n_visits=4, nside=NSIDE, order="nested")
    assert explicit.band == "FUV" and explicit.n_visits == 4 and explicit.snr_threshold == 3.0


def test_the_stopping_rule_takes_its_settings_from_the_configuration(configured):
    rule = StoppingRule.from_config()
    assert (rule.chunk_size, rule.k_target, rule.n_max) == tuple(
        config[f"phase_space.stopping.{key}"] for key in ("chunk_size", "k_target", "n_max")
    )
    configured(stopping__k_target=7, stopping__chunk_size=50, stopping__n_max=200)
    assert StoppingRule.from_config() == StoppingRule(chunk_size=50, k_target=7, n_max=200)
    assert StoppingRule.from_config(k_target=9).k_target == 9


def test_the_tables_are_sized_by_the_configuration(schedule, configured):
    configured(numerics__z_table_min=1e-4, numerics__z_table_points=64, numerics__volume_points=100)
    sampler = CanonicalSampler(schedule, nside=NSIDE, order="nested")
    assert sampler._limit.z_min == 1e-4
    assert len(sampler._z_grid) == 101


def test_a_rebuilt_sampler_keeps_what_was_recorded_not_what_the_configuration_says_now(schedule, configured):
    recorded = CanonicalSampler(schedule, snr_threshold=4.0, nside=NSIDE, order="nested", schedule_name=None).settings
    configured(detection__snr_threshold=9.0, detection__n_visits=3)
    rebuilt = CanonicalSampler.from_settings(recorded, schedule=schedule)
    assert rebuilt.snr_threshold == 4.0 and rebuilt.n_visits == recorded["n_visits"]


# ---------------------------------------------------------------------------- #
# The overlay                                                                  #
# ---------------------------------------------------------------------------- #
def test_the_overlay_takes_its_grid_and_draws_from_the_configuration(configured):
    bandpass = synphot.SpectralElement(synphot.Gaussian1D, mean=1500 * u.AA, stddev=100 * u.AA)
    configured(overlay__n_draws=7, overlay__n_time=25, overlay__t_min=0.5, overlay__t_max_factor=2.0)
    from_config = measure_class(TidalDisruptionEvent, bandpass, rng=0)
    explicit = measure_class(
        TidalDisruptionEvent,
        bandpass,
        n_draws=7,
        n_time=25,
        t_min=0.5 * u.day,
        t_max=2.0 * TidalDisruptionEvent.DEFAULT_DURATION,
        rng=0,
    )
    assert len(from_config) == 7
    for column in ("log_luminosity", "log_half_time"):
        np.testing.assert_array_equal(from_config[column], explicit[column])


# ---------------------------------------------------------------------------- #
# The defaults of a kind                                                       #
# ---------------------------------------------------------------------------- #
def test_a_kind_takes_its_own_section_before_the_shared_one_and_a_parent_kind_before_the_shared_one(configured):
    from uvex_transients.phase_space._config import default
    from uvex_transients.phase_space.sampling import BracketedBreakoutSampler, BreakoutSampler, SurveySettings

    assert CanonicalSampler.kinds() == ("canonical",)
    assert BracketedBreakoutSampler.kinds() == ("breakout_bracketed", "breakout")

    # The breakout has its own n_visits, so the shared one does not reach it or the kind built on it
    assert default("detection", "n_visits", BreakoutSampler.kinds()) == config["phase_space.breakout.n_visits"]
    assert (
        SurveySettings.default("n_visits", BracketedBreakoutSampler.kinds()) == config["phase_space.breakout.n_visits"]
    )
    assert SurveySettings.default("n_visits", CanonicalSampler.kinds()) == config["phase_space.detection.n_visits"]

    # The most specific kind wins
    configured(breakout_bracketed__n_visits=3)
    assert SurveySettings.default("n_visits", BracketedBreakoutSampler.kinds()) == 3
    assert SurveySettings.default("n_visits", BreakoutSampler.kinds()) == config["phase_space.breakout.n_visits"]


def test_the_survey_settings_are_resolved_in_one_place_and_refuse_what_they_do_not_know():
    from uvex_transients.phase_space.sampling import SurveySettings

    settings = SurveySettings.resolve(("canonical",), band="NUV", n_visits=3, nside=NSIDE, order="nested")
    assert (settings.band, settings.n_visits, settings.nside, settings.order) == ("NUV", 3, NSIDE, "nested")
    assert settings.snr_threshold == config["phase_space.detection.snr_threshold"] and settings.detection is None
    assert set(settings.as_dict()) == {*SurveySettings.SECTIONS, "nside", "order"}
    with pytest.raises(TypeError, match="bogus"):
        SurveySettings.resolve(bogus=1)
    with pytest.raises(ValueError, match="cut"):
        SurveySettings.resolve(detection=[{"cut": "no_such_cut"}])
