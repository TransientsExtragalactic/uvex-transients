"""Tests that the phase-space defaults come from the configuration, and that an argument overrides them."""

import numpy as np
import pytest
import synphot
from astropy import units as u

pytest.importorskip("sklearn")

from uvex_transients.phase_space import map as map_module  # noqa: E402
from uvex_transients.phase_space import overlay as overlay_module  # noqa: E402
from uvex_transients.phase_space import transient as transient_module  # noqa: E402
from uvex_transients.phase_space.overlay import measure_class  # noqa: E402
from uvex_transients.phase_space.sampling import base as base_module  # noqa: E402
from uvex_transients.phase_space.sampling import CanonicalSampler, StoppingRule  # noqa: E402
from uvex_transients.phase_space.sampling import core as core_module  # noqa: E402
from uvex_transients.phase_space.transient import _CanonicalTransient  # noqa: E402
from uvex_transients.transients.TDEs import TidalDisruptionEvent  # noqa: E402
from uvex_transients.utils import config  # noqa: E402

NSIDE = 32

# Every key of ``config["phase_space"]`` the code reads, with the type it is read as
KEYS = {
    "transient.spectral_index": float,
    "transient.rise_ratio": float,
    "transient.fade_mag": float,
    "breakout.log_L0": list,
    "breakout.log_t0": list,
    "breakout.n_visits": int,
    "breakout.time_bins": int,
    "breakout.z_max": float,
    "breakout.table_points": int,
    "recovery.mission": str,
    "recovery.band": str,
    "recovery.snr_threshold": float,
    "recovery.n_visits": int,
    "recovery.exclude_first_visit_detections": bool,
    "recovery.mag_limit": float,
    "recovery.time_bins": int,
    "recovery.chunk_size": int,
    "recovery.k_target": int,
    "recovery.n_max": int,
    "numerics.exposure_spacing": float,
    "numerics.z_table_min": float,
    "numerics.z_table_points": int,
    "numerics.volume_z_min": float,
    "numerics.volume_points": int,
    "sampler.n_initial": int,
    "sampler.batch_size": int,
    "sampler.max_points": int,
    "sampler.tolerance": float,
    "sampler.acquisition": str,
    "sampler.n_candidates": int,
    "sampler.floor": float,
    "sampler.n_restarts": int,
    "map.log_luminosity": list,
    "map.log_half_time": list,
    "map.grid_size": int,
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
        for module in (
            map_module,
            overlay_module,
            transient_module,
            base_module,
            core_module,
        ):
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
    assert default.spectral_index == config["phase_space.transient.spectral_index"]
    assert default.rise_ratio == config["phase_space.transient.rise_ratio"]
    assert default.fade_mag == config["phase_space.transient.fade_mag"]

    configured(transient__spectral_index=-0.5, transient__rise_ratio=0.3, transient__fade_mag=4.0)
    changed = _CanonicalTransient()
    assert (changed.spectral_index, changed.rise_ratio, changed.fade_mag) == (-0.5, 0.3, 4.0)
    # An argument is not overridden by the configuration
    assert _CanonicalTransient(spectral_index=1.0).spectral_index == 1.0


# ---------------------------------------------------------------------------- #
# The sampler and the stopping rule                                            #
# ---------------------------------------------------------------------------- #
def test_a_sampler_given_nothing_takes_every_setting_from_the_configuration(schedule):
    settings = CanonicalSampler(schedule, nside=NSIDE, order="nested").settings
    for key in (
        "mission",
        "band",
        "snr_threshold",
        "n_visits",
        "exclude_first_visit_detections",
        "mag_limit",
        "time_bins",
    ):
        assert settings[key] == config[f"phase_space.recovery.{key}"], key
    for key in ("spectral_index", "rise_ratio", "fade_mag"):
        assert settings[key] == config[f"phase_space.transient.{key}"], key


def test_a_sampler_follows_a_changed_configuration_and_an_argument_overrides_it(schedule, configured):
    configured(
        recovery__band="NUV",
        recovery__snr_threshold=3.0,
        recovery__n_visits=1,
        recovery__exclude_first_visit_detections=False,
        transient__spectral_index=-1.0,
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
        config[f"phase_space.recovery.{key}"] for key in ("chunk_size", "k_target", "n_max")
    )
    configured(recovery__k_target=7, recovery__chunk_size=50, recovery__n_max=200)
    assert StoppingRule.from_config() == StoppingRule(chunk_size=50, k_target=7, n_max=200)
    assert StoppingRule.from_config(k_target=9).k_target == 9


def test_the_tables_are_sized_by_the_configuration(schedule, configured):
    configured(numerics__z_table_min=1e-4, numerics__z_table_points=64, numerics__volume_points=100)
    sampler = CanonicalSampler(schedule, nside=NSIDE, order="nested")
    assert sampler._limit.z_min == 1e-4
    assert len(sampler._z_grid) == 101


def test_a_rebuilt_sampler_keeps_what_was_recorded_not_what_the_configuration_says_now(schedule, configured):
    recorded = CanonicalSampler(schedule, snr_threshold=4.0, nside=NSIDE, order="nested", schedule_name=None).settings
    configured(recovery__snr_threshold=9.0, recovery__n_visits=3)
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
