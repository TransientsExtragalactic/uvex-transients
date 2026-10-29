"""Tests that the phase-space defaults come from the configuration, and that an argument overrides them."""

import numpy as np
import pytest
import synphot
from astropy import units as u

pytest.importorskip("sklearn")

from uvex_transients.phase_space import evaluator as evaluator_module  # noqa: E402
from uvex_transients.phase_space import overlay as overlay_module  # noqa: E402
from uvex_transients.phase_space import sampler as sampler_module  # noqa: E402
from uvex_transients.phase_space import transient as transient_module  # noqa: E402
from uvex_transients.phase_space.evaluator import RecoveryEvaluator  # noqa: E402
from uvex_transients.phase_space.overlay import measure_class  # noqa: E402
from uvex_transients.phase_space.sampler import PhaseSpaceSampler  # noqa: E402
from uvex_transients.phase_space.transient import _CanonicalTransient  # noqa: E402
from uvex_transients.transients.TDEs import TidalDisruptionEvent  # noqa: E402
from uvex_transients.utils import config  # noqa: E402

NSIDE = 32
BOUNDS = [(41.0, 43.0), (0.0, 2.0)]

# Every key of ``config["phase_space"]`` the code reads, with the type it is read as
KEYS = {
    "transient.spectral_index": float,
    "transient.rise_ratio": float,
    "transient.fade_mag": float,
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
        for module in (evaluator_module, overlay_module, sampler_module, transient_module):
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
# The evaluator                                                                #
# ---------------------------------------------------------------------------- #
def test_an_evaluator_given_nothing_takes_every_setting_from_the_configuration(schedule):
    settings = RecoveryEvaluator(schedule, nside=NSIDE, order="nested").settings
    for key in (
        "mission",
        "band",
        "snr_threshold",
        "n_visits",
        "exclude_first_visit_detections",
        "mag_limit",
        "time_bins",
        "chunk_size",
        "k_target",
        "n_max",
    ):
        assert settings[key] == config[f"phase_space.recovery.{key}"], key
    for key in ("spectral_index", "rise_ratio", "fade_mag"):
        assert settings[key] == config[f"phase_space.transient.{key}"], key


def test_an_evaluator_follows_a_changed_configuration_and_an_argument_overrides_it(schedule, configured):
    configured(
        recovery__band="NUV",
        recovery__snr_threshold=3.0,
        recovery__n_visits=1,
        recovery__exclude_first_visit_detections=False,
        recovery__k_target=7,
        recovery__chunk_size=50,
        recovery__n_max=200,
        transient__spectral_index=-1.0,
    )
    evaluator = RecoveryEvaluator(schedule, nside=NSIDE, order="nested")
    assert evaluator.band == "NUV" and evaluator.snr_threshold == 3.0 and evaluator.n_visits == 1
    assert evaluator.exclude_first_visit_detections is False
    assert (evaluator.k_target, evaluator.chunk_size, evaluator.n_max) == (7, 50, 200)
    assert evaluator.spectral_index == -1.0 and evaluator.settings["spectral_index"] == -1.0

    explicit = RecoveryEvaluator(schedule, band="FUV", n_visits=4, nside=NSIDE, order="nested")
    assert explicit.band == "FUV" and explicit.n_visits == 4 and explicit.snr_threshold == 3.0


def test_the_tables_are_sized_by_the_configuration(schedule, configured):
    configured(numerics__z_table_min=1e-4, numerics__z_table_points=64, numerics__volume_points=100)
    evaluator = RecoveryEvaluator(schedule, nside=NSIDE, order="nested")
    assert evaluator.redshift_limit.z_min == 1e-4
    assert len(evaluator._z_grid) == 101


def test_a_rebuilt_evaluator_keeps_what_was_recorded_not_what_the_configuration_says_now(schedule, configured):
    recorded = RecoveryEvaluator(schedule, snr_threshold=4.0, nside=NSIDE, order="nested", schedule_name=None).settings
    configured(recovery__snr_threshold=9.0, recovery__n_visits=3)
    rebuilt = RecoveryEvaluator.from_settings(recorded, schedule=schedule)
    assert rebuilt.snr_threshold == 4.0 and rebuilt.n_visits == recorded["n_visits"]


# ---------------------------------------------------------------------------- #
# The sampler                                                                  #
# ---------------------------------------------------------------------------- #
class _Evaluator:
    """A callable that reports what it is expected to do, as `RecoveryEvaluator` does."""

    k_target, chunk_size, n_max = 11, 222, 3333

    def __call__(self, x):
        return {"k": 1, "n": 2}


def test_a_sampler_given_nothing_takes_its_settings_from_the_configuration(configured):
    sampler = PhaseSpaceSampler(lambda x: {"k": 1, "n": 2}, BOUNDS)
    assert sampler.settings["n_initial"] == config["phase_space.sampler.n_initial"]
    assert sampler.settings["acquisition"] == config["phase_space.sampler.acquisition"]
    assert sampler.settings["max_points"] == config["phase_space.sampler.max_points"]

    configured(sampler__n_initial=5, sampler__batch_size=3, sampler__acquisition="variance", sampler__floor=2e-3)
    changed = PhaseSpaceSampler(lambda x: {"k": 1, "n": 2}, BOUNDS)
    assert (changed.n_initial, changed.batch_size, changed.acquisition, changed.floor) == (5, 3, "variance", 2e-3)
    assert PhaseSpaceSampler(lambda x: {"k": 1, "n": 2}, BOUNDS, n_initial=9).n_initial == 9


def test_a_sampler_expects_of_its_evaluator_what_the_evaluator_does():
    sampler = PhaseSpaceSampler(_Evaluator(), BOUNDS)
    assert (sampler.k_target, sampler.n_min, sampler.n_max) == (11.0, 222.0, 3333.0)

    # An explicit value wins, and a callable that says nothing falls back to the recovery settings of the configuration
    assert PhaseSpaceSampler(_Evaluator(), BOUNDS, k_target=5).k_target == 5.0
    plain = PhaseSpaceSampler(lambda x: {"k": 1, "n": 2}, BOUNDS)
    assert plain.k_target == config["phase_space.recovery.k_target"]
    assert plain.n_min == config["phase_space.recovery.chunk_size"]
    assert plain.n_max == config["phase_space.recovery.n_max"]


class _Scales(_Evaluator):
    """The noise-free scales a map table asks of an evaluator."""

    def redshift_limit(self, log_luminosity):
        return 0.1 + 0 * np.asarray(log_luminosity)

    def coverage(self, log_luminosity, log_half_time):
        return 0.5 + 0 * np.asarray(log_luminosity) + 0 * np.asarray(log_half_time)

    volume_time = coverage

    def all_sky_volume_time(self, log_luminosity):
        return 1.0 + 0 * np.asarray(log_luminosity)


def test_the_map_grid_takes_its_size_from_the_configuration(configured):
    def evaluate(sampler):
        return {"k": 3, "n": 100}

    sampler = PhaseSpaceSampler(_Scales(), BOUNDS, n_initial=4, max_points=4, n_restarts=0, seed=1)
    sampler.run()
    configured(map__grid_size=5)
    assert len(sampler.map_table()) == 25
    assert len(sampler.map_table(4)) == 16


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
