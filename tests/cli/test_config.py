"""Tests for `uvex_transients.cli.config.RunConfig`."""

import io
from pathlib import Path

import pytest
from astropy import units as u
from astropy.cosmology import FLRW
from m4opt.missions._uvex import uvex

import uvex_transients.transients as transients_pkg
from uvex_transients.cli.config import _KNOWN_TRANSIENT_MODULES, RunConfig
from uvex_transients.cli.yaml_tags import get_run_yaml
from uvex_transients.models.core.priors import NormalPrior
from uvex_transients.transients.TDEs import TidalDisruptionEvent


def _config_from(doc: str) -> RunConfig:
    """Build a `RunConfig` directly from a YAML string, without touching disk."""
    raw = get_run_yaml().load(io.StringIO(doc)) or {}
    return RunConfig(raw)


MINIMAL_TRANSIENTS = "transients:\n  tde:\n    class: TidalDisruptionEvent\n"


def test_schedule_path_and_fov_path_round_trip(tmp_path, make_schedule):
    """`schedule: {path, fov_path}` reads back a schedule written via `SurveySchedule.to_disk`."""
    schedule = make_schedule()
    table_path = tmp_path / "schedule.ecsv"
    fov_path = tmp_path / "schedule.reg"
    schedule.to_disk(table_path, fov_path=fov_path)

    config = _config_from(f"schedule:\n  path: {table_path}\n  fov_path: {fov_path}\n")
    assert len(config.schedule.observe_rows) == len(schedule.observe_rows)


def test_schedule_rejects_more_than_one_source():
    """Giving both `name:` and `url:` (or any two of the three groups) raises."""
    config = _config_from("schedule:\n  name: uvex_initial_main\n  url: https://example.com/x.ecsv\n")
    with pytest.raises(ValueError, match="at most one of"):
        _ = config.schedule


def test_schedule_path_requires_fov_path():
    """`path:` without a companion `fov_path:` raises."""
    config = _config_from("schedule:\n  path: /tmp/does-not-matter.ecsv\n")
    with pytest.raises(ValueError, match="also requires 'fov_path'"):
        _ = config.schedule


def test_mission_defaults_to_uvex():
    """An entirely absent `mission:` section defaults to `uvex`."""
    config = _config_from("x: 1\n")
    assert config.mission is uvex


def test_mission_unknown_name_raises():
    """An unrecognized `mission:` name raises, listing the real mission names."""
    config = _config_from("mission: not_a_real_mission\n")
    with pytest.raises(ValueError, match="Unknown mission 'not_a_real_mission'"):
        _ = config.mission


def test_transients_section_required():
    """A config with no `transients:` section raises when `.transients` is accessed."""
    config = _config_from("x: 1\n")
    with pytest.raises(ValueError, match="missing a 'transients:' section"):
        _ = config.transients


def test_transients_resolves_class_and_overrides():
    """`class:`, `z_limit:`, and `parameters:` (raw value + `!prior`) overrides all apply."""
    doc = """
transients:
  tde:
    class: TidalDisruptionEvent
    z_limit: 0.5
    parameters:
      sigma_rise: 2.0 day
      amplitude: !prior {type: normal, mean: 44.0, sigma: 0.1}
"""
    config = _config_from(doc)
    transient = config.transients["tde"]

    assert isinstance(transient, TidalDisruptionEvent)
    assert transient.redshift_limit == 0.5
    assert transient.sed["sigma_rise"].fixed_value == 2.0 * u.day
    assert transient.sed["amplitude"].prior == NormalPrior(mean=44.0, sigma=0.1)


def test_transients_unknown_class_raises():
    """An unrecognized `class:` name raises, listing the registered transient classes."""
    config = _config_from("transients:\n  x:\n    class: NotARealTransient\n")
    with pytest.raises(ValueError, match="unknown transient class 'NotARealTransient'"):
        _ = config.transients


def test_transients_unknown_parameter_raises():
    """An unrecognized parameter name in `parameters:` raises, listing the real ones."""
    doc = "transients:\n  tde:\n    class: TidalDisruptionEvent\n    parameters:\n      bogus: 1.0\n"
    config = _config_from(doc)
    with pytest.raises(KeyError, match="has no parameter named 'bogus'"):
        _ = config.transients


def test_transients_missing_class_key_raises():
    """A transient entry with no `class:` key raises."""
    config = _config_from("transients:\n  tde:\n    z_limit: 0.5\n")
    with pytest.raises(ValueError, match="missing required key 'class'"):
        _ = config.transients


def test_transients_cosmology_override():
    """A `cosmology:` override (via `!astropy_cosmology`) is forwarded to the transient's constructor."""
    doc = f"{MINIMAL_TRANSIENTS.rstrip()}\n    cosmology: !astropy_cosmology {{name: Planck18}}\n"
    config = _config_from(doc)
    assert isinstance(config.transients["tde"].cosmology, FLRW)


def test_generate_requires_time_bins():
    """`generate:` without `time_bins:` raises."""
    config = _config_from("generate:\n  nside: 32\n")
    with pytest.raises(ValueError, match="missing required key 'time_bins'"):
        _ = config.generate


def test_generate_section_required():
    """A config with no `generate:` section raises, naming the `generate` command."""
    config = _config_from("x: 1\n")
    with pytest.raises(ValueError, match="required by the 'generate' command"):
        _ = config.generate


def test_generate_parses_optional_fields():
    """Optional `generate:` fields parse through; unset ones stay `None`."""
    config = _config_from("generate:\n  time_bins: 10\n  nside: 32\n")
    assert config.generate.time_bins == 10
    assert config.generate.nside == 32
    assert config.generate.downsample is None


def test_cuts_unknown_type_raises():
    """An unrecognized cut `type:` raises, listing `SurveySimulator.available_cuts()`."""
    config = _config_from("cuts:\n  cut_1:\n    type: not_a_real_cut\n")
    with pytest.raises(ValueError, match="unknown cut type 'not_a_real_cut'"):
        _ = config.cuts


def test_cuts_reserved_param_name_raises():
    """A cut `params:` block using a reserved name (`catalog`/`mission`) raises."""
    config = _config_from("cuts:\n  cut_1:\n    type: snr\n    mission: uvex\n")
    with pytest.raises(ValueError, match="reserved name"):
        _ = config.cuts


def test_cuts_resolves_type_and_params_in_order():
    """Cuts parse with their declared `type:`/params and preserve YAML declaration order."""
    doc = """
cuts:
  cut_1:
    type: limiting_magnitude
    mag_limit: 25.0
  cut_2:
    type: snr
    snr_threshold: 5.0
"""
    config = _config_from(doc)
    assert list(config.cuts) == ["cut_1", "cut_2"]
    assert config.cuts["cut_1"].type == "limiting_magnitude"
    assert config.cuts["cut_1"].params == {"mag_limit": 25.0}


def test_photometry_defaults_when_omitted():
    """An entirely absent `photometry:` section defaults every field to `None`."""
    config = _config_from("x: 1\n")
    assert config.photometry.bands is None
    assert config.photometry.n_sigma is None


def test_keep_intermediate_defaults_to_true_when_omitted():
    """An entirely absent `keep_intermediate:` key defaults to `True`."""
    config = _config_from("x: 1\n")
    assert config.keep_intermediate is True


def test_keep_intermediate_reads_the_config_value():
    """`keep_intermediate: false` resolves to `False`."""
    config = _config_from("keep_intermediate: false\n")
    assert config.keep_intermediate is False


def test_config_missing_unrelated_section_still_works():
    """A config with only `transients:`/`schedule:` still resolves `.transients` fine, with no `cuts:`/`generate:`."""
    config = _config_from(MINIMAL_TRANSIENTS)
    assert "tde" in config.transients
    with pytest.raises(ValueError, match="required by the 'cut' command"):
        _ = config.cuts


def test_known_transient_modules_matches_the_package_directory():
    """`_KNOWN_TRANSIENT_MODULES` must list every `.py` file that defines a transient type.

    A future 6th transient type living in a module not listed here would silently never
    register (see `_import_known_transients`'s docstring) -- this fails loudly instead.
    """
    package_dir = Path(transients_pkg.__file__).parent
    actual_modules = {path.stem for path in package_dir.glob("*.py") if path.stem not in {"__init__", "base"}}
    assert actual_modules == set(_KNOWN_TRANSIENT_MODULES)


#: Registered transient classes deliberately left out of `configs/full_run.yaml`: bespoke, first-principles
#: models of one phase of a population another class already covers (counting both would double-count it).
_FULL_RUN_EXCLUDED_CLASSES = {"ShockCoolingIIb"}


def test_full_run_config_lists_every_registered_transient_class():
    """`configs/full_run.yaml` promises "every registered class" -- fail loudly if a new one is missing from it."""
    from uvex_transients.cli.config import _import_known_transients
    from uvex_transients.transients.base import TransientBase

    _import_known_transients()
    registered = {name for name in TransientBase.registry() if not name.startswith("_")}

    config_path = Path(__file__).resolve().parents[2] / "configs" / "full_run.yaml"
    raw = get_run_yaml().load(config_path.read_text())
    listed = {entry["class"] for entry in raw["transients"].values()}

    assert listed == registered - _FULL_RUN_EXCLUDED_CLASSES, (
        f"missing from full_run.yaml: {sorted(registered - _FULL_RUN_EXCLUDED_CLASSES - listed)}; "
        f"unknown to the registry: {sorted(listed - registered)}"
    )
