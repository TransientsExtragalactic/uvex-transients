"""Tests for the ``phase-space`` command and `uvex_transients.cli.phase_space`."""

import pytest
from astropy import units as u
from click.testing import CliRunner

pytest.importorskip("sklearn")

from uvex_transients.cli.main import cli  # noqa: E402
from uvex_transients.cli.phase_space import PhaseSpaceConfig  # noqa: E402
from uvex_transients.utils import config  # noqa: E402

CONFIG_TEMPLATE = """
schedule:
  path: {schedule_path}
  fov_path: {fov_path}
mission: uvex_fast
seed: 3
overlay: {{n_draws: 20}}
recovery: {{snr_threshold: 5.0, n_visits: 1, mag_limit: 24.0, time_bins: 3, chunk_size: 40, k_target: 2, n_max: 80, nside: 32, order: nested}}
sampler: {{n_initial: 4, batch_size: 2, max_points: 4}}
map: {{log_luminosity: [43.5, 45.0], log_half_time: [1.5, 2.5], grid_size: 5}}
maps:
  - id: fuv_test
    recovery: {{band: FUV}}
  - id: nuv_test
    recovery: {{band: NUV, snr_threshold: 3.0}}
"""


@pytest.fixture
def config_path(tmp_path, make_schedule):
    """A phase-space run-config over a synthetic schedule, with two small maps."""
    schedule_path, fov_path = tmp_path / "schedule.ecsv", tmp_path / "schedule.reg"
    make_schedule().to_disk(schedule_path, fov_path=fov_path)
    path = tmp_path / "phase_space.yaml"
    path.write_text(CONFIG_TEMPLATE.format(schedule_path=schedule_path, fov_path=fov_path))
    return path


def _parse(text: str) -> PhaseSpaceConfig:
    from uvex_transients.cli.yaml_tags import get_run_yaml

    return PhaseSpaceConfig(get_run_yaml().load(text))


# ---------------------------------------------------------------------------- #
# The run-config                                                               #
# ---------------------------------------------------------------------------- #
def test_the_shipped_run_config_parses_and_asks_for_the_agreed_detection_maps():
    from pathlib import Path

    parsed = PhaseSpaceConfig.from_yaml(Path(__file__).parents[2] / "configs" / "phase_space.yaml")
    assert [spec.id for spec in parsed.maps] == ["fuv_detection", "nuv_detection"]
    assert [spec.band for spec in parsed.maps] == ["FUV", "NUV"]
    for spec in parsed.maps:
        assert spec.recovery["snr_threshold"] == 5.0 and spec.recovery["n_visits"] == 2
        assert spec.recovery["mag_limit"] == 25.0
    assert parsed.overlay and parsed.schedule_name == config["schedules.default_schedule"]


def test_a_map_overrides_the_shared_settings_key_by_key():
    parsed = _parse(
        """
recovery: {snr_threshold: 5.0, n_visits: 2}
maps:
  - id: a
    recovery: {band: FUV}
  - id: b
    recovery: {band: NUV, n_visits: 1}
"""
    )
    a, b = parsed.maps
    assert a.recovery == {"snr_threshold": 5.0, "n_visits": 2, "band": "FUV"}
    assert b.recovery == {"snr_threshold": 5.0, "n_visits": 1, "band": "NUV"}
    assert a.bounds is None and a.grid_size is None


def test_the_map_section_sets_the_bounds_and_the_grid():
    parsed = _parse("map: {log_luminosity: [40, 42], grid_size: 9}\nmaps: [{id: a}]")
    (spec,) = parsed.maps
    assert spec.grid_size == 9
    assert spec.bounds[0] == (40, 42) and spec.bounds[1] == tuple(config["phase_space.map.log_half_time"])


def test_every_problem_is_reported_at_once():
    with pytest.raises(ValueError) as error:
        _parse(
            """
recovery: {snr: 5.0}
maps:
  - id: ok
  - id: ok
  - id: ../bad
  - recovery: {bogus: 1}
"""
        )
    message = str(error.value)
    assert "unknown 'recovery:' key(s) ['snr']" in message
    assert "the id is used twice" in message
    assert "'../bad'" in message and "None" in message


@pytest.mark.parametrize("text", ["maps: []", "seed: 1", "bogus: 1\nmaps: [{id: a}]"])
def test_a_run_config_needs_maps_and_knows_its_keys(text):
    with pytest.raises(ValueError):
        _parse(text)


def test_select_keeps_the_order_of_the_file_and_names_what_is_missing():
    parsed = _parse("maps: [{id: a}, {id: b}, {id: c}]")
    assert [spec.id for spec in parsed.select(["c", "a"])] == ["a", "c"]
    assert [spec.id for spec in parsed.select(())] == ["a", "b", "c"]
    with pytest.raises(ValueError, match=r"\['x'\]"):
        parsed.select(["x"])


def test_the_rate_and_overlay_options():
    parsed = _parse("rate: 100\noverlay: {n_draws: 7}\nmaps: [{id: a}]")
    assert parsed.rate.unit == u.Gpc**-3 / u.yr and parsed.rate.value == 100
    assert parsed.overlay and parsed.overlay_draws == 7
    off = _parse("maps: [{id: a}]")
    assert not off.overlay and off.rate is None


# ---------------------------------------------------------------------------- #
# The command                                                                  #
# ---------------------------------------------------------------------------- #
def test_the_command_is_listed():
    result = CliRunner().invoke(cli, ["--help"])
    assert "phase-space" in result.output


def test_dry_run_reports_each_map_and_writes_nothing(config_path, tmp_path):
    out = tmp_path / "out"
    result = CliRunner().invoke(cli, ["phase-space", str(config_path), "--out-dir", str(out), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "fuv_test: FUV, SNR >= 5 on 1 visit(s), 24 AB" in result.output
    assert "nuv_test: NUV, SNR >= 3 on 1 visit(s), 24 AB" in result.output
    assert not out.exists()


def test_dry_run_fails_on_a_band_the_mission_lacks_or_on_existing_output(config_path, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(config_path.read_text().replace("band: NUV,", "band: XRAY,"))
    result = CliRunner().invoke(cli, ["phase-space", str(bad), "--out-dir", str(tmp_path / "out"), "--dry-run"])
    assert result.exit_code == 1 and "XRAY" in result.output

    taken = tmp_path / "taken"
    (taken / "fuv_test").mkdir(parents=True)
    (taken / "fuv_test" / "points.ecsv").write_text("x")
    result = CliRunner().invoke(cli, ["phase-space", str(config_path), "--out-dir", str(taken), "--dry-run"])
    assert result.exit_code == 1 and "already exist" in result.output
    ok = CliRunner().invoke(cli, ["phase-space", str(config_path), "--out-dir", str(taken), "--dry-run", "--overwrite"])
    assert ok.exit_code == 0


def test_a_bad_config_is_a_click_error_not_a_traceback(config_path, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("maps: []\n")
    result = CliRunner().invoke(cli, ["phase-space", str(bad), "--out-dir", str(tmp_path / "out")])
    assert result.exit_code == 1 and "maps" in result.output
    result = CliRunner().invoke(cli, ["phase-space", str(config_path), "--out-dir", str(tmp_path), "--map", "nope"])
    assert result.exit_code == 1 and "nope" in result.output


def test_a_run_writes_one_directory_per_map(config_path, tmp_path):
    out = tmp_path / "out"
    result = CliRunner().invoke(cli, ["phase-space", str(config_path), "--out-dir", str(out), "--map", "fuv_test"])
    assert result.exit_code == 0, result.output
    assert sorted(path.name for path in out.iterdir()) == ["fuv_test"]
    assert sorted(path.name for path in (out / "fuv_test").iterdir()) == [
        "map.ecsv",
        "map.png",
        "overlay.ecsv",
        "points.ecsv",
    ]

    from astropy.table import Table

    points = Table.read(out / "fuv_test" / "points.ecsv", format="ascii.ecsv")
    assert len(points) == 4 and points.meta["evaluator"]["settings"]["band"] == "FUV"
    assert points.meta["evaluator"]["settings"]["mag_limit"] == 24.0
    assert len(Table.read(out / "fuv_test" / "map.ecsv", format="ascii.ecsv")) == 25  # 5 x 5 grid


def test_an_existing_map_is_refused_unless_overwrite_or_resume(config_path, tmp_path):
    out = tmp_path / "out"
    args = ["phase-space", str(config_path), "--out-dir", str(out), "--map", "fuv_test"]
    assert CliRunner().invoke(cli, args).exit_code == 0

    refused = CliRunner().invoke(cli, args)
    assert refused.exit_code == 1 and "overwrite" in refused.output
    assert CliRunner().invoke(cli, [*args, "--overwrite"]).exit_code == 0
    assert CliRunner().invoke(cli, [*args, "--resume"]).exit_code == 0


# ---------------------------------------------------------------------------- #
# A sampler of another kind                                                    #
# ---------------------------------------------------------------------------- #
BREAKOUT_MAP = """
  - id: breakout_test
    kind: breakout
    recovery: {band: FUV, mag_limit: 25.0}
    settings: {z_max: 8.0, table_points: 9}
    map: {bounds: {log_L0: [45.0, 46.5], log_t0: [3.0, 4.5]}, grid_size: 5}
"""


@pytest.fixture
def breakout_config_path(config_path):
    path = config_path.with_name("breakout.yaml")
    path.write_text(config_path.read_text() + BREAKOUT_MAP)
    return path


def test_a_map_is_canonical_unless_it_names_a_kind(config_path):
    parsed = PhaseSpaceConfig.from_yaml(config_path)
    assert {spec.kind for spec in parsed.maps} == {"canonical"}
    assert all(spec.settings == {} and spec.axis_bounds is None for spec in parsed.maps)


def test_a_map_of_another_kind_has_its_settings_and_axis_bounds(breakout_config_path):
    spec = PhaseSpaceConfig.from_yaml(breakout_config_path).select(["breakout_test"])[0]
    assert spec.kind == "breakout"
    assert spec.settings == {"z_max": 8.0, "table_points": 9}
    assert spec.axis_bounds == {"log_L0": (45.0, 46.5), "log_t0": (3.0, 4.5)}
    assert spec.grid_size == 5 and spec.recovery["band"] == "FUV"


def test_a_kind_can_be_given_at_the_top_level_and_overridden_by_a_map():
    parsed = _parse("kind: breakout\nmaps:\n  - id: a\n  - id: b\n    kind: canonical\n")
    assert [spec.kind for spec in parsed.maps] == ["breakout", "canonical"]


def test_an_unknown_kind_is_reported_with_the_known_ones():
    with pytest.raises(ValueError, match=r"unknown kind 'nothing'.*breakout"):
        _parse("maps:\n  - id: a\n    kind: nothing\n")


def test_the_transient_section_is_only_for_the_canonical_kind():
    with pytest.raises(ValueError, match="transient.*canonical"):
        _parse("maps:\n  - id: a\n    kind: breakout\n    transient: {rise_ratio: 0.2}\n")


def test_axis_bounds_have_to_map_names_to_pairs():
    with pytest.raises(ValueError, match="bounds"):
        _parse("maps:\n  - id: a\n    kind: breakout\n    map: {bounds: [1, 2]}\n")


def test_settings_are_free_form_and_the_dry_run_names_the_kind(breakout_config_path, tmp_path):
    result = CliRunner().invoke(
        cli, ["phase-space", str(breakout_config_path), "--out-dir", str(tmp_path / "o"), "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "breakout_test [breakout]: FUV" in result.output


def test_a_breakout_map_is_run_as_an_experiment_and_writes_its_files(breakout_config_path, tmp_path):
    from astropy.table import Table

    out = tmp_path / "out"
    args = ["phase-space", str(breakout_config_path), "--out-dir", str(out), "--map", "breakout_test"]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert sorted(path.name for path in (out / "breakout_test").iterdir()) == ["map.ecsv", "map.png", "points.ecsv"]

    points = Table.read(out / "breakout_test" / "points.ecsv", format="ascii.ecsv")
    assert points.meta["schema_version"] == 2 and points.meta["sampler"]["kind"] == "breakout"
    assert points.meta["sampler"]["settings"]["z_max"] == 8.0
    assert points.colnames[:2] == ["log_L0", "log_t0"] and len(points) == 4
    assert points["log_L0"].min() >= 45.0 and points["log_t0"].min() >= 3.0

    grid = Table.read(out / "breakout_test" / "map.ecsv", format="ascii.ecsv")
    assert len(grid) == 25 and "recovered_volume_time" in grid.colnames


def test_a_setting_the_sampler_does_not_take_and_an_axis_it_does_not_have_are_errors(breakout_config_path, tmp_path):
    for old, new, message in (
        ("z_max: 8.0", "no_such_setting: 1", "does not take"),
        ("log_L0: [45.0, 46.5]", "log_Q: [45.0, 46.5]", "no axis"),
    ):
        bad = tmp_path / "bad.yaml"
        bad.write_text(breakout_config_path.read_text().replace(old, new))
        result = CliRunner().invoke(
            cli, ["phase-space", str(bad), "--out-dir", str(tmp_path / "o2"), "--map", "breakout_test"]
        )
        assert result.exit_code != 0 and message in str(result.output) + str(result.exception)
