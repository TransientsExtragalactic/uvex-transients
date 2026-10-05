"""`click.testing.CliRunner` smoke tests for `uvex_transients.cli.main`."""

from pathlib import Path

from astropy.table import QTable
from click.testing import CliRunner

from uvex_transients.cli.main import cli
from uvex_transients.simulation.event_catalog import EventCatalog

CONFIG_TEMPLATE = """
schedule:
  path: {schedule_path}
  fov_path: {fov_path}

transients:
  tde:
    class: TidalDisruptionEvent

generate:
  time_bins: 2
  nside: 16
  seed: 1

steps:
  - id: cut_1
    type: cut
    cut: limiting_magnitude
    input: baseline
    params:
      mag_limit: 25.0
  - id: cut_2
    type: cut
    cut: snr
    input: cut_1
    params:
      snr_threshold: 5.0
  - id: phot
    type: action
    action: photometry
    inputs:
      catalog: cut_2
"""


def _write_config(tmp_path, make_schedule) -> str:
    """Write a synthetic schedule and a matching run-config YAML to `tmp_path`; return the config path."""
    schedule = make_schedule()
    schedule_path = tmp_path / "schedule.ecsv"
    fov_path = tmp_path / "schedule.reg"
    schedule.to_disk(schedule_path, fov_path=fov_path)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG_TEMPLATE.format(schedule_path=schedule_path, fov_path=fov_path))
    return str(config_path)


def test_help_shows_the_logo_banner():
    """`--help` prints the package's ASCII banner above the usual click help text."""
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "generate" in result.output
    assert "cut" in result.output
    # The logo is drawn from block characters, not literal text -- just check it's non-trivially long.
    assert result.output.count("\n") > 15


def test_bare_invocation_shows_help_too():
    """Invoking with no subcommand also shows the help/banner (click's `no_args_is_help` default)."""
    result = CliRunner().invoke(cli, [])
    assert "generate" in result.output


def test_generate_writes_a_catalog(tmp_path, make_schedule):
    """`generate` writes an `EventCatalog` file readable back via `EventCatalog.from_disk`."""
    config_path = _write_config(tmp_path, make_schedule)
    out_path = tmp_path / "catalog.ecsv"

    result = CliRunner().invoke(cli, ["generate", config_path, "--out", str(out_path)])

    assert result.exit_code == 0, result.output
    assert out_path.exists()
    catalog = EventCatalog.from_disk(out_path)
    assert catalog.nside == 16


def test_run_end_to_end_produces_every_stage_file(tmp_path, make_schedule):
    """`run` chains generate -> both cuts -> photometry, writing every checkpointed step's file to `--out-dir`."""
    config_path = _write_config(tmp_path, make_schedule)
    out_dir = tmp_path / "results"

    result = CliRunner().invoke(cli, ["run", config_path, "--out-dir", str(out_dir)])

    assert result.exit_code == 0, result.output
    assert (out_dir / "baseline.ecsv").exists()
    assert (out_dir / "exposure.ecsv").exists()
    assert (out_dir / "01_cut_1.ecsv").exists()
    assert (out_dir / "02_cut_2.ecsv").exists()
    phot_path = out_dir / "03_phot.ecsv"
    assert phot_path.exists()
    phot = QTable.read(phot_path)
    assert set(phot.colnames) >= {"event_id", "obs_time", "band", "snr"}


def test_run_writes_event_summary_when_configured(tmp_path, make_schedule):
    """A config with an `event_summary` action step makes `run` also write its checkpoint."""
    config_path = _write_config(tmp_path, make_schedule)
    Path(config_path).write_text(
        Path(config_path).read_text()
        + """
  - id: es
    type: action
    action: event_summary
    inputs:
      catalog: cut_2
      exposure: exposure
      photometry: phot
    params:
      snr_threshold: 5.0
      footprints: []
"""
    )
    out_dir = tmp_path / "results"

    result = CliRunner().invoke(cli, ["run", config_path, "--out-dir", str(out_dir)])

    assert result.exit_code == 0, result.output
    es_path = out_dir / "04_es.ecsv"
    assert es_path.exists()
    table = QTable.read(es_path)
    assert set(table.colnames) >= {"event_id", "weight", "n_det", "t_first_det", "t_alert", "peak_snr"}


def test_run_no_keep_intermediate_writes_only_checkpointed_steps(tmp_path, make_schedule):
    """`run --no-keep-intermediate` skips every step whose own `checkpoint:` is unset."""
    config_path = _write_config(tmp_path, make_schedule)
    sparse_out_dir = tmp_path / "sparse"

    result = CliRunner().invoke(cli, ["run", config_path, "--out-dir", str(sparse_out_dir), "--no-keep-intermediate"])

    assert result.exit_code == 0, result.output
    assert not (sparse_out_dir / "baseline.ecsv").exists()
    assert not (sparse_out_dir / "01_cut_1.ecsv").exists()
    assert not (sparse_out_dir / "02_cut_2.ecsv").exists()
    assert not (sparse_out_dir / "03_phot.ecsv").exists()
    # `exposure.ecsv` is always written, unconditionally.
    assert (sparse_out_dir / "exposure.ecsv").exists()


def test_run_keep_intermediate_config_key_is_honored_without_the_cli_flag(tmp_path, make_schedule):
    """A config's `keep_intermediate: false` suppresses stage files even without `--no-keep-intermediate`."""
    config_path = _write_config(tmp_path, make_schedule)
    Path(config_path).write_text(Path(config_path).read_text() + "\nkeep_intermediate: false\n")
    out_dir = tmp_path / "results"

    result = CliRunner().invoke(cli, ["run", config_path, "--out-dir", str(out_dir)])

    assert result.exit_code == 0, result.output
    assert not (out_dir / "baseline.ecsv").exists()
    assert not (out_dir / "03_phot.ecsv").exists()
    assert (out_dir / "exposure.ecsv").exists()


def test_run_keep_intermediate_cli_flag_overrides_the_config_key(tmp_path, make_schedule):
    """`--keep-intermediate` on the CLI overrides a config's `keep_intermediate: false`."""
    config_path = _write_config(tmp_path, make_schedule)
    Path(config_path).write_text(Path(config_path).read_text() + "\nkeep_intermediate: false\n")
    out_dir = tmp_path / "results"

    result = CliRunner().invoke(cli, ["run", config_path, "--out-dir", str(out_dir), "--keep-intermediate"])

    assert result.exit_code == 0, result.output
    assert (out_dir / "baseline.ecsv").exists()


def test_run_resumes_from_an_existing_checkpoint_without_overwrite(tmp_path, make_schedule):
    """Rerunning `run` without `--overwrite` loads an already-checkpointed step instead of recomputing it."""
    config_path = _write_config(tmp_path, make_schedule)
    out_dir = tmp_path / "results"

    first = CliRunner().invoke(cli, ["run", config_path, "--out-dir", str(out_dir)])
    assert first.exit_code == 0, first.output
    original_mag_limit = (out_dir / "01_cut_1.ecsv").read_text()

    second = CliRunner().invoke(cli, ["run", config_path, "--out-dir", str(out_dir)])
    assert second.exit_code == 0, second.output
    assert "loaded from checkpoint" in second.output
    assert (out_dir / "01_cut_1.ecsv").read_text() == original_mag_limit


def test_a_step_specific_checkpoint_filename_is_honored(tmp_path, make_schedule):
    """A step's own `checkpoint: "<name>"` overrides the default numbered filename."""
    config_path = _write_config(tmp_path, make_schedule)
    Path(config_path).write_text(
        Path(config_path)
        .read_text()
        .replace(
            "  - id: cut_1\n    type: cut\n    cut: limiting_magnitude\n    input: baseline\n    params:\n"
            "      mag_limit: 25.0\n",
            "  - id: cut_1\n    type: cut\n    cut: limiting_magnitude\n    input: baseline\n"
            "    checkpoint: mag_cut.ecsv\n    params:\n      mag_limit: 25.0\n",
        )
    )
    out_dir = tmp_path / "results"

    result = CliRunner().invoke(cli, ["run", config_path, "--out-dir", str(out_dir)])

    assert result.exit_code == 0, result.output
    assert (out_dir / "mag_cut.ecsv").exists()
    assert not (out_dir / "01_cut_1.ecsv").exists()


# --------------------------------------------------------------------------- #
# --dry-run                                                                   #
# --------------------------------------------------------------------------- #
def test_generate_dry_run_reports_the_plan_and_writes_nothing(tmp_path, make_schedule):
    """`generate --dry-run` validates the config, prints the plan, and creates no output file."""
    config_path = _write_config(tmp_path, make_schedule)
    out_path = tmp_path / "catalog.ecsv"

    result = CliRunner().invoke(cli, ["generate", config_path, "--out", str(out_path), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "dry run: generate" in result.output
    assert "TidalDisruptionEvent" in result.output
    assert "time_bins=2" in result.output
    assert f"would write  {out_path}" in result.output
    assert "nothing was sampled or written" in result.output
    assert not out_path.exists()


def test_run_dry_run_lists_every_stage_file_and_creates_no_directory(tmp_path, make_schedule):
    """`run --dry-run` lists each step and every stage file it would write, without even creating OUT_DIR."""
    config_path = _write_config(tmp_path, make_schedule)
    out_dir = tmp_path / "results"

    result = CliRunner().invoke(cli, ["run", config_path, "--out-dir", str(out_dir), "--dry-run"])

    assert result.exit_code == 0, result.output
    for name in ("baseline.ecsv", "exposure.ecsv", "01_cut_1.ecsv", "02_cut_2.ecsv", "03_phot.ecsv"):
        assert f"would write  {out_dir / name}" in result.output
    assert "limiting_magnitude" in result.output
    assert not out_dir.exists()


def test_run_dry_run_no_keep_intermediate_lists_only_exposure(tmp_path, make_schedule):
    """`run --dry-run --no-keep-intermediate` lists only the always-written `exposure.ecsv`."""
    config_path = _write_config(tmp_path, make_schedule)
    out_dir = tmp_path / "results"

    result = CliRunner().invoke(
        cli, ["run", config_path, "--out-dir", str(out_dir), "--dry-run", "--no-keep-intermediate"]
    )

    assert result.exit_code == 0, result.output
    assert f"would write  {out_dir / 'exposure.ecsv'}" in result.output
    for name in ("baseline.ecsv", "01_cut_1.ecsv", "02_cut_2.ecsv", "03_phot.ecsv"):
        assert name not in result.output


def test_dry_run_flags_an_existing_output_and_fails(tmp_path, make_schedule):
    """A dry run exits non-zero when the real command would refuse to overwrite an existing file, unless `--overwrite`."""
    config_path = _write_config(tmp_path, make_schedule)
    out_path = tmp_path / "catalog.ecsv"
    out_path.write_text("already here")

    blocked = CliRunner().invoke(cli, ["generate", config_path, "--out", str(out_path), "--dry-run"])
    assert blocked.exit_code == 1
    assert "WOULD FAIL" in blocked.output

    allowed = CliRunner().invoke(cli, ["generate", config_path, "--out", str(out_path), "--dry-run", "--overwrite"])
    assert allowed.exit_code == 0, allowed.output
    assert out_path.read_text() == "already here"


def test_dry_run_reports_a_bad_config_as_a_clean_error(tmp_path, make_schedule):
    """Config errors surface in a dry run as a short message rather than a traceback."""
    config_path = _write_config(tmp_path, make_schedule)
    bad = tmp_path / "bad.yaml"
    bad.write_text(open(config_path).read().replace("TidalDisruptionEvent", "NotARealTransient"))

    result = CliRunner().invoke(cli, ["generate", str(bad), "--out", str(tmp_path / "c.ecsv"), "--dry-run"])

    assert result.exit_code != 0
    assert "dry run failed" in result.output
    assert "NotARealTransient" in result.output


def test_cut_dry_run_rejects_an_unknown_step_id(tmp_path, make_schedule):
    """`cut --dry-run` validates the requested step ids against the config, without needing to read the catalog."""
    config_path = _write_config(tmp_path, make_schedule)
    catalog_path = tmp_path / "in.ecsv"
    catalog_path.write_text("placeholder")  # only has to exist; a dry run never reads it

    ok = CliRunner().invoke(
        cli, ["cut", config_path, "cut_2", "--in", str(catalog_path), "--out", str(tmp_path / "o.ecsv"), "--dry-run"]
    )
    assert ok.exit_code == 0, ok.output
    assert "steps (1, in order)" in ok.output

    bad = CliRunner().invoke(
        cli, ["cut", config_path, "nope", "--in", str(catalog_path), "--out", str(tmp_path / "o.ecsv"), "--dry-run"]
    )
    assert bad.exit_code != 0
    assert "No step with id 'nope'" in bad.output
