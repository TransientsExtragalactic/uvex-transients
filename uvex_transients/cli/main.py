"""`click`-based CLI entry point for `uvex_transients` (console script: ``uvex-transients``)."""

from pathlib import Path

import click

from ..simulation.event_catalog import EventCatalog
from ..simulation.exposure_catalog import ExposureCatalog
from ..utils import configure_logging
from . import pipeline, steps
from .config import RunConfig

_LOGO_PATH = Path(__file__).resolve().parents[1] / "_logo.txt"


def _logo_text() -> str:
    """
    Read the package's ASCII banner, or an empty string if it's ever missing (never fatal).

    Returns
    -------
    str
        The banner text, or ``""`` if `_LOGO_PATH` doesn't exist or can't be read.
    """
    try:
        return _LOGO_PATH.read_text()
    except OSError:
        return ""


class _LogoGroup(click.Group):
    """A `click.Group` that prints the package's ASCII banner above its usual help text."""

    def format_help(self, ctx, formatter):
        """
        Print the package's ASCII banner, then delegate to `click.Group.format_help`.

        Parameters
        ----------
        ctx : click.Context
            The current click context.
        formatter : click.HelpFormatter
            The formatter to write help text to.
        """
        logo = _logo_text()
        if logo:
            formatter.write(logo)
            formatter.write("\n")
        super().format_help(ctx, formatter)


CONFIG_ARGUMENT = click.argument(
    "config_path", metavar="CONFIG", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
OVERWRITE_OPTION = click.option("--overwrite", is_flag=True, default=False, help="Overwrite an existing output file.")
DRY_RUN_OPTION = click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Validate CONFIG and report what would run, without sampling, computing, or writing anything.",
)


def _dry_run(config: RunConfig, command: str, outputs, overwrite: bool, step_ids=None) -> None:
    """
    Print `pipeline.dry_run_report` and exit non-zero if the real command would fail on an existing output.

    Parameters
    ----------
    config : RunConfig
        The parsed run-config.
    command : str
        The CLI command being dry-run (e.g. ``"generate"``).
    outputs : list of Path
        The output path(s) the real command would write.
    overwrite : bool
        Whether the real command would be allowed to overwrite existing outputs.
    step_ids : list of str, optional
        For ``"cut"``/``"photometry"``, the step id(s) that would run.
    """
    try:
        lines, ok = pipeline.dry_run_report(config, command, outputs=outputs, overwrite=overwrite, step_ids=step_ids)
    except (ValueError, KeyError, OSError) as error:
        raise click.ClickException(f"dry run failed: {error}") from error
    for line in lines:
        click.echo(line)
    if not ok:
        raise click.exceptions.Exit(1)


@click.group(cls=_LogoGroup)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default=None,
    help="Logging verbosity. Defaults to the configured system.logging.main.level.",
)
def cli(log_level: str | None):
    """
    Simulate UVEX transient populations against a survey schedule.

    Parameters
    ----------
    log_level : str, optional
        If given, overrides the configured logging level for this invocation.
    """
    if log_level is not None:
        configure_logging(log_level.upper())


@cli.command("generate")
@CONFIG_ARGUMENT
@click.option("--out", "out_path", required=True, type=click.Path(dir_okay=False, path_type=Path))
@OVERWRITE_OPTION
@DRY_RUN_OPTION
def generate_command(config_path: Path, out_path: Path, overwrite: bool, dry_run: bool) -> None:
    """
    Sample a Monte Carlo event catalog (needs CONFIG's schedule/transients/mission/generate sections).

    Parameters
    ----------
    config_path : Path
        Path to the run-config YAML file (``CONFIG``).
    out_path : Path
        Destination path for the generated event catalog.
    overwrite : bool
        Whether to overwrite an existing file at `out_path`.
    dry_run : bool
        If True, validate and report without sampling or writing anything.

    Returns
    -------
    None
        Exits the process via ``click`` on failure; otherwise returns nothing.
    """
    config = RunConfig.from_yaml(config_path)
    if dry_run:
        return _dry_run(config, "generate", [out_path], overwrite)
    catalog = pipeline.run_generate(config)
    catalog.to_disk(out_path, overwrite=overwrite)
    click.echo(f"generate: {len(catalog)} events -> {out_path}")


@cli.command("cut")
@CONFIG_ARGUMENT
@click.argument("step_ids", metavar="[STEP_ID]...", nargs=-1)
@click.option("--in", "in_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", "out_path", required=True, type=click.Path(dir_okay=False, path_type=Path))
@OVERWRITE_OPTION
@DRY_RUN_OPTION
def cut_command(
    config_path: Path, step_ids: tuple[str, ...], in_path: Path, out_path: Path, overwrite: bool, dry_run: bool
) -> None:
    """
    Run one or more of CONFIG's declared ``cut``-type steps against a catalog, chained in order.

    STEP_ID names are ids from CONFIG's ``steps:`` list (each must be a ``type: cut`` step);
    with none given, every declared ``cut`` step runs, in declared order. Unlike the full
    ``run``/``steps:`` graph, this ignores each step's own configured ``input:`` and instead
    threads ``--in`` through every selected step.

    Parameters
    ----------
    config_path : Path
        Path to the run-config YAML file (``CONFIG``).
    step_ids : tuple of str
        Cut step ids to run, from CONFIG's ``steps:`` list (``STEP_ID``); empty runs every
        declared ``cut`` step.
    in_path : Path
        Path to the input event catalog.
    out_path : Path
        Destination path for the filtered catalog.
    overwrite : bool
        Whether to overwrite an existing file at `out_path`.
    dry_run : bool
        If True, validate and report without filtering or writing anything.

    Returns
    -------
    None
        Exits the process via ``click`` on failure; otherwise returns nothing.
    """
    config = RunConfig.from_yaml(config_path)
    if dry_run:
        return _dry_run(config, "cut", [out_path], overwrite, step_ids=list(step_ids) or None)
    catalog = EventCatalog.from_disk(in_path)
    result = pipeline.run_cut_steps(config, catalog, step_ids=list(step_ids) or None)
    result.to_disk(out_path, overwrite=overwrite)
    click.echo(f"cut: {len(result)}/{len(catalog)} events survived -> {out_path}")


@cli.command("photometry")
@CONFIG_ARGUMENT
@click.argument("step_id", metavar="STEP_ID")
@click.option("--in", "in_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", "out_path", required=True, type=click.Path(dir_okay=False, path_type=Path))
@OVERWRITE_OPTION
@DRY_RUN_OPTION
def photometry_command(
    config_path: Path, step_id: str, in_path: Path, out_path: Path, overwrite: bool, dry_run: bool
) -> None:
    """
    Run synthetic photometry over every event in a catalog, per one of CONFIG's declared steps.

    STEP_ID is an id from CONFIG's ``steps:`` list, naming a ``type: action``,
    ``action: photometry`` step (for its ``bands``/``n_sigma`` params).

    Parameters
    ----------
    config_path : Path
        Path to the run-config YAML file (``CONFIG``).
    step_id : str
        The photometry step's own id in CONFIG's ``steps:`` list.
    in_path : Path
        Path to the input event catalog.
    out_path : Path
        Destination path for the photometry table.
    overwrite : bool
        Whether to overwrite an existing file at `out_path`.
    dry_run : bool
        If True, validate and report without simulating or writing anything.

    Returns
    -------
    None
        Exits the process via ``click`` on failure; otherwise returns nothing.
    """
    config = RunConfig.from_yaml(config_path)
    if dry_run:
        return _dry_run(config, "photometry", [out_path], overwrite, step_ids=[step_id])
    catalog = EventCatalog.from_disk(in_path)
    phot = pipeline.run_photometry_step(config, step_id, catalog)
    phot.write(out_path, overwrite=overwrite)
    click.echo(f"photometry: {len(phot)} rows -> {out_path}")


KEEP_INTERMEDIATE_OPTION = click.option(
    "--keep-intermediate/--no-keep-intermediate",
    default=None,
    help="Keep (or discard) each step's artifact under OUT_DIR, per its own 'checkpoint:' (falling back to "
    "this run's 'keep_intermediate:', itself defaulting to keeping everything). With "
    "--no-keep-intermediate, only steps that explicitly set 'checkpoint: true' (or a filename) are written.",
)


@cli.command("run")
@CONFIG_ARGUMENT
@click.option("--out-dir", "out_dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@OVERWRITE_OPTION
@KEEP_INTERMEDIATE_OPTION
@DRY_RUN_OPTION
def run_command(
    config_path: Path, out_dir: Path, overwrite: bool, keep_intermediate: bool | None, dry_run: bool
) -> None:
    """
    Sample a baseline catalog, tabulate exposure, then run every declared step in CONFIG's ``steps:`` list.

    Writes each checkpointed artifact to OUT_DIR as it goes.

    Parameters
    ----------
    config_path : Path
        Path to the run-config YAML file (``CONFIG``).
    out_dir : Path
        Directory to write each checkpointed step's artifact into.
    overwrite : bool
        Whether to overwrite existing files in `out_dir` (and, symmetrically, whether an
        existing checkpoint file is trusted and loaded instead of its step being rerun).
    keep_intermediate : bool, optional
        The default "should this step be checkpointed" answer for a step whose own
        ``checkpoint:`` is unset. If `None` (the default), falls back to the config's
        ``keep_intermediate:`` (itself defaulting to `True`).
    dry_run : bool
        If True, validate and report without running any step or writing anything.

    Returns
    -------
    None
        Exits the process via ``click`` on failure; otherwise returns nothing.
    """
    logo = _logo_text()
    if logo:
        click.echo(logo)

    config = RunConfig.from_yaml(config_path)
    keep = config.keep_intermediate if keep_intermediate is None else keep_intermediate

    if dry_run:
        stage_files = ["exposure.ecsv"]
        if keep:
            stage_files.append("baseline.ecsv")
        for _step, path in steps.checkpoint_targets(config, keep):
            if path is not None:
                stage_files.append(str(path))
        return _dry_run(config, "run", [out_dir / name for name in stage_files], overwrite)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = out_dir / "baseline.ecsv"
    if baseline_path.exists() and not overwrite:
        catalog = EventCatalog.from_disk(baseline_path)
        click.echo(f"generate: {len(catalog)} events -> {baseline_path.name} (loaded from checkpoint)")
    else:
        catalog = pipeline.run_generate(config)
        if keep:
            catalog.to_disk(baseline_path, overwrite=overwrite)
            click.echo(f"generate: {len(catalog)} events -> {baseline_path.name}")
        else:
            click.echo(f"generate: {len(catalog)} events")

    exposure_path = out_dir / "exposure.ecsv"
    if exposure_path.exists() and not overwrite:
        exposure = ExposureCatalog.from_disk(exposure_path)
        click.echo(f"exposure: {len(exposure)} (type, bin) rows -> {exposure_path.name} (loaded from checkpoint)")
    else:
        exposure = pipeline.run_exposure(config)
        exposure.to_disk(exposure_path, overwrite=overwrite)
        click.echo(f"exposure: {len(exposure)} (type, bin) rows -> {exposure_path.name}")

    store = {"baseline": catalog, "exposure": exposure}
    step_runner = steps.run_steps(config, store, out_dir=out_dir, overwrite=overwrite, keep_intermediate=keep)
    for step, artifact, path in step_runner:
        size = f"{len(artifact)} rows" if hasattr(artifact, "__len__") else "1 artifact"
        where = f" -> {path.name}" if path is not None else ""
        click.echo(f"{step.type} {step.id}: {size}{where}")
