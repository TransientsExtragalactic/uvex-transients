"""`click`-based CLI entry point for `uvex_transients` (console script: ``uvex-transients``)."""

from pathlib import Path

import click

from ..simulation.event_catalog import EventCatalog
from ..simulation.exposure_catalog import ExposureCatalog
from ..simulation.photometry_catalog import PhotometryCatalog
from . import pipeline
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


def _dry_run(config: RunConfig, command: str, outputs, overwrite: bool, cut_names=None) -> None:
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
    cut_names : list of str, optional
        For the ``"cut"`` command, the cut names that would run.
    """
    try:
        lines, ok = pipeline.dry_run_report(config, command, outputs=outputs, overwrite=overwrite, cut_names=cut_names)
    except (ValueError, KeyError, OSError) as error:
        raise click.ClickException(f"dry run failed: {error}") from error
    for line in lines:
        click.echo(line)
    if not ok:
        raise click.exceptions.Exit(1)


@click.group(cls=_LogoGroup)
def cli():
    """Simulate UVEX transient populations against a survey schedule."""


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
@click.argument("names", nargs=-1)
@click.option("--in", "in_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", "out_path", required=True, type=click.Path(dir_okay=False, path_type=Path))
@OVERWRITE_OPTION
@DRY_RUN_OPTION
def cut_command(
    config_path: Path, names: tuple[str, ...], in_path: Path, out_path: Path, overwrite: bool, dry_run: bool
) -> None:
    """
    Run one or more of CONFIG's declared cuts against a catalog, chained in order.

    NAMES are keys from CONFIG's ``cuts:`` section; with none given, every declared cut
    runs, in declared order.

    Parameters
    ----------
    config_path : Path
        Path to the run-config YAML file (``CONFIG``).
    names : tuple of str
        Cut names to run, from CONFIG's ``cuts:`` section (``NAMES``); empty runs every
        declared cut.
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
        return _dry_run(config, "cut", [out_path], overwrite, cut_names=list(names) or None)
    catalog = EventCatalog.from_disk(in_path)
    result = pipeline.run_cuts(config, catalog, names=list(names) or None)
    result.to_disk(out_path, overwrite=overwrite)
    click.echo(f"cut: {len(result)}/{len(catalog)} events survived -> {out_path}")


@cli.command("photometry")
@CONFIG_ARGUMENT
@click.option("--in", "in_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", "out_path", required=True, type=click.Path(dir_okay=False, path_type=Path))
@OVERWRITE_OPTION
@DRY_RUN_OPTION
def photometry_command(config_path: Path, in_path: Path, out_path: Path, overwrite: bool, dry_run: bool) -> None:
    """
    Run synthetic photometry over every event in a catalog.

    Parameters
    ----------
    config_path : Path
        Path to the run-config YAML file (``CONFIG``).
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
        return _dry_run(config, "photometry", [out_path], overwrite)
    catalog = EventCatalog.from_disk(in_path)
    phot = pipeline.run_photometry(config, catalog)
    phot.write(out_path, overwrite=overwrite)
    click.echo(f"photometry: {len(phot)} rows -> {out_path}")


@cli.command("detection-counts")
@CONFIG_ARGUMENT
@click.option(
    "--catalog",
    "catalog_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="The full, pre-cut event catalog from 'generate' (e.g. 00_generated.ecsv from 'run' with "
    "intermediates kept), NOT the post-cut catalog --photometry was computed over: n_total must "
    "count every sampled event, including ones later cut or never observed, or 'fraction' is "
    "computed against the wrong denominator.",
)
@click.option(
    "--photometry",
    "photometry_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Photometry table, as written by the 'photometry' command.",
)
@click.option(
    "--exposure",
    "exposure_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Exposure catalog, as written by 'run' (or SurveySimulator.compute_effective_exposure).",
)
@click.option("--out", "out_path", required=True, type=click.Path(dir_okay=False, path_type=Path))
@OVERWRITE_OPTION
@DRY_RUN_OPTION
def detection_counts_command(
    config_path: Path,
    catalog_path: Path,
    photometry_path: Path,
    exposure_path: Path,
    out_path: Path,
    overwrite: bool,
    dry_run: bool,
) -> None:
    """
    Estimate, per transient type, how many events show N_det >= k detected epochs.

    Combines CATALOG, PHOTOMETRY, and EXPOSURE per the config's 'detection_counts:' section
    (see `PhotometryCatalog.compute_detection_count_table`); the output table carries both
    Clopper-Pearson confidence bounds and exposure-scaled expected event counts, not just raw
    Monte Carlo catalog counts.

    Parameters
    ----------
    config_path : Path
        Path to the run-config YAML file (``CONFIG``).
    catalog_path : Path
        Path to the full, pre-cut event catalog from the 'generate' stage (e.g.
        ``00_generated.ecsv`` from 'run' with intermediates kept) -- *not* the post-cut catalog
        `photometry_path` was computed over. `PhotometryCatalog.compute_detection_count_table`
        needs every sampled event, including ones later cut or never observed, to compute
        `n_total`/`fraction` correctly; passing the post-cut catalog silently inflates
        `expected_events` by roughly ``1/efficiency``.
    photometry_path : Path
        Path to the photometry table.
    exposure_path : Path
        Path to the exposure catalog.
    out_path : Path
        Destination path for the detection-count table.
    overwrite : bool
        Whether to overwrite an existing file at `out_path`.
    dry_run : bool
        If True, validate and report without computing or writing anything.

    Returns
    -------
    None
        Exits the process via ``click`` on failure; otherwise returns nothing.
    """
    config = RunConfig.from_yaml(config_path)
    if dry_run:
        return _dry_run(config, "detection-counts", [out_path], overwrite)
    catalog = EventCatalog.from_disk(catalog_path)
    photometry = PhotometryCatalog.from_disk(photometry_path)
    exposure = ExposureCatalog.from_disk(exposure_path)
    table = pipeline.run_detection_counts(config, catalog, exposure, photometry)
    table.write(out_path, overwrite=overwrite)
    click.echo(f"detection-counts: {len(table)} (type, k) row(s) -> {out_path}")


KEEP_INTERMEDIATE_OPTION = click.option(
    "--keep-intermediate/--no-keep-intermediate",
    default=None,
    help="Keep (or discard) each stage's catalog under OUT_DIR; defaults to the config's "
    "'keep_intermediate:' (itself defaulting to keeping everything). With --no-keep-intermediate, "
    "only the final event catalog (final_catalog.ecsv) and the photometry table are written.",
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
    Chain generate -> every declared cut -> photometry in one process, writing each stage's catalog to OUT_DIR.

    Parameters
    ----------
    config_path : Path
        Path to the run-config YAML file (``CONFIG``).
    out_dir : Path
        Directory to write each stage's catalog into.
    overwrite : bool
        Whether to overwrite existing files in `out_dir`.
    keep_intermediate : bool, optional
        Whether to write each *intermediate* stage's catalog to `out_dir`. If `None` (the
        default), falls back to the config's ``keep_intermediate:`` (itself defaulting to
        `True`). The final event catalog and the photometry table are always written either way.
    dry_run : bool
        If True, validate and report without running any stage or writing anything.

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
        stage_files = []
        if keep:
            stage_files.append("00_generated.ecsv")
            if config.has_section("cuts"):
                stage_files += [f"{i:02d}_{key}.ecsv" for i, key in enumerate(config.cuts, start=1)]
        else:
            stage_files.append("final_catalog.ecsv")
        stage_files.append("photometry.ecsv")
        stage_files += ["exposure.ecsv", "yield_summary.ecsv", "yield_summary.txt"]
        if config.has_section("detection_counts"):
            stage_files.append("detection_counts.ecsv")
        return _dry_run(config, "run", [out_dir / name for name in stage_files], overwrite)
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog = pipeline.run_generate(config)
    raw_catalog = catalog
    if keep:
        catalog.to_disk(out_dir / "00_generated.ecsv", overwrite=overwrite)
        click.echo(f"generate: {len(catalog)} events -> 00_generated.ecsv")
    else:
        click.echo(f"generate: {len(catalog)} events")

    exposure = pipeline.run_exposure(config)
    exposure_path = out_dir / "exposure.ecsv"
    exposure.to_disk(exposure_path, overwrite=overwrite)
    click.echo(f"exposure: {len(exposure)} (type, bin) rows -> {exposure_path.name}")

    if config.has_section("cuts"):
        for i, key in enumerate(config.cuts, start=1):
            catalog = pipeline.run_cuts(config, catalog, names=[key])
            if keep:
                stage_path = out_dir / f"{i:02d}_{key}.ecsv"
                catalog.to_disk(stage_path, overwrite=overwrite)
                click.echo(f"cut {key}: {len(catalog)} events -> {stage_path.name}")
            else:
                click.echo(f"cut {key}: {len(catalog)} events")

    if not keep:
        final_path = out_dir / "final_catalog.ecsv"
        catalog.to_disk(final_path, overwrite=overwrite)
        click.echo(f"catalog: {len(catalog)} events -> {final_path.name}")

    yield_table = pipeline.run_yield_summary(config, raw_catalog, catalog, exposure)
    yield_ecsv_path = out_dir / "yield_summary.ecsv"
    yield_table.to_disk(yield_ecsv_path, overwrite=overwrite)
    yield_ascii_path = out_dir / "yield_summary.txt"
    yield_table.to_ascii(yield_ascii_path, overwrite=overwrite)
    click.echo(f"yield: {len(yield_table)} transient type(s) -> {yield_ecsv_path.name}, {yield_ascii_path.name}")

    phot = pipeline.run_photometry(config, catalog)
    phot_path = out_dir / "photometry.ecsv"
    phot.write(phot_path, overwrite=overwrite)
    click.echo(f"photometry: {len(phot)} rows -> {phot_path.name}")

    if config.has_section("detection_counts"):
        detection_counts = pipeline.run_detection_counts(config, raw_catalog, exposure, phot)
        detection_counts_path = out_dir / "detection_counts.ecsv"
        detection_counts.write(detection_counts_path, overwrite=overwrite)
        click.echo(f"detection-counts: {len(detection_counts)} (type, k) row(s) -> {detection_counts_path.name}")
