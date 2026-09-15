"""`click`-based CLI entry point for `uvex_transients` (console script: ``uvex-transients``)."""

from pathlib import Path

import click

from ..simulation.event_catalog import EventCatalog
from . import pipeline
from .config import RunConfig

_LOGO_PATH = Path(__file__).resolve().parents[1] / "_logo.txt"


def _logo_text() -> str:
    """Read the package's ASCII banner, or an empty string if it's ever missing (never fatal)."""
    try:
        return _LOGO_PATH.read_text()
    except OSError:
        return ""


class _LogoGroup(click.Group):
    """A `click.Group` that prints the package's ASCII banner above its usual help text."""

    def format_help(self, ctx, formatter):
        logo = _logo_text()
        if logo:
            formatter.write(logo)
            formatter.write("\n")
        super().format_help(ctx, formatter)


CONFIG_ARGUMENT = click.argument(
    "config_path", metavar="CONFIG", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
OVERWRITE_OPTION = click.option("--overwrite", is_flag=True, default=False, help="Overwrite an existing output file.")


@click.group(cls=_LogoGroup)
def cli():
    """Simulate UVEX transient populations against a survey schedule."""


@cli.command("generate")
@CONFIG_ARGUMENT
@click.option("--out", "out_path", required=True, type=click.Path(dir_okay=False, path_type=Path))
@OVERWRITE_OPTION
def generate_command(config_path: Path, out_path: Path, overwrite: bool) -> None:
    """Sample a Monte Carlo event catalog (needs CONFIG's schedule/transients/mission/generate sections)."""
    config = RunConfig.from_yaml(config_path)
    catalog = pipeline.run_generate(config)
    catalog.to_disk(out_path, overwrite=overwrite)
    click.echo(f"generate: {len(catalog)} events -> {out_path}")


@cli.command("cut")
@CONFIG_ARGUMENT
@click.argument("names", nargs=-1)
@click.option("--in", "in_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", "out_path", required=True, type=click.Path(dir_okay=False, path_type=Path))
@OVERWRITE_OPTION
def cut_command(config_path: Path, names: tuple[str, ...], in_path: Path, out_path: Path, overwrite: bool) -> None:
    """
    Run one or more of CONFIG's declared cuts against a catalog, chained in order.

    NAMES are keys from CONFIG's ``cuts:`` section; with none given, every declared cut
    runs, in declared order.
    """
    config = RunConfig.from_yaml(config_path)
    catalog = EventCatalog.from_disk(in_path)
    result = pipeline.run_cuts(config, catalog, names=list(names) or None)
    result.to_disk(out_path, overwrite=overwrite)
    click.echo(f"cut: {len(result)}/{len(catalog)} events survived -> {out_path}")


@cli.command("photometry")
@CONFIG_ARGUMENT
@click.option("--in", "in_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", "out_path", required=True, type=click.Path(dir_okay=False, path_type=Path))
@OVERWRITE_OPTION
def photometry_command(config_path: Path, in_path: Path, out_path: Path, overwrite: bool) -> None:
    """Run synthetic photometry over every event in a catalog."""
    config = RunConfig.from_yaml(config_path)
    catalog = EventCatalog.from_disk(in_path)
    phot = pipeline.run_photometry(config, catalog)
    phot.write(out_path, overwrite=overwrite)
    click.echo(f"photometry: {len(phot)} rows -> {out_path}")


@cli.command("run")
@CONFIG_ARGUMENT
@click.option("--out-dir", "out_dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@OVERWRITE_OPTION
def run_command(config_path: Path, out_dir: Path, overwrite: bool) -> None:
    """Chain generate -> every declared cut -> photometry in one process, writing each stage's catalog to OUT_DIR."""
    config = RunConfig.from_yaml(config_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog = pipeline.run_generate(config)
    catalog.to_disk(out_dir / "00_generated.ecsv", overwrite=overwrite)
    click.echo(f"generate: {len(catalog)} events -> 00_generated.ecsv")

    if config.has_section("cuts"):
        for i, key in enumerate(config.cuts, start=1):
            catalog = pipeline.run_cuts(config, catalog, names=[key])
            stage_path = out_dir / f"{i:02d}_{key}.ecsv"
            catalog.to_disk(stage_path, overwrite=overwrite)
            click.echo(f"cut {key}: {len(catalog)} events -> {stage_path.name}")

    phot = pipeline.run_photometry(config, catalog)
    phot_path = out_dir / "photometry.ecsv"
    phot.write(phot_path, overwrite=overwrite)
    click.echo(f"photometry: {len(phot)} rows -> {phot_path.name}")
