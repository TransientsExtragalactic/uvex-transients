"""
The ``uvex-transients phase-space`` command group.

A recovery map is a directory of files, and each command is one stage that reads some of them and writes others (see
`uvex_transients.phase_space.stages`), so a stage can be run, repeated or changed without the ones before it:

``run CONFIG --out-dir DIR``
    Simulate the maps of a run-config and write only their ``points.ecsv``, the one expensive file. ``--workers``
    measures the points of a batch in parallel, and ``--resume`` continues a run that stopped.
``fit DIR...``
    Fit the map of the points and write ``map.ecsv`` (and ``overlay.ecsv``). Nothing is simulated.
``plot DIR...``
    Draw ``map.png`` from the files alone.
``compare BASELINE COMPARISON --out-dir DIR``
    Difference two maps, in dex of the recovered volume-time.
``info PATH...``
    Say what a map is, from the header of its points file, and which files it has.
``kinds``
    List the registered kinds of sampler and the settings each takes.
``build CONFIG --out-dir DIR``
    ``run``, ``fit`` and ``plot`` in one go. It is what ``phase-space CONFIG`` means, so the invocation that existed
    before the group still works.
"""

import inspect
from pathlib import Path

import click
from astropy import units as u

from .phase_space import PhaseSpaceConfig, dry_run_report, run_maps

__all__ = ["phase_space_group"]

_SKLEARN_HINT = "Install the 'phase_space' extra: pip install 'uvex_transients[phase_space]'."


class _DefaultGroup(click.Group):
    """A `click.Group` whose command is ``build`` when the first argument is not the name of one."""

    default = "build"

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
        from .main import _logo_text

        logo = _logo_text()
        if logo:
            formatter.write(logo)
            formatter.write("\n")
        super().format_help(ctx, formatter)

    def resolve_command(self, ctx, args):
        """
        Resolve a command, taking ``phase-space CONFIG ...`` to be ``phase-space build CONFIG ...``.

        Parameters
        ----------
        ctx : click.Context
            The current click context.
        args : list of str
            The arguments after the group's own.

        Returns
        -------
        tuple
            The name of the command, the command and the arguments left for it.

        Raises
        ------
        click.UsageError
            If the first argument is neither a command nor an existing file.
        """
        if args and args[0] not in self.commands:
            if not Path(args[0]).is_file():
                raise click.UsageError(
                    f"No such command or run-config {args[0]!r}. Commands: {', '.join(sorted(self.commands))}."
                )
            return self.default, self.commands[self.default], args
        return super().resolve_command(ctx, args)


@click.group("phase-space", cls=_DefaultGroup, context_settings={"help_option_names": ["-h", "--help"]})
def phase_space_group() -> None:
    """
    Make, fit, draw and compare the survey's recovery maps.

    A map is a directory of files, and each command is a stage over them: ``run`` simulates (the expensive part),
    ``fit`` fits the map, ``plot`` draws it and ``compare`` differences two. ``phase-space CONFIG --out-dir DIR`` alone
    is ``build``: all of the first three.
    """


# ============================================ #
# Helpers                                      #
# ============================================ #
def _banner() -> None:
    """Print the package's ASCII banner, if it has one."""
    from .main import _logo_text

    logo = _logo_text()
    if logo:
        click.echo(logo)


def _guard(function, *args, **kwargs):
    """
    Call a function, turning what a user can cause into a one-line `click.ClickException`.

    Parameters
    ----------
    function : callable
        What to call.
    *args, **kwargs
        Its arguments.

    Returns
    -------
    object
        What `function` returns.

    Raises
    ------
    click.ClickException
        If it raises a `FileNotFoundError`, `FileExistsError` or `ValueError`, or needs scikit-learn and the package
        is not installed.
    """
    try:
        return function(*args, **kwargs)
    except ImportError as error:
        raise click.ClickException(f"{error}. {_SKLEARN_HINT}") from error
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        raise click.ClickException(str(error)) from error


MAPS_ARGUMENT = click.argument("paths", metavar="DIR...", nargs=-1, required=True, type=click.Path(path_type=Path))
OVERWRITE_OPTION = click.option("--overwrite", is_flag=True, default=False, help="Replace the files that exist.")


def _simulation_options(function):
    """Give a command the arguments and options of a simulation of a run-config."""
    for decorator in reversed(
        (
            click.argument(
                "config_path", metavar="CONFIG", type=click.Path(exists=True, dir_okay=False, path_type=Path)
            ),
            click.option("--out-dir", "out_dir", required=True, type=click.Path(file_okay=False, path_type=Path)),
            click.option(
                "--map",
                "map_ids",
                multiple=True,
                metavar="ID",
                help="Make only this map of CONFIG's 'maps:' list (repeatable). The default is every map.",
            ),
            OVERWRITE_OPTION,
            click.option(
                "--resume",
                is_flag=True,
                default=False,
                help="Continue each map from the points.ecsv already in its directory, instead of refusing.",
            ),
            click.option(
                "--workers",
                type=click.IntRange(min=0),
                default=1,
                show_default=True,
                help="Processes that measure the points of a batch (0: one per CPU). For the maps of a sampler other "
                "than the canonical one; the canonical map is made by the sequential loop its published files were "
                "made with.",
            ),
            click.option("--progress", is_flag=True, default=False, help="Show the progress bars of the simulation."),
            click.option(
                "--dry-run",
                is_flag=True,
                default=False,
                help="Validate CONFIG and report what would run, without loading the schedule, sampling or writing.",
            ),
        )
    ):
        function = decorator(function)
    return function


def _simulate(
    config_path: Path,
    out_dir: Path,
    map_ids: tuple[str, ...],
    overwrite: bool,
    resume: bool,
    workers: int,
    progress: bool,
    dry_run: bool,
    products: bool,
) -> None:
    """
    Run the maps of a run-config, or report what would be run.

    Parameters
    ----------
    config_path : Path
        The phase-space run-config.
    out_dir : Path
        Where each map's subdirectory goes.
    map_ids : tuple of str
        The maps to make; empty for all.
    overwrite, resume : bool
        As for the command.
    workers : int
        The processes that measure a batch.
    progress : bool
        Whether to show the progress bars.
    dry_run : bool
        Whether to report only.
    products : bool
        Whether to fit and draw each map after sampling it.
    """
    _banner()
    try:
        phase_space_config = PhaseSpaceConfig.from_yaml(config_path)
        specs = phase_space_config.select(map_ids)
    except ValueError as error:
        raise click.ClickException(str(error)) from error

    if dry_run:
        try:
            lines, ok = dry_run_report(phase_space_config, out_dir, specs, overwrite, resume)
        except (ValueError, KeyError, OSError) as error:
            raise click.ClickException(f"dry run failed: {error}") from error
        for line in lines:
            click.echo(line)
        if not ok:
            raise click.exceptions.Exit(1)
        return

    results = _guard(
        run_maps,
        phase_space_config,
        out_dir,
        specs,
        overwrite=overwrite,
        resume=resume,
        progress=progress,
        workers=workers,
        products=products,
    )
    for map_id, files in results.items():
        click.echo(f"phase-space {map_id}: {len(files)} file(s) -> {out_dir / map_id}")


# ============================================ #
# Simulating                                   #
# ============================================ #
@phase_space_group.command("build")
@_simulation_options
def build_command(**options) -> None:
    """
    Simulate, fit and draw the maps of CONFIG: ``run``, ``fit`` and ``plot`` in one go.

    Each map is written to ``OUT_DIR/<id>/`` as ``points.ecsv``, ``map.ecsv`` and ``map.png``, and ``overlay.ecsv`` for
    the canonical map when the run-config asks for it. This is what ``phase-space CONFIG`` runs. Needs the
    ``phase_space`` extra.

    Parameters
    ----------
    **options
        The arguments and options of the command.
    """
    _simulate(products=True, **options)


@phase_space_group.command("run")
@_simulation_options
def run_command(**options) -> None:
    """
    Simulate the maps of CONFIG and write only their ``points.ecsv``.

    The points are the one expensive file, and are saved after every batch, so a run that stops can be continued with
    ``--resume``. ``fit`` and ``plot`` make the rest from them. Needs the ``phase_space`` extra.

    Parameters
    ----------
    **options
        The arguments and options of the command.
    """
    _simulate(products=False, **options)


# ============================================ #
# Fitting, drawing, comparing                  #
# ============================================ #
@phase_space_group.command("fit")
@MAPS_ARGUMENT
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="A run-config, for the schedule, mission, rate and overlay it names. Without it the schedule is the one the "
    "points file names in the registry.",
)
@click.option("--grid-size", type=click.IntRange(min=2), default=None, help="Points per axis of the map.")
@click.option("--rate", type=float, default=None, help="A volumetric rate in Gpc^-3 yr^-1; adds expected_recoveries.")
@click.option(
    "--overlay/--no-overlay",
    default=None,
    help="Measure the shipped classes for overlay.ecsv (the canonical map only). The default is what --config says.",
)
@click.option("--overlay-draws", type=click.IntRange(min=1), default=None, help="SED draws per class of the overlay.")
@OVERWRITE_OPTION
def fit_command(
    paths: tuple[Path, ...],
    config_path: Path | None,
    grid_size: int | None,
    rate: float | None,
    overlay: bool | None,
    overlay_draws: int | None,
    overwrite: bool,
) -> None:
    """
    Fit the map of the points in each DIR and write its ``map.ecsv``.

    DIR is a map (a directory with a ``points.ecsv``) or the output directory of a run, and then each map in it. The
    experiment is rebuilt from the header of the points and nothing is simulated. Needs the ``phase_space`` extra.

    Parameters
    ----------
    paths : tuple of Path
        The directories.
    config_path : Path or None
        A run-config for the schedule and the defaults of the rate and overlay.
    grid_size : int or None
        Points per axis of the map.
    rate : float or None
        The volumetric rate, in Gpc^-3 yr^-1.
    overlay : bool or None
        Whether to measure the classes. `None` takes the run-config's, and then does it only for the canonical map.
    overlay_draws : int or None
        SED draws per class.
    overwrite : bool
        Whether to replace files that exist.
    """
    from uvex_transients.phase_space.stages import describe, find_maps, fit_map

    directories = _guard(find_maps, paths)
    context = {}
    run_config = None
    if config_path is not None:
        try:
            run_config = PhaseSpaceConfig.from_yaml(config_path)
            context = {"schedule": run_config.schedule, "mission": run_config.mission}
        except ValueError as error:
            raise click.ClickException(str(error)) from error
    rate_quantity = rate / u.Gpc**3 / u.yr if rate is not None else getattr(run_config, "rate", None)
    asked = overlay if overlay is not None else bool(getattr(run_config, "overlay", False))
    draws = overlay_draws if overlay_draws is not None else getattr(run_config, "overlay_draws", None)
    seed = getattr(run_config, "seed", None)

    for directory in directories:
        info = _guard(describe, directory)
        # An overlay that came from the run-config is for the canonical maps; one asked for outright is for this map
        make_overlay = asked and (overlay is True or info.flavor == "legacy")
        written = _guard(
            fit_map,
            directory,
            grid_size=grid_size,
            rate=rate_quantity,
            overlay=make_overlay,
            overlay_draws=draws,
            seed=seed,
            overwrite=overwrite,
            **context,
        )
        click.echo(f"fit {directory}: {', '.join(path.name for path in written.values())}")


@phase_space_group.command("plot")
@MAPS_ARGUMENT
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write the figure here instead of to map.png in the directory (for one map).",
)
@click.option("--title", default=None, help="A title. The default says what the map is.")
@click.option("--show-points", is_flag=True, default=False, help="Mark the measured points.")
@click.option(
    "--overlay/--no-overlay",
    default=True,
    help="Shade the classes of overlay.ecsv, if the map has one.",
    show_default=True,
)
def plot_command(
    paths: tuple[Path, ...], out_path: Path | None, title: str | None, show_points: bool, overlay: bool
) -> None:
    """
    Draw the map of each DIR from its ``map.ecsv`` and ``points.ecsv``, and write ``map.png``.

    DIR is a map or the output directory of a run. It needs matplotlib, and no schedule or scikit-learn, so it works on
    the files of a release.

    Parameters
    ----------
    paths : tuple of Path
        The directories.
    out_path : Path or None
        Where to write the figure.
    title : str or None
        The title.
    show_points : bool
        Whether to mark the measured points.
    overlay : bool
        Whether to shade the classes.
    """
    from uvex_transients.phase_space.stages import find_maps, plot_files

    directories = _guard(find_maps, paths)
    if out_path is not None and len(directories) != 1:
        raise click.ClickException(f"--out is for one map, and {len(directories)} were given.")
    for directory in directories:
        figure = _guard(plot_files, directory, path=out_path, title=title, show_points=show_points, overlay=overlay)
        click.echo(f"plot {directory}: {figure}")


@phase_space_group.command("compare")
@click.argument("baseline", type=click.Path(exists=True, path_type=Path))
@click.argument("comparison", type=click.Path(exists=True, path_type=Path))
@click.option("--out-dir", "out_dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--floor", type=float, default=None, help="The recovery fraction below which a map recovers nothing.")
@click.option(
    "--no-check-meta",
    is_flag=True,
    default=False,
    help="Compare maps even if they do not measure the same axes (another band or shape).",
)
@click.option("--no-plot", is_flag=True, default=False, help="Write difference.ecsv only.")
@click.option("--title", default=None, help="A title. The default names the two maps.")
@OVERWRITE_OPTION
def compare_command(
    baseline: Path,
    comparison: Path,
    out_dir: Path,
    floor: float | None,
    no_check_meta: bool,
    no_plot: bool,
    title: str | None,
    overwrite: bool,
) -> None:
    """
    Difference the maps of two directories, in dex of the recovered volume-time.

    The difference is positive where COMPARISON recovers more than BASELINE. It is written to ``OUT_DIR`` as
    ``difference.ecsv`` and ``difference.png``, from the two ``map.ecsv`` alone. Maps that do not measure the same
    axes, band or shape are refused unless ``--no-check-meta``.

    Parameters
    ----------
    baseline, comparison : Path
        The directories of the two maps.
    out_dir : Path
        Where to write the files.
    floor : float or None
        The recovery fraction below which a map recovers nothing.
    no_check_meta : bool
        Whether to skip the check that the maps measure the same axes.
    no_plot : bool
        Whether to skip the figure.
    title : str or None
        The title.
    overwrite : bool
        Whether to replace files that exist.
    """
    from uvex_transients.phase_space.stages import compare_maps, find_maps

    (baseline_dir,), (comparison_dir,) = (_guard(find_maps, [path]) for path in (baseline, comparison))
    written = _guard(
        compare_maps,
        baseline_dir,
        comparison_dir,
        out_dir,
        floor=floor,
        check_meta=not no_check_meta,
        plot=not no_plot,
        title=title,
        overwrite=overwrite,
    )
    click.echo(f"compare: {', '.join(str(path) for path in written.values())}")


# ============================================ #
# Looking                                      #
# ============================================ #
@phase_space_group.command("info")
@MAPS_ARGUMENT
def info_command(paths: tuple[Path, ...]) -> None:
    """
    Say what each map is, from the header of its points file, and which files it has.

    DIR is a map or the output directory of a run. Nothing is simulated or fitted.

    Parameters
    ----------
    paths : tuple of Path
        The directories.
    """
    from uvex_transients.phase_space.stages import describe, find_maps

    for directory in _guard(find_maps, paths):
        info = _guard(describe, directory)
        header = info.header
        old = " (earlier points format)" if info.flavor == "legacy" else ""
        click.echo(str(directory))
        click.echo(f"  kind      {info.kind}{old}")
        if info.band:
            click.echo(f"  band      {info.band}")
        for axis in info.space:
            unit = f" {axis.unit}" if axis.unit else ""
            click.echo(f"  axis      {axis.name}: {axis.lo:g} to {axis.hi:g} ({axis.scale}{unit})")
        click.echo(f"  points    {info.n_points} ({info.n_tries:,} tries)")
        click.echo("  rule      " + ", ".join(f"{key}={value}" for key, value in header["rule"].items()))
        click.echo("  campaign  " + ", ".join(f"{key}={value}" for key, value in header["campaign"].items()))
        click.echo(f"  files     {', '.join(info.files)}")


@phase_space_group.command("kinds")
def kinds_command() -> None:
    """List the registered kinds of sampler, and the settings each takes under ``settings:`` or ``recovery:``."""
    from uvex_transients.phase_space.sampling import available_samplers, get_sampler

    for kind in sorted(available_samplers()):
        sampler_class = get_sampler(kind)
        summary = (inspect.getdoc(sampler_class) or "").split("\n\n")[0].replace("\n", " ")
        click.echo(f"{kind}: {summary}")
        parameters = [
            parameter
            for name, parameter in inspect.signature(sampler_class.__init__).parameters.items()
            if name not in ("self", "schedule", "mission", "kwargs", "schedule_name")
            and parameter.kind is not inspect.Parameter.VAR_KEYWORD
        ]
        if parameters:
            click.echo("  takes: " + ", ".join(parameter.name for parameter in parameters))
