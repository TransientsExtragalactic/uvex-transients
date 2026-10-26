"""
Map UVEX's recovery of a flat-spectrum top-hat transient over luminosity and rest-frame duration.

Every point of the map is a transient with a flat spectrum (the same L_nu at all frequencies) and a constant luminosity
for a rest-frame duration. Its luminosity is given in erg/s for the band: L_band = L_nu times the equivalent bandwidth
of the band (the width, in frequency, of a top hat of the band's peak throughput that passes as much). At a point,
events are drawn volume-weighted in redshift out to the redshift where the source falls below a chosen depth
(``--mag-limit``), on the footprint the schedule covers, and the ones the survey recovers are counted (see
``uvex_transients.phase_space``). The expensive points are placed by a Gaussian process on the logit of the recovery
fraction, which picks each batch where one more simulation teaches the most about the map.

Three things are written to the output directory:

``points.ecsv``
    Every simulated point: its coordinates, ``k`` recoveries out of ``n`` events, the redshift limit, the footprint
    coverage and the effective volume-time. Written after every batch, and read back with ``--resume``.
``map.ecsv``
    The fitted map on a grid, each recovery quantity with a 1 sigma interval (``_lo``, ``_hi``):

    ``f``
        The recovery fraction in the sampled footprint. It depends on ``--time-bins``, because the number of bins sets
        how much of the sky observed outside an event's lifetime is in the footprint.
    ``efficiency``
        ``f * coverage``: the probability that an event exploding at a random position and time during the survey,
        out to the redshift limit, is recovered. It does not depend on ``--time-bins``, but it does depend on the
        redshift limit (``--mag-limit``, ``--z-max``), through the volume it is averaged over.
    ``recovered_volume_time``
        ``f * volume_time``, in Mpc^3 yr: the expected number of recoveries at one event per Mpc^3 per yr. It depends on
        neither choice above, and scales linearly with a rate.
    ``volume_time``, ``all_sky_volume_time``, ``coverage``, ``z_limit``, ``z_limit_capped``
        The noise-free scales the quantities above are built from (``coverage = volume_time / all_sky_volume_time``),
        and whether the redshift limit was capped by ``--z-max``.
    ``sigma_log``
        The remaining uncertainty of ``ln(f + epsilon)``, which is what the sampling spends points to reduce.
    ``expected_recoveries``
        Only with ``--rate``: the rate times ``recovered_volume_time``.
``map.png``
    ``f``, ``efficiency``, ``recovered_volume_time`` and ``sigma_log``.

Only events brighter than ``--mag-limit`` at their own redshift are drawn, so recoveries of fainter ones are not
counted. The default is about the single-dwell depth, and the survey recovers some events fainter than that, so deeper
limits (about 25.5) give larger recovered volume-times, at the price of more events per point.

``recovered_volume_time`` and ``expected_recoveries`` assume a comoving rate that is constant in redshift, so they are
not yields for an evolving population. Above the luminosity (the dashed line of the figure) where the redshift limit is
capped by ``--z-max``, they are lower bounds. Only events exploding during the survey window are counted. A map is for
one band, ``--band FUV`` (the default) or ``--band NUV``, which both detects the events and defines the luminosity.

Reload a finished run in Python, with no simulation, to evaluate any map anywhere, at any resolution::

    from astropy import units as u
    from uvex_transients.phase_space.sampler import (
        PhaseSpaceSampler,
    )

    sampler = PhaseSpaceSampler.load(
        "results/phase_space/points.ecsv"
    )  # rebuilds the model and the evaluator
    eta = sampler.evaluate_efficiency_map(
        log_luminosity, log_duration
    )  # arrays broadcast against each other
    n, lo, hi = (
        sampler.evaluate_expected_recoveries_map(
            43.0,
            1.5,
            10 / u.Gpc**3 / u.yr,
            interval=True,
        )
    )
    table = sampler.map_table(
        200, rate=10 / u.Gpc**3 / u.yr
    )  # the contents of map.ecsv, on any grid

Usage::

    python scripts/phase_space_map.py --out-dir results/phase_space/
    python scripts/phase_space_map.py --band NUV --max-points 120 --out-dir results/phase_space_nuv/
    python scripts/phase_space_map.py --out-dir results/phase_space/ --resume --max-points 200
"""

import argparse
import logging
import time
from pathlib import Path

from astropy import units as u
from tqdm import tqdm

from uvex_transients.phase_space.evaluator import RecoveryEvaluator
from uvex_transients.phase_space.plotting import plot_map
from uvex_transients.phase_space.sampler import PhaseSpaceSampler
from uvex_transients.surveys.utils import get_schedule
from uvex_transients.utils import config, logger

NAMES = ("log_luminosity", "log_duration")


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command line parser.

    Returns
    -------
    argparse.ArgumentParser
        The parser, with the options of every stage of the script, from survey choice to output.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])

    survey = parser.add_argument_group("survey")
    survey.add_argument("--schedule", default=None, help="schedule name (default: the configured default)")
    survey.add_argument("--mission", default="uvex_fast", help="mission name (default: uvex_fast)")
    survey.add_argument(
        "--band", default="FUV", help="band that recovers the events and defines the luminosity (default: FUV)"
    )

    recovery = parser.add_argument_group("what counts as recovered")
    recovery.add_argument("--snr", type=float, default=5.0, help="SNR threshold of a visit (default: 5)")
    recovery.add_argument("--n-visits", type=int, default=2, help="visits that must reach it (default: 2)")
    recovery.add_argument(
        "--keep-first-visit",
        action="store_true",
        help="count a detection that is only on a field's first visit (default: drop it)",
    )

    sampling = parser.add_argument_group("sampling of one point")
    sampling.add_argument(
        "--mag-limit",
        type=float,
        default=24.5,
        help="AB depth setting z_lim; only events brighter than it are counted (default: 24.5, one dwell)",
    )
    sampling.add_argument("--z-max", type=float, default=2.0, help="largest redshift limit of a point (default: 2)")
    sampling.add_argument("--time-bins", type=int, default=12, help="explosion-time bins (default: 12)")
    sampling.add_argument("--chunk-size", type=int, default=1000, help="events per chunk (default: 1000)")
    sampling.add_argument("--k-target", type=int, default=30, help="recoveries to stop at (default: 30)")
    sampling.add_argument("--n-max", type=int, default=20000, help="most events per point (default: 20000)")

    space = parser.add_argument_group("the map")
    space.add_argument(
        "--log-l",
        nargs=2,
        type=float,
        default=(41.0, 43.5),
        metavar=("LO", "HI"),
        help="log10 L_band / (erg/s) range (default: 41 43.5)",
    )
    space.add_argument(
        "--log-dt",
        nargs=2,
        type=float,
        default=(-1.0, 2.5),
        metavar=("LO", "HI"),
        help="log10 rest-frame duration / day range (default: -1 2.5)",
    )

    active = parser.add_argument_group("active learning")
    active.add_argument("--n-initial", type=int, default=24, help="space-filling points first (default: 24)")
    active.add_argument("--batch-size", type=int, default=4, help="points per round (default: 4)")
    active.add_argument("--max-points", type=int, default=120, help="most points to simulate (default: 120)")
    active.add_argument(
        "--tolerance",
        type=float,
        default=0.1,
        help="stop when the largest uncertainty of ln(p + floor) is below this (default: 0.1)",
    )
    active.add_argument("--acquisition", choices=("variance", "imse"), default="imse", help="default: imse")

    output = parser.add_argument_group("output")
    output.add_argument("--out-dir", type=Path, default=Path("results/phase_space"), help="output directory")
    output.add_argument("--resume", action="store_true", help="continue from the points.ecsv in --out-dir")
    output.add_argument(
        "--rate",
        type=float,
        default=None,
        help="volumetric rate in Gpc^-3 yr^-1; adds expected_recoveries to map.ecsv and its contours to map.png",
    )
    output.add_argument("--grid-size", type=int, default=60, help="grid points per axis of the map (default: 60)")
    output.add_argument("--seed", type=int, default=0, help="root seed (default: 0)")
    output.add_argument("--progress", action="store_true", help="show the progress bars of the simulation")
    output.add_argument("--no-plot", action="store_true", help="do not draw map.png")
    return parser


def main() -> None:
    """Run the map and write its outputs."""
    args = build_parser().parse_args()
    if not args.progress:
        # Some bars in the simulation pass `disable` explicitly, so a default would not silence them.
        original_init = tqdm.__init__

        def quiet_init(self, *init_args, **init_kwargs):
            init_kwargs["disable"] = True
            original_init(self, *init_args, **init_kwargs)

        tqdm.__init__ = quiet_init
    logging.getLogger("uvex_transients").setLevel(logging.INFO)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    points_path = args.out_dir / "points.ecsv"

    start = time.perf_counter()
    schedule = get_schedule(name=args.schedule)
    evaluator = RecoveryEvaluator(
        schedule,
        args.mission,
        band=args.band,
        snr_threshold=args.snr,
        n_visits=args.n_visits,
        exclude_first_visit_detections=not args.keep_first_visit,
        mag_limit=args.mag_limit,
        z_max=args.z_max,
        time_bins=args.time_bins,
        chunk_size=args.chunk_size,
        k_target=args.k_target,
        n_max=args.n_max,
        seed=args.seed,
        schedule_name=args.schedule or config["schedules.default_schedule"],
    )

    settings = {
        "names": NAMES,
        "n_initial": args.n_initial,
        "batch_size": args.batch_size,
        "max_points": args.max_points,
        "tolerance": args.tolerance,
        "acquisition": args.acquisition,
        "k_target": args.k_target,
        "n_min": args.chunk_size,
        "n_max": args.n_max,
        "seed": args.seed,
    }
    bounds = [tuple(args.log_l), tuple(args.log_dt)]
    if args.resume and points_path.exists():
        sampler = PhaseSpaceSampler.load(points_path, evaluator, bounds=bounds, **settings)
        logger.info("Resuming from %d points in %s.", len(sampler), points_path)
    else:
        sampler = PhaseSpaceSampler(evaluator, bounds, **settings)

    sampler.run(callback=lambda s: s.save(points_path))
    elapsed = time.perf_counter() - start

    points = sampler.table
    rate = None if args.rate is None else args.rate / u.Gpc**3 / u.yr
    grid = sampler.map_table(args.grid_size, rate=rate)
    grid.write(args.out_dir / "map.ecsv", format="ascii.ecsv", overwrite=True)

    title = (
        f"{args.band}: SNR >= {args.snr:g} on {args.n_visits} visit(s), {len(points)} points, "
        f"{int(points['n'].sum()):,} events, {elapsed / 60:.1f} min"
    )
    if not args.no_plot:
        plot_map(grid, points, args.out_dir / "map.png", title)

    print(f"\n{title}\nWrote {points_path}, {args.out_dir / 'map.ecsv'}" + ("" if args.no_plot else ", map.png"))


if __name__ == "__main__":
    main()
