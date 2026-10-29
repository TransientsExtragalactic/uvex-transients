"""
Map UVEX's recovery of a canonical transient over peak luminosity and rest-frame half-decline time.

Every point of the map is a transient with a fast-rise, exponential-decay light curve and a power-law spectrum, with
two free quantities: its peak luminosity in the band (erg/s) and its half-decline time (days), the time from the peak
until the luminosity has halved. At a point, events are drawn volume-weighted in redshift out to where the source falls
below a chosen depth (``--mag-limit``), on the footprint the schedule covers, and the ones the survey recovers are
counted (see ``uvex_transients.phase_space``). A Gaussian process on the recovery fraction places the expensive points,
each batch where one more simulation teaches the most about the map. Any other SED can be measured the same way and
placed on the map (``SpectralModel.peak_luminosity`` and ``decline_time``).

Every option that is not given takes its value from ``config["phase_space"]``, which also explains each of them.

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
        redshift limit (``--mag-limit``), through the volume it is averaged over.
    ``recovered_volume_time``
        ``f * volume_time``, in Mpc^3 yr: the expected number of recoveries at one event per Mpc^3 per yr. It depends on
        neither choice above, and scales linearly with a rate.
    ``volume_time``, ``all_sky_volume_time``, ``coverage``, ``z_limit``
        The noise-free scales the quantities above are built from (``coverage = volume_time / all_sky_volume_time``).
    ``sigma_log``
        The remaining uncertainty of ``ln(f + epsilon)``, which is what the sampling spends points to reduce.
    ``expected_recoveries``
        Only with ``--rate``: the rate times ``recovered_volume_time``.
``map.png``
    ``f``, ``efficiency``, ``recovered_volume_time`` and ``sigma_log``. With ``--overlay``, the region holding 68% of
    every shipped transient class is shaded on it, from the same two quantities measured on draws from the class's SED
    priors in the map's band (``overlay.ecsv`` holds the measurements). ``--show-points`` marks the simulated points.

Only events brighter than ``--mag-limit`` at their own redshift are drawn, so recoveries of fainter ones are not
counted. The default is about the single-dwell depth, and the survey recovers some events fainter than that, so deeper
limits (about 25.5) give larger recovered volume-times, at the price of more events per point.

``recovered_volume_time`` and ``expected_recoveries`` assume a comoving rate that is constant in redshift, so they are
not yields for an evolving population. Only events exploding during the survey window are counted. A map is for
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
        log_luminosity, log_half_time
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

import numpy as np
from astropy import units as u
from astropy.table import Table
from tqdm import tqdm

from uvex_transients.phase_space.evaluator import RecoveryEvaluator
from uvex_transients.phase_space.overlay import measure_class, shipped_classes
from uvex_transients.phase_space.plotting import plot_map
from uvex_transients.phase_space.sampler import PhaseSpaceSampler
from uvex_transients.surveys.utils import get_schedule
from uvex_transients.utils import config, logger

#: Names of the two coordinates of the map, as they appear in the files.
NAMES = ("log_luminosity", "log_half_time")


def _default(text: str, key: str) -> str:
    """Return a help string that ends with the configured default, ``config["phase_space.<key>"]``."""
    return f"{text} (default: {config[f'phase_space.{key}']})"


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command line parser.

    Every option that has a configured default is `None` here and resolved by the library, so that the configuration is
    the one place the defaults live; the help text shows the current values.

    Returns
    -------
    argparse.ArgumentParser
        The parser, with the options of every stage of the script, from survey choice to output.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])

    survey = parser.add_argument_group("survey")
    survey.add_argument("--schedule", default=None, help="schedule name (default: the configured default)")
    survey.add_argument("--mission", default=None, help=_default("mission name", "recovery.mission"))
    survey.add_argument(
        "--band",
        default=None,
        help=_default("band that recovers the events and defines the luminosity", "recovery.band"),
    )

    transient = parser.add_argument_group("the transient")
    transient.add_argument(
        "--spectral-index",
        type=float,
        default=None,
        help=_default(
            "index alpha of the spectrum, L_nu proportional to nu^alpha; 0 is flat", "transient.spectral_index"
        ),
    )
    transient.add_argument(
        "--rise-ratio",
        type=float,
        default=None,
        help=_default("rise time over decay time of the light curve", "transient.rise_ratio"),
    )
    transient.add_argument(
        "--fade-mag",
        type=float,
        default=None,
        help=_default("magnitudes below its peak an event is followed to", "transient.fade_mag"),
    )

    recovery = parser.add_argument_group("what counts as recovered")
    recovery.add_argument(
        "--snr", type=float, default=None, help=_default("SNR threshold of a visit", "recovery.snr_threshold")
    )
    recovery.add_argument(
        "--n-visits", type=int, default=None, help=_default("visits that must reach it", "recovery.n_visits")
    )
    recovery.add_argument(
        "--keep-first-visit",
        action="store_true",
        help="count a detection that is only on a field's first visit (otherwise it is dropped, as configured)",
    )

    sampling = parser.add_argument_group("sampling of one point")
    sampling.add_argument(
        "--mag-limit",
        type=float,
        default=None,
        help=_default("AB depth setting z_lim; only events brighter than it are counted", "recovery.mag_limit"),
    )
    sampling.add_argument(
        "--time-bins", type=int, default=None, help=_default("explosion-time bins", "recovery.time_bins")
    )
    sampling.add_argument(
        "--chunk-size", type=int, default=None, help=_default("events per chunk", "recovery.chunk_size")
    )
    sampling.add_argument(
        "--k-target", type=int, default=None, help=_default("recoveries to stop at", "recovery.k_target")
    )
    sampling.add_argument("--n-max", type=int, default=None, help=_default("most events per point", "recovery.n_max"))

    space = parser.add_argument_group("the map")
    space.add_argument(
        "--log-l",
        nargs=2,
        type=float,
        default=None,
        metavar=("LO", "HI"),
        help=_default("log10 L_band / (erg/s) range", "map.log_luminosity"),
    )
    space.add_argument(
        "--log-dt",
        nargs=2,
        type=float,
        default=None,
        metavar=("LO", "HI"),
        help=_default("log10 rest-frame half-decline time / day range", "map.log_half_time"),
    )

    active = parser.add_argument_group("active learning")
    active.add_argument(
        "--n-initial", type=int, default=None, help=_default("space-filling points first", "sampler.n_initial")
    )
    active.add_argument("--batch-size", type=int, default=None, help=_default("points per round", "sampler.batch_size"))
    active.add_argument(
        "--max-points", type=int, default=None, help=_default("most points to simulate", "sampler.max_points")
    )
    active.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help=_default("stop when the largest uncertainty of ln(p + floor) is below this", "sampler.tolerance"),
    )
    active.add_argument(
        "--acquisition",
        choices=("variance", "imse"),
        default=None,
        help=_default("which point is worth simulating next", "sampler.acquisition"),
    )

    output = parser.add_argument_group("output")
    output.add_argument("--out-dir", type=Path, default=Path("results/phase_space"), help="output directory")
    output.add_argument("--resume", action="store_true", help="continue from the points.ecsv in --out-dir")
    output.add_argument(
        "--rate",
        type=float,
        default=None,
        help="volumetric rate in Gpc^-3 yr^-1; adds expected_recoveries to map.ecsv and its contours to map.png",
    )
    output.add_argument(
        "--grid-size", type=int, default=None, help=_default("grid points per axis of the map", "map.grid_size")
    )
    output.add_argument("--seed", type=int, default=0, help="root seed (default: 0)")
    output.add_argument("--progress", action="store_true", help="show the progress bars of the simulation")
    output.add_argument("--no-plot", action="store_true", help="do not draw map.png")
    output.add_argument(
        "--overlay", action="store_true", help="measure the shipped transient classes and draw them on map.png"
    )
    output.add_argument("--show-points", action="store_true", help="mark the simulated points on map.png")
    output.add_argument(
        "--overlay-draws", type=int, default=None, help=_default("SED draws per class", "overlay.n_draws")
    )
    return parser


def main() -> None:
    """Run the map and write its outputs."""
    args = build_parser().parse_args()

    # Some bars in the simulation pass `disable` explicitly, so a default would not silence them
    if not args.progress:
        original_init = tqdm.__init__

        def quiet_init(self, *init_args, **init_kwargs):
            init_kwargs["disable"] = True
            original_init(self, *init_args, **init_kwargs)

        tqdm.__init__ = quiet_init
    logging.getLogger("uvex_transients").setLevel(logging.INFO)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    points_path = args.out_dir / "points.ecsv"

    # ---------------------------------------------- #
    # The run                                        #
    # ---------------------------------------------- #
    start = time.perf_counter()
    schedule = get_schedule(name=args.schedule)
    evaluator = RecoveryEvaluator(
        schedule,
        args.mission,
        band=args.band,
        spectral_index=args.spectral_index,
        rise_ratio=args.rise_ratio,
        fade_mag=args.fade_mag,
        snr_threshold=args.snr,
        n_visits=args.n_visits,
        exclude_first_visit_detections=False if args.keep_first_visit else None,
        mag_limit=args.mag_limit,
        time_bins=args.time_bins,
        chunk_size=args.chunk_size,
        k_target=args.k_target,
        n_max=args.n_max,
        seed=args.seed,
        schedule_name=args.schedule or config["schedules.default_schedule"],
    )

    # The sampler takes what the evaluator is expected to do (its chunk size, target and cap) from the evaluator itself
    settings = {
        "names": NAMES,
        "n_initial": args.n_initial,
        "batch_size": args.batch_size,
        "max_points": args.max_points,
        "tolerance": args.tolerance,
        "acquisition": args.acquisition,
        "seed": args.seed,
    }
    bounds = [
        tuple(args.log_l or config["phase_space.map.log_luminosity"]),
        tuple(args.log_dt or config["phase_space.map.log_half_time"]),
    ]
    if args.resume and points_path.exists():
        sampler = PhaseSpaceSampler.load(points_path, evaluator, bounds=bounds, **settings)
        logger.info("Resuming from %d points in %s.", len(sampler), points_path)
    else:
        sampler = PhaseSpaceSampler(evaluator, bounds, **settings)

    # The points are saved after every batch, so a run that stops can be resumed
    sampler.run(callback=lambda s: s.save(points_path))
    elapsed = time.perf_counter() - start

    # ---------------------------------------------- #
    # The outputs                                    #
    # ---------------------------------------------- #
    points = sampler.table
    rate = None if args.rate is None else args.rate / u.Gpc**3 / u.yr
    grid = sampler.map_table(args.grid_size, rate=rate)
    grid.write(args.out_dir / "map.ecsv", format="ascii.ecsv", overwrite=True)

    title = (
        f"{evaluator.band}: SNR >= {evaluator.snr_threshold:g} on {evaluator.n_visits} visit(s), {len(points)} points, "
        f"{int(points['n'].sum()):,} events, {elapsed / 60:.1f} min"
    )

    # The shipped classes, measured in the band of the map
    overlays = None
    if args.overlay:
        bandpass = evaluator.mission.detector.bandpasses[evaluator.band]
        overlays = [
            measure_class(cls, bandpass, n_draws=args.overlay_draws, rng=args.seed)
            for cls in shipped_classes().values()
        ]
        stacked = Table(
            {
                "class": np.concatenate([[o.meta["class"]] * len(o) for o in overlays]),
                **{
                    name: np.concatenate([np.asarray(o[name]) for o in overlays])
                    for name in ("log_luminosity", "log_half_time", "measured")
                },
            }
        )
        stacked.write(args.out_dir / "overlay.ecsv", format="ascii.ecsv", overwrite=True)

    if not args.no_plot:
        plot_map(grid, points, args.out_dir / "map.png", title, overlays=overlays, show_points=args.show_points)

    print(f"\n{title}\nWrote {points_path}, {args.out_dir / 'map.ecsv'}" + ("" if args.no_plot else ", map.png"))


if __name__ == "__main__":
    main()
