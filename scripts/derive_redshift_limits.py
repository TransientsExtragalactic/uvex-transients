"""
Derive the redshift limit of every transient class from its SED.

Each class's ``DEFAULT_Z_LIM`` is the redshift beyond which, with 95% confidence, at most 1% of
events drawn from the SED's priors are brighter than 24.5 AB in either UVEX band (see
:meth:`~uvex_transients.transients.base.ExtragalacticTransient.get_detection_horizon`), rounded up to two
significant figures.
This script reproduces those numbers. Re-run it after changing an SED's priors, a class's duration, or
the magnitude limit, and copy the printed values into the classes.

It works in two passes per class. A coarse pass finds roughly where the limit lies. A fine pass then
evaluates a dense redshift grid in a window around that answer, because the limit is rounded up to a
grid redshift and a coarse grid overestimates it by up to one grid step. The fine pass also uses a denser
time grid. A class takes a few minutes, and the classes run in parallel.

The result is reproducible to the spacing of the fine grid, about 2.5%, not exactly. The window is centered
on the coarse answer, which depends on the redshift grid the coarse pass started from, so a rerun can land
one grid point away and move the last digit of the rounded value (for example 2.0 against 2.1 for the TDEs).
Any value from either run is a valid limit, since each is rounded up. The values in the transient classes were
produced with 1000 draws, 150 and 300 time points, and the settings below.

Usage::

    python scripts/derive_redshift_limits.py                    # every class
    python scripts/derive_redshift_limits.py Kilonova TypeIa    # chosen classes
    python scripts/derive_redshift_limits.py --mag-limit 25.0
"""

import argparse
import importlib
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from astropy import units as u

CLASSES = {
    "Kilonova": ("uvex_transients.transients.kilonovae", "Kilonova"),
    "LFBOT": ("uvex_transients.transients.LFBOTs", "LuminousFastBlueOpticalTransient"),
    "TDE": ("uvex_transients.transients.TDEs", "TidalDisruptionEvent"),
    "TypeIIP": ("uvex_transients.transients.supernovae", "TypeIIPSNe"),
    "TypeIIPExcess": ("uvex_transients.transients.supernovae", "TypeIIPExcessSNe"),
    "ShockCoolingIIb": ("uvex_transients.transients.supernovae", "ShockCoolingIIb"),
    "TypeIIb": ("uvex_transients.transients.supernovae", "TypeIIbSNe"),
    "TypeIb": ("uvex_transients.transients.supernovae", "TypeIbSNe"),
    "TypeIc": ("uvex_transients.transients.supernovae", "TypeIcSNe"),
    "TypeIcBL": ("uvex_transients.transients.supernovae", "TypeIcBLSNe"),
    "MagnetarSLSNe": ("uvex_transients.transients.supernovae", "MagnetarSLSNe"),
    "TypeIa": ("uvex_transients.transients.supernovae", "TypeIaSNe"),
}

CONFIDENCE = 0.95
TOLERANCE = 0.01
N_SAMPLES = 1000
T_MIN = 1e-3 * u.day


def round_up(value: float, significant: int = 2) -> float:
    """
    Round `value` up to `significant` significant figures.

    Parameters
    ----------
    value : float
        The number to round.
    significant : int, optional
        Number of significant figures to keep. The default is 2.

    Returns
    -------
    float
        The smallest number with `significant` significant figures that is at least `value`.
    """
    scale = 10.0 ** (np.floor(np.log10(value)) - (significant - 1))
    return round(float(np.ceil(value / scale - 1e-9) * scale), 10)


def derive(name: str, mag_limit: float) -> tuple[str, float, float]:
    """
    Derive the redshift limit of one transient class.

    Parameters
    ----------
    name : str
        Key of the class in ``CLASSES``.
    mag_limit : float
        The AB magnitude limit, in either UVEX band.

    Returns
    -------
    name : str
        The class key, passed through so that results can be matched up when run in parallel.
    coarse : float
        The limit from the coarse pass.
    fine : float
        The limit from the fine pass, which is the one to adopt.
    """
    from m4opt.missions import uvex

    from uvex_transients.missions import downsample_mission

    logging.disable(logging.INFO)
    module, class_name = CLASSES[name]
    transient = getattr(importlib.import_module(module), class_name)()
    bandpasses = downsample_mission(uvex).detector.bandpasses

    # Coarse pass: where is the limit, roughly? The grid spans a factor of 16 around the class's current limit,
    # and extends itself upward if that does not bracket the answer.
    curve, _ = transient.get_detection_horizon(
        mag_limit,
        bandpasses,
        z_min=transient.redshift_limit / 4,
        z_max=4 * transient.redshift_limit,
        confidence=CONFIDENCE,
        tolerance=TOLERANCE,
        n_samples=N_SAMPLES,
        n_z=40,
        n_time=150,
        t_min=T_MIN,
        rng=0,
        progress=False,
    )
    coarse = float(curve["z_limit"][0])

    # Fine pass: a dense redshift grid around the coarse answer, and a denser time grid. Draws that
    # cross below the window are rounded up to its first point, which cannot change the limit.
    grid = transient.sed.get_effective_peak_magnitudes(
        np.geomspace(coarse / 3, 1.5 * coarse, 60),
        bandpasses,
        t_min=T_MIN,
        t_max=transient.duration_limit,
        n_time=300,
        n_samples=N_SAMPLES,
        rng=0,
        progress=False,
    )
    fine = float(
        transient.sed.get_observability_curve(mag_limit, grid=grid, confidence=CONFIDENCE, tolerance=TOLERANCE)[
            "z_limit"
        ][0]
    )
    return name, coarse, fine


def main() -> None:
    """Derive and print the limits."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("classes", nargs="*", choices=list(CLASSES), help="classes to derive (default: all)")
    parser.add_argument("--mag-limit", type=float, default=24.5, help="AB magnitude limit (default: 24.5)")
    parser.add_argument("--workers", type=int, default=8, help="parallel classes (default: 8)")
    args = parser.parse_args()

    names = args.classes or list(CLASSES)
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(derive, names, [args.mag_limit] * len(names)))

    print(f"\nRedshift limits at {args.mag_limit:g} AB ({CONFIDENCE:.0%} confidence, {TOLERANCE:.0%} tolerance)")
    print(f"{'class':<18}{'coarse':>10}{'fine':>10}{'rounded up':>12}")
    for name, coarse, fine in results:
        print(f"{name:<18}{coarse:>10.3f}{fine:>10.3f}{round_up(fine):>12g}")


if __name__ == "__main__":
    main()
