"""
Build the stored detection-horizon curve of every shipped extragalactic transient class.

Each curve is written to ``uvex_transients/transients/horizon_curves/<ClassName>.ecsv``, where
:attr:`~uvex_transients.transients.base.ExtragalacticTransient.horizon_curve` reads it. Only the default
state of each class is stored: its own priors, duration and cosmology, in the ``uvex_fast`` bands, with the settings in
``uvex_transients.transients.horizon.SETTINGS``. Nothing here takes options for anything else.

Re-run it after changing a shipped class's priors or duration. The test
``tests/transients/test_horizon_curve.py::test_stored_curves_match_their_classes`` fails, naming the stale classes,
when it is needed. It cannot see a change to an SED's code, so rebuild by hand after one. A class takes seconds to a
minute and the classes run in parallel.

Usage::

    python scripts/build_horizon_curves.py                   # every shipped class
    python scripts/build_horizon_curves.py TypeIaSNe TDE     # the named classes (by class name)
"""

import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import uvex_transients.transients  # noqa: F401  (importing the package registers every shipped class)
from uvex_transients.transients.base import ExtragalacticTransient, TransientBase
from uvex_transients.transients.horizon import CURVE_DIR, HorizonCurve, curve_path


def shipped_classes() -> dict[str, type[ExtragalacticTransient]]:
    """Return every extragalactic class in the package that ships a stored horizon curve, by class name.

    Returns
    -------
    dict
        Keys are class names, values are the classes themselves.
    """
    return {
        name: cls
        for name, cls in TransientBase.registry().items()
        if issubclass(cls, ExtragalacticTransient)
        and cls.HAS_STORED_HORIZON_CURVE
        and cls.__module__.startswith("uvex_transients.transients.")
    }


def build(name: str) -> tuple[str, float, float]:
    """
    Generate and store the curve of one class.

    Parameters
    ----------
    name : str
        The class name.

    Returns
    -------
    name : str
        The class name, passed through so that results can be matched up when run in parallel.
    z_24_5 : float
        The curve's redshift limit at 24.5 AB, for the printed summary.
    seconds : float
        How long the class took.
    """
    logging.disable(logging.INFO)
    start = time.perf_counter()
    curve = HorizonCurve.generate(shipped_classes()[name](), progress=False)
    curve.write(curve_path(name))
    return name, curve.z_limit(24.5), time.perf_counter() - start


def main() -> None:
    """Build the requested curves, reporting each class as it finishes, and print a summary."""
    classes = shipped_classes()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("classes", nargs="*", choices=sorted(classes), help="classes to build (default: all)")
    parser.add_argument("--workers", type=int, default=8, help="parallel classes (default: 8)")
    args = parser.parse_args()

    names = args.classes or sorted(classes)
    workers = min(args.workers, len(names))
    print(f"Building {len(names)} horizon curve(s) with {workers} worker(s) into {CURVE_DIR}", flush=True)

    results, failures = {}, {}
    start = time.perf_counter()
    with ProcessPoolExecutor(workers) as pool:
        futures = {pool.submit(build, name): name for name in names}
        for done, future in enumerate(as_completed(futures), start=1):
            name = futures[future]
            try:
                _, z_24_5, seconds = future.result()
            except Exception as error:  # report it and carry on, so one bad class does not discard the others
                failures[name] = error
                print(f"[{done}/{len(names)}] {name}: FAILED ({type(error).__name__}: {error})", flush=True)
                continue
            results[name] = z_24_5
            print(
                f"[{done}/{len(names)}] {name}: {seconds:5.1f} s, z_limit(24.5 AB) = {z_24_5:.3f}"
                f"  ({time.perf_counter() - start:.0f} s elapsed)",
                flush=True,
            )

    print(f"\n{'class':<28}{'z_limit at 24.5 AB':>20}{'DEFAULT_Z_LIM':>16}")
    for name in sorted(results):
        print(f"{name:<28}{results[name]:>20.3f}{classes[name].DEFAULT_Z_LIM:>16g}")
    print(f"\nDone in {time.perf_counter() - start:.0f} s: {len(results)} built, {len(failures)} failed.")
    if failures:
        raise SystemExit(f"Failed: {', '.join(sorted(failures))}")


if __name__ == "__main__":
    main()
