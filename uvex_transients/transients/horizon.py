r"""
Stored detection-horizon curves of the transient classes.

A `HorizonCurve` is the redshift limit :math:`z_L(m)` of a transient population as a function of the AB magnitude limit
:math:`m`: beyond :math:`z_L`, with 95% confidence, at most 1% of events drawn from the SED's priors are brighter than
:math:`m` in either UVEX band. The calculation is in `uvex_transients.models.core.detection_horizon`; this module
stores and checks its result.

A curve takes seconds to a minute to compute, so each shipped class stores its own as
``horizon_curves/<ClassName>.ecsv``, which `~uvex_transients.transients.base.ExtragalacticTransient.horizon_curve`
reads. A custom class, or a shipped one
whose priors or duration changed, generates its curve on first use. Which is which is decided by a *fingerprint*, a
fixed-seed draw from the class's priors and its duration, stored with the curve and compared on use. ``make horizons``
rebuilds the stored files, and the test suite fails, naming the class, when one is stale.
"""

from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.table import QTable
from numpy.typing import ArrayLike, NDArray

from uvex_transients.missions import uvex_fast
from uvex_transients.utils import logger

__all__ = ["HorizonCurve", "curve_path", "fingerprint"]

# ========================================= #
# Settings / Configuration                  #
# ========================================= #
CURVE_DIR = Path(__file__).parent / "horizon_curves"
"""pathlib.Path: Directory holding the stored curve of every shipped transient class."""

# The constants below define what a stored curve is. They are deliberately not in the user configuration: changing
# one changes the science of every stored curve, so it has to come with a rebuild (`make horizons`), and the
# tests check that the stored files were built with exactly these values.

MAG_LIMITS = np.arange(21.0, 27.0 + 1e-9, 0.5)
"""numpy.ndarray: The single-dwell AB depths a curve is tabulated at.

21 AB is brighter than any UVEX selection and 27 AB deeper than any realistic one. Between rows `HorizonCurve.z_limit`
interpolates; shallower than the first row it returns the first redshift, deeper than the last it raises.
"""

SETTINGS = {
    "confidence": 0.95,
    "tolerance": 0.01,
    "n_samples": 500,
    "n_z": 40,
    "n_time": 60,
}
"""dict: The settings a stored curve is generated with, recorded in its metadata.

``confidence`` and ``tolerance`` are the guarantee. ``n_samples`` is the number of draws, whose cost is linear and
which cannot be below 299. ``n_z`` is the number of logarithmic redshifts each draw is followed across; the crossing is
interpolated between them, which keeps the limit within about 2%. ``n_time`` is the number of logarithmic rest-frame
times from `T_MIN` to the duration, and it must resolve each light curve's peak, since a missed peak makes an event look
fainter. They are chosen for a limit good to a few percent in redshift, not for the tightest possible guarantee.
"""

T_MIN = 1e-3 * u.day
"""~astropy.units.Quantity: Earliest rest-frame time of the light-curve grid (86 s). Only sets where the grid starts."""

Z_MIN = 0.003
"""float: Smallest redshift of the grid (about 13 Mpc). A limit cannot be reported below it."""

SEED = 0
"""int: Seed of every random draw, so that generating a curve twice gives the same table."""

# The fingerprint (see `fingerprint`) is this many fixed-seed draws from the priors, compared with a relative tolerance
# that absorbs floating-point noise between platforms but not a real change.
_FINGERPRINT_DRAWS = 16
_FINGERPRINT_RTOL = 1e-6


# =========================================================================== #
# Helpers                                                                     #
# =========================================================================== #
def curve_path(name: str) -> Path:
    """
    Return where the stored curve of a transient class lives.

    Parameters
    ----------
    name : str
        The class name, for example ``"TypeIaSNe"``.

    Returns
    -------
    pathlib.Path
        ``horizon_curves/<name>.ecsv``. The file may not exist.
    """
    return CURVE_DIR / f"{name}.ecsv"


def fingerprint(transient) -> NDArray:
    """
    Summarize the part of a transient that its curve depends on: the duration and the priors.

    The summary is the duration in days followed by a fixed-seed draw from the SED's parameter priors. Any
    change to a prior, a pinned parameter or the duration changes it. Nothing else is covered: not the SED's
    code, the cosmology or the bands.

    Parameters
    ----------
    transient : ~uvex_transients.transients.base.ExtragalacticTransient
        The transient to summarize.

    Returns
    -------
    numpy.ndarray
        A one-dimensional float array.
    """
    draws = transient.sed.sample_parameters(size=_FINGERPRINT_DRAWS, rng=SEED)
    parameters = [np.asarray(getattr(draws[name], "value", draws[name]), dtype=np.float64) for name in sorted(draws)]
    duration = np.array([transient.duration_limit.to_value(u.day)])
    return np.concatenate([duration, *parameters]).astype(np.float64)


# =========================================================================== #
# Horizon Curve                                                               #
# =========================================================================== #
class HorizonCurve:
    """
    The redshift limit of a transient population as a function of the AB magnitude limit.

    Parameters
    ----------
    table : ~astropy.table.QTable
        One row per magnitude limit, with columns ``mag_limit`` (in AB magnitudes, strictly increasing) and
        ``z_limit`` (finite, non-decreasing). Its ``meta`` records how the curve was made; see `generate`.

    Raises
    ------
    ValueError
        If the columns are missing, the magnitudes are not strictly increasing, or the redshifts are not finite
        and non-decreasing.

    Notes
    -----
    A curve is a single-dwell statement. It says nothing about a schedule, extinction or sky position. See
    :ref:`user_guide_transients_redshift_limits`.

    Examples
    --------
    .. code-block:: python

        from uvex_transients.transients.TDEs import (
            TidalDisruptionEvent,
        )

        tde = TidalDisruptionEvent()
        tde.redshift_limit = tde.horizon_curve.z_limit(
            23.0
        )
    """

    def __init__(self, table: QTable):
        missing = [name for name in ("mag_limit", "z_limit") if name not in table.colnames]
        if missing:
            raise ValueError(f"A horizon curve needs columns 'mag_limit' and 'z_limit'; missing {missing}.")

        mag_limit = np.asarray(getattr(table["mag_limit"], "value", table["mag_limit"]), dtype=np.float64)
        z_limit = np.asarray(table["z_limit"], dtype=np.float64)
        if mag_limit.ndim != 1 or mag_limit.size < 2 or np.any(np.diff(mag_limit) <= 0):
            raise ValueError("'mag_limit' must be a 1-d, strictly increasing array of at least two magnitudes.")
        if not np.all(np.isfinite(z_limit)) or np.any(z_limit <= 0) or np.any(np.diff(z_limit) < 0):
            raise ValueError("'z_limit' must be finite, positive and non-decreasing.")

        self._mag_limit = mag_limit
        self._z_limit = z_limit
        self._meta = dict(table.meta)

    # ---------------------------------------- #
    # Properties                               #
    # ---------------------------------------- #
    @property
    def mag_limits(self) -> NDArray[np.float64]:
        """numpy.ndarray: The tabulated AB magnitude limits."""
        return self._mag_limit.copy()

    @property
    def z_limits(self) -> NDArray[np.float64]:
        """numpy.ndarray: The redshift limit at each of `mag_limits`."""
        return self._z_limit.copy()

    @property
    def meta(self) -> dict:
        """dict: How the curve was made: its settings, duration, cosmology and fingerprint."""
        return dict(self._meta)

    @property
    def table(self) -> QTable:
        """~astropy.table.QTable: The curve as a table, with its metadata."""
        return QTable(
            {"mag_limit": self._mag_limit * u.ABmag, "z_limit": self._z_limit},
            meta=dict(self._meta),
        )

    def __repr__(self) -> str:
        """Summarize the curve's range."""
        return (
            f"HorizonCurve({self._meta.get('class', '?')}: {self._mag_limit[0]:g} to {self._mag_limit[-1]:g} AB, "
            f"z = {self._z_limit[0]:.3g} to {self._z_limit[-1]:.3g})"
        )

    # ---------------------------------------- #
    # Lookup                                   #
    # ---------------------------------------- #
    def z_limit(self, mag_limit: ArrayLike | u.Quantity) -> float | NDArray[np.float64]:
        r"""
        Look up the redshift limit for a magnitude limit.

        The curve is interpolated linearly in :math:`\log z` against the magnitude, which is exact for
        the nearby Universe, where :math:`m = 5\log_{10} z + \mathrm{const}`. A limit shallower than the
        first tabulated magnitude returns the first redshift, which is an overestimate and so the safe
        direction.

        Parameters
        ----------
        mag_limit : float, array-like or ~astropy.units.Quantity
            The AB magnitude limit(s).

        Returns
        -------
        float or numpy.ndarray
            The redshift limit(s), in the shape of `mag_limit`.

        Raises
        ------
        ValueError
            If a limit is deeper than the deepest tabulated magnitude, since the curve cannot bound the
            population there. Regenerate the curve with a deeper `MAG_LIMITS`.
        """
        mag = np.asarray(getattr(mag_limit, "value", mag_limit), dtype=np.float64)
        if np.any(np.isnan(mag)) or np.any(mag > self._mag_limit[-1]):
            raise ValueError(f"mag_limit={mag_limit!r} is outside the curve, which reaches {self._mag_limit[-1]:g} AB.")
        z = np.exp(np.interp(mag, self._mag_limit, np.log(self._z_limit)))
        return float(z) if z.ndim == 0 else z

    # ---------------------------------------- #
    # Validity                                 #
    # ---------------------------------------- #
    def matches(self, transient) -> bool:
        """
        Check whether this curve still describes a transient's duration and priors.

        Parameters
        ----------
        transient : ~uvex_transients.transients.base.ExtragalacticTransient
            The transient to check against.

        Returns
        -------
        bool
            `True` if the stored fingerprint equals the transient's current one (see `fingerprint`).
        """
        stored = self._meta.get("fingerprint")
        if stored is None:
            return False
        stored = np.asarray(stored, dtype=np.float64)
        current = fingerprint(transient)
        return stored.shape == current.shape and bool(np.allclose(stored, current, rtol=_FINGERPRINT_RTOL, atol=0.0))

    # ---------------------------------------- #
    # Input and output                         #
    # ---------------------------------------- #
    @classmethod
    def read(cls, path: str | Path) -> "HorizonCurve | None":
        """
        Read a stored curve.

        Parameters
        ----------
        path : str or pathlib.Path
            An ECSV file written by `write`.

        Returns
        -------
        HorizonCurve or None
            The curve, or `None` if the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            return None
        return cls(QTable.read(path, format="ascii.ecsv"))

    def write(self, path: str | Path) -> None:
        """
        Write the curve as an ECSV file, replacing any existing one.

        Parameters
        ----------
        path : str or pathlib.Path
            The destination.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.table.write(path, format="ascii.ecsv", overwrite=True)

    # ---------------------------------------- #
    # Generation                               #
    # ---------------------------------------- #
    @classmethod
    def generate(
        cls,
        transient,
        *,
        mag_limits: ArrayLike | None = None,
        progress: bool = True,
        **settings,
    ) -> "HorizonCurve":
        """
        Compute the curve of a transient in the ``uvex_fast`` bands, which takes seconds to a minute.

        Runs `~uvex_transients.models.core.detection_horizon.detection_horizon` on the transient's priors, duration
        and cosmology, with this module's settings and seed. One grid serves every magnitude limit, and the crossings
        are interpolated, so the limit is good to about 2% but is not guaranteed to be an overestimate.

        Parameters
        ----------
        transient : ~uvex_transients.transients.base.ExtragalacticTransient
            The transient to compute the curve of. Its priors, duration and cosmology are used.
        mag_limits : array-like, optional
            The AB magnitude limits to tabulate. Defaults to `MAG_LIMITS`.
        progress : bool, optional
            Whether to show a progress bar. The default is `True`.
        **settings
            Overrides of `SETTINGS` (``confidence``, ``tolerance``, ``n_samples``, ``n_z``, ``n_time``). The
            defaults are what the stored curves use, so overriding them is meant for tests.

        Returns
        -------
        HorizonCurve
            The curve. Nothing is written to disk.

        Raises
        ------
        TypeError
            If `settings` has a key that is not in `SETTINGS`.
        RuntimeError
            If the redshift grid, after being extended several times, still does not reach some limit.
        """
        unknown = sorted(set(settings) - set(SETTINGS))
        if unknown:
            raise TypeError(f"Unknown setting(s) {unknown}; available: {sorted(SETTINGS)}.")
        settings = {**SETTINGS, **settings}
        t_min = T_MIN if transient.HORIZON_T_MIN is None else transient.HORIZON_T_MIN
        limits = MAG_LIMITS if mag_limits is None else np.atleast_1d(np.asarray(mag_limits, dtype=np.float64))

        logger.info(
            "Generating the detection-horizon curve of %s (%d draws, %d redshifts x %d times). "
            "This takes seconds to a minute.",
            type(transient).__name__,
            settings["n_samples"],
            settings["n_z"],
            settings["n_time"],
        )
        table, _ = transient.get_detection_horizon(
            limits,
            uvex_fast.detector.bandpasses,
            z_min=Z_MIN,
            t_min=t_min,
            rng=SEED,
            progress=progress,
            interpolate=True,
            cosmology=transient.cosmology,
            **settings,
        )
        if not np.all(np.isfinite(table["z_limit"])):
            raise RuntimeError(
                f"The horizon of {type(transient).__name__} was not bracketed for mag_limit in "
                f"{limits[~np.isfinite(table['z_limit'])]}; the population may be too bright for the redshift grid."
            )

        meta = {
            "class": type(transient).__name__,
            "duration_day": float(transient.duration_limit.to_value(u.day)),
            "cosmology": transient.cosmology.name,
            "seed": SEED,
            "t_min_day": float(t_min.to_value(u.day)),
            "z_min": Z_MIN,
            "criterion": "any",
            "interpolate": True,
            "bands": list(table.meta["bands"]),
            "fingerprint": [float(x) for x in fingerprint(transient)],
            **{key: (int(v) if isinstance(v, (int, np.integer)) else float(v)) for key, v in settings.items()},
        }
        return cls(QTable({"mag_limit": table["mag_limit"], "z_limit": table["z_limit"]}, meta=meta))
