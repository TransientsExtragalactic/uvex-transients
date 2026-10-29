"""
Active learning of a binomial recovery map with a Gaussian process.

`PhaseSpaceSampler` maps a recovery fraction over a rectangle of a two dimensional space (here, luminosity and
half-decline time) when each evaluation is expensive and returns only a binomial count: ``k`` recoveries out of ``n``
simulated events. Instead of filling a grid, it fits a Gaussian process to the points so far and puts each new batch
where one more simulation teaches the most about the map.

The sampler knows nothing about transients. It needs a callable that takes a point and returns ``k`` and ``n``
(`~uvex_transients.phase_space.evaluator.RecoveryEvaluator` is one), and, for the maps that need more than the fitted
recovery fraction, an evaluator that also supplies the noise-free scales of the survey.
"""

import importlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.units import Quantity
from numpy.typing import NDArray
from scipy.special import expit
from scipy.stats import qmc

from uvex_transients.utils import config, logger
from uvex_transients.utils.stat_utils import binomial_logit

__all__ = ["PhaseSpaceSampler"]

# ============================================ #
# Constants                                    #
# ============================================ #
#: The acquisition rules (see `PhaseSpaceSampler`).
_ACQUISITIONS = ("variance", "imse")

#: Recoveries are clipped this far from 0 and 1 where the logit and its weights would diverge.
_P_CLIP = 1e-6

#: What an evaluator has to provide for the maps that need more than the fitted recovery fraction.
_SCALE_METHODS = ("coverage", "volume_time", "all_sky_volume_time", "redshift_limit")

#: The only place a file may name an evaluator class from.
_EVALUATOR_PACKAGE = "uvex_transients."

#: Default kernel: an amplitude times an anisotropic Matern 5/2 on the unit square, as (initial value, bounds).
_AMPLITUDE = (1.0, (1e-2, 1e2))
_LENGTH_SCALE = (0.3, (0.03, 10.0))
_MATERN_NU = 2.5


def _configured(value, key: str):
    """Return `value`, or ``config["phase_space.<key>"]`` if it is `None`."""
    return config[f"phase_space.{key}"] if value is None else value


def _expected(value, evaluate, attribute: str, key: str):
    """Return `value`, or the `attribute` of the evaluator, or ``config["phase_space.<key>"]``, in that order."""
    if value is not None:
        return value
    return getattr(evaluate, attribute, config[f"phase_space.{key}"])


class _MissingEvaluator:
    """Stands in for the evaluator of a sampler loaded from a file that does not record one."""

    def __call__(self, x):
        raise RuntimeError("This sampler has no evaluator, so it cannot simulate more points; pass `evaluate`.")


def _require_sklearn():
    """Import `sklearn.gaussian_process`, with a pointer to the optional extra if it is missing."""
    try:
        from sklearn import gaussian_process
    except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
        raise ImportError(
            "PhaseSpaceSampler needs scikit-learn; install it with `pip install 'uvex_transients[phase_space]'`."
        ) from exc
    return gaussian_process


# ============================================ #
# Sampler                                      #
# ============================================ #
class PhaseSpaceSampler:
    r"""
    Adaptively map a recovery fraction over a rectangle of a two dimensional space.

    Every argument that is `None` (the default) takes ``config["phase_space.sampler.<name>"]``.

    Parameters
    ----------
    evaluate : callable
        ``evaluate(x)`` takes a length two array (a point of the space, in the units of `bounds`) and returns a
        mapping with at least the integer entries ``"k"`` (recoveries) and ``"n"`` (events simulated). Every other
        entry is stored as an extra column of `table`.
    bounds : sequence of (float, float)
        ``[(lo, hi), (lo, hi)]``, the rectangle to map. The Gaussian process works on it rescaled to the unit square.
    names : sequence of str, optional
        The names of the two coordinates, used as column names. Defaults to ``("x0", "x1")``.
    n_initial : int, optional
        Size of the scrambled Sobol design evaluated before the first fit.
    batch_size : int, optional
        Points chosen per round.
    max_points : int, optional
        Stop once this many points have been evaluated.
    tolerance : float, optional
        Stop once the largest posterior standard deviation of :math:`\ln(p+\epsilon)` over the candidates falls below
        this, a relative uncertainty on the recovery.
    acquisition : {"variance", "imse"}, optional
        The acquisition rule; see Notes.
    n_candidates : int, optional
        Size of the Sobol candidate set the acquisition is evaluated on each round.
    floor : float, optional
        The recovery :math:`\epsilon` below which the log uncertainty is not chased.
    k_target, n_min, n_max : float, optional
        What the evaluator is expected to do: simulate at least `n_min` events, and top up to `k_target` recoveries or
        `n_max` events. They only predict the noise of a point before it is measured, so that a batch can be built;
        the real ``k`` and ``n`` always enter the fit. By default they are taken from the evaluator (its ``k_target``,
        ``chunk_size`` and ``n_max``), or from ``config["phase_space.recovery"]`` if it has none.
    seed : int or numpy.random.SeedSequence, optional
        Seed for the designs, candidate sets and the kernel optimizer.
    kernel : sklearn.gaussian_process.kernels.Kernel, optional
        Kernel on the unit square, for the standardized logit. The default is an amplitude times an anisotropic
        Matern 5/2.
    n_restarts : int, optional
        Restarts of the kernel hyperparameter optimizer.
    map_fn : callable, optional
        ``map_fn(evaluate, points)`` returning the results in order; replace it to evaluate a batch in parallel. The
        default is the builtin `map`.
    table : astropy.table.Table, optional
        Previously evaluated points (see `table`) to resume from.

    Attributes
    ----------
    table : astropy.table.Table
        One row per evaluated point, in evaluation order.

    Notes
    -----
    The model is a Gaussian process on the logit of the recovery fraction, :math:`y_i=\mathrm{logit}\,\hat p_i` with
    :math:`\hat p_i=(k_i+1/2)/(n_i+1)`, and a per-point noise variance :math:`1/[(n_i+1)\hat p_i(1-\hat p_i)]` from the
    delta method (see `~uvex_transients.utils.stat_utils.binomial_logit`).

    The next points maximize an acquisition that is a function of the posterior only, never of an unseen outcome:

    ``"variance"``
        :math:`w(\mathbf x)\,\sigma_g^2(\mathbf x)`, the posterior variance of :math:`\ln(p+\epsilon)` by the delta
        method, with :math:`w=[p(1-p)/(p+\epsilon)]^2`. The floor :math:`\epsilon` is the smallest recovery a single
        point can resolve, so effort is not spent separating a recovery of :math:`10^{-6}` from zero.
    ``"imse"``
        The reduction, from observing at the candidate, of that same weighted variance integrated over the whole
        space. It costs one covariance matrix per round and values a point by what it teaches about the whole map.

    A batch is built greedily. After each pick the posterior covariance is updated as though the point had been
    measured (the variance does not depend on the outcome), so the points of a batch spread out and can be evaluated
    in parallel.
    """

    def __init__(
        self,
        evaluate: Callable[[NDArray], Mapping],
        bounds: Sequence[tuple[float, float]],
        *,
        names: Sequence[str] = ("x0", "x1"),
        n_initial: int | None = None,
        batch_size: int | None = None,
        max_points: int | None = None,
        tolerance: float | None = None,
        acquisition: str | None = None,
        n_candidates: int | None = None,
        floor: float | None = None,
        k_target: float | None = None,
        n_min: float | None = None,
        n_max: float | None = None,
        seed: int | np.random.SeedSequence | None = None,
        kernel=None,
        n_restarts: int | None = None,
        map_fn: Callable = map,
        table: Table | None = None,
    ):
        # Anything not given comes from the configuration, and what the evaluator is expected to do from the
        # evaluator itself, so that the two cannot disagree
        n_initial = _configured(n_initial, "sampler.n_initial")
        batch_size = _configured(batch_size, "sampler.batch_size")
        max_points = _configured(max_points, "sampler.max_points")
        tolerance = _configured(tolerance, "sampler.tolerance")
        acquisition = _configured(acquisition, "sampler.acquisition")
        n_candidates = _configured(n_candidates, "sampler.n_candidates")
        floor = _configured(floor, "sampler.floor")
        n_restarts = _configured(n_restarts, "sampler.n_restarts")
        k_target = _expected(k_target, evaluate, "k_target", "recovery.k_target")
        n_min = _expected(n_min, evaluate, "chunk_size", "recovery.chunk_size")
        n_max = _expected(n_max, evaluate, "n_max", "recovery.n_max")

        # Validate everything before anything is stored
        bounds = np.asarray(bounds, dtype=np.float64)
        if bounds.shape != (2, 2) or not np.all(bounds[:, 1] > bounds[:, 0]):
            raise ValueError(f"`bounds` must be [(lo, hi), (lo, hi)] with lo < hi, got {bounds.tolist()}.")
        if len(names) != 2 or len(set(names)) != 2:
            raise ValueError(f"`names` must be two distinct strings, got {list(names)}.")
        if acquisition not in _ACQUISITIONS:
            raise ValueError(f"`acquisition` must be one of {_ACQUISITIONS}, got {acquisition!r}.")
        for label, value in (("n_initial", n_initial), ("batch_size", batch_size), ("max_points", max_points)):
            if int(value) != value or value < 1:
                raise ValueError(f"`{label}` must be a positive integer, got {value!r}.")
        if not 0.0 < floor < 1.0:
            raise ValueError(f"`floor` must lie in (0, 1), got {floor!r}.")

        self._evaluate = evaluate
        self._bounds = bounds
        self._names = tuple(names)
        self.n_initial = int(n_initial)
        self.batch_size = int(batch_size)
        self.max_points = int(max_points)
        self.tolerance = float(tolerance)
        self.acquisition = acquisition
        self.n_candidates = int(n_candidates)
        self.floor = float(floor)
        self.k_target = float(k_target)
        self.n_min = float(n_min)
        self.n_max = float(n_max)
        self._kernel = kernel
        self._n_restarts = int(n_restarts)
        self._map_fn = map_fn
        self._seed = np.random.SeedSequence(seed) if not isinstance(seed, np.random.SeedSequence) else seed

        # The Gaussian process is fitted on first use
        self._gp = None
        self._y_mean = 0.0
        self._y_scale = 1.0

        # Resume from the points of an earlier run
        self._rows: list[dict] = []
        if table is not None:
            self._rows = [{name: row[name] for name in table.colnames} for row in table]
            for name in (*self._names, "k", "n"):
                if name not in table.colnames:
                    raise ValueError(f"Cannot resume from a table with no {name!r} column.")

    # -------------------------------------- #
    # Data                                   #
    # -------------------------------------- #
    @property
    def bounds(self) -> NDArray[np.float64]:
        """numpy.ndarray: The mapped rectangle, ``[[lo, hi], [lo, hi]]``."""
        return self._bounds.copy()

    @property
    def names(self) -> tuple[str, str]:
        """Tuple of str: The coordinate names."""
        return self._names

    @property
    def settings(self) -> dict:
        """dict: The settings that decide where points go and when to stop, as the constructor takes them."""
        return {
            "n_initial": self.n_initial,
            "batch_size": self.batch_size,
            "max_points": self.max_points,
            "tolerance": self.tolerance,
            "acquisition": self.acquisition,
            "n_candidates": self.n_candidates,
            "floor": self.floor,
            "k_target": self.k_target,
            "n_min": self.n_min,
            "n_max": self.n_max,
            "n_restarts": self._n_restarts,
        }

    def _evaluator_spec(self) -> dict | None:
        """Describe the evaluator for the file header, or return `None` if it does not record its `settings`."""
        settings = getattr(self._evaluate, "settings", None)
        if not isinstance(settings, Mapping):
            return None
        cls = type(self._evaluate)
        return {"class": f"{cls.__module__}.{cls.__qualname__}", "settings": dict(settings)}

    @property
    def table(self) -> Table:
        """
        astropy.table.Table: Every evaluated point, with the extra entries of `evaluate` as columns.

        Its ``meta`` holds the bounds and names, this sampler's `settings`, and, if the evaluator records its
        ``settings``, which evaluator to rebuild; that is what lets `load` restore a working sampler from a file.
        """
        if self._rows:
            table = Table(rows=self._rows)
        else:
            table = Table(names=(*self._names, "k", "n"), dtype=(float, float, int, int))
        table.meta["bounds"] = self._bounds.tolist()
        table.meta["names"] = list(self._names)
        table.meta["sampler"] = self.settings
        spec = self._evaluator_spec()
        if spec is not None:
            table.meta["evaluator"] = spec
        return table

    def __len__(self) -> int:
        return len(self._rows)

    def save(self, path: str | Path) -> None:
        """
        Write `table`, with the settings needed to `load` it, as an ECSV file, replacing any existing one.

        Parameters
        ----------
        path : str or pathlib.Path
            The destination.
        """
        self.table.write(Path(path), format="ascii.ecsv", overwrite=True)

    @classmethod
    def load(
        cls,
        path: str | Path,
        evaluate: Callable[[NDArray], Mapping] | None = None,
        *,
        evaluator_kwargs: Mapping | None = None,
        **kwargs,
    ) -> "PhaseSpaceSampler":
        """
        Restore a sampler from a file written by `save`.

        The points are loaded, and the bounds, names and settings are restored. If the file records an evaluator
        (see `table`), it is rebuilt, so that the sampler can evaluate every map and `run` further. The model is
        fitted on first use.

        Parameters
        ----------
        path : str or pathlib.Path
            An ECSV file written by `save`.
        evaluate : callable, optional
            As in the constructor. By default the evaluator recorded in the file is rebuilt, or, if there is none,
            a stand-in that raises if asked to simulate.
        evaluator_kwargs : mapping, optional
            Passed to the recorded evaluator's ``from_settings``, for what the file could not record, such as the
            ``schedule`` of a schedule that is not in the registry.
        **kwargs
            Any other constructor argument, replacing the one recorded in the file.

        Returns
        -------
        PhaseSpaceSampler
            A sampler holding the stored points.

        Raises
        ------
        ValueError
            If the file names an evaluator class from outside ``uvex_transients``.
        """
        table = Table.read(Path(path), format="ascii.ecsv")
        meta = table.meta

        # What the file records stands unless the caller overrides it
        kwargs.setdefault("bounds", meta["bounds"])
        kwargs.setdefault("names", tuple(meta["names"]))
        for key, value in meta.get("sampler", {}).items():
            kwargs.setdefault(key, value)

        # Rebuild the evaluator the file names, but only from inside this package
        if evaluate is None:
            spec = meta.get("evaluator")
            if spec is None:
                evaluate = _MissingEvaluator()
            else:
                module_name, _, class_name = spec["class"].rpartition(".")
                if not module_name.startswith(_EVALUATOR_PACKAGE):
                    raise ValueError(
                        f"Refusing to import the evaluator {spec['class']!r}; it is not in uvex_transients."
                    )
                evaluator_class = getattr(importlib.import_module(module_name), class_name)
                evaluate = evaluator_class.from_settings(spec["settings"], **dict(evaluator_kwargs or {}))
        return cls(evaluate, table=table, **kwargs)

    def _unit(self, x: NDArray) -> NDArray:
        """Rescale points from `bounds` to the unit square."""
        return (np.asarray(x, dtype=np.float64) - self._bounds[:, 0]) / (self._bounds[:, 1] - self._bounds[:, 0])

    def _physical(self, u: NDArray) -> NDArray:
        """Rescale points from the unit square to `bounds`."""
        return self._bounds[:, 0] + np.asarray(u, dtype=np.float64) * (self._bounds[:, 1] - self._bounds[:, 0])

    def _training_data(self) -> tuple[NDArray, NDArray, NDArray]:
        """Return the unit-square inputs, the logit target and its noise variance, over every point so far."""
        x = np.array([[row[name] for name in self._names] for row in self._rows], dtype=np.float64)
        k = np.array([row["k"] for row in self._rows])
        n = np.array([row["n"] for row in self._rows])
        y, variance = binomial_logit(k, n)
        return self._unit(x), y, variance

    # -------------------------------------- #
    # Model                                  #
    # -------------------------------------- #
    def fit(self) -> None:
        """
        Fit the Gaussian process to every point evaluated so far.

        The target is centred and scaled by hand, and the noise variances are scaled with it, so that the noise
        stays in the units of the data.
        """
        if len(self._rows) < 2:
            raise ValueError("Need at least two evaluated points to fit.")
        gaussian_process = _require_sklearn()
        kernels = gaussian_process.kernels

        # Standardize the target by hand, and scale the noise variances with it so that they stay in its units
        x, y, variance = self._training_data()
        self._y_mean = float(np.mean(y))
        self._y_scale = float(np.std(y)) or 1.0

        kernel = self._kernel
        if kernel is None:
            kernel = kernels.ConstantKernel(*_AMPLITUDE) * kernels.Matern(
                length_scale=[_LENGTH_SCALE[0]] * 2, length_scale_bounds=_LENGTH_SCALE[1], nu=_MATERN_NU
            )

        # A child of the seed makes the optimizer's restarts reproducible without sharing a stream with the designs
        child = self._seed.spawn(1)[0]
        self._gp = gaussian_process.GaussianProcessRegressor(
            kernel=kernel,
            alpha=variance / self._y_scale**2,
            normalize_y=False,
            n_restarts_optimizer=self._n_restarts,
            random_state=int(child.generate_state(1)[0]),
        )
        self._gp.fit(x, (y - self._y_mean) / self._y_scale)
        logger.debug("PhaseSpaceSampler: fit %d points, kernel %s.", len(x), self._gp.kernel_)

    def predict(self, x: NDArray, *, cov: bool = False):
        """
        Predict the logit of the recovery fraction at points of the space.

        Parameters
        ----------
        x : array-like
            Points in the units of `bounds`, shape ``(m, 2)``.
        cov : bool, optional
            If `True`, return the full posterior covariance instead of the standard deviation.

        Returns
        -------
        mean : numpy.ndarray
            Posterior mean of the logit, shape ``(m,)``.
        spread : numpy.ndarray
            Posterior standard deviation of the logit, shape ``(m,)``, or its covariance, ``(m, m)``.
        """
        if self._gp is None:
            self.fit()
        x = np.atleast_2d(x)
        if cov:
            mean, covariance = self._gp.predict(self._unit(x), return_cov=True)
            return mean * self._y_scale + self._y_mean, covariance * self._y_scale**2
        mean, std = self._gp.predict(self._unit(x), return_std=True)
        return mean * self._y_scale + self._y_mean, std * self._y_scale

    def recovery(self, x: NDArray, *, n_sigma: float = 1.0) -> dict[str, NDArray]:
        """
        Predict the recovery fraction, with an interval, at points of the space.

        Parameters
        ----------
        x : array-like
            Points in the units of `bounds`, shape ``(m, 2)``.
        n_sigma : float, optional
            Half-width of the interval, in posterior standard deviations of the logit. The default is 1.

        Returns
        -------
        dict
            ``"p"`` (the posterior median recovery), ``"p_lo"``, ``"p_hi"`` (the interval) and
            ``"sigma_logit"`` (the posterior standard deviation of the logit), each of shape ``(m,)``.
        """
        mean, std = self.predict(x)
        return {
            "p": expit(mean),
            "p_lo": expit(mean - n_sigma * std),
            "p_hi": expit(mean + n_sigma * std),
            "sigma_logit": std,
        }

    def log_uncertainty(self, x: NDArray) -> NDArray[np.float64]:
        r"""
        Return the posterior standard deviation of :math:`\ln(p+\epsilon)` at points of the space.

        This is the quantity `tolerance` bounds and the acquisition spends points to reduce: the uncertainty of the
        log recovery, with the floor :math:`\epsilon` (`floor`) so that a recovery of zero is not chased.

        Parameters
        ----------
        x : array-like
            Points in the units of `bounds`, shape ``(m, 2)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(m,)``.
        """
        mean, std = self.predict(x)
        return np.sqrt(self._weight(expit(mean))) * std

    # -------------------------------------- #
    # Maps                                   #
    # -------------------------------------- #
    @staticmethod
    def _broadcast(log_luminosity, log_half_time):
        """Broadcast the coordinates, and return them as arrays with the points stacked as rows."""
        log_luminosity, log_half_time = np.broadcast_arrays(
            np.asarray(log_luminosity, dtype=np.float64), np.asarray(log_half_time, dtype=np.float64)
        )
        return log_luminosity, log_half_time, np.column_stack([log_luminosity.ravel(), log_half_time.ravel()])

    def _scales(self):
        """Return the evaluator, which supplies the noise-free scales of the maps, if it can."""
        missing = [name for name in _SCALE_METHODS if not hasattr(self._evaluate, name)]
        if missing:
            raise RuntimeError(
                f"This map needs an evaluator with {missing}, such as RecoveryEvaluator. "
                "Load a file written by a run of one, or pass `evaluate`."
            )
        return self._evaluate

    def _scaled_recovery(self, log_luminosity, log_half_time, scale, interval):
        """Return the recovery fraction (and its interval) times `scale`, which is `None` or per point."""
        log_luminosity, log_half_time, points = self._broadcast(log_luminosity, log_half_time)
        recovery = self.recovery(points)
        factor = 1.0 if scale is None else np.asarray(scale(log_luminosity, log_half_time), dtype=np.float64).ravel()
        values = [(recovery[key] * factor).reshape(log_luminosity.shape) for key in ("p", "p_lo", "p_hi")]
        return tuple(values) if interval else values[0]

    def evaluate_recovery_map(self, log_luminosity, log_half_time, *, interval: bool = False):
        r"""
        Evaluate the recovery fraction :math:`f` in the sampled footprint.

        It is conditional on the footprint that was sampled, so it depends on the number of time bins of the run; see
        `evaluate_efficiency_map` for the version that does not.

        Parameters
        ----------
        log_luminosity, log_half_time : array-like
            The coordinates, in the units of `bounds`, broadcast against each other.
        interval : bool, optional
            If `True`, return ``(median, lower, upper)``, with the interval one posterior standard deviation of the
            logit wide. The default returns the median alone.

        Returns
        -------
        numpy.ndarray or tuple of numpy.ndarray
            The shape of the broadcast coordinates.
        """
        return self._scaled_recovery(log_luminosity, log_half_time, None, interval)

    def evaluate_efficiency_map(self, log_luminosity, log_half_time, *, interval: bool = False):
        r"""
        Evaluate the efficiency :math:`\eta = f\,\mathcal E/(4\pi\,T)` over the whole sky and survey window.

        The probability that an event exploding at a random position and time during the survey, out to the
        redshift limit, is recovered. It does not depend on the number of time bins, but does depend on the
        redshift limit. Needs an evaluator, see `_scales`.

        Parameters
        ----------
        log_luminosity, log_half_time : array-like
            The coordinates, in the units of `bounds`, broadcast against each other.
        interval : bool, optional
            As in `evaluate_recovery_map`.

        Returns
        -------
        numpy.ndarray or tuple of numpy.ndarray
            The shape of the broadcast coordinates.
        """
        return self._scaled_recovery(log_luminosity, log_half_time, self._scales().coverage, interval)

    def evaluate_recovered_volume_time_map(self, log_luminosity, log_half_time, *, interval: bool = False):
        r"""
        Evaluate the recovered volume-time :math:`f\,\mathcal V T`, in Mpc^3 yr.

        The expected number of recoveries at one event per Mpc^3 per yr, for a rate constant in redshift. It does not
        depend on the number of time bins. It is a lower bound only for luminosities so high that the redshift limit
        reaches where the band leaves the spectrum (log L_band of about 49). Needs an evaluator, see `_scales`.

        Parameters
        ----------
        log_luminosity, log_half_time : array-like
            The coordinates, in the units of `bounds`, broadcast against each other.
        interval : bool, optional
            As in `evaluate_recovery_map`.

        Returns
        -------
        numpy.ndarray or tuple of numpy.ndarray
            The shape of the broadcast coordinates.
        """
        return self._scaled_recovery(log_luminosity, log_half_time, self._scales().volume_time, interval)

    def evaluate_expected_recoveries_map(
        self, log_luminosity, log_half_time, rate: Quantity, *, interval: bool = False
    ):
        """
        Evaluate the expected number of recoveries over the survey, for a population of one luminosity and duration.

        The recovered volume-time times a rate that is constant in redshift. See `evaluate_recovered_volume_time_map`.

        Parameters
        ----------
        log_luminosity, log_half_time : array-like
            The coordinates, in the units of `bounds`, broadcast against each other.
        rate : ~astropy.units.Quantity
            The comoving volumetric rate, with units of inverse volume per time, for example ``10 / u.Gpc**3 / u.yr``.
        interval : bool, optional
            As in `evaluate_recovery_map`.

        Returns
        -------
        numpy.ndarray or tuple of numpy.ndarray
            The shape of the broadcast coordinates.
        """
        per_mpc3_yr = u.Quantity(rate).to_value(u.Mpc**-3 * u.yr**-1)
        volume_time = self._scales().volume_time
        return self._scaled_recovery(
            log_luminosity, log_half_time, lambda a, b: per_mpc3_yr * volume_time(a, b), interval
        )

    def evaluate_uncertainty_map(self, log_luminosity, log_half_time):
        r"""
        Evaluate the posterior standard deviation of :math:`\ln(f+\epsilon)`; see `log_uncertainty`.

        Parameters
        ----------
        log_luminosity, log_half_time : array-like
            The coordinates, in the units of `bounds`, broadcast against each other.

        Returns
        -------
        numpy.ndarray
            The shape of the broadcast coordinates.
        """
        log_luminosity, _, points = self._broadcast(log_luminosity, log_half_time)
        return self.log_uncertainty(points).reshape(log_luminosity.shape)

    def evaluate_coverage_map(self, log_luminosity, log_half_time):
        """
        Evaluate the fraction of the sky and survey window that the sampled footprint covers (noise-free).

        Parameters
        ----------
        log_luminosity, log_half_time : array-like
            The coordinates, in the units of `bounds`, broadcast against each other.

        Returns
        -------
        numpy.ndarray
            Between 0 and 1, the shape of the broadcast coordinates.
        """
        return self._scales().coverage(log_luminosity, log_half_time)

    def evaluate_volume_time_map(self, log_luminosity, log_half_time):
        """
        Evaluate the volume-time of the sampled footprint over the survey, in Mpc^3 yr (noise-free).

        Parameters
        ----------
        log_luminosity, log_half_time : array-like
            The coordinates, in the units of `bounds`, broadcast against each other.

        Returns
        -------
        numpy.ndarray
            The shape of the broadcast coordinates.
        """
        return self._scales().volume_time(log_luminosity, log_half_time)

    def evaluate_all_sky_volume_time_map(self, log_luminosity):
        """
        Evaluate the volume-time of the whole sky over the survey, in Mpc^3 yr (noise-free).

        Parameters
        ----------
        log_luminosity : array-like
            The luminosity coordinate, in the units of `bounds`.

        Returns
        -------
        numpy.ndarray
            The shape of `log_luminosity`.
        """
        return self._scales().all_sky_volume_time(log_luminosity)

    def evaluate_z_limit_map(self, log_luminosity):
        """
        Evaluate the redshift limit of a luminosity (noise-free).

        Parameters
        ----------
        log_luminosity : array-like
            The luminosity coordinate, in the units of `bounds`.

        Returns
        -------
        numpy.ndarray
            The shape of `log_luminosity`.
        """
        return self._scales().redshift_limit(log_luminosity)

    def map_table(
        self,
        grid_size: int | None = None,
        rate: Quantity | None = None,
        bounds: Sequence[tuple[float, float]] | None = None,
    ) -> Table:
        """
        Evaluate every map on a regular grid, with intervals, as one table.

        Parameters
        ----------
        grid_size : int, optional
            Points per axis. If `None` (the default), uses ``config["phase_space.map.grid_size"]``.
        rate : ~astropy.units.Quantity, optional
            A volumetric rate; adds the ``expected_recoveries`` columns. See `evaluate_expected_recoveries_map`.
        bounds : sequence of (float, float), optional
            The rectangle to evaluate, which can differ from `bounds`. The default is `bounds`.

        Returns
        -------
        astropy.table.Table
            One row per grid point, in ``ij`` order (the first coordinate varies slowest), with the coordinates,
            ``z_limit``, ``coverage``, ``volume_time``, ``all_sky_volume_time``, ``sigma_log`` and
            ``f``, ``efficiency`` and ``recovered_volume_time`` (each with ``_lo`` and ``_hi``). Its ``meta`` holds what
            the evaluator records about the axes, such as the band, which the figure uses to label them.
        """
        scales = self._scales()
        grid_size = _configured(grid_size, "map.grid_size")
        (l_lo, l_hi), (d_lo, d_hi) = self._bounds if bounds is None else np.asarray(bounds, dtype=np.float64)
        log_l, log_dt = np.meshgrid(
            np.linspace(l_lo, l_hi, grid_size), np.linspace(d_lo, d_hi, grid_size), indexing="ij"
        )
        luminosity, duration = log_l.ravel(), log_dt.ravel()

        table = Table()
        table.meta.update(getattr(scales, "table_meta", {}))
        table[self._names[0]], table[self._names[1]] = luminosity, duration

        # The noise-free scales, from the evaluator
        table["z_limit"] = self.evaluate_z_limit_map(luminosity)
        table["coverage"] = self.evaluate_coverage_map(luminosity, duration)
        table["volume_time"] = self.evaluate_volume_time_map(luminosity, duration)
        table["all_sky_volume_time"] = self.evaluate_all_sky_volume_time_map(luminosity)
        table["sigma_log"] = self.evaluate_uncertainty_map(luminosity, duration)

        # The fitted quantities, each with its interval
        for name, evaluate in (
            ("f", self.evaluate_recovery_map),
            ("efficiency", self.evaluate_efficiency_map),
            ("recovered_volume_time", self.evaluate_recovered_volume_time_map),
        ):
            for suffix, values in zip(("", "_lo", "_hi"), evaluate(luminosity, duration, interval=True)):
                table[name + suffix] = values
        if rate is not None:
            for suffix, values in zip(
                ("", "_lo", "_hi"), self.evaluate_expected_recoveries_map(luminosity, duration, rate, interval=True)
            ):
                table["expected_recoveries" + suffix] = values
        return table

    # -------------------------------------- #
    # Acquisition                            #
    # -------------------------------------- #
    def _weight(self, p: NDArray) -> NDArray:
        r"""Return :math:`w=[p(1-p)/(p+\epsilon)]^2`, the squared slope of :math:`\ln(p+\epsilon)` in the logit."""
        p = np.clip(p, _P_CLIP, 1.0 - _P_CLIP)
        return (p * (1.0 - p) / (p + self.floor)) ** 2

    def _expected_noise(self, p: NDArray) -> NDArray:
        """Predict the logit noise variance of a point measured at recovery `p`, given how the evaluator tops up."""
        p = np.clip(p, _P_CLIP, 1.0 - _P_CLIP)
        n_expected = np.clip(self.k_target / p, self.n_min, self.n_max)
        return 1.0 / (n_expected * p * (1.0 - p))

    def _sobol(self, n: int) -> NDArray:
        """Draw `n` points of a fresh scrambled Sobol design over the unit square."""
        child = self._seed.spawn(1)[0]
        sobol = qmc.Sobol(d=2, scramble=True, seed=np.random.default_rng(child))
        return sobol.random_base2(int(np.ceil(np.log2(n))))[:n]

    def propose(self, batch_size: int | None = None) -> tuple[NDArray, float]:
        r"""
        Choose the next batch of points, fitting the model first.

        Parameters
        ----------
        batch_size : int, optional
            Number of points. Defaults to `batch_size`.

        Returns
        -------
        points : numpy.ndarray
            The chosen points in the units of `bounds`, shape ``(batch_size, 2)``.
        uncertainty : float
            The largest posterior standard deviation of :math:`\ln(p+\epsilon)` over the candidates, before the
            batch, which is what `tolerance` is compared to.
        """
        batch_size = self.batch_size if batch_size is None else int(batch_size)
        self.fit()

        # The posterior over a fresh candidate set, and what a point there would be worth
        candidates = self._sobol(self.n_candidates)
        mean, covariance = self.predict(self._physical(candidates), cov=True)
        p = expit(mean)
        weight = self._weight(p)
        noise = self._expected_noise(p)

        # The largest uncertainty before the batch, which is what `tolerance` is compared to
        uncertainty = float(np.sqrt(np.max(weight * np.diag(covariance))))

        chosen: list[int] = []
        for _ in range(min(batch_size, len(candidates))):
            diagonal = np.diag(covariance)
            if self.acquisition == "variance":
                score = weight * diagonal
            else:
                score = (weight[:, None] * covariance**2).sum(axis=0) / (diagonal + noise)

            # Never the same candidate twice
            score[chosen] = -np.inf
            best = int(np.argmax(score))
            chosen.append(best)

            # Conditioning on a measurement changes the covariance but not the mean, so the batch can be built
            # without knowing any outcome.
            column = covariance[:, best].copy()
            covariance = covariance - np.outer(column, column) / (column[best] + noise[best])

        return self._physical(candidates[chosen]), uncertainty

    # -------------------------------------- #
    # Running                                #
    # -------------------------------------- #
    def _evaluate_points(self, points: NDArray) -> None:
        """Evaluate points and append their results to the stored rows."""
        results = list(self._map_fn(self._evaluate, [np.asarray(point, dtype=np.float64) for point in points]))
        for point, result in zip(points, results):
            for key in ("k", "n"):
                if key not in result:
                    raise KeyError(f"`evaluate` must return {key!r}; got keys {sorted(result)}.")

            # The coordinates, then whatever else the evaluator reported as extra columns
            row = {self._names[0]: float(point[0]), self._names[1]: float(point[1])}
            row.update({key: value for key, value in result.items() if key not in row})
            row["k"], row["n"] = int(row["k"]), int(row["n"])
            self._rows.append(row)

    def _evaluate_initial_design(self) -> None:
        """Evaluate a scrambled Sobol design to bring the number of points up to `n_initial`."""
        n_new = self.n_initial - len(self._rows)
        design = self._sobol(n_new)
        logger.info("PhaseSpaceSampler: evaluating an initial design of %d points.", n_new)
        self._evaluate_points(self._physical(design))

    def run(self, callback: Callable[["PhaseSpaceSampler"], None] | None = None) -> "PhaseSpaceSampler":
        """
        Evaluate points until `max_points` is reached or the map is converged to `tolerance`.

        The initial design is evaluated first if there are fewer than `n_initial` points. Each round then fits the
        model, and stops without spending the batch if the uncertainty is already below `tolerance`.

        Parameters
        ----------
        callback : callable, optional
            Called with this sampler after the initial design and after every batch, for example to
            ``sampler.save`` a checkpoint.

        Returns
        -------
        PhaseSpaceSampler
            This sampler, so that ``sampler.run().table`` works.
        """
        if len(self._rows) < self.n_initial:
            self._evaluate_initial_design()
            if callback is not None:
                callback(self)

        # Rounds of propose, evaluate and refit, until the budget is spent or the map is good enough
        while len(self._rows) < self.max_points:
            n_new = min(self.batch_size, self.max_points - len(self._rows))
            points, uncertainty = self.propose(n_new)
            if uncertainty < self.tolerance:
                logger.info("PhaseSpaceSampler: converged (%.3f < %.3f).", uncertainty, self.tolerance)
                break
            logger.info(
                "PhaseSpaceSampler: %d points, uncertainty %.3f (tolerance %.3f); evaluating %d more.",
                len(self._rows),
                uncertainty,
                self.tolerance,
                n_new,
            )
            self._evaluate_points(points)
            if callback is not None:
                callback(self)

        self.fit()
        return self
