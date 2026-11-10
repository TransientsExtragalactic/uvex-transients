r"""
The map: the data and the model of a phase-space recovery map, and the hook that chooses the next points.

A `PhaseSpaceMap` holds what has been measured so far, ``k`` successes of ``n`` tries at a set of points of a
`~uvex_transients.phase_space.space.Space`, and models the success fraction over the whole space with a Gaussian
process. It does one thing beyond that, which is to say where the next measurements should be (`PhaseSpaceMap.propose`).
It knows nothing about transients, surveys or how a point is measured: a point comes in as coordinates and counts, and
the quantities derived from the fraction (volume-times, efficiencies, redshift limits) are made by
`~uvex_transients.phase_space.samplers.products.Product` objects that are handed the map's estimate. The loop that
evaluates the points the map asks for is not here either; see `~uvex_transients.phase_space.campaign.run_campaign`.

The model is a Gaussian process on the logit of the success fraction, :math:`y_i=\mathrm{logit}\,\hat p_i` with
:math:`\hat p_i=(k_i+1/2)/(n_i+1)`, and a per-point noise variance :math:`1/[(n_i+1)\hat p_i(1-\hat p_i)]` from the
delta method (see `~uvex_transients.utils.stat_utils.binomial_logit`). It works on the unit cube of the space, so any
number of axes will do.

The next points maximize an acquisition that is a function of the posterior only, never of an unseen outcome:

``"variance"``
    :math:`w(\mathbf x)\,\sigma_g^2(\mathbf x)`, the posterior variance of :math:`\ln(p+\epsilon)` by the delta method,
    with :math:`w=[p(1-p)/(p+\epsilon)]^2`. The floor :math:`\epsilon` is the smallest recovery a single point can
    resolve, so effort is not spent separating a recovery of :math:`10^{-6}` from zero.
``"imse"``
    The reduction, from observing at the candidate, of that same weighted variance integrated over the whole space. It
    costs one covariance matrix per round and values a point by what it teaches about the whole map.

A batch is built greedily. After each pick the posterior covariance is updated as though the point had been measured
(the variance does not depend on the outcome), so the points of a batch spread out and can be evaluated in parallel.
"""

from collections.abc import Callable, Sequence

import numpy as np
from astropy.table import Table
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit

from uvex_transients.phase_space.samplers.products import Estimate, Product
from uvex_transients.phase_space.space import Space
from uvex_transients.utils import config, logger
from uvex_transients.utils.stat_utils import binomial_logit

__all__ = ["PhaseSpaceMap"]

# ============================================ #
# Constants                                    #
# ============================================ #
#: The acquisition rules (see the module documentation).
_ACQUISITIONS = ("variance", "imse")

#: Recoveries are clipped this far from 0 and 1 where the logit and its weights would diverge.
_P_CLIP = 1e-6

#: Default kernel: an amplitude times an anisotropic Matern 5/2 on the unit cube, as (initial value, bounds).
_AMPLITUDE = (1.0, (1e-2, 1e2))
_LENGTH_SCALE = (0.3, (0.03, 10.0))
_MATERN_NU = 2.5


def _configured(value, key: str):
    """Return `value`, or ``config["phase_space.<key>"]`` if it is `None`."""
    return config[f"phase_space.{key}"] if value is None else value


def _require_sklearn():
    """Import `sklearn.gaussian_process`, with a pointer to the optional extra if it is missing."""
    try:
        from sklearn import gaussian_process
    except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
        raise ImportError(
            "PhaseSpaceMap needs scikit-learn; install it with `pip install 'uvex_transients[phase_space]'`."
        ) from exc
    return gaussian_process


# ============================================ #
# Map                                          #
# ============================================ #
class PhaseSpaceMap:
    r"""
    The measured points of a map and the Gaussian process model of the success fraction over their space.

    Every argument that is `None` (the default) takes ``config["phase_space.sampler.<name>"]``.

    Parameters
    ----------
    space : ~uvex_transients.phase_space.space.Space
        The axes the map is over, with the ranges it covers. Any number of axes.
    acquisition : {"variance", "imse"}, optional
        The acquisition rule; see the module documentation.
    n_candidates : int, optional
        Size of the Sobol candidate set the acquisition is evaluated on each round.
    floor : float, optional
        The success fraction :math:`\epsilon` below which the log uncertainty is not chased.
    n_restarts : int, optional
        Restarts of the kernel hyperparameter optimizer.
    kernel : sklearn.gaussian_process.kernels.Kernel, optional
        Kernel on the unit cube, for the standardized logit. The default is an amplitude times an anisotropic Matern
        5/2.
    seed : int or numpy.random.SeedSequence, optional
        Seed for the designs, candidate sets and the kernel optimizer. Each of them draws a fresh child of it, so a map
        is reproducible but they do not share a stream.
    table : astropy.table.Table, optional
        Previously measured points (see `table`) to start from.

    Raises
    ------
    ValueError
        If `acquisition` or `floor` is out of range, or `table` lacks a column of the space or ``k`` or ``n``.
    """

    def __init__(
        self,
        space: Space,
        *,
        acquisition: str | None = None,
        n_candidates: int | None = None,
        floor: float | None = None,
        n_restarts: int | None = None,
        kernel=None,
        seed: int | np.random.SeedSequence | None = None,
        table: Table | None = None,
    ):
        acquisition = _configured(acquisition, "sampler.acquisition")
        n_candidates = _configured(n_candidates, "sampler.n_candidates")
        floor = _configured(floor, "sampler.floor")
        n_restarts = _configured(n_restarts, "sampler.n_restarts")

        # Validate everything before anything is stored
        if acquisition not in _ACQUISITIONS:
            raise ValueError(f"`acquisition` must be one of {_ACQUISITIONS}, got {acquisition!r}.")
        if not 0.0 < floor < 1.0:
            raise ValueError(f"`floor` must lie in (0, 1), got {floor!r}.")

        self._space = space
        self.acquisition = acquisition
        self.n_candidates = int(n_candidates)
        self.floor = float(floor)
        self._kernel = kernel
        self._n_restarts = int(n_restarts)
        self._seed = np.random.SeedSequence(seed) if not isinstance(seed, np.random.SeedSequence) else seed

        # The Gaussian process is fitted on first use
        self._gp = None
        self._y_mean = 0.0
        self._y_scale = 1.0

        # Resume from the points of an earlier run
        self._rows: list[dict] = []
        if table is not None:
            self._rows = [{name: row[name] for name in table.colnames} for row in table]
            for name in (*space.names, "k", "n"):
                if name not in table.colnames:
                    raise ValueError(f"Cannot resume from a table with no {name!r} column.")

    # -------------------------------------- #
    # Data                                   #
    # -------------------------------------- #
    @property
    def space(self) -> Space:
        """Space: The axes the map is over."""
        return self._space

    @property
    def settings(self) -> dict:
        """dict: The settings of the model and the acquisition, as the constructor takes them."""
        return {
            "acquisition": self.acquisition,
            "n_candidates": self.n_candidates,
            "floor": self.floor,
            "n_restarts": self._n_restarts,
        }

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def points(self) -> NDArray[np.float64]:
        """numpy.ndarray: The measured points in the coordinates of the space, shape ``(len(self), ndim)``."""
        return np.array([[row[name] for name in self._space.names] for row in self._rows], dtype=np.float64).reshape(
            len(self._rows), self._space.ndim
        )

    @property
    def table(self) -> Table:
        """astropy.table.Table: Every measured point, with the extra entries of its outcome as columns."""
        if self._rows:
            return Table(rows=self._rows)
        return Table(names=(*self._space.names, "k", "n"), dtype=(*(float,) * self._space.ndim, int, int))

    def add(self, x: ArrayLike, k: int, n: int, **extras) -> None:
        """
        Add a measured point.

        The model is not refitted until it is next used.

        Parameters
        ----------
        x : array-like
            The point, in the coordinates of the space, shape ``(ndim,)``.
        k, n : int
            The successes and the tries at the point.
        **extras
            Any other entries of the point, which become columns of `table`.

        Raises
        ------
        ValueError
            If the point is not of the space, or the counts do not satisfy ``0 <= k <= n``.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self._space.ndim,):
            raise ValueError(f"A point of this space has shape ({self._space.ndim},), got {x.shape}.")
        k, n = int(k), int(n)
        if n < 1 or not 0 <= k <= n:
            raise ValueError(f"Need 0 <= k <= n and n >= 1, got k={k} and n={n}.")
        row = {name: float(value) for name, value in zip(self._space.names, x)}
        row["k"], row["n"] = k, n
        row.update({key: value for key, value in extras.items() if key not in row})
        self._rows.append(row)

    def _training_data(self) -> tuple[NDArray, NDArray, NDArray]:
        """Return the unit-cube inputs, the logit target and its noise variance, over every point so far."""
        x = self.points
        k = np.array([row["k"] for row in self._rows])
        n = np.array([row["n"] for row in self._rows])
        y, variance = binomial_logit(k, n)
        return self._space.to_unit(x), y, variance

    # -------------------------------------- #
    # Model                                  #
    # -------------------------------------- #
    def fit(self) -> None:
        """
        Fit the Gaussian process to every point measured so far.

        The target is centred and scaled by hand, and the noise variances are scaled with it, so that the noise
        stays in the units of the data.

        Raises
        ------
        ValueError
            If fewer than two points have been measured.
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
                length_scale=[_LENGTH_SCALE[0]] * self._space.ndim, length_scale_bounds=_LENGTH_SCALE[1], nu=_MATERN_NU
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
        logger.debug("PhaseSpaceMap: fit %d points, kernel %s.", len(x), self._gp.kernel_)

    def predict(self, x: ArrayLike, *, cov: bool = False):
        """
        Predict the logit of the success fraction at points of the space.

        Parameters
        ----------
        x : array-like
            Points in the coordinates of the space, shape ``(m, ndim)``.
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
            mean, covariance = self._gp.predict(self._space.to_unit(x), return_cov=True)
            return mean * self._y_scale + self._y_mean, covariance * self._y_scale**2
        mean, std = self._gp.predict(self._space.to_unit(x), return_std=True)
        return mean * self._y_scale + self._y_mean, std * self._y_scale

    def recovery(self, x: ArrayLike, *, n_sigma: float = 1.0) -> dict[str, NDArray]:
        """
        Predict the success fraction, with an interval, at points of the space.

        Parameters
        ----------
        x : array-like
            Points in the coordinates of the space, shape ``(m, ndim)``.
        n_sigma : float, optional
            Half-width of the interval, in posterior standard deviations of the logit. The default is 1.

        Returns
        -------
        dict
            ``"p"`` (the posterior median), ``"p_lo"``, ``"p_hi"`` (the interval) and ``"sigma_logit"`` (the posterior
            standard deviation of the logit), each of shape ``(m,)``.
        """
        mean, std = self.predict(x)
        return {
            "p": expit(mean),
            "p_lo": expit(mean - n_sigma * std),
            "p_hi": expit(mean + n_sigma * std),
            "sigma_logit": std,
        }

    def estimate(self, x: ArrayLike, *, n_sigma: float = 1.0) -> Estimate:
        """
        Estimate the success fraction, with an interval, at points of any leading shape.

        This is what a `~uvex_transients.phase_space.samplers.products.Product` is given.

        Parameters
        ----------
        x : array-like
            Points in the coordinates of the space, shape ``(..., ndim)``.
        n_sigma : float, optional
            Half-width of the interval, in posterior standard deviations of the logit. The default is 1.

        Returns
        -------
        ~uvex_transients.phase_space.samplers.products.Estimate
            The estimate, of shape ``(...)``.
        """
        points = np.asarray(x, dtype=np.float64)
        if points.ndim == 0 or points.shape[-1] != self._space.ndim:
            raise ValueError(f"Points of this space have {self._space.ndim} coordinates, got shape {points.shape}.")
        shape = points.shape[:-1]
        recovery = self.recovery(points.reshape(-1, self._space.ndim), n_sigma=n_sigma)
        return Estimate(*(recovery[key].reshape(shape) for key in ("p", "p_lo", "p_hi")))

    def log_uncertainty(self, x: ArrayLike) -> NDArray[np.float64]:
        r"""
        Return the posterior standard deviation of :math:`\ln(p+\epsilon)` at points of the space.

        This is the quantity a tolerance bounds and the acquisition spends points to reduce: the uncertainty of the
        log success fraction, with the floor :math:`\epsilon` so that a fraction of zero is not chased.

        Parameters
        ----------
        x : array-like
            Points in the coordinates of the space, shape ``(m, ndim)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(m,)``.
        """
        mean, std = self.predict(x)
        return np.sqrt(self._weight(expit(mean))) * std

    # -------------------------------------- #
    # Derived maps                           #
    # -------------------------------------- #
    def evaluate(
        self, x: ArrayLike, products: Sequence[Product] = (), *, n_sigma: float = 1.0
    ) -> dict[str, NDArray[np.float64]]:
        """
        Evaluate the success fraction and derived quantities at points of the space.

        Parameters
        ----------
        x : array-like
            Points in the coordinates of the space, shape ``(..., ndim)``.
        products : sequence of Product, optional
            The derived quantities to make from the estimate and the points.
        n_sigma : float, optional
            Half-width of the interval, in posterior standard deviations of the logit.

        Returns
        -------
        dict of str to numpy.ndarray
            ``f``, ``f_lo`` and ``f_hi`` (the success fraction and its interval), ``sigma_log`` (see
            `log_uncertainty`), and every column of every product, each of shape ``(...)``.
        """
        x = np.asarray(x, dtype=np.float64)
        estimate = self.estimate(x, n_sigma=n_sigma)
        columns = {
            "f": estimate.p,
            "f_lo": estimate.p_lo,
            "f_hi": estimate.p_hi,
            "sigma_log": self.log_uncertainty(x.reshape(-1, self._space.ndim)).reshape(estimate.shape),
        }
        for product in products:
            columns.update(product(x, estimate))
        return columns

    def grid_table(
        self,
        size: int | Sequence[int],
        products: Sequence[Product] = (),
        *,
        bounds: ArrayLike | None = None,
        n_sigma: float = 1.0,
    ) -> Table:
        """
        Evaluate the map on a regular grid, with its derived quantities, as one table.

        Parameters
        ----------
        size : int or sequence of int
            Points per axis; see `~uvex_transients.phase_space.space.Space.grid`.
        products : sequence of Product, optional
            The derived quantities to make; see `evaluate`.
        bounds : array-like, optional
            The rectangle to evaluate, shape ``(ndim, 2)``, which can differ from the ranges of the space.
        n_sigma : float, optional
            Half-width of the interval, in posterior standard deviations of the logit.

        Returns
        -------
        astropy.table.Table
            One row per grid point, in the order of `~uvex_transients.phase_space.space.Space.grid`, with the
            coordinates, then the columns of `evaluate`. Its ``meta`` holds the ``space`` that was evaluated and the
            ``products`` (name, label and unit).
        """
        grid_space = self._space if bounds is None else self._space.with_bounds(bounds)
        points = self._space.grid(size, bounds)
        columns = self.evaluate(points, products, n_sigma=n_sigma)

        table = Table()
        for index, name in enumerate(self._space.names):
            table[name] = points[:, index]
        for name, values in columns.items():
            table[name] = values
        table.meta["space"] = grid_space.to_dict()
        table.meta["products"] = [{"name": p.name, "label": p.label, "unit": p.unit} for p in products]
        return table

    # -------------------------------------- #
    # Acquisition                            #
    # -------------------------------------- #
    def _weight(self, p: NDArray) -> NDArray:
        r"""Return :math:`w=[p(1-p)/(p+\epsilon)]^2`, the squared slope of :math:`\ln(p+\epsilon)` in the logit."""
        p = np.clip(p, _P_CLIP, 1.0 - _P_CLIP)
        return (p * (1.0 - p) / (p + self.floor)) ** 2

    @staticmethod
    def _expected_noise(p: NDArray, expected_n: Callable[[NDArray], ArrayLike]) -> NDArray:
        """Predict the logit noise variance of a point measured at success fraction `p`, given its expected tries."""
        p = np.clip(p, _P_CLIP, 1.0 - _P_CLIP)
        return 1.0 / (np.asarray(expected_n(p)) * p * (1.0 - p))

    def _sobol(self, n: int) -> NDArray:
        """Draw `n` points of a fresh scrambled Sobol design over the unit cube."""
        return self._space.sobol(n, seed=self._seed.spawn(1)[0])

    def initial_design(self, n: int) -> NDArray[np.float64]:
        """
        Draw a scrambled Sobol design over the space, for the points measured before the first fit.

        Parameters
        ----------
        n : int
            The number of points.

        Returns
        -------
        numpy.ndarray
            Points in the coordinates of the space, shape ``(n, ndim)``.
        """
        return self._space.from_unit(self._sobol(n))

    def propose(self, batch_size: int, expected_n: Callable[[NDArray], ArrayLike]) -> tuple[NDArray, float]:
        r"""
        Choose the next batch of points, fitting the model first. This is the hook that drives a campaign.

        Parameters
        ----------
        batch_size : int
            Number of points.
        expected_n : callable
            ``expected_n(p)`` gives how many tries a point of success fraction `p` is expected to get, which the
            acquisition needs to predict the noise of a point before it is measured. A
            `~uvex_transients.phase_space.samplers.stopping.StoppingRule` supplies it.

        Returns
        -------
        points : numpy.ndarray
            The chosen points in the coordinates of the space, shape ``(batch_size, ndim)``.
        uncertainty : float
            The largest posterior standard deviation of :math:`\ln(p+\epsilon)` over the candidates, before the
            batch, which is what a tolerance is compared to.
        """
        self.fit()

        # The posterior over a fresh candidate set, and what a point there would be worth
        candidates = self._sobol(self.n_candidates)
        mean, covariance = self.predict(self._space.from_unit(candidates), cov=True)
        p = expit(mean)
        weight = self._weight(p)
        noise = self._expected_noise(p, expected_n)

        # The largest uncertainty before the batch, which is what a tolerance is compared to
        uncertainty = float(np.sqrt(np.max(weight * np.diag(covariance))))

        chosen: list[int] = []
        for _ in range(min(int(batch_size), len(candidates))):
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

        return self._space.from_unit(candidates[chosen]), uncertainty
