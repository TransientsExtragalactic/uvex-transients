r"""
Statistical prior distributions for :mod:`uvex_transients.models`.

This module defines the :class:`Prior` interface and a collection of
one-dimensional probability distributions used to describe model-parameter
priors and generate random realizations.

Continuous priors are defined primarily through their log-probability density
and support. From these, the base class provides probability-density,
cumulative-distribution, and generic numerical-sampling functionality.
Subclasses may override the sampling implementation when a more efficient
analytic or library-backed generator is available.

The module also supports degenerate and discrete priors, for which probability
mass functions and specialized sampling behavior replace the usual continuous
density interface.

All priors operate on plain numerical coordinates. Physical units,
parameter transformations, and other model-specific semantics are handled by
the surrounding parameter infrastructure.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from scipy import integrate, stats
from scipy.special import logsumexp
from scipy.stats.sampling import NumericalInversePolynomial

from uvex_transients.utils import get_rng

__all__ = [
    "ConstantPrior",
    "DiscretePrior",
    "ExponentialPrior",
    "LogNormalPrior",
    "MixturePrior",
    "NormalPrior",
    "PowerLawPrior",
    "Prior",
    "TruncatedNormalPrior",
    "UniformPrior",
]


class _LogPDFDistribution:
    """
    Adapts a :class:`Prior` to the scalar ``pdf``/``logpdf``/``support`` protocol.

    Required by :class:`~scipy.stats.sampling.NumericalInversePolynomial`.
    UNU.RAN evaluates the density one Python float at a time, whereas
    :meth:`Prior._logpdf` is vectorized; this adapter bridges the two.

    Parameters
    ----------
    prior : Prior
        The prior to adapt.
    """

    __slots__ = ("_prior",)

    def __init__(self, prior: "Prior") -> None:
        """
        Store the `Prior` to adapt.

        Parameters
        ----------
        prior : Prior
            The prior to adapt.
        """
        self._prior = prior

    def logpdf(self, x: float) -> float:
        """
        Evaluate the prior's log-density at one scalar point.

        Parameters
        ----------
        x : float
            The point at which to evaluate the log-density.

        Returns
        -------
        float
            `Prior._logpdf` evaluated at `x`.
        """
        return float(self._prior._logpdf(np.asarray(x, dtype=np.float64)))

    def support(self) -> tuple[float, float]:
        """
        Return the adapted prior's support.

        Returns
        -------
        tuple of float
            The ``(lower, upper)`` support, from `Prior.support`.
        """
        return self._prior.support


@dataclass(frozen=True, slots=True)
class Prior(ABC):
    """
    Abstract base class for one-dimensional statistical priors.

    A prior represents a probability distribution over real-valued numerical
    coordinates. The primary purpose of a prior is to generate random samples
    through the :meth:`sample` method.

    Subclasses should be implemented as frozen dataclasses and are responsible
    for validating their own parameters (:meth:`_validate`) and providing the
    distribution's log-density (:meth:`_logpdf`). Everything else — sampling,
    :meth:`pdf`, :meth:`cdf`, :meth:`logpdf`, :meth:`logcdf` — is derived from
    ``_logpdf`` automatically.

    See Also
    --------
    scipy.stats.sampling.NumericalInversePolynomial :
        Backs the generic :meth:`_sample` fallback.

    Notes
    -----
    A minimal implementation looks like

    .. code-block:: python

        @dataclass(frozen=True)
        class NormalPrior(Prior):
            mean: float
            sigma: float

            def _validate(self):
                if self.sigma <= 0:
                    raise ValueError(
                        "`sigma` must be positive."
                    )

            def _logpdf(self, x):
                return scipy.stats.norm.logpdf(
                    x, loc=self.mean, scale=self.sigma
                )

    This is already enough for :meth:`sample` to work, via numerical
    inversion of the CDF built from ``_logpdf``. If a fast closed-form or
    ``scipy.stats`` sampler is available, override :meth:`_sample` too:

    .. code-block:: python

            def _sample(self, rng, size):
                return rng.normal(
                    self.mean, self.sigma, size=size
                )
    """

    # ----------------------------------- #
    # Class-Level Parameters              #
    # ----------------------------------- #
    DISTRIBUTION_NAME: ClassVar[str] = "prior"
    """str: The public-facing name of this distribution prior class."""

    _REGISTRY: ClassVar[dict[str, type["Prior"]]] = {}
    """dict[str, type[Prior]]: Every concrete `Prior` subclass, keyed by `DISTRIBUTION_NAME`; see `registry`."""

    # Cache for the generic numerical-inversion sampler (see `_sample`). Built
    # lazily on first use since constructing it requires evaluating `_logpdf`
    # across the support, which is wasted work for subclasses that override
    # `_sample` with a closed-form generator.
    #
    # NOTE: uses `default_factory` rather than a plain `default`. For an
    # `init=False` field, dataclasses normally leaves a plain default as a
    # class attribute rather than assigning it in `__init__`; but on a
    # `slots=True` base class the slot descriptor displaces that class
    # attribute, so subclass instances would have no value at all until one
    # was explicitly set. `default_factory` forces an explicit `__init__`
    # assignment, sidestepping the issue.
    _sampler: NumericalInversePolynomial | None = field(
        default_factory=lambda: None,
        init=False,
        repr=False,
        compare=False,
    )

    # ----------------------------------- #
    # Registration                        #
    # ----------------------------------- #
    def __init_subclass__(cls, **kwargs) -> None:
        """
        Register a concrete subclass in `_REGISTRY`, keyed by its own `DISTRIBUTION_NAME`.

        Parameters
        ----------
        **kwargs
            Forwarded to :meth:`object.__init_subclass__` unchanged; this
            class declares no class-keyword-argument options of its own.
        """
        # Explicit two-argument `super()` -- `Prior` is a `@dataclass(slots=True)` class,
        # which rebuilds the class object to add `__slots__`; a zero-argument `super()`
        # here would close over the pre-rebuild `Prior`, raising a `TypeError` at every
        # subclass definition (a known dataclass+slots interaction, not specific to this
        # method).
        super(Prior, cls).__init_subclass__(**kwargs)
        if ABC in cls.__bases__:
            # `cls` is itself still abstract -- not a real, sampleable distribution yet.
            return
        Prior._REGISTRY[cls.DISTRIBUTION_NAME] = cls

    @classmethod
    def registry(cls) -> dict[str, type["Prior"]]:
        """dict[str, type[Prior]]: A copy of every concrete `Prior` subclass, keyed by `DISTRIBUTION_NAME`."""
        return dict(cls._REGISTRY)

    # ----------------------------------- #
    # Initialization / Dunders            #
    # ----------------------------------- #
    # All of the prior subclasses are dataclasses, so we simply validate the
    # provided parameters in the __post_init__ hook and then provide the __call__
    # interface to the pdf function.
    def __post_init__(self) -> None:
        """
        Validate the distribution parameters.

        This method is called automatically after the dataclass constructor has
        assigned all fields. Validation is delegated to the subclass through
        :meth:`_validate`.
        """
        self._validate()

    def __call__(
        self,
        size: int = 1,
        *,
        rng: np.random.Generator | int | None = None,
    ) -> NDArray[np.float64]:
        """
        Draw random samples from the prior.

        This is a convenience alias for :meth:`sample`.

        Parameters
        ----------
        size : int, optional
            Number of samples to draw.
        rng : numpy.random.Generator, int, or None, optional
            Random-number source; see :meth:`sample`.

        Returns
        -------
        numpy.ndarray
            A one-dimensional ``float64`` array with shape ``(size,)``.
        """
        return self.sample(size=size, rng=rng)

    def __copy__(self) -> "Prior":
        """
        Return a copy with a freshly rebuilt (not shared) sampler cache.

        Returns
        -------
        Prior
            A new instance with the same field values, via `dataclasses.replace`.
        """
        # `replace` re-runs `__init__`/`_validate`, so `_sampler` (init=False)
        # is rebuilt from its `default_factory` rather than shared with the
        # original — the cached sampler closes over `_LogPDFDistribution(self)`,
        # and we don't want the copy silently pinning the original alive.
        return replace(self)

    def __deepcopy__(self, memo: dict) -> "Prior":
        """
        Return a deep copy, with every field (and the sampler cache) independently copied.

        Parameters
        ----------
        memo : dict
            The `copy.deepcopy` memo dict, used to preserve shared/cyclic
            references and avoid copying the same object twice.

        Returns
        -------
        Prior
            A new, independent instance.
        """
        if id(self) in memo:
            return memo[id(self)]

        result = replace(
            self,
            **{f.name: deepcopy(getattr(self, f.name), memo) for f in fields(self) if f.init},
        )
        memo[id(self)] = result
        return result

    # ----------------------------------- #
    # Abstract Methods                    #
    # ----------------------------------- #
    @abstractmethod
    def _validate(self) -> None:
        """
        Validate the distribution parameters.

        This method is called automatically during object construction after
        all dataclass fields have been initialized.

        Implementations should raise informative exceptions whenever the
        distribution parameters are invalid.
        """
        ...

    @abstractmethod
    def _logpdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluate the log probability density function.

        The single source of truth for the distribution: :meth:`pdf`,
        :meth:`cdf`, :meth:`logpdf`, :meth:`logcdf`, and the generic
        :meth:`_sample` fallback are all derived from this method.

        Parameters
        ----------
        x : numpy.ndarray
            Array of points at which to evaluate the log-density.

        Returns
        -------
        numpy.ndarray
            The log-density evaluated at each point in ``x``, with the same
            shape. Should be ``-inf`` outside :attr:`support`.
        """
        ...

    # ------------------------------------------- #
    # Statistical Methods                         #
    # ------------------------------------------- #
    @property
    def support(self) -> tuple[float, float]:
        """
        Tuple of float: The ``(lower, upper)`` support of the distribution.

        Subclasses whose support is not the entire real line (bounded,
        positive-only, etc.) should override this property; it is used both
        to bound the CDF integration in :meth:`cdf` and, via
        :class:`_LogPDFDistribution`, to seed the domain of the generic
        :meth:`_sample` fallback.
        """
        return (-np.inf, np.inf)

    def logpdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluate the log probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log-density.

        Returns
        -------
        numpy.ndarray
            The log-density at each point in ``x``.
        """
        return np.asarray(self._logpdf(np.asarray(x, dtype=np.float64)))

    def pdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluate the probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the density.

        Returns
        -------
        numpy.ndarray
            The density at each point in ``x``.
        """
        return np.exp(self.logpdf(x))

    def cdf(self, x: NDArray[np.float64]) -> NDArray[np.float64] | float:
        """
        Evaluate the cumulative distribution function.

        Computed by numerically integrating :meth:`pdf` from the lower edge
        of :attr:`support` up to each point in ``x`` (:func:`scipy.integrate.quad`).
        This is intended for introspection/diagnostics (e.g. plotting); it is
        not used by :meth:`sample`, which relies on
        :class:`~scipy.stats.sampling.NumericalInversePolynomial` instead.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the CDF.

        Returns
        -------
        numpy.ndarray or float
            The CDF at each point in ``x``. Returns a scalar if input was a scalar,
            otherwise returns an array with the same shape as the input.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        x_ndim = x_arr.ndim
        lower, _ = self.support

        def _pdf_scalar(t: float) -> float:
            """
            Evaluate the density at one scalar point, for `scipy.integrate.quad`.

            Parameters
            ----------
            t : float
                The point at which to evaluate the density.

            Returns
            -------
            float
                :meth:`pdf` at `t`.
            """
            return float(np.exp(self._logpdf(np.asarray(t, dtype=np.float64))))

        result = np.array([integrate.quad(_pdf_scalar, lower, xi)[0] for xi in np.atleast_1d(x_arr)])

        if x_ndim == 0:
            return float(result[0])
        else:
            return result.reshape(x_arr.shape)

    def logcdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluate the log cumulative distribution function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log-CDF.

        Returns
        -------
        numpy.ndarray
            The log-CDF at each point in ``x``.
        """
        with np.errstate(divide="ignore"):
            return np.log(self.cdf(x))

    def _logpmf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluate the log probability *mass* function.

        Only meaningful for discrete priors (see :class:`DiscretePrior`); the
        base implementation raises, since :class:`Prior` models continuous
        distributions by default.

        Parameters
        ----------
        x : numpy.ndarray
            Points at which to evaluate the log-pmf.

        Returns
        -------
        numpy.ndarray
            The log-pmf at each point in ``x``.

        Raises
        ------
        NotImplementedError
            Always, for a continuous distribution.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} is a continuous distribution and has no "
            "probability mass function; use `logpdf`/`pdf` instead."
        )

    def logpmf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluate the log probability mass function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log-pmf.

        Returns
        -------
        numpy.ndarray
            The log-pmf at each point in ``x``.

        Raises
        ------
        NotImplementedError
            If the distribution is continuous (see :meth:`_logpmf`).
        """
        return np.asarray(self._logpmf(np.asarray(x, dtype=np.float64)))

    def pmf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluate the probability mass function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the pmf.

        Returns
        -------
        numpy.ndarray
            The pmf at each point in ``x``.

        Raises
        ------
        NotImplementedError
            If the distribution is continuous (see :meth:`_logpmf`).
        """
        return np.exp(self.logpmf(x))

    # ----------------------------------- #
    # Public Methods                      #
    # ----------------------------------- #
    @property
    def name(self) -> str:
        """str: Human-readable name of the distribution."""
        return self.DISTRIBUTION_NAME

    def _sample(
        self,
        rng: np.random.Generator,
        size: int,
    ) -> NDArray[np.float64]:
        """
        Draw samples via numerical inversion of the CDF implied by :meth:`_logpdf`.

        This is the generic fallback: it builds (and caches) a
        :class:`~scipy.stats.sampling.NumericalInversePolynomial` sampler from
        :meth:`_logpdf` and :attr:`support` on first use. It requires no
        additional work from subclasses beyond ``_logpdf``, but is slower and
        only approximate compared to a closed-form generator. Subclasses with
        a known fast sampler (e.g. delegating to ``scipy.stats`` or
        ``numpy.random.Generator`` directly) should override this method.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random-number generator.
        size : int
            Number of samples to draw.

        Returns
        -------
        numpy.ndarray
            One-dimensional array containing exactly ``size`` samples.
        """
        sampler = self._sampler
        if sampler is None:
            # scipy-stubs' `_PINVDist` protocol requires `pdf`/`cdf`/a `support`
            # attribute, but scipy itself (see `NumericalInversePolynomial`'s
            # docstring) also accepts a `logpdf` method plus a `support()`
            # method -- which is what `_LogPDFDistribution` provides.
            sampler = NumericalInversePolynomial(
                _LogPDFDistribution(self)  # type: ignore[arg-type]
            )
            object.__setattr__(self, "_sampler", sampler)

        # scipy-stubs' `size`-and-`random_state` overload of `rvs` only
        # covers the scalar (`size=None`) case, even though scipy itself
        # (see `NumericalInversePolynomial`'s docstring) accepts both
        # together for any `size`. Splitting this into a `set_random_state`
        # call followed by a separate `rvs(size=size)` would be racy if
        # `self` (and so this cached `_sampler`) is ever shared/sampled
        # concurrently -- keep it as one atomic call.
        # `NumericalInversePolynomial` samples a continuous distribution, so
        # this is always float64 in practice; the stub's return type is
        # widened to cover discrete `Method` subclasses too.
        return np.asarray(
            sampler.rvs(size=size, random_state=rng),  # type: ignore[call-overload]
            dtype=np.float64,
        )

    @staticmethod
    def _validate_size(size: int) -> int:
        """
        Validate and normalize the requested sample size.

        Parameters
        ----------
        size : int
            Requested number of samples.

        Returns
        -------
        int
            The validated size, coerced to a plain Python ``int``.

        Raises
        ------
        TypeError
            If ``size`` is a ``bool`` or otherwise not integer-like.
        ValueError
            If ``size`` is not a positive integer.
        """
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)):
            raise TypeError(f"`size` must be an integer, got {type(size).__name__}.")

        if size < 1:
            raise ValueError(f"`size` must be a positive integer, got {size}.")

        return int(size)

    @staticmethod
    def _get_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
        """
        Resolve ``rng`` to a :class:`numpy.random.Generator`.

        Thin wrapper around :func:`uvex_transients.utils.get_rng`, giving
        :meth:`sample` a single, overridable hook for constructing the
        random-number generator.

        Parameters
        ----------
        rng : numpy.random.Generator, int, or None
            Random-number source; see :func:`uvex_transients.utils.get_rng`.

        Returns
        -------
        numpy.random.Generator
            The resolved generator.
        """
        return get_rng(rng)

    def sample(
        self,
        size: int = 1,
        *,
        rng: np.random.Generator | int | None = None,
    ) -> NDArray[np.float64]:
        """
        Draw random samples from the prior.

        Parameters
        ----------
        size : int, optional
            Number of samples to draw. Must be a positive integer. The default
            is ``1``.

        rng : numpy.random.Generator, int, or None, optional
            Random-number source.

            Passing a ``Generator`` is recommended when sampling from multiple
            priors so that all draws come from the same reproducible random
            sequence.

        Returns
        -------
        numpy.ndarray
            A one-dimensional ``float64`` array with shape ``(size,)``.

        Raises
        ------
        ValueError
            If the subclass returns an incorrect number of samples.
        """
        size = self._validate_size(size)
        rng = self._get_rng(rng)

        samples = np.asarray(
            self._sample(rng=rng, size=size),
            dtype=np.float64,
        )

        if samples.size != size:
            raise ValueError(f"{self.__class__.__name__} returned {samples.size} samples, but {size} were requested.")

        return samples.reshape(size)


@dataclass(frozen=True)
class ConstantPrior(Prior):
    """
    Prior which always returns a single constant value.

    Attributes
    ----------
    value : float
        Constant value to return.

    Notes
    -----
    A point mass has no probability density, so :meth:`~Prior.pdf`,
    :meth:`~Prior.cdf`, and :meth:`~Prior.logpdf` all raise
    :class:`NotImplementedError` for this class. Sampling does not go through
    :meth:`~Prior._logpdf` at all.
    """

    DISTRIBUTION_NAME = "constant"

    value: float

    def _validate(self) -> None:
        """Check that :attr:`value` is finite."""
        if not np.isfinite(self.value):
            raise ValueError("`value` must be finite.")

    @property
    def support(self) -> tuple[float, float]:
        """Tuple of float: ``(value, value)`` -- a degenerate point mass has no width."""
        return (self.value, self.value)

    def _logpdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Raise, since a point mass has no probability density.

        Parameters
        ----------
        x : numpy.ndarray
            Unused; every call raises.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(
            "ConstantPrior is a degenerate point mass and has no probability density; "
            "use `sample()`, which does not require `_logpdf`."
        )

    def _sample(
        self,
        rng: np.random.Generator,
        size: int,
    ) -> NDArray[np.float64]:
        """
        Return :attr:`value`, repeated `size` times.

        Parameters
        ----------
        rng : numpy.random.Generator
            Unused; every draw is the same constant.
        size : int
            Number of (identical) samples to return.

        Returns
        -------
        numpy.ndarray
            `value`, repeated `size` times.
        """
        return np.full(size, self.value, dtype=float)


@dataclass(frozen=True)
class UniformPrior(Prior):
    """
    Uniform prior over the interval ``[lower, upper)``.

    Attributes
    ----------
    lower : float
        Lower bound.
    upper : float
        Upper bound. Must satisfy ``upper > lower``.
    """

    DISTRIBUTION_NAME = "uniform"

    lower: float
    upper: float

    def _validate(self) -> None:
        """Check that :attr:`lower`/:attr:`upper` are finite and ``upper > lower``."""
        if not np.isfinite(self.lower):
            raise ValueError("`lower` must be finite.")

        if not np.isfinite(self.upper):
            raise ValueError("`upper` must be finite.")

        if self.upper <= self.lower:
            raise ValueError("`upper` must be greater than `lower`.")

    @property
    def support(self) -> tuple[float, float]:
        """Tuple of float: ``(lower, upper)``."""
        return (self.lower, self.upper)

    def _logpdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluate the uniform log-density via :func:`scipy.stats.uniform.logpdf`.

        Parameters
        ----------
        x : numpy.ndarray
            Points at which to evaluate the log-density.

        Returns
        -------
        numpy.ndarray
            The log-density at each point in `x`.
        """
        return stats.uniform.logpdf(x, loc=self.lower, scale=self.upper - self.lower)

    def _sample(
        self,
        rng: np.random.Generator,
        size: int,
    ) -> NDArray[np.float64]:
        """
        Draw samples via :meth:`numpy.random.Generator.uniform`.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random-number generator.
        size : int
            Number of samples to draw.

        Returns
        -------
        numpy.ndarray
            `size` samples drawn uniformly from ``[lower, upper)``.
        """
        return rng.uniform(
            self.lower,
            self.upper,
            size=size,
        )


@dataclass(frozen=True)
class NormalPrior(Prior):
    """
    Gaussian prior.

    Attributes
    ----------
    mean : float
        Mean of the distribution.
    sigma : float
        Standard deviation.
    """

    DISTRIBUTION_NAME = "normal"

    mean: float
    sigma: float

    def _validate(self) -> None:
        """Check that :attr:`mean` is finite and :attr:`sigma` is positive."""
        if not np.isfinite(self.mean):
            raise ValueError("`mean` must be finite.")

        if self.sigma <= 0:
            raise ValueError("`sigma` must be positive.")

    def _logpdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluate the Gaussian log-density via :func:`scipy.stats.norm.logpdf`.

        Parameters
        ----------
        x : numpy.ndarray
            Points at which to evaluate the log-density.

        Returns
        -------
        numpy.ndarray
            The log-density at each point in `x`.
        """
        return stats.norm.logpdf(x, loc=self.mean, scale=self.sigma)

    def _sample(
        self,
        rng: np.random.Generator,
        size: int,
    ) -> NDArray[np.float64]:
        r"""
        Draw samples via :meth:`numpy.random.Generator.normal`.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random-number generator.
        size : int
            Number of samples to draw.

        Returns
        -------
        numpy.ndarray
            `size` samples drawn from :math:`\mathcal{N}(\mathrm{mean}, \mathrm{sigma})`.
        """
        return rng.normal(
            self.mean,
            self.sigma,
            size=size,
        )


@dataclass(frozen=True)
class LogNormalPrior(Prior):
    r"""
    Log-normal prior.

    Samples are generated as

    .. math::

        x = \\exp(y),

    where

    .. math::

        y \\sim \\mathcal{N}(\\mu, \\sigma).

    Attributes
    ----------
    mean : float
        Mean of the underlying normal distribution.
    sigma : float
        Standard deviation of the underlying normal distribution.
    """

    DISTRIBUTION_NAME = "lognormal"

    mean: float
    sigma: float

    def _validate(self) -> None:
        """Check that :attr:`mean` is finite and :attr:`sigma` is positive."""
        if not np.isfinite(self.mean):
            raise ValueError("`mean` must be finite.")

        if self.sigma <= 0:
            raise ValueError("`sigma` must be positive.")

    @property
    def support(self) -> tuple[float, float]:
        """Tuple of float: ``(0.0, inf)``."""
        return (0.0, np.inf)

    def _logpdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluate the log-normal log-density via :func:`scipy.stats.lognorm.logpdf`.

        Parameters
        ----------
        x : numpy.ndarray
            Points at which to evaluate the log-density.

        Returns
        -------
        numpy.ndarray
            The log-density at each point in `x`.
        """
        return stats.lognorm.logpdf(x, s=self.sigma, scale=np.exp(self.mean))

    def _sample(
        self,
        rng: np.random.Generator,
        size: int,
    ) -> NDArray[np.float64]:
        """
        Draw samples via :meth:`numpy.random.Generator.lognormal`.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random-number generator.
        size : int
            Number of samples to draw.

        Returns
        -------
        numpy.ndarray
            `size` samples drawn from this log-normal distribution.
        """
        return rng.lognormal(
            self.mean,
            self.sigma,
            size=size,
        )


@dataclass(frozen=True)
class TruncatedNormalPrior(Prior):
    """
    Truncated normal prior.

    Attributes
    ----------
    mean : float
        Mean of the parent normal distribution.
    sigma : float
        Standard deviation.
    lower : float
        Lower truncation bound.
    upper : float
        Upper truncation bound.
    """

    DISTRIBUTION_NAME = "truncated_normal"

    mean: float
    sigma: float
    lower: float
    upper: float

    def _validate(self) -> None:
        """Check that :attr:`sigma` is positive and ``upper > lower``."""
        if self.sigma <= 0:
            raise ValueError("`sigma` must be positive.")

        if self.upper <= self.lower:
            raise ValueError("`upper` must exceed `lower`.")

    @property
    def support(self) -> tuple[float, float]:
        """Tuple of float: ``(lower, upper)``."""
        return (self.lower, self.upper)

    def _logpdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluate the truncated-normal log-density via :func:`scipy.stats.truncnorm.logpdf`.

        Parameters
        ----------
        x : numpy.ndarray
            Points at which to evaluate the log-density.

        Returns
        -------
        numpy.ndarray
            The log-density at each point in `x`.
        """
        a = (self.lower - self.mean) / self.sigma
        b = (self.upper - self.mean) / self.sigma

        return stats.truncnorm.logpdf(x, a, b, loc=self.mean, scale=self.sigma)

    def _sample(
        self,
        rng: np.random.Generator,
        size: int,
    ) -> NDArray[np.float64]:
        r"""
        Draw samples via rejection sampling from the parent normal distribution.

        Repeatedly draws from :math:`\mathcal{N}(\mathrm{mean}, \mathrm{sigma})`
        and keeps only the draws falling inside ``[lower, upper]``, until `size`
        accepted samples have been collected.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random-number generator.
        size : int
            Number of samples to draw.

        Returns
        -------
        numpy.ndarray
            `size` accepted samples.
        """
        samples = np.empty(size)

        n = 0
        while n < size:
            candidate = rng.normal(
                self.mean,
                self.sigma,
                size=size - n,
            )

            accepted = candidate[(candidate >= self.lower) & (candidate <= self.upper)]

            m = accepted.size

            samples[n : n + m] = accepted
            n += m

        return samples


@dataclass(frozen=True)
class ExponentialPrior(Prior):
    """
    Exponential prior.

    Attributes
    ----------
    scale : float
        Exponential scale length.
    """

    DISTRIBUTION_NAME = "exponential"

    scale: float

    def _validate(self) -> None:
        """Check that :attr:`scale` is positive."""
        if self.scale <= 0:
            raise ValueError("`scale` must be positive.")

    @property
    def support(self) -> tuple[float, float]:
        """Tuple of float: ``(0.0, inf)``."""
        return (0.0, np.inf)

    def _logpdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluate the exponential log-density via :func:`scipy.stats.expon.logpdf`.

        Parameters
        ----------
        x : numpy.ndarray
            Points at which to evaluate the log-density.

        Returns
        -------
        numpy.ndarray
            The log-density at each point in `x`.
        """
        return stats.expon.logpdf(x, scale=self.scale)

    def _sample(
        self,
        rng: np.random.Generator,
        size: int,
    ) -> NDArray[np.float64]:
        """
        Draw samples via :meth:`numpy.random.Generator.exponential`.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random-number generator.
        size : int
            Number of samples to draw.

        Returns
        -------
        numpy.ndarray
            `size` samples drawn from this exponential distribution.
        """
        return rng.exponential(
            self.scale,
            size=size,
        )


@dataclass(frozen=True)
class PowerLawPrior(Prior):
    r"""
    Continuous power-law prior.

    The probability density is

    .. math::

        p(x) \\propto x^{-\\alpha},

    over the interval ``[lower, upper]``.

    Attributes
    ----------
    alpha : float
        Power-law index.
    lower : float
        Lower bound.
    upper : float
        Upper bound.
    """

    DISTRIBUTION_NAME = "power_law"

    alpha: float
    lower: float
    upper: float

    def _validate(self) -> None:
        """Check that :attr:`lower` is positive and ``upper > lower``."""
        if self.lower <= 0:
            raise ValueError("`lower` must be positive.")

        if self.upper <= self.lower:
            raise ValueError("`upper` must exceed `lower`.")

    @property
    def support(self) -> tuple[float, float]:
        """Tuple of float: ``(lower, upper)``."""
        return (self.lower, self.upper)

    def _logpdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluate the power-law log-density in closed form.

        Handles the ``alpha == 1`` case (log-uniform) separately, since the
        general normalization has a removable singularity there.

        Parameters
        ----------
        x : numpy.ndarray
            Points at which to evaluate the log-density.

        Returns
        -------
        numpy.ndarray
            The log-density at each point in `x`; ``-inf`` outside
            ``[lower, upper]``.
        """
        x = np.asarray(x, dtype=np.float64)
        inside = (x >= self.lower) & (x <= self.upper)

        with np.errstate(divide="ignore", invalid="ignore"):
            if np.isclose(self.alpha, 1.0):
                log_norm = -np.log(np.log(self.upper / self.lower))
                log_density = log_norm - np.log(x)
            else:
                exponent = 1.0 - self.alpha
                log_norm = np.log(np.abs(exponent)) - np.log(np.abs(self.upper**exponent - self.lower**exponent))
                log_density = log_norm - self.alpha * np.log(x)

        return np.where(inside, log_density, -np.inf)

    def _sample(
        self,
        rng: np.random.Generator,
        size: int,
    ) -> NDArray[np.float64]:
        """
        Draw samples via inverse-CDF transform of a uniform draw.

        Closed-form for a power law, so no numerical inversion is needed.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random-number generator.
        size : int
            Number of samples to draw.

        Returns
        -------
        numpy.ndarray
            `size` samples drawn from this power-law distribution.
        """
        u = rng.random(size)

        if np.isclose(self.alpha, 1.0):
            return self.lower * (self.upper / self.lower) ** u

        exponent = 1.0 - self.alpha

        return (u * (self.upper**exponent - self.lower**exponent) + self.lower**exponent) ** (1.0 / exponent)


@dataclass(frozen=True)
class DiscretePrior(Prior):
    """
    Discrete weighted prior.

    Attributes
    ----------
    values : ndarray
        Possible sampled values.
    probabilities : ndarray
        Relative probabilities of each value.

    Notes
    -----
    A discrete distribution has a probability *mass* function, not a
    density: use :meth:`~Prior.logpmf`/:meth:`~Prior.pmf` rather than
    :meth:`~Prior.logpdf`/:meth:`~Prior.pdf`, which raise
    :class:`NotImplementedError` for this class.
    """

    DISTRIBUTION_NAME = "discrete"

    values: np.ndarray
    probabilities: np.ndarray

    def _validate(self) -> None:
        """Check that :attr:`values`/:attr:`probabilities` match in length and form a valid distribution."""
        if len(self.values) != len(self.probabilities):
            raise ValueError("`values` and `probabilities` must have the same length.")

        if np.any(self.probabilities < 0):
            raise ValueError("Probabilities must be non-negative.")

        if np.sum(self.probabilities) <= 0:
            raise ValueError("At least one probability must be positive.")

    def _logpdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Raise, since a discrete distribution has no probability density.

        Parameters
        ----------
        x : numpy.ndarray
            Unused; every call raises.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(
            "DiscretePrior is a discrete distribution and has no probability density; use `logpmf`/`pmf` instead."
        )

    def _logpmf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluate the log probability mass at each point in `x`.

        Points not matching any of :attr:`values` (within floating-point
        tolerance) get probability 0 (log-probability ``-inf``).

        Parameters
        ----------
        x : numpy.ndarray
            Points at which to evaluate the log-pmf.

        Returns
        -------
        numpy.ndarray
            The log-pmf at each point in `x`, normalized so :attr:`probabilities`
            sums to 1.
        """
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        values = np.asarray(self.values, dtype=np.float64)
        p = np.asarray(self.probabilities, dtype=np.float64)
        p = p / p.sum()

        matches = np.isclose(x[:, None], values[None, :])
        prob = np.where(matches, p[None, :], 0.0).sum(axis=1)

        with np.errstate(divide="ignore"):
            return np.log(prob)

    def _sample(
        self,
        rng: np.random.Generator,
        size: int,
    ) -> NDArray[np.float64]:
        """
        Draw samples via :meth:`numpy.random.Generator.choice`.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random-number generator.
        size : int
            Number of samples to draw.

        Returns
        -------
        numpy.ndarray
            `size` samples drawn from :attr:`values`, weighted by
            (normalized) :attr:`probabilities`.
        """
        p = self.probabilities / np.sum(self.probabilities)

        return rng.choice(
            self.values,
            size=size,
            p=p,
        )


@dataclass(frozen=True)
class MixturePrior(Prior):
    r"""
    Weighted mixture of component priors.

    The density is the weighted sum of the components' densities,

    .. math::

        p(x) = \sum_i w_i\,p_i(x),

    with :attr:`weights` normalized to sum to 1.

    Attributes
    ----------
    components : tuple of Prior
        The mixture's component distributions.
    weights : ndarray
        Relative weight of each component, in the same order as :attr:`components`. Need not sum to
        1; normalized internally.
    """

    DISTRIBUTION_NAME = "mixture"

    components: tuple["Prior", ...]
    weights: np.ndarray

    def _validate(self) -> None:
        """Check that :attr:`components`/:attr:`weights` match in length and form a valid mixture."""
        if len(self.components) == 0:
            raise ValueError("`components` must contain at least one prior.")

        if len(self.components) != len(self.weights):
            raise ValueError("`components` and `weights` must have the same length.")

        if np.any(np.asarray(self.weights) < 0):
            raise ValueError("Weights must be non-negative.")

        if np.sum(self.weights) <= 0:
            raise ValueError("At least one weight must be positive.")

    @property
    def support(self) -> tuple[float, float]:
        """Tuple of float: the union of every component's support, ``(min(lowers), max(uppers))``."""
        lowers, uppers = zip(*(component.support for component in self.components))
        return (min(lowers), max(uppers))

    def _logpdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluate the mixture log-density as the log-sum-exp of the weighted component log-densities.

        Parameters
        ----------
        x : numpy.ndarray
            Points at which to evaluate the log-density.

        Returns
        -------
        numpy.ndarray
            The log-density at each point in `x`.
        """
        weights = np.asarray(self.weights, dtype=np.float64)
        log_weights = np.log(weights / weights.sum())

        component_logpdfs = np.stack(
            [component._logpdf(x) + log_weight for component, log_weight in zip(self.components, log_weights)],
            axis=0,
        )
        return logsumexp(component_logpdfs, axis=0)

    def _sample(
        self,
        rng: np.random.Generator,
        size: int,
    ) -> NDArray[np.float64]:
        """
        Draw samples by first choosing a component per draw, then sampling from it.

        The number of draws assigned to each component is fixed by a single multinomial draw
        (rather than choosing a component independently per sample), so the total is always exactly
        `size`.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random-number generator.
        size : int
            Number of samples to draw.

        Returns
        -------
        numpy.ndarray
            `size` samples drawn from the mixture.
        """
        weights = np.asarray(self.weights, dtype=np.float64)
        counts = rng.multinomial(size, weights / weights.sum())

        samples = np.concatenate(
            [component.sample(size=count, rng=rng) for component, count in zip(self.components, counts) if count > 0]
        )
        rng.shuffle(samples)
        return samples
