r"""
Base class for time- and frequency-dependent spectral models.

A :class:`SpectralModel` describes how a source's spectral luminosity,
:math:`L_\nu(\nu, t)`, evolves with observer-frame frequency :math:`\nu` and
time since explosion :math:`t`. From that single quantity, this base class
derives everything else a user typically needs to turn a physical model into
something observable: a bolometric light curve, a normalized spectral shape,
a redshifted and distance-diluted flux, a throughput-weighted band flux, AB
magnitudes, and a :class:`~synphot.SourceSpectrum` ready to feed into a
detector simulation.

To define a new model, subclass :class:`SpectralModel` and implement
:meth:`~SpectralModel._eval`, the natural-log spectral luminosity in cgs
units.

Every other quantity (fluxes, magnitudes, band-integrated fluxes, bolometric
luminosity) is derived automatically. All of these public methods come in a
consistent family of four, distinguished by suffix:

.. list-table::
   :header-rows: 1

   * - Suffix
     - Inputs / outputs
     - Example
   * - ``_log_cgs``
     - unit-free numbers, natural log
     - :meth:`SpectralModel.eval_log_cgs`
   * - ``_log``
     - physical :class:`~astropy.units.Quantity`, natural log
     - :meth:`SpectralModel.eval_log`
   * - ``_cgs``
     - unit-free numbers, linear scale
     - :meth:`SpectralModel.eval_cgs`
   * - (none)
     - physical ``Quantity``, linear scale
     - :meth:`SpectralModel.eval`

:class:`Lightcurve` and :class:`Spectrum` are the time-only and
frequency-only halves of the same idea: a :class:`Lightcurve` is just
:math:`L_\mathrm{bol}(t)`, and a :class:`Spectrum` is just a shape
:math:`S(\nu)` (not necessarily normalized to 1). :class:`ComposedSpectralModel`
combines one of each into a full :class:`SpectralModel`, with
:math:`L_\nu(\nu, t) = L_\mathrm{bol}(t) \cdot S(\nu) / \int S(\nu')\,d\nu'`.
"""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from copy import copy, deepcopy
from dataclasses import replace
from typing import ClassVar, Self

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.cosmology import FLRW
from astropy.modeling import Model
from astropy.table import QTable
from astropy.time import Time
from astropy.units import Quantity
from m4opt.synphot import Detector
from scipy.integrate import quad_vec
from synphot import SourceSpectrum, SpectralElement
from synphot import units as synphot_units
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from uvex_transients.dust import attenuation_callable
from uvex_transients.utils import get_rng, logger
from uvex_transients.utils.cosmology import resolve_cosmological_distances

from .._constants import AB_MAG_ZERO_POINT, H_CGS
from .._typing import (
    CGSParameterValue,
    FloatArray,
    FloatResult,
    NumericalInput,
    OverrideValue,
    ParameterSamples,
    ParameterValue,
    PhysicalInput,
    RNGInput,
)
from .._utils import (
    _BOL_FLUX_UNIT,
    _BOL_LUM_UNIT,
    _NEGLIGIBLE_FLUX_JY,
    _PLACEHOLDER_OBSERVER_LOCATION,
    _PLACEHOLDER_OBSTIME,
    _SED_SHAPE_UNIT,
    _SPEC_FLUX_UNIT,
    _SPEC_LUM_UNIT,
    _flat_flux_source_spectrum,
    hz_per_unit,
    measure_bands,
    model_class_from_kernel,
    normalize_times,
    photometry_table,
    resolve_bands,
    to_cgs_value,
)
from ._tolerance import (
    Criterion,
    EffectivePeakGrid,
    first_crossing,
    min_samples,
    reduce_bands,
    resolve_bandpasses,
    resolve_time_grid,
    tolerance_limit,
    wilks_rank,
)
from .parameters import Parameter

__all__ = ["ComposedSpectralModel", "Lightcurve", "SpectralModel", "Spectrum"]

# Loose enough that it only fires on genuine `quad_vec` non-convergence (its own default
# `epsrel` is 1e-8), not routine floating-point noise -- see `_warn_if_not_converged`.
_QUAD_VEC_WARN_RTOL = 1e-4

# Target size of the largest intermediate array in `get_effective_peak_magnitudes`.
_CHUNK_BYTES = 8_000_000


def _warn_if_not_converged(context: str, integral: FloatArray, err: float) -> None:
    """
    Log a warning if a `scipy.integrate.quad_vec` result's reported error is suspiciously large.

    `quad_vec` returns its integral and a scalar error-norm estimate but never raises on
    poor convergence, so silent non-convergence in a normalization/bolometric integral
    would otherwise be invisible until it shows up as a subtly wrong downstream flux.

    Parameters
    ----------
    context : str
        Short label identifying the caller, used to prefix the warning message.
    integral : numpy.ndarray
        The `quad_vec` result being checked.
    err : float
        `quad_vec`'s own scalar error-norm estimate for `integral`.
    """
    scale = max(float(np.max(np.abs(integral))), np.finfo(float).tiny)
    if err > _QUAD_VEC_WARN_RTOL * scale:
        logger.warning(
            "%s: quad_vec integral may not have converged (error estimate %.3g, relative to integral magnitude %.3g).",
            context,
            err,
            scale,
        )


class _ModelBase(Mapping[str, Parameter], ABC):
    r"""
    Shared parameter storage, construction, and bookkeeping.

    Common base for :class:`SpectralModel`, :class:`Lightcurve`, and
    :class:`Spectrum`. Houses everything that doesn't depend on what a
    subclass physically computes: parameter declaration and validation
    (:attr:`_DEFAULT_PARAMETERS`, :attr:`_DOMAIN`), construction and
    copying, the :class:`~collections.abc.Mapping` interface, parameter
    packing, and parameter sampling. A subclass supplies only the physics --
    its own ``_eval``-style abstract method(s) mapping numerical inputs to a
    natural-log cgs result, the public ``eval*`` family that wraps them (see
    the module docstring for that four-way convention), and
    ``simulate``.

    Not part of the public API; use :class:`SpectralModel`,
    :class:`Lightcurve`, or :class:`Spectrum` instead.

    Parameters
    ----------
    **overrides : Parameter, ~astropy.units.Quantity, float, or int
        Per-parameter overrides. See :meth:`__init__`.
    """

    # -------------------------------------- #
    # Class-Level Parameters                 #
    # -------------------------------------- #
    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {}
    """dict of Parameter: This model's parameters, and their default order.

    Deep-copied into each instance's own parameter set upon construction, so
    mutating one instance's parameters never affects the class default or any
    other instance.
    """

    _DOMAIN: ClassVar[tuple[Quantity, Quantity] | None] = None
    """tuple of Quantity, or None: The ``(low, high)`` frequency range a subclass integrates over, if any.

    ``None`` (the default) means this subclass has no frequency-domain
    integral to bound, and :meth:`__init_subclass__` skips the associated
    validation entirely -- true of :class:`Lightcurve`, which has no notion
    of frequency at all. :class:`SpectralModel` and :class:`Spectrum`
    override this with an actual ``(low, high)`` Quantity pair; see their
    own ``_DOMAIN`` documentation for what it bounds.
    """

    # -------------------------------------- #
    # Subclass Validation                    #
    # -------------------------------------- #
    def __init_subclass__(cls, **kwargs) -> None:
        """
        Validate a subclass's :attr:`_DEFAULT_PARAMETERS` and, if set, :attr:`_DOMAIN`.

        Parameters
        ----------
        **kwargs
            Forwarded to :meth:`object.__init_subclass__` unchanged; this
            class declares no class-keyword-argument options of its own.
        """
        super().__init_subclass__(**kwargs)

        for name, parameter in cls._DEFAULT_PARAMETERS.items():
            if not isinstance(parameter, Parameter):
                raise TypeError(
                    f"{cls.__name__}._DEFAULT_PARAMETERS[{name!r}] must be a Parameter "
                    f"instance, got {type(parameter).__name__}."
                )

        if cls._DOMAIN is None:
            return

        low, high = cls._DOMAIN
        if not (isinstance(low, Quantity) and isinstance(high, Quantity)):
            raise TypeError(f"{cls.__name__}._DOMAIN must be a (low, high) pair of Quantity objects.")
        if low.unit.physical_type != "frequency" or high.unit.physical_type != "frequency":
            raise TypeError(f"{cls.__name__}._DOMAIN must be expressed in frequency units.")
        if high <= low:
            raise ValueError(f"{cls.__name__}._DOMAIN must satisfy high > low, got ({low}, {high}).")

    # -------------------------------------- #
    # Construction and Copying               #
    # -------------------------------------- #
    def _init_parameters(self, overrides: Mapping[str, OverrideValue]) -> None:
        """
        Deep-copy :attr:`_DEFAULT_PARAMETERS` and apply constructor ``overrides`` on top.

        Parameters
        ----------
        overrides : mapping of str to (Parameter, Quantity, float, or int)
            Per-parameter overrides, keyed by parameter name. See :meth:`__init__`.
        """
        self._parameters = deepcopy(self._DEFAULT_PARAMETERS)

        unknown = [name for name in overrides if name not in self._parameters]
        if unknown:
            raise KeyError(
                f"{self.__class__.__name__} has no parameter(s) named {unknown}. "
                f"Valid parameters are {tuple(self._parameters)}."
            )

        for name, value in overrides.items():
            if isinstance(value, Parameter):
                self._parameters[name] = value
            else:
                self._parameters[name].fix(value)

    def __init__(self, **overrides: OverrideValue) -> None:
        """
        Create a model instance, optionally overriding some of its default parameters.

        Parameters
        ----------
        **overrides
            Per-parameter overrides, keyed by parameter name (must match one
            of this model's parameter names). Passing a plain value (a
            :class:`~astropy.units.Quantity`, ``float``, or ``int``) fixes
            that parameter to it; passing a
            :class:`~uvex_transients.models.core.parameters.Parameter` replaces the
            default parameter entirely (e.g. to use a different prior). In
            the latter case, the *same* ``Parameter`` instance is stored
            (not copied) -- passing one object to two model instances links
            them, so that fixing or sampling the parameter through either
            model affects both.

        Raises
        ------
        KeyError
            If ``overrides`` names a parameter this model doesn't have.
        """
        super().__init__()

        self._init_parameters(overrides)

    def __copy__(self) -> Self:
        """
        Return a shallow copy: a new instance sharing this one's `Parameter` objects.

        Returns
        -------
        instance of this class
            A new instance whose `_parameters` dict is a fresh mapping, but
            whose individual `Parameter` values are the *same* objects as
            `self`'s -- fixing or sampling a shared parameter through either
            instance affects both.
        """
        new = self.__class__.__new__(self.__class__)
        new._parameters = copy(self._parameters)
        return new

    def __deepcopy__(self, memo: dict) -> Self:
        """
        Return a deep copy: a new instance with its own independent `Parameter` objects.

        Parameters
        ----------
        memo : dict
            The `copy.deepcopy` memo dict, used to preserve shared/cyclic
            references and avoid copying the same object twice.

        Returns
        -------
        instance of this class
            A new instance with an independently deep-copied `_parameters` dict.
        """
        if id(self) in memo:
            return memo[id(self)]

        new = self.__class__.__new__(self.__class__)
        new._parameters = deepcopy(self._parameters, memo)
        memo[id(self)] = new
        return new

    # -------------------------------------- #
    # Mapping Interface                      #
    # -------------------------------------- #
    def __len__(self) -> int:
        """
        Return the number of parameters this model has.

        Returns
        -------
        int
            ``len(self._parameters)``.
        """
        return len(self._parameters)

    def __iter__(self) -> Iterator[str]:
        """
        Iterate over this model's parameter names, in declaration order.

        Returns
        -------
        Iterator of str
            An iterator over `_parameters`' keys.
        """
        return iter(self._parameters)

    def __getitem__(self, key: str) -> Parameter:
        """
        Look up one of this model's `Parameter` objects by name.

        Parameters
        ----------
        key : str
            The parameter name.

        Returns
        -------
        Parameter
            The named parameter.
        """
        return self._parameters[key]

    def __setitem__(self, key: str, value: Parameter) -> None:
        """
        Refuse item assignment; parameters may only be modified in place.

        Parameters
        ----------
        key : str
            The parameter name (unused; every call raises).
        value : Parameter
            The value that would have been assigned (unused; every call raises).

        Raises
        ------
        TypeError
            Always -- model parameters cannot be replaced this way.
        """
        raise TypeError("Model parameters cannot be replaced directly. Modify the existing parameter instead.")

    def __delitem__(self, key: str) -> None:
        """
        Refuse item deletion; a model's parameter set is fixed at construction.

        Parameters
        ----------
        key : str
            The parameter name that would have been deleted (unused; every call raises).

        Raises
        ------
        TypeError
            Always -- model parameters cannot be deleted.
        """
        raise TypeError("Model parameters cannot be deleted.")

    def __repr__(self) -> str:
        """
        Return a multi-line representation listing every parameter's fixed value or prior.

        Returns
        -------
        str
            One line per parameter, showing either its fixed value
            (``name: fixed=...``) or its prior (``name: free, prior=...``).
        """
        rows = []
        for name, parameter in self._parameters.items():
            if parameter.is_fixed:
                rows.append(f"    {name}: fixed={parameter.fixed_value!r}")
            else:
                rows.append(f"    {name}: free, prior={parameter.prior.name}")

        return f"{self.__class__.__name__}(\n" + "\n".join(rows) + "\n)"

    # -------------------------------------- #
    # Parameter Packing                       #
    # -------------------------------------- #
    @classmethod
    def pack_params_to_arrays(cls, **parameters: ParameterValue) -> tuple[ParameterValue, ...]:
        """
        Convert a dict of parameter values into an ordered sequence.

        The output is a plain tuple of the input values, reordered to match
        this model's parameter order -- not stacked into a single array
        (parameters may carry different, incompatible units).

        Parameters
        ----------
        **parameters
            Parameter values, keyed by name. Must supply a value for every
            one of this model's parameters.

        Returns
        -------
        tuple
            The values from ``parameters``, in this model's parameter order.

        Raises
        ------
        KeyError
            If ``parameters`` is missing a value for one or more of this
            model's parameters.

        See Also
        --------
        unpack_params_from_arrays : The inverse conversion.
        """
        missing = [name for name in cls._DEFAULT_PARAMETERS if name not in parameters]
        if missing:
            raise KeyError(f"Missing value(s) for parameter(s) {missing} required by {cls.__name__}.")

        return tuple(parameters[name] for name in cls._DEFAULT_PARAMETERS)

    @classmethod
    def unpack_params_from_arrays(cls, *parameters: ParameterValue) -> dict[str, ParameterValue]:
        """
        Convert an ordered sequence of parameter values back into a dict.

        The inverse of :meth:`pack_params_to_arrays`.

        Parameters
        ----------
        *parameters
            Parameter values, one per parameter of this model, in this
            model's parameter order.

        Returns
        -------
        dict
            ``{name: value}`` for each of this model's parameters.

        Raises
        ------
        ValueError
            If the number of positional values doesn't match the number of
            parameters.

        See Also
        --------
        pack_params_to_arrays : The inverse conversion.
        """
        names = tuple(cls._DEFAULT_PARAMETERS)
        if len(parameters) != len(names):
            raise ValueError(
                f"{cls.__name__} expected {len(names)} positional parameter value(s) {names}, got {len(parameters)}."
            )

        return dict(zip(names, parameters))

    # -------------------------------------- #
    # Parameter Sampling                     #
    # -------------------------------------- #
    def sample_parameters(
        self,
        size: int = 1,
        *,
        rng: RNGInput = None,
        parameters: list[str] | None = None,
    ) -> ParameterSamples:
        """
        Draw random samples of some or all of this model's parameters.

        Parameters
        ----------
        size : int
            Number of samples to draw per parameter.
        rng : numpy.random.Generator, int, or None
            Random-number source, forwarded to each
            :meth:`~uvex_transients.models.core.parameters.Parameter.sample`. Passing
            a shared :class:`~numpy.random.Generator` is recommended so that
            every parameter's draws come from the same reproducible stream.
        parameters : list of str, optional
            Names of the parameters to sample. If ``None`` (the default),
            every parameter is sampled.

        Returns
        -------
        dict
            ``{name: samples}`` for each requested parameter, each with
            shape ``(size,)``.
        """
        if parameters is None:
            parameters = list(self._parameters.keys())

        return {
            parameter_name: parameter.sample(size=size, rng=rng)
            for parameter_name, parameter in self._parameters.items()
            if parameter_name in parameters
        }


class Lightcurve(_ModelBase):
    r"""
    Abstract base class for time-dependent bolometric luminosity models.

    A :class:`Lightcurve` describes how a source's bolometric
    (frequency-integrated) luminosity, :math:`L_\mathrm{bol}(t)`, evolves
    with time since explosion :math:`t`. It shares :class:`SpectralModel`'s
    parameter storage, evaluation-family conventions, and sampling
    machinery, but carries none of :class:`SpectralModel`'s :math:`\nu`-dependent or
    observed-frame machinery (spectral shape, flux, band flux, magnitudes).

    To define a new model, subclass :class:`Lightcurve` and implement
    :meth:`_eval`, the natural log of :math:`L_\mathrm{bol}(t)` in cgs
    units. Every other quantity (the ``*_log_cgs``/``*_log``/``*_cgs``/plain
    family described in the module docstring) is derived automatically.

    Pairing a :class:`Lightcurve` with a :class:`Spectrum` via
    :class:`ComposedSpectralModel` turns it into a full :math:`L_\nu(\nu,
    t)` :class:`SpectralModel`.

    See Also
    --------
    Spectrum : The frequency-only counterpart this mirrors.
    ComposedSpectralModel : Combines a Lightcurve and a Spectrum into a full SED.
    """

    # -------------------------------------- #
    # Bolometric Luminosity: L_bol(t)         #
    # -------------------------------------- #
    @classmethod
    @abstractmethod
    def _eval(cls, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        Evaluate the natural log of :math:`L_\mathrm{bol}(t)`, in cgs units.

        This is the one method every model must implement. ``t`` and every
        parameter value are combined using plain NumPy broadcasting -- no
        axes are inserted automatically.

        Parameters
        ----------
        t : numpy.ndarray
            Time since explosion, in seconds. Always non-negative.
        **parameters
            This model's parameter values, in cgs units, broadcastable
            against ``t``.

        Returns
        -------
        numpy.ndarray
            The natural log of :math:`L_\mathrm{bol}(t)`, in erg/s.
        """
        ...

    @classmethod
    def eval_log_cgs(cls, t: NumericalInput, **parameters: CGSParameterValue) -> FloatResult:
        r"""
        Natural log of the bolometric luminosity, taking and returning plain cgs numbers.

        Parameters
        ----------
        t : array-like
            Time since explosion, in seconds.
        **parameters
            This model's parameter values, in cgs units. To evaluate several
            parameter realizations at once, give each parameter a leading
            batch axis.

        Returns
        -------
        numpy.ndarray or float
            The natural log of :math:`L_\mathrm{bol}(t)`, in erg/s. A scalar
            is returned if the result is 0-dimensional.
        """
        t_arr = np.asarray(t, dtype=np.float64)
        cgs_parameters: dict[str, CGSParameterValue] = {
            name: np.asarray(value, dtype=np.float64) for name, value in parameters.items()
        }

        result = cls._eval(t_arr, **cgs_parameters)
        return result.item() if result.ndim == 0 else result

    @classmethod
    def eval_log(cls, t: Quantity, **parameters: ParameterValue) -> FloatResult:
        r"""
        Natural log of the bolometric luminosity, given physical-unit inputs.

        Parameters
        ----------
        t : ~astropy.units.Quantity
            Time since explosion, with time units.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        numpy.ndarray or float
            The natural log of :math:`L_\mathrm{bol}(t)`, in erg/s.

        Raises
        ------
        TypeError
            If ``t`` is not a Quantity with time units.
        """
        if not isinstance(t, Quantity) or t.unit.physical_type != "time":
            raise TypeError("`t` must be an astropy Quantity with time units.")

        cgs_parameters: dict[str, CGSParameterValue] = {name: to_cgs_value(value) for name, value in parameters.items()}

        return cls.eval_log_cgs(t.cgs.value, **cgs_parameters)

    @classmethod
    def eval_cgs(cls, t: NumericalInput, **parameters: CGSParameterValue) -> FloatResult:
        r"""
        Bolometric luminosity, taking and returning plain cgs numbers.

        Parameters
        ----------
        t : array-like
            Time since explosion, in seconds.
        **parameters
            This model's parameter values, in cgs units. See :meth:`eval_log_cgs`.

        Returns
        -------
        numpy.ndarray or float
            :math:`L_\mathrm{bol}(t)`, in erg/s. A scalar is returned if the
            result is 0-dimensional.
        """
        return np.exp(cls.eval_log_cgs(t, **parameters))

    @classmethod
    def eval(cls, t: Quantity, **parameters: ParameterValue) -> Quantity:
        r"""
        Evaluate the bolometric luminosity at the given time.

        Parameters
        ----------
        t : ~astropy.units.Quantity
            Time since explosion.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`L_\mathrm{bol}(t)`, in erg/s.
        """
        return np.exp(cls.eval_log(t, **parameters)) * _BOL_LUM_UNIT

    @classmethod
    def eval_from_arrays(cls, t: Quantity, *parameters: ParameterValue) -> Quantity:
        r"""
        Positional-argument form of :meth:`eval`.

        Equivalent to ``cls.eval(t, **cls.unpack_params_from_arrays(*parameters))``.

        Parameters
        ----------
        t : ~astropy.units.Quantity
            Time since explosion.
        *parameters
            This model's parameter values, one per parameter, in this
            model's parameter order. See :meth:`unpack_params_from_arrays`.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`L_\mathrm{bol}(t)`, in erg/s.

        See Also
        --------
        pack_params_to_arrays : The inverse conversion, dict -> ordered sequence.
        unpack_params_from_arrays : Ordered sequence -> dict, used internally here.
        """
        return cls.eval(t, **cls.unpack_params_from_arrays(*parameters))

    # -------------------------------------- #
    # Simulation                              #
    # -------------------------------------- #
    def simulate(self, t: Quantity, size: int = 1, *, rng: RNGInput = None) -> Quantity:
        r"""
        Draw random parameter realizations and evaluate the model at the given time.

        Equivalent to ``self.eval(t, **self.sample_parameters(size=size, rng=rng))``.

        Parameters
        ----------
        t : ~astropy.units.Quantity
            Time since explosion. See :meth:`eval`.
        size : int
            Number of realizations to draw.
        rng : numpy.random.Generator, int, or None
            Random-number source, forwarded to :meth:`sample_parameters`.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`L_\mathrm{bol}(t)`, in erg/s.

        See Also
        --------
        eval : The underlying evaluation.
        sample_parameters : The underlying sampling.
        """
        return self.eval(t, **self.sample_parameters(size=size, rng=rng))


class Spectrum(_ModelBase):
    r"""
    Abstract base class for frequency-dependent spectral shape models.

    A :class:`Spectrum` describes how a source's light is distributed across
    frequency, as a *shape* :math:`S(\nu)`. It
    shares :class:`SpectralModel`'s parameter storage, evaluation-family
    conventions, and sampling machinery, but -- having no notion of time,
    redshift, or distance -- carries none of :class:`SpectralModel`'s
    :math:`t`-dependent or observed-frame machinery (bolometric luminosity,
    flux, band flux, magnitudes).

    To define a new model, subclass :class:`Spectrum` and implement
    :meth:`_eval`, the natural log of :math:`S(\nu)` in cgs units. Unlike
    :class:`SpectralModel`'s spectral shape, :math:`S(\nu)` need not
    integrate to any particular value by construction --
    :meth:`eval_normalization` computes whatever :math:`\int S(\nu)\,d\nu`
    actually is, which is exactly the factor :class:`ComposedSpectralModel`
    divides out to combine a :class:`Spectrum` with a :class:`Lightcurve`'s
    :math:`L_\mathrm{bol}(t)` into an exactly normalized :math:`L_\nu(\nu,
    t)`.

    See Also
    --------
    Lightcurve : The time-only counterpart this mirrors.
    ComposedSpectralModel : Combines a Lightcurve and a Spectrum into a full SED.
    """

    _DOMAIN: ClassVar[tuple[Quantity, Quantity]] = (0 * u.Hz, np.inf * u.Hz)
    """tuple of Quantity: The ``(low, high)`` frequency range integrated over
    when computing :meth:`eval_normalization` (see :meth:`_eval_normalization`).

    Only consulted by the default, numerical-quadrature implementation of
    :meth:`_eval_normalization`. A subclass that overrides that method with a
    closed-form expression does not need this to be meaningful.
    """

    # -------------------------------------- #
    # Spectral Shape: S(nu)                   #
    # -------------------------------------- #
    @classmethod
    @abstractmethod
    def _eval(cls, nu: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        Evaluate the natural log of :math:`S(\nu)`, in cgs units.

        This is the one method every model must implement. ``nu`` and every
        parameter value are combined using plain NumPy broadcasting -- no
        axes are inserted automatically. There is no requirement that
        :math:`S(\nu)` integrate to any particular value; see
        :meth:`eval_normalization`.

        Parameters
        ----------
        nu : numpy.ndarray
            Frequency, in Hz.
        **parameters
            This model's parameter values, in cgs units, broadcastable
            against ``nu``.

        Returns
        -------
        numpy.ndarray
            The natural log of :math:`S(\nu)`, in 1/Hz.
        """
        ...

    @classmethod
    def eval_log_cgs(cls, nu: NumericalInput, **parameters: CGSParameterValue) -> FloatResult:
        r"""
        Natural log of the spectral shape, taking and returning plain cgs numbers.

        Parameters
        ----------
        nu : array-like
            Frequency, in Hz.
        **parameters
            This model's parameter values, in cgs units. To evaluate several
            parameter realizations at once, give each parameter a leading
            batch axis.

        Returns
        -------
        numpy.ndarray or float
            The natural log of :math:`S(\nu)`, in 1/Hz. A scalar is returned
            if the result is 0-dimensional.
        """
        nu_arr = np.asarray(nu, dtype=np.float64)
        cgs_parameters: dict[str, CGSParameterValue] = {
            name: np.asarray(value, dtype=np.float64) for name, value in parameters.items()
        }

        result = cls._eval(nu_arr, **cgs_parameters)
        return result.item() if result.ndim == 0 else result

    @classmethod
    def eval_log(cls, nu: Quantity, **parameters: ParameterValue) -> FloatResult:
        r"""
        Natural log of the spectral shape, given physical-unit inputs.

        Parameters
        ----------
        nu : ~astropy.units.Quantity
            Frequency, with frequency units.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        numpy.ndarray or float
            The natural log of :math:`S(\nu)`, in 1/Hz.

        Raises
        ------
        TypeError
            If ``nu`` is not a Quantity with frequency units.
        """
        if not isinstance(nu, Quantity) or nu.unit.physical_type != "frequency":
            raise TypeError("`nu` must be an astropy Quantity with frequency units.")

        cgs_parameters: dict[str, CGSParameterValue] = {name: to_cgs_value(value) for name, value in parameters.items()}

        return cls.eval_log_cgs(nu.cgs.value, **cgs_parameters)

    @classmethod
    def eval_cgs(cls, nu: NumericalInput, **parameters: CGSParameterValue) -> FloatResult:
        r"""
        Spectral shape, taking and returning plain cgs numbers.

        Parameters
        ----------
        nu : array-like
            Frequency, in Hz.
        **parameters
            This model's parameter values, in cgs units. See :meth:`eval_log_cgs`.

        Returns
        -------
        numpy.ndarray or float
            :math:`S(\nu)`, in 1/Hz. A scalar is returned if the result is
            0-dimensional.
        """
        return np.exp(cls.eval_log_cgs(nu, **parameters))

    @classmethod
    def eval(cls, nu: Quantity, **parameters: ParameterValue) -> Quantity:
        r"""
        Evaluate the spectral shape at the given frequency.

        Parameters
        ----------
        nu : ~astropy.units.Quantity
            Frequency at which to evaluate the shape.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`S(\nu)`, in 1/Hz.
        """
        return np.exp(cls.eval_log(nu, **parameters)) * _SED_SHAPE_UNIT

    @classmethod
    def eval_from_arrays(cls, nu: Quantity, *parameters: ParameterValue) -> Quantity:
        r"""
        Positional-argument form of :meth:`eval`.

        Equivalent to ``cls.eval(nu, **cls.unpack_params_from_arrays(*parameters))``.

        Parameters
        ----------
        nu : ~astropy.units.Quantity
            Frequency at which to evaluate the shape.
        *parameters
            This model's parameter values, one per parameter, in this
            model's parameter order. See :meth:`unpack_params_from_arrays`.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`S(\nu)`, in 1/Hz.

        See Also
        --------
        pack_params_to_arrays : The inverse conversion, dict -> ordered sequence.
        unpack_params_from_arrays : Ordered sequence -> dict, used internally here.
        """
        return cls.eval(nu, **cls.unpack_params_from_arrays(*parameters))

    # -------------------------------------- #
    # Normalization: integral of S(nu) dnu    #
    # -------------------------------------- #
    @classmethod
    def _eval_normalization(cls, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        Evaluate the natural log of :math:`\int S(\nu)\,d\nu`, in cgs units.

        Default implementation: broadcasts ``parameters`` together, then
        numerically integrates :math:`\int \exp(\mathtt{\_eval}(\nu))\,d\nu`
        over :attr:`_DOMAIN` (:func:`scipy.integrate.quad_vec`, which
        evaluates the whole broadcast array at each trial frequency at once
        rather than looping over realizations one at a time).

        A model whose shape integral has a closed form (e.g. one already
        normalized to 1 by construction) should override this method
        directly, for both speed and exactness.

        Parameters
        ----------
        **parameters
            This model's parameter values, in cgs units, broadcastable
            against one another.

        Returns
        -------
        numpy.ndarray
            The natural log of :math:`\int S(\nu)\,d\nu`, dimensionless.
        """
        param_arrays = [np.asarray(value, dtype=np.float64) for value in parameters.values()]
        param_grids = dict(zip(parameters, np.broadcast_arrays(*param_arrays))) if param_arrays else {}

        lo, hi = cls._DOMAIN

        def integrand(nu: float) -> FloatArray:
            r"""
            Evaluate :math:`\exp(\mathtt{\_eval}(\nu))` at one trial frequency, for `quad_vec`.

            Parameters
            ----------
            nu : float
                Trial frequency, in Hz, supplied by `quad_vec`.

            Returns
            -------
            numpy.ndarray
                :math:`S(\nu)` broadcast against `param_grids`.
            """
            return np.exp(cls._eval(np.asarray(nu, dtype=np.float64), **param_grids))

        integral, err = quad_vec(integrand, float(to_cgs_value(lo)), float(to_cgs_value(hi)))
        _warn_if_not_converged(f"{cls.__name__}._eval_normalization", integral, err)

        return np.log(integral)

    @classmethod
    def eval_normalization_log_cgs(cls, **parameters: CGSParameterValue) -> FloatResult:
        r"""
        Natural log of the shape's frequency integral, taking and returning plain cgs numbers.

        Parameters
        ----------
        **parameters
            This model's parameter values, in cgs units. To evaluate several
            parameter realizations at once, give each parameter a leading
            batch axis.

        Returns
        -------
        numpy.ndarray or float
            The natural log of :math:`\int S(\nu)\,d\nu`. A scalar is
            returned if the result is 0-dimensional.
        """
        cgs_parameters: dict[str, CGSParameterValue] = {
            name: np.asarray(value, dtype=np.float64) for name, value in parameters.items()
        }

        result = cls._eval_normalization(**cgs_parameters)
        return result.item() if result.ndim == 0 else result

    @classmethod
    def eval_normalization_log(cls, **parameters: ParameterValue) -> FloatResult:
        r"""
        Natural log of the shape's frequency integral, given physical-unit inputs.

        Parameters
        ----------
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        numpy.ndarray or float
            The natural log of :math:`\int S(\nu)\,d\nu`.
        """
        cgs_parameters: dict[str, CGSParameterValue] = {name: to_cgs_value(value) for name, value in parameters.items()}

        return cls.eval_normalization_log_cgs(**cgs_parameters)

    @classmethod
    def eval_normalization_cgs(cls, **parameters: CGSParameterValue) -> FloatResult:
        r"""
        Return the shape's frequency integral as plain cgs numbers; see :meth:`eval_normalization_log_cgs`.

        Parameters
        ----------
        **parameters
            This model's parameter values, in cgs units. See :meth:`eval_normalization_log_cgs`.

        Returns
        -------
        numpy.ndarray or float
            :math:`\int S(\nu)\,d\nu`. A scalar is returned if the result is
            0-dimensional.
        """
        return np.exp(cls.eval_normalization_log_cgs(**parameters))

    @classmethod
    def eval_normalization(cls, **parameters: ParameterValue) -> Quantity:
        r"""
        Evaluate :math:`\int S(\nu)\,d\nu`.

        This is exactly the factor by which :math:`S(\nu)` must be divided
        to turn it into a shape that integrates to 1 -- the normalization
        :class:`ComposedSpectralModel` applies when combining a
        :class:`Spectrum` with a :class:`Lightcurve`'s
        :math:`L_\mathrm{bol}(t)` to get :math:`L_\nu(\nu, t)`.

        Parameters
        ----------
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`\int S(\nu)\,d\nu`, dimensionless.
        """
        return np.exp(cls.eval_normalization_log(**parameters)) * u.dimensionless_unscaled

    # -------------------------------------- #
    # Simulation                              #
    # -------------------------------------- #
    def simulate(self, nu: Quantity, size: int = 1, *, rng: RNGInput = None) -> Quantity:
        r"""
        Draw random parameter realizations and evaluate the model at the given frequency.

        Equivalent to ``self.eval(nu, **self.sample_parameters(size=size, rng=rng))``.

        Parameters
        ----------
        nu : ~astropy.units.Quantity
            Frequency at which to evaluate the model. See :meth:`eval`.
        size : int
            Number of realizations to draw.
        rng : numpy.random.Generator, int, or None
            Random-number source, forwarded to :meth:`sample_parameters`.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`S(\nu)`, in 1/Hz.

        See Also
        --------
        eval : The underlying evaluation.
        sample_parameters : The underlying sampling.
        """
        return self.eval(nu, **self.sample_parameters(size=size, rng=rng))


class SpectralModel(_ModelBase):
    r"""
    Abstract base class for time-dependent spectral energy distribution models.

    A ``SpectralModel`` represents the intrinsic spectral luminosity

    .. math::

        L_\nu(\nu, t),

    as a function of source-frame frequency :math:`\nu` and time since
    explosion :math:`t`. From this fundamental quantity, the base class
    provides a consistent interface for computing bolometric luminosities,
    normalized spectral shapes, observed flux densities, band-averaged
    fluxes, apparent magnitudes, and synthetic
    :class:`~synphot.SourceSpectrum` objects.

    Notes
    -----
    The core subclassing interface consists of:

    - :attr:`_DEFAULT_PARAMETERS` for declaring model parameters;
    - :attr:`_DOMAIN` for the frequency range used by the default bolometric
      integration; and
    - :meth:`_eval` for evaluating
      :math:`\log L_\nu(\nu, t)` in unit-stripped cgs coordinates.

    Every other public method -- including :meth:`as_astropy_model`
    and :meth:`as_source_spectrum`, which build synthetic spectra -- is
    derived automatically from :meth:`_eval` and requires no per-subclass
    override.

    Most user-facing quantities are available in parallel interfaces:

    ``*_log_cgs``
        Unit-stripped cgs inputs and logarithmic numerical output.

    ``*_log``
        Physical-unit inputs and logarithmic numerical output.

    ``*_cgs``
        Unit-stripped cgs inputs and linear numerical output.

    no suffix
        Physical-unit inputs and physical-unit output.

    Because magnitudes are already logarithmic quantities, their API consists
    only of unit-stripped ``*_cgs`` and unit-aware forms.

    All of the above are :class:`classmethod`\ s: they operate purely on the
    ``**parameters`` values passed in, not on any particular instance's
    stored parameter configuration. Only parameter *storage* -- inherited
    from :class:`_ModelBase` -- requires an instance.
    """

    _DOMAIN: ClassVar[tuple[Quantity, Quantity]] = (0 * u.Hz, np.inf * u.Hz)
    """tuple of Quantity: The ``(low, high)`` frequency range integrated over
    when computing the bolometric luminosity (see :meth:`_eval_bolometric`).

    Only consulted by the default, numerical-quadrature implementation of
    :meth:`_eval_bolometric`. A subclass that overrides that method with a
    closed-form expression does not need this to be meaningful.
    """

    # -------------------------------------- #
    # Spectral Luminosity: L_nu(nu, t)        #
    # -------------------------------------- #
    @classmethod
    @abstractmethod
    def _eval(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        Evaluate the natural log of :math:`L_\nu(\nu, t)`, in cgs units.

        This is the one method every model must implement. ``nu``, ``t``,
        and every parameter value are combined using plain NumPy
        broadcasting -- no axes are inserted automatically. See the module
        docstring for what that means in practice.

        Parameters
        ----------
        nu : numpy.ndarray
            Frequency, in Hz.
        t : numpy.ndarray
            Time since explosion, in seconds. Always non-negative.
        **parameters
            This model's parameter values, in cgs units, broadcastable
            against ``nu`` and ``t``.

        Returns
        -------
        numpy.ndarray
            The natural log of :math:`L_\nu(\nu, t)`, in erg/s/Hz.
        """
        ...

    @classmethod
    def eval_log_cgs(cls, nu: NumericalInput, t: NumericalInput, **parameters: CGSParameterValue) -> FloatResult:
        """
        Natural log of the spectral luminosity, taking and returning plain cgs numbers.

        Parameters
        ----------
        nu : array-like
            Frequency, in Hz.
        t : array-like
            Time since explosion, in seconds.
        **parameters
            This model's parameter values, in cgs units. To evaluate several
            parameter realizations at once, give each parameter a leading
            batch axis.

        Returns
        -------
        numpy.ndarray or float
            The natural log of the spectral luminosity, in erg/s/Hz. A
            scalar is returned if the result is 0-dimensional.
        """
        nu_arr = np.asarray(nu, dtype=np.float64)
        t_arr = np.asarray(t, dtype=np.float64)
        cgs_parameters: dict[str, CGSParameterValue] = {
            name: np.asarray(value, dtype=np.float64) for name, value in parameters.items()
        }

        result = cls._eval(nu_arr, t_arr, **cgs_parameters)
        return result.item() if result.ndim == 0 else result

    @classmethod
    def eval_log(cls, nu: Quantity, t: Quantity, **parameters: ParameterValue) -> FloatResult:
        """
        Natural log of the spectral luminosity, given physical-unit inputs.

        Parameters
        ----------
        nu : ~astropy.units.Quantity
            Frequency, with frequency units.
        t : ~astropy.units.Quantity
            Time since explosion, with time units.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        numpy.ndarray or float
            The natural log of the spectral luminosity, in erg/s/Hz.

        Raises
        ------
        TypeError
            If ``nu``/``t`` are not Quantities with frequency/time units.
        """
        if not isinstance(nu, Quantity) or nu.unit.physical_type != "frequency":
            raise TypeError("`nu` must be an astropy Quantity with frequency units.")
        if not isinstance(t, Quantity) or t.unit.physical_type != "time":
            raise TypeError("`t` must be an astropy Quantity with time units.")

        cgs_parameters: dict[str, CGSParameterValue] = {name: to_cgs_value(value) for name, value in parameters.items()}

        return cls.eval_log_cgs(nu.cgs.value, t.cgs.value, **cgs_parameters)

    @classmethod
    def eval_cgs(cls, nu: NumericalInput, t: NumericalInput, **parameters: CGSParameterValue) -> FloatResult:
        r"""
        Spectral luminosity, taking and returning plain cgs numbers.

        Parameters
        ----------
        nu : array-like
            Frequency, in Hz.
        t : array-like
            Time since explosion, in seconds.
        **parameters
            This model's parameter values, in cgs units. See :meth:`eval_log_cgs`.

        Returns
        -------
        numpy.ndarray or float
            :math:`L_\nu(\nu, t)`, in erg/s/Hz. A scalar is returned if the
            result is 0-dimensional.
        """
        return np.exp(cls.eval_log_cgs(nu, t, **parameters))

    @classmethod
    def eval(cls, nu: Quantity, t: Quantity, **parameters: ParameterValue) -> Quantity:
        r"""
        Evaluate the spectral luminosity at the given frequency and time.

        Parameters
        ----------
        nu : ~astropy.units.Quantity
            Frequency at which to evaluate the model.
        t : ~astropy.units.Quantity
            Time since explosion.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`L_\nu(\nu, t)`, in erg/s/Hz.
        """
        return np.exp(cls.eval_log(nu, t, **parameters)) * _SPEC_LUM_UNIT

    @classmethod
    def eval_from_arrays(cls, nu: Quantity, t: Quantity, *parameters: ParameterValue) -> Quantity:
        r"""
        Positional-argument form of :meth:`eval`.

        Equivalent to ``cls.eval(nu, t, **cls.unpack_params_from_arrays(*parameters))``.
        Useful when parameter values are already stored as a plain sequence
        (e.g. rows of an array) rather than a dict.

        Parameters
        ----------
        nu : ~astropy.units.Quantity
            Frequency at which to evaluate the model.
        t : ~astropy.units.Quantity
            Time since explosion.
        *parameters
            This model's parameter values, one per parameter, in this
            model's parameter order. See :meth:`unpack_params_from_arrays`.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`L_\nu(\nu, t)`, in erg/s/Hz.

        See Also
        --------
        pack_params_to_arrays : The inverse conversion, dict -> ordered sequence.
        unpack_params_from_arrays : Ordered sequence -> dict, used internally here.
        """
        return cls.eval(nu, t, **cls.unpack_params_from_arrays(*parameters))

    # -------------------------------------- #
    # Bolometric Luminosity: L_bol(t)         #
    # -------------------------------------- #
    @classmethod
    def _eval_bolometric(cls, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        Evaluate the natural log of the bolometric luminosity, in cgs units.

        Default implementation: broadcasts ``t`` and ``parameters`` together,
        then numerically integrates :math:`\int \exp(\mathtt{\_eval}(\nu,
        t))\,d\nu` over :attr:`_DOMAIN` (:func:`scipy.integrate.quad_vec`,
        which evaluates the whole broadcast array at each trial frequency at
        once rather than looping over realizations/times one at a time).

        A model for which this integral has a closed form should override
        this method directly, for both speed and exactness.

        Parameters
        ----------
        t : numpy.ndarray
            Time since explosion, in seconds, broadcastable against
            ``parameters``.
        **parameters
            This model's parameter values, in cgs units, broadcastable
            against ``t``.

        Returns
        -------
        numpy.ndarray
            The natural log of :math:`L_\mathrm{bol}(t)`, in erg/s.
        """
        t_grid, *param_arrays = np.broadcast_arrays(np.asarray(t, dtype=np.float64), *parameters.values())
        param_grids = dict(zip(parameters, param_arrays))

        lo, hi = cls._DOMAIN

        def integrand(nu: float) -> FloatArray:
            r"""
            Evaluate :math:`\exp(\mathtt{\_eval}(\nu, t))` at one trial frequency, for `quad_vec`.

            Parameters
            ----------
            nu : float
                Trial frequency, in Hz, supplied by `quad_vec`.

            Returns
            -------
            numpy.ndarray
                :math:`L_\nu(\nu, t)` broadcast against `t_grid`/`param_grids`.
            """
            return np.exp(cls._eval(np.asarray(nu, dtype=np.float64), t_grid, **param_grids))

        integral, err = quad_vec(integrand, float(to_cgs_value(lo)), float(to_cgs_value(hi)))
        _warn_if_not_converged(f"{cls.__name__}._eval_bolometric", integral, err)

        return np.log(integral)

    @classmethod
    def eval_bolometric_log_cgs(cls, t: NumericalInput, **parameters: CGSParameterValue) -> FloatResult:
        """
        Natural log of the bolometric luminosity, taking and returning plain cgs numbers.

        Parameters
        ----------
        t : array-like
            Time since explosion, in seconds.
        **parameters
            This model's parameter values, in cgs units. To evaluate several
            parameter realizations at once, give each parameter a leading
            batch axis.

        Returns
        -------
        numpy.ndarray or float
            The natural log of the bolometric luminosity, in erg/s. A scalar
            is returned if the result is 0-dimensional.
        """
        t_arr = np.asarray(t, dtype=np.float64)
        cgs_parameters: dict[str, CGSParameterValue] = {
            name: np.asarray(value, dtype=np.float64) for name, value in parameters.items()
        }

        result = cls._eval_bolometric(t_arr, **cgs_parameters)
        return result.item() if result.ndim == 0 else result

    @classmethod
    def eval_bolometric_log(cls, t: Quantity, **parameters: ParameterValue) -> FloatResult:
        """
        Natural log of the bolometric luminosity, given physical-unit inputs.

        Parameters
        ----------
        t : ~astropy.units.Quantity
            Time since explosion.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        numpy.ndarray or float
            The natural log of the bolometric luminosity, in erg/s.

        Raises
        ------
        TypeError
            If ``t`` is not a Quantity with time units.
        """
        if not isinstance(t, Quantity) or t.unit.physical_type != "time":
            raise TypeError("`t` must be an astropy Quantity with time units.")

        cgs_parameters: dict[str, CGSParameterValue] = {name: to_cgs_value(value) for name, value in parameters.items()}

        return cls.eval_bolometric_log_cgs(t.cgs.value, **cgs_parameters)

    @classmethod
    def eval_bolometric_cgs(cls, t: NumericalInput, **parameters: CGSParameterValue) -> FloatResult:
        """
        Bolometric luminosity, taking and returning plain cgs numbers.

        Parameters
        ----------
        t : array-like
            Time since explosion, in seconds.
        **parameters
            This model's parameter values, in cgs units. See :meth:`eval_log_cgs`.

        Returns
        -------
        numpy.ndarray or float
            The bolometric luminosity, in erg/s. A scalar is returned if the
            result is 0-dimensional.
        """
        return np.exp(cls.eval_bolometric_log_cgs(t, **parameters))

    @classmethod
    def eval_bolometric(cls, t: Quantity, **parameters: ParameterValue) -> Quantity:
        r"""
        Evaluate the bolometric luminosity at the given time.

        Parameters
        ----------
        t : ~astropy.units.Quantity
            Time since explosion.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`L_\mathrm{bol}(t)`, in erg/s.
        """
        return np.exp(cls.eval_bolometric_log(t, **parameters)) * _BOL_LUM_UNIT

    # -------------------------------------- #
    # Normalized Spectral Shape: S(nu, t)    #
    # -------------------------------------- #
    @classmethod
    def _eval_spectrum(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        Evaluate the natural log of the normalized spectral shape, in cgs units.

        Default implementation: :math:`\log S(\nu, t) = \mathtt{\_eval}(\nu,
        t) - \mathtt{\_eval\_bolometric}(t)`, i.e. :math:`S(\nu, t) =
        L_\nu(\nu, t)/L_\mathrm{bol}(t)`, computed as a log-space subtraction
        rather than a linear-space division so it stays accurate regardless
        of :math:`L_\nu`'s dynamic range. This integrates to exactly 1 over
        :math:`\nu`, for any :math:`t`, by construction.

        A model whose spectral shape is already known independently of its
        bolometric integral (e.g. one built from a separately normalized
        template spectrum) should override this method directly, skipping
        the bolometric-integral subtraction entirely.

        Parameters
        ----------
        nu : numpy.ndarray
            Frequency, in Hz.
        t : numpy.ndarray
            Time since explosion, in seconds.
        **parameters
            This model's parameter values, in cgs units, broadcastable
            against ``nu``/``t``.

        Returns
        -------
        numpy.ndarray
            The natural log of :math:`S(\nu, t)`, in 1/Hz.
        """
        return cls._eval(nu, t, **parameters) - cls._eval_bolometric(t, **parameters)

    @classmethod
    def eval_spectrum_log_cgs(
        cls, nu: NumericalInput, t: NumericalInput, **parameters: CGSParameterValue
    ) -> FloatResult:
        """
        Natural log of the normalized spectral shape, taking and returning plain cgs numbers.

        Parameters
        ----------
        nu : array-like
            Frequency, in Hz.
        t : array-like
            Time since explosion, in seconds.
        **parameters
            This model's parameter values, in cgs units. To evaluate several
            parameter realizations at once, give each parameter a leading
            batch axis.

        Returns
        -------
        numpy.ndarray or float
            The natural log of the normalized spectral shape. A scalar is
            returned if the result is 0-dimensional.
        """
        nu_arr = np.asarray(nu, dtype=np.float64)
        t_arr = np.asarray(t, dtype=np.float64)
        cgs_parameters: dict[str, CGSParameterValue] = {
            name: np.asarray(value, dtype=np.float64) for name, value in parameters.items()
        }

        result = cls._eval_spectrum(nu_arr, t_arr, **cgs_parameters)
        return result.item() if result.ndim == 0 else result

    @classmethod
    def eval_spectrum_log(cls, nu: Quantity, t: Quantity, **parameters: ParameterValue) -> FloatResult:
        """
        Natural log of the normalized spectral shape, given physical-unit inputs.

        Parameters
        ----------
        nu : ~astropy.units.Quantity
            Frequency at which to evaluate the shape.
        t : ~astropy.units.Quantity
            Time since explosion.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        numpy.ndarray or float
            The natural log of the normalized spectral shape, in 1/Hz.

        Raises
        ------
        TypeError
            If ``nu``/``t`` are not Quantities with frequency/time units.
        """
        if not isinstance(nu, Quantity) or nu.unit.physical_type != "frequency":
            raise TypeError("`nu` must be an astropy Quantity with frequency units.")
        if not isinstance(t, Quantity) or t.unit.physical_type != "time":
            raise TypeError("`t` must be an astropy Quantity with time units.")

        cgs_parameters: dict[str, CGSParameterValue] = {name: to_cgs_value(value) for name, value in parameters.items()}

        return cls.eval_spectrum_log_cgs(nu.cgs.value, t.cgs.value, **cgs_parameters)

    @classmethod
    def eval_spectrum_cgs(cls, nu: NumericalInput, t: NumericalInput, **parameters: CGSParameterValue) -> FloatResult:
        """
        Return the normalized spectral shape as plain cgs numbers; see :meth:`eval_log_cgs`.

        Parameters
        ----------
        nu : array-like
            Frequency, in Hz.
        t : array-like
            Time since explosion, in seconds.
        **parameters
            This model's parameter values, in cgs units. See :meth:`eval_log_cgs`.

        Returns
        -------
        numpy.ndarray or float
            The normalized spectral shape, in 1/Hz. A scalar is returned if
            the result is 0-dimensional.
        """
        return np.exp(cls.eval_spectrum_log_cgs(nu, t, **parameters))

    @classmethod
    def eval_spectrum(cls, nu: Quantity, t: Quantity, **parameters: ParameterValue) -> Quantity:
        r"""
        Evaluate the normalized spectral shape at the given frequency and time.

        Parameters
        ----------
        nu : ~astropy.units.Quantity
            Frequency at which to evaluate the shape.
        t : ~astropy.units.Quantity
            Time since explosion.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`S(\nu, t)`, in 1/Hz. Integrates to 1 over :math:`\nu`
            for any fixed :math:`t`.
        """
        return np.exp(cls.eval_spectrum_log(nu, t, **parameters)) * _SED_SHAPE_UNIT

    # -------------------------------------- #
    # Synthetic Spectrum Generation          #
    # -------------------------------------- #
    # At the m4opt.synphot level, operations are performed on Synphot / Astropy Model objects,
    # which are not immediately compatible with the machinery of the SpectralModel class. These
    # methods allow a user to provide a set of parameters and generate spectra objects.
    @classmethod
    def as_astropy_model(
        cls,
        x_type: str = "lambda",
        y_type: str = "lambda",
        *,
        y_kind: str = "energy",
        wave_unit: str | u.UnitBase = u.AA,
        freq_unit: str | u.UnitBase = u.Hz,
        redshift: PhysicalInput | None = None,
        luminosity_distance: Quantity | None = None,
        angular_diameter_distance: Quantity | None = None,
        proper_distance: Quantity | None = None,
        cosmology: FLRW | None = None,
        **parameters: ParameterValue,
    ) -> Model:
        r"""
        Build an :class:`~astropy.modeling.Model` of this :class:`SpectralModel` for a given parameter set.

        This method is the single entry point for converting a :class:`SpectralModel` into
        a model compatible with :mod:`synphot` / :mod:`astropy.modeling` / :mod:`m4opt.synphot`.
        Several parameters can be modified to specify exactly what type of spectral model to
        generate:

        - ``x_type``: May be either ``"lambda"`` or ``"nu"`` to control what input the
          model expects.
        - ``y_type``: May be either ``"lambda"`` or ``"nu"``. If ``"lambda"``, then :math:`F_\lambda` is
          generated, otherwise :math:`F_\nu` is generated.
        - ``y_kind``: May be either ``"energy"`` or ``"photon"``. If ``"energy"``, then the output is
          an energy-flux density (:math:`F_\nu`/:math:`F_\lambda`). If ``"photon"``, then the output is
          a photon count flux density.

        Additionally, any of a number of cosmological parameters may be specified to provide the distance
        from the source. If any of these are provided, the model will generate a flux density. Otherwise
        a luminosity density is produced.

        Parameters
        ----------
        x_type : str
            ``"lambda"`` (the default) if the model's first input is a
            wavelength, or ``"nu"`` if it is a frequency.
        y_type : str
            ``"lambda"`` (the default) if the output is expressed per unit
            wavelength, or ``"nu"`` if per unit frequency.
        y_kind : str
            ``"energy"`` (the default) for an energy-flux density, or
            ``"photon"`` for a photon-count-flux density (dividing by the
            photon energy :math:`h\nu`).
        wave_unit : str or ~astropy.units.UnitBase
            The wavelength unit used wherever ``x_type``/``y_type`` is
            ``"lambda"``.
        freq_unit : str or ~astropy.units.UnitBase
            The frequency unit used wherever ``x_type``/``y_type`` is
            ``"nu"``.
        redshift : array-like or ~astropy.units.Quantity, optional
            If any of ``redshift``/``luminosity_distance``/
            ``angular_diameter_distance``/``proper_distance`` is given,
            exactly one of them must be given; the rest (and
            ``cosmology``) are as in :meth:`flux_log`, and the output is
            the observed, diluted flux. If none of the four is given, the
            output is the rest-frame luminosity and ``cosmology`` is
            ignored.
        luminosity_distance : ~astropy.units.Quantity, optional
            See ``redshift`` above.
        angular_diameter_distance : ~astropy.units.Quantity, optional
            See ``redshift`` above.
        proper_distance : ~astropy.units.Quantity, optional
            See ``redshift`` above.
        cosmology : ~astropy.cosmology.FLRW, optional
            See ``redshift`` above.
        **parameters
            This model's parameter values, either
            :class:`~astropy.units.Quantity` or already unit-stripped cgs
            values (see :meth:`eval_log_cgs`). May carry leading batch
            axes.

        Returns
        -------
        ~astropy.modeling.Model
            Callable as ``model(x, t)``, with ``x``/``t`` either bare
            numbers (``x`` in ``wave_unit``/``freq_unit``, ``t`` in
            seconds) or :class:`~astropy.units.Quantity`. Unit-attached
            only when at least one of ``x``/``t`` was itself a
            ``Quantity`` (a property of
            :meth:`astropy.modeling.Model.__call__`, not something this
            method controls).

        Raises
        ------
        ValueError
            If ``x_type``/``y_type`` is not ``"lambda"``/``"nu"``, if
            ``y_kind`` is not ``"energy"``/``"photon"``, or if
            ``wave_unit``/``freq_unit`` is not a wavelength/frequency unit.
        """
        # Resolve and validate the x_type, y_type and y_kind parameters.
        if x_type not in ("nu", "lambda"):
            raise ValueError(f"x_type must be 'nu' or 'lambda', got {x_type!r}.")
        if y_type not in ("nu", "lambda"):
            raise ValueError(f"y_type must be 'nu' or 'lambda', got {y_type!r}.")
        if y_kind not in ("energy", "photon"):
            raise ValueError(f"y_kind must be 'energy' or 'photon', got {y_kind!r}.")

        x_is_wavelength = x_type == "lambda"
        y_is_wavelength = y_type == "lambda"

        x_unit = u.Unit(wave_unit if x_is_wavelength else freq_unit)
        y_unit = u.Unit(wave_unit if y_is_wavelength else freq_unit)

        x_coefficient = hz_per_unit(x_unit, is_wavelength=x_is_wavelength)
        y_coefficient = hz_per_unit(y_unit, is_wavelength=y_is_wavelength)

        # Determine if we are generating a flux density or luminosity. This is determined
        # by the specification / lack of the redshift or other cosmological parameters.
        _is_luminosity = (
            redshift is None
            and luminosity_distance is None
            and angular_diameter_distance is None
            and proper_distance is None
        )

        # Convert ALL of the provided parameters to their CGS value so that we
        # do not need to do on-the-fly unit conversions in performance critical
        # call sequences.
        cgs_parameters: dict[str, CGSParameterValue] = {name: to_cgs_value(value) for name, value in parameters.items()}

        # Determine the "native" evaluation function. This is the F_nu(nu) in units of
        # erg / cm^2 / Hz / s if we are computing a flux, or a luminosity L_nu(nu) in
        # erg / Hz / s if we are computing a luminosity.
        #
        # Once this has been generated, we can modify it to generate the correct units
        # and other properties.
        if not _is_luminosity:
            # Resolve the cosmological distances and extract them.
            distances = resolve_cosmological_distances(
                redshift=redshift,
                luminosity_distance=luminosity_distance,
                angular_diameter_distance=angular_diameter_distance,
                proper_distance=proper_distance,
                cosmology=cosmology,
            )
            redshift_cgs = np.asarray(distances["redshift"], dtype=np.float64)
            luminosity_distance_cgs = distances["luminosity_distance"].cgs.value
            _denominator_unit = u.s**-1 * u.cm**-2

            # Construct the native evaluation function.
            def _eval_native(nu_hz: FloatArray, t: FloatArray) -> FloatResult:
                return cls.flux_cgs(nu_hz, t, redshift_cgs, luminosity_distance_cgs, **cgs_parameters)
        else:
            # We are not producing a flux. We can just generate the function.
            _denominator_unit = u.s**-1

            def _eval_native(nu_hz: FloatArray, t: FloatArray) -> FloatResult:
                return cls.eval_cgs(nu_hz, t, **cgs_parameters)

        # Generate the modifications to the function to coerce x to nu.
        if x_is_wavelength:

            def _x_to_nu(x: FloatArray) -> FloatArray:
                return x_coefficient / x
        else:

            def _x_to_nu(x: FloatArray) -> FloatArray:
                return x_coefficient * x

        # Generate the modification to the function to coerce F_nu to y.
        if y_is_wavelength:

            def _to_dlambda(nu_hz: FloatArray, y: FloatResult) -> FloatArray:
                return np.asarray(y * (nu_hz**2 / y_coefficient))
        else:

            def _to_dlambda(nu_hz: FloatArray, y: FloatResult) -> FloatArray:
                return np.asarray(y * y_coefficient)

        if y_kind == "energy":
            _numerator_unit = u.erg

            def _to_output_flux(nu_hz: FloatArray, y: FloatResult) -> FloatArray:
                return np.asarray(y)
        else:
            _numerator_unit = u.photon

            def _to_output_flux(nu_hz: FloatArray, y: FloatResult) -> FloatArray:
                return y / (H_CGS * nu_hz)

        # Generate the final evaluator.
        def _evaluate(x: FloatArray, t: FloatArray) -> FloatResult:
            """
            Evaluate the composed model at unit-stripped ``(x, t)``, in the resolved output units.

            Chains `_x_to_nu`, `_eval_native`, `_to_dlambda`, and
            `_to_output_flux` (each resolved above, based on `x_type`/
            `y_type`/`y_kind`) into the single callable passed to
            :class:`~astropy.modeling.custom_model`.

            Parameters
            ----------
            x : numpy.ndarray
                Wavelength or frequency, in `wave_unit`/`freq_unit`, per `x_type`.
            t : numpy.ndarray
                Time since explosion, in seconds.

            Returns
            -------
            float or numpy.ndarray
                The evaluated flux/luminosity density, unit-stripped, in `output_unit`.
            """
            nu_hz = _x_to_nu(x)
            y = _eval_native(nu_hz, t)
            y = _to_dlambda(nu_hz, y)
            y = _to_output_flux(nu_hz, y)
            return y

        output_unit = _numerator_unit / y_unit * _denominator_unit

        model_class = model_class_from_kernel(
            "_SpectralAstropyModel",
            inputs={("wave" if x_is_wavelength else "nu"): x_unit, "t": u.s},
            outputs={"y": output_unit},
            evaluate=_evaluate,
        )
        return model_class()

    @classmethod
    def as_source_spectrum(
        cls,
        t: PhysicalInput,
        *,
        redshift: PhysicalInput | None = None,
        luminosity_distance: Quantity | None = None,
        angular_diameter_distance: Quantity | None = None,
        proper_distance: Quantity | None = None,
        cosmology: FLRW | None = None,
        ebv: float | Quantity | FloatArray | None = None,
        dust_law: str | None = None,
        **parameters: ParameterValue,
    ) -> SourceSpectrum:
        r"""
        Build a :class:`~synphot.SourceSpectrum` giving the observed flux at one fixed time :math:`t`.

        A thin wrapper around :meth:`as_astropy_model`: fixes ``t`` (so the
        result is a function of wavelength alone, the shape :mod:`synphot`
        requires), and fixes ``x_type``/``y_type``/``y_kind`` to
        wavelength-in/photon-out, in Angstrom -- because
        :class:`~synphot.SourceSpectrum` always samples its wrapped model
        in *wavelength* space, and always treats the model's raw return
        value as already being expressed in :mod:`synphot`'s internal
        :data:`~synphot.units.PHOTLAM` (*photon*-count flux density per
        unit wavelength, not the energy-flux :math:`F_\lambda` one might
        expect) -- :meth:`~synphot.SourceSpectrum.__call__` does not
        consult a wrapped model's declared output units to convert.

        Requires an observed (diluted) flux -- at least one of
        ``redshift``/the distance keywords -- since a
        :class:`~synphot.SourceSpectrum` is meant to be a real per-area
        flux for :class:`~m4opt.synphot.Detector` to consume, not a
        rest-frame luminosity.

        Foreground (e.g. Milky Way) dust attenuation is optional: pass an
        already-resolved ``ebv`` (see :func:`~uvex_transients.dust.resolve_ebv`
        for turning a dust map and sky position into one, as a separate prior
        step) and it's folded into the native photon flux for you, inside this
        method's own evaluation kernel, before the result is wrapped as a
        :class:`~synphot.SourceSpectrum` -- applied here rather than through
        :mod:`synphot`'s own spectrum-composition operators
        (``spectrum * extinction``), and resolved to
        :func:`~uvex_transients.dust.log_attenuation`'s callable form (via
        :func:`~uvex_transients.dust.attenuation_callable`) internally, right
        here, rather than something a caller ever has to build. Doing it here
        rather than by composing two :class:`~astropy.modeling.Model`
        instances is what lets ``ebv`` stay vector-valued (e.g. one row per
        event) without resurrecting :mod:`synphot`'s ``n_models=1``
        restriction, exactly like every other parameter here (see
        :func:`~uvex_transients.models._utils.model_class_from_kernel`) -- the
        attenuation is just one more array multiplied in via plain NumPy
        broadcasting, not a second composed model.

        Parameters
        ----------
        t : array-like or Quantity
            Observed time since explosion, either a
            :class:`~astropy.units.Quantity` with time units or an already
            unit-stripped cgs (seconds) value. May carry leading batch
            axes, broadcastable against ``parameters``.
        redshift : array-like or ~astropy.units.Quantity, optional
            Exactly one of ``redshift`` or the three distance keywords must
            be given; the rest are derived from it using ``cosmology``. See
            :meth:`as_astropy_model`.
        luminosity_distance : ~astropy.units.Quantity, optional
            See ``redshift`` above.
        angular_diameter_distance : ~astropy.units.Quantity, optional
            See ``redshift`` above.
        proper_distance : ~astropy.units.Quantity, optional
            See ``redshift`` above.
        cosmology : ~astropy.cosmology.FLRW, optional
            See ``redshift`` above.
        ebv : float, ~astropy.units.Quantity, or array-like, optional
            Already-resolved, dimensionless E(B-V) (see
            :func:`~uvex_transients.dust.resolve_ebv`). ``None`` (the default)
            applies no dust attenuation.
        dust_law : str, optional
            Passed through to :func:`~uvex_transients.dust.get_dust_law`;
            the configured default (``config["physics.default_dust_law"]``)
            is almost always the right choice.
        **parameters
            This model's parameter values, either
            :class:`~astropy.units.Quantity` or already unit-stripped cgs
            values (see :meth:`eval_log_cgs`). May carry leading batch
            axes.

        Returns
        -------
        ~synphot.SourceSpectrum
            Callable as ``spectrum(wave)``, returning the observed flux
            density at ``t``.

        Raises
        ------
        ValueError
            If none of ``redshift``/``luminosity_distance``/
            ``angular_diameter_distance``/``proper_distance`` is given.
        """
        if (
            redshift is None
            and luminosity_distance is None
            and angular_diameter_distance is None
            and proper_distance is None
        ):
            raise ValueError(
                "as_source_spectrum requires an observed flux -- pass "
                "`redshift=` or one of the distance keywords accepted by "
                "as_astropy_model. A SourceSpectrum must be a real per-area "
                "flux, not a rest-frame luminosity."
            )

        model = cls.as_astropy_model(
            x_type="lambda",
            y_type="lambda",
            y_kind="photon",
            wave_unit=u.AA,
            redshift=redshift,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            cosmology=cosmology,
            **parameters,
        )
        t_cgs = to_cgs_value(t)
        log_attenuation = None if ebv is None else attenuation_callable(ebv, dust_law)

        if log_attenuation is None:

            def evaluate(wave: FloatArray) -> FloatResult:
                return model(wave, t_cgs)
        else:

            def evaluate(wave: FloatArray) -> FloatResult:
                return model(wave, t_cgs) * np.exp(log_attenuation(wave * u.AA))

        model_class = model_class_from_kernel(
            "_FixedTimeSourceSpectrumModel",
            inputs={"wave": u.AA},
            outputs={"y": synphot_units.PHOTLAM},
            evaluate=evaluate,
        )
        return SourceSpectrum(model_class())

    # -------------------------------------- #
    # Observed Flux Density: F_nu(nu, t)      #
    # -------------------------------------- #
    @classmethod
    def _eval_flux(
        cls,
        nu: FloatArray,
        t: FloatArray,
        redshift: FloatArray,
        luminosity_distance: FloatArray,
        *,
        log_attenuation: FloatArray | None = None,
        **parameters: CGSParameterValue,
    ) -> FloatArray:
        r"""
        Evaluate the natural log of the observed flux density, in cgs units.

        Implements the K-correction-free relation (Hogg 1999):

        .. math::

            F_\nu(\nu, t) = (1+z) \cdot L_\nu\big(\nu(1+z),\ t/(1+z)\big) / (4\pi D_L^2)
                \cdot \exp(\ell(\nu))

        :math:`\nu(1+z)` and :math:`t/(1+z)` convert the observed
        frequency/time to the rest-frame values :meth:`_eval` expects; the
        :math:`(1+z)` prefactor accounts for observed-bandwidth compression.
        :math:`\ell(\nu)` is `log_attenuation`, an observed-frame effect
        (e.g. Milky Way foreground dust) applied here rather than inside
        :meth:`_eval`, so it does not also leak into
        :meth:`_eval_bolometric`/:meth:`as_astropy_model`'s (undiluted) normalization.

        Parameters
        ----------
        nu, t : numpy.ndarray
            Observed frequency (Hz) and time since explosion (s).
        redshift : numpy.ndarray
            Cosmological redshift, dimensionless.
        luminosity_distance : numpy.ndarray
            Luminosity distance, in cm.
        log_attenuation : numpy.ndarray, optional
            Natural log of an observed-frame multiplicative attenuation
            (e.g. the log of Milky Way dust transmission), added directly to
            the log flux. ``None`` (the default) applies no attenuation.
        **parameters
            This model's parameter values, in cgs units, broadcastable
            against ``nu``/``t``/``redshift``/``luminosity_distance``.

        Returns
        -------
        numpy.ndarray
            The natural log of :math:`F_\nu`, in erg/s/cm^2/Hz.
        """
        log_flux = (
            np.log1p(redshift)
            + cls._eval(nu * (1.0 + redshift), t / (1.0 + redshift), **parameters)
            - np.log(4.0 * np.pi)
            - 2.0 * np.log(luminosity_distance)
        )

        if log_attenuation is not None:
            log_flux = log_flux + log_attenuation

        return log_flux

    @classmethod
    def flux_log_cgs(
        cls,
        nu: NumericalInput,
        t: NumericalInput,
        redshift: NumericalInput,
        luminosity_distance: NumericalInput,
        *,
        log_attenuation: NumericalInput | None = None,
        **parameters: CGSParameterValue,
    ) -> FloatResult:
        r"""
        Natural log of the observed flux density, taking and returning plain cgs numbers.

        Unlike :meth:`flux_log`, ``redshift``/``luminosity_distance`` are not
        resolved from a cosmology here -- pass already-computed values
        directly (e.g. a precomputed per-event grid).

        Parameters
        ----------
        nu, t : array-like
            Observed frequency (Hz) and time since explosion (s).
        redshift : array-like
            Cosmological redshift, dimensionless.
        luminosity_distance : array-like
            Luminosity distance, in cm.
        log_attenuation : array-like, optional
            Natural log of an observed-frame multiplicative attenuation,
            added directly to the log flux. Must already broadcast against
            the natural output shape of this call.
        **parameters
            This model's parameter values, in cgs units.

        Returns
        -------
        numpy.ndarray or float
            The natural log of :math:`F_\nu`, in erg/s/cm^2/Hz. A scalar is
            returned if the result is 0-dimensional.
        """
        nu_arr = np.asarray(nu, dtype=np.float64)
        t_arr = np.asarray(t, dtype=np.float64)
        redshift_arr = np.asarray(redshift, dtype=np.float64)
        luminosity_distance_arr = np.asarray(luminosity_distance, dtype=np.float64)
        cgs_parameters: dict[str, CGSParameterValue] = {
            name: np.asarray(value, dtype=np.float64) for name, value in parameters.items()
        }

        result = cls._eval_flux(
            nu_arr,
            t_arr,
            redshift_arr,
            luminosity_distance_arr,
            log_attenuation=None if log_attenuation is None else np.asarray(log_attenuation, dtype=np.float64),
            **cgs_parameters,
        )
        return result.item() if result.ndim == 0 else result

    @classmethod
    def flux_log(
        cls,
        nu: Quantity,
        t: Quantity,
        *,
        redshift: PhysicalInput | None = None,
        luminosity_distance: Quantity | None = None,
        angular_diameter_distance: Quantity | None = None,
        proper_distance: Quantity | None = None,
        cosmology: FLRW | None = None,
        log_attenuation: NumericalInput | None = None,
        **parameters: ParameterValue,
    ) -> FloatResult:
        r"""
        Natural log of the observed flux density, given physical-unit inputs.

        Parameters
        ----------
        nu : ~astropy.units.Quantity
            Observed frequency.
        t : ~astropy.units.Quantity
            Observed time since explosion.
        redshift : array-like or ~astropy.units.Quantity, optional
            Exactly one of ``redshift`` or the three distance keywords must
            be given; the rest are derived from it using ``cosmology`` (see
            :func:`~uvex_transients.utils.cosmology.resolve_cosmological_distances`).
            ``cosmology`` defaults to that function's configured default.
        luminosity_distance : ~astropy.units.Quantity, optional
            See ``redshift`` above.
        angular_diameter_distance : ~astropy.units.Quantity, optional
            See ``redshift`` above.
        proper_distance : ~astropy.units.Quantity, optional
            See ``redshift`` above.
        cosmology : ~astropy.cosmology.FLRW, optional
            See ``redshift`` above.
        log_attenuation : array-like, optional
            See :meth:`flux_log_cgs`.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        numpy.ndarray or float
            The natural log of :math:`F_\nu`, in erg/s/cm^2/Hz.
        """
        if not isinstance(nu, Quantity) or nu.unit.physical_type != "frequency":
            raise TypeError("`nu` must be an astropy Quantity with frequency units.")
        if not isinstance(t, Quantity) or t.unit.physical_type != "time":
            raise TypeError("`t` must be an astropy Quantity with time units.")

        distances = resolve_cosmological_distances(
            redshift=redshift,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            cosmology=cosmology,
        )
        cgs_parameters: dict[str, CGSParameterValue] = {name: to_cgs_value(value) for name, value in parameters.items()}

        return cls.flux_log_cgs(
            nu.cgs.value,
            t.cgs.value,
            np.asarray(distances["redshift"], dtype=np.float64),
            distances["luminosity_distance"].cgs.value,
            log_attenuation=log_attenuation,
            **cgs_parameters,
        )

    @classmethod
    def flux_cgs(
        cls,
        nu: NumericalInput,
        t: NumericalInput,
        redshift: NumericalInput,
        luminosity_distance: NumericalInput,
        *,
        log_attenuation: NumericalInput | None = None,
        **parameters: CGSParameterValue,
    ) -> FloatResult:
        r"""
        Observed flux density, taking and returning plain cgs numbers.

        Parameters
        ----------
        nu : array-like
            Observed frequency, in Hz.
        t : array-like
            Observed time since explosion, in seconds.
        redshift : array-like
            Cosmological redshift, dimensionless.
        luminosity_distance : array-like
            Luminosity distance, in cm.
        log_attenuation : array-like, optional
            See :meth:`flux_log_cgs`.
        **parameters
            This model's parameter values, in cgs units. See :meth:`flux_log_cgs`.

        Returns
        -------
        numpy.ndarray or float
            :math:`F_\nu(\nu, t)`, in erg/s/cm^2/Hz. A scalar is returned if
            the result is 0-dimensional.
        """
        return np.exp(
            cls.flux_log_cgs(
                nu,
                t,
                redshift,
                luminosity_distance,
                log_attenuation=log_attenuation,
                **parameters,
            )
        )

    @classmethod
    def flux(
        cls,
        nu: Quantity,
        t: Quantity,
        *,
        redshift: PhysicalInput | None = None,
        luminosity_distance: Quantity | None = None,
        angular_diameter_distance: Quantity | None = None,
        proper_distance: Quantity | None = None,
        cosmology: FLRW | None = None,
        log_attenuation: NumericalInput | None = None,
        **parameters: ParameterValue,
    ) -> Quantity:
        r"""
        Evaluate the observed flux density at the given frequency and time.

        Parameters
        ----------
        nu : ~astropy.units.Quantity
            Observed frequency.
        t : ~astropy.units.Quantity
            Observed time since explosion.
        redshift : array-like or ~astropy.units.Quantity, optional
            See :meth:`flux_log`.
        luminosity_distance : ~astropy.units.Quantity, optional
            See :meth:`flux_log`.
        angular_diameter_distance : ~astropy.units.Quantity, optional
            See :meth:`flux_log`.
        proper_distance : ~astropy.units.Quantity, optional
            See :meth:`flux_log`.
        cosmology : ~astropy.cosmology.FLRW, optional
            See :meth:`flux_log`.
        log_attenuation : array-like, optional
            See :meth:`flux_log_cgs`.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`F_\nu(\nu, t)`, in erg/s/cm^2/Hz.
        """
        return (
            np.exp(
                cls.flux_log(
                    nu,
                    t,
                    redshift=redshift,
                    luminosity_distance=luminosity_distance,
                    angular_diameter_distance=angular_diameter_distance,
                    proper_distance=proper_distance,
                    cosmology=cosmology,
                    log_attenuation=log_attenuation,
                    **parameters,
                )
            )
            * _SPEC_FLUX_UNIT
        )

    # -------------------------------------- #
    # Observed Bolometric Flux: F_bol(t)      #
    # -------------------------------------- #
    @classmethod
    def _eval_flux_bolometric(
        cls,
        t: FloatArray,
        redshift: FloatArray,
        luminosity_distance: FloatArray,
        **parameters: CGSParameterValue,
    ) -> FloatArray:
        r"""
        Evaluate the natural log of the observed bolometric flux, in cgs units.

        .. math::

            F_\mathrm{bol}(t) = L_\mathrm{bol}(t/(1+z)) / (4\pi D_L^2)

        No :math:`(1+z)` prefactor appears here: integrating
        :meth:`_eval_flux`'s :math:`F_\nu(\nu_\mathrm{obs})` over all
        observed frequency and substituting :math:`\nu_\mathrm{emit} =
        \nu_\mathrm{obs}(1+z)` makes that factor cancel exactly against the
        Jacobian of the substitution.

        Parameters
        ----------
        t : numpy.ndarray
            Observed time since explosion, in seconds.
        redshift : numpy.ndarray
            Cosmological redshift, dimensionless, broadcastable against ``t``.
        luminosity_distance : numpy.ndarray
            Luminosity distance, in cm, broadcastable against ``t``.
        **parameters
            This model's parameter values, in cgs units, broadcastable
            against ``t``.

        Returns
        -------
        numpy.ndarray
            The natural log of the observed bolometric flux, in erg/s/cm^2.
        """
        return (
            cls._eval_bolometric(t / (1.0 + redshift), **parameters)
            - np.log(4.0 * np.pi)
            - 2.0 * np.log(luminosity_distance)
        )

    @classmethod
    def flux_bolometric_log_cgs(
        cls,
        t: NumericalInput,
        redshift: NumericalInput,
        luminosity_distance: NumericalInput,
        **parameters: CGSParameterValue,
    ) -> FloatResult:
        """
        Natural log of the observed bolometric flux, taking and returning plain cgs numbers.

        Parameters
        ----------
        t : array-like
            Time since explosion, in seconds.
        redshift : array-like
            Cosmological redshift, dimensionless.
        luminosity_distance : array-like
            Luminosity distance, in cm.
        **parameters
            This model's parameter values, in cgs units; must already
            broadcast against ``t``.

        Returns
        -------
        numpy.ndarray or float
            The natural log of the observed bolometric flux, in erg/s/cm^2.
            A scalar is returned if the result is 0-dimensional.
        """
        t_arr = np.asarray(t, dtype=np.float64)
        redshift_arr = np.asarray(redshift, dtype=np.float64)
        luminosity_distance_arr = np.asarray(luminosity_distance, dtype=np.float64)
        cgs_parameters: dict[str, CGSParameterValue] = {
            name: np.asarray(value, dtype=np.float64) for name, value in parameters.items()
        }

        result = cls._eval_flux_bolometric(t_arr, redshift_arr, luminosity_distance_arr, **cgs_parameters)

        return result.item() if result.ndim == 0 else result

    @classmethod
    def flux_bolometric_log(
        cls,
        t: Quantity,
        *,
        redshift: PhysicalInput | None = None,
        luminosity_distance: Quantity | None = None,
        angular_diameter_distance: Quantity | None = None,
        proper_distance: Quantity | None = None,
        cosmology: FLRW | None = None,
        **parameters: ParameterValue,
    ) -> FloatResult:
        """
        Natural log of the observed bolometric flux, given physical-unit inputs.

        Parameters
        ----------
        t : ~astropy.units.Quantity
            Observed time since explosion.
        redshift : array-like or ~astropy.units.Quantity, optional
            See :meth:`flux_log`.
        luminosity_distance : ~astropy.units.Quantity, optional
            See :meth:`flux_log`.
        angular_diameter_distance : ~astropy.units.Quantity, optional
            See :meth:`flux_log`.
        proper_distance : ~astropy.units.Quantity, optional
            See :meth:`flux_log`.
        cosmology : ~astropy.cosmology.FLRW, optional
            See :meth:`flux_log`.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        numpy.ndarray or float
            The natural log of the observed bolometric flux, in erg/s/cm^2.

        Raises
        ------
        TypeError
            If ``t`` is not a Quantity with time units.
        """
        if not isinstance(t, Quantity) or t.unit.physical_type != "time":
            raise TypeError("`t` must be an astropy Quantity with time units.")

        distances = resolve_cosmological_distances(
            redshift=redshift,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            cosmology=cosmology,
        )

        cgs_parameters: dict[str, CGSParameterValue] = {name: to_cgs_value(value) for name, value in parameters.items()}

        return cls.flux_bolometric_log_cgs(
            t.cgs.value,
            np.asarray(distances["redshift"], dtype=np.float64),
            distances["luminosity_distance"].cgs.value,
            **cgs_parameters,
        )

    @classmethod
    def flux_bolometric_cgs(
        cls,
        t: NumericalInput,
        redshift: NumericalInput,
        luminosity_distance: NumericalInput,
        **parameters: CGSParameterValue,
    ) -> FloatResult:
        """
        Observed bolometric flux, taking and returning plain cgs numbers.

        Parameters
        ----------
        t : array-like
            Time since explosion, in seconds.
        redshift : array-like
            Cosmological redshift, dimensionless.
        luminosity_distance : array-like
            Luminosity distance, in cm.
        **parameters
            This model's parameter values, in cgs units. See :meth:`flux_log_cgs`.

        Returns
        -------
        numpy.ndarray or float
            The observed bolometric flux, in erg/s/cm^2. A scalar is
            returned if the result is 0-dimensional.
        """
        return np.exp(cls.flux_bolometric_log_cgs(t, redshift, luminosity_distance, **parameters))

    @classmethod
    def flux_bolometric(
        cls,
        t: Quantity,
        *,
        redshift: PhysicalInput | None = None,
        luminosity_distance: Quantity | None = None,
        angular_diameter_distance: Quantity | None = None,
        proper_distance: Quantity | None = None,
        cosmology: FLRW | None = None,
        **parameters: ParameterValue,
    ) -> Quantity:
        r"""
        Evaluate the observed bolometric flux at the given time.

        Parameters
        ----------
        t : ~astropy.units.Quantity
            Observed time since explosion.
        redshift : array-like or ~astropy.units.Quantity, optional
            See :meth:`flux_log`.
        luminosity_distance : ~astropy.units.Quantity, optional
            See :meth:`flux_log`.
        angular_diameter_distance : ~astropy.units.Quantity, optional
            See :meth:`flux_log`.
        proper_distance : ~astropy.units.Quantity, optional
            See :meth:`flux_log`.
        cosmology : ~astropy.cosmology.FLRW, optional
            See :meth:`flux_log`.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`F_\mathrm{bol}(t)`, in erg/s/cm^2.
        """
        return (
            np.exp(
                cls.flux_bolometric_log(
                    t,
                    redshift=redshift,
                    luminosity_distance=luminosity_distance,
                    angular_diameter_distance=angular_diameter_distance,
                    proper_distance=proper_distance,
                    cosmology=cosmology,
                    **parameters,
                )
            )
            * _BOL_FLUX_UNIT
        )

    # -------------------------------------- #
    # Observed Band-Averaged Flux: F_band(t)  #
    # -------------------------------------- #
    @classmethod
    def flux_band_log_cgs(
        cls,
        nu: NumericalInput,
        throughput: NumericalInput,
        t: NumericalInput,
        redshift: NumericalInput,
        luminosity_distance: NumericalInput,
        *,
        log_attenuation: NumericalInput | None = None,
        **parameters: CGSParameterValue,
    ) -> FloatResult:
        r"""
        Natural log of the throughput-weighted mean flux density over a band, plain cgs numbers.

        .. math::

            \log \bar{F}_\nu = \log \frac{\int F_\nu(\nu, t) \cdot T(\nu)\,d\nu}{\int T(\nu)\,d\nu}

        using :meth:`flux_cgs`'s (redshifted, distance-diluted) :math:`F_\nu`,
        integrated over the observed-frame frequency grid ``nu`` and bandpass
        response ``throughput`` by trapezoidal quadrature. Dividing by the
        integrated throughput keeps the result dimensionally a flux density,
        directly comparable to :meth:`flux_cgs`.

        Because this method integrates away a frequency axis, ``t``,
        ``redshift``, ``luminosity_distance``, and every parameter are each
        given one trailing axis internally so they broadcast against the
        ``nu`` grid; any shape you would otherwise pass unchanged to
        :meth:`flux_log_cgs` still works here.

        Parameters
        ----------
        nu : array-like
            Observed frequency grid to integrate over, in Hz, shape ``(K,)``.
            Need not be sorted.
        throughput : array-like
            Dimensionless bandpass response at each ``nu`` sample, shape
            ``(K,)``.
        t : array-like
            Observed time since explosion, in seconds, any shape.
        redshift : array-like
            Cosmological redshift, dimensionless, any shape.
        luminosity_distance : array-like
            Luminosity distance, in cm, any shape.
        log_attenuation : array-like, optional
            Natural log of an observed-frame multiplicative attenuation,
            sampled at the same ``nu`` grid (its last axis must have length
            ``K``, in ``nu``'s original, pre-sort order). Any leading axes
            broadcast against ``t``/``redshift``/``luminosity_distance``/the
            parameters.
        **parameters
            This model's parameter values, in cgs units, any shape.

        Returns
        -------
        numpy.ndarray
            The natural log of :math:`\bar{F}_\nu`, in erg/s/cm^2/Hz, with
            the broadcast shape of ``t``/``redshift``/``luminosity_distance``/
            the parameters (the frequency axis is integrated away).
        """
        nu = np.asarray(nu, dtype=np.float64)
        throughput = np.asarray(throughput, dtype=np.float64)
        order = np.argsort(nu)
        nu_sorted = nu[order]
        throughput_sorted = throughput[order]

        band_ready_parameters: dict[str, CGSParameterValue] = {
            name: np.asarray(value, dtype=np.float64)[..., np.newaxis] for name, value in parameters.items()
        }

        log_flux_density = cls._eval_flux(
            nu_sorted,
            np.asarray(t, dtype=np.float64)[..., np.newaxis],
            np.asarray(redshift, dtype=np.float64)[..., np.newaxis],
            np.asarray(luminosity_distance, dtype=np.float64)[..., np.newaxis],
            log_attenuation=(
                None if log_attenuation is None else np.asarray(log_attenuation, dtype=np.float64)[..., order]
            ),
            **band_ready_parameters,
        )

        numerator = np.trapezoid(np.exp(log_flux_density) * throughput_sorted, nu_sorted, axis=-1)
        denominator = np.trapezoid(throughput_sorted, nu_sorted)

        return np.log(numerator / denominator)

    @classmethod
    def flux_band_log(
        cls,
        nu: Quantity,
        throughput: NumericalInput,
        t: Quantity,
        *,
        redshift: PhysicalInput | None = None,
        luminosity_distance: Quantity | None = None,
        angular_diameter_distance: Quantity | None = None,
        proper_distance: Quantity | None = None,
        cosmology: FLRW | None = None,
        log_attenuation: NumericalInput | None = None,
        **parameters: ParameterValue,
    ) -> FloatResult:
        r"""
        Natural log of the band-averaged observed flux density, given physical-unit inputs.

        Parameters
        ----------
        nu : ~astropy.units.Quantity
            Observed frequency grid to integrate over. Need not be sorted.
        throughput : array-like
            Dimensionless bandpass response at each ``nu`` sample.
        t : ~astropy.units.Quantity
            Observed time since explosion.
        redshift : array-like or ~astropy.units.Quantity, optional
            See :meth:`flux_log`.
        luminosity_distance : ~astropy.units.Quantity, optional
            See :meth:`flux_log`.
        angular_diameter_distance : ~astropy.units.Quantity, optional
            See :meth:`flux_log`.
        proper_distance : ~astropy.units.Quantity, optional
            See :meth:`flux_log`.
        cosmology : ~astropy.cosmology.FLRW, optional
            See :meth:`flux_log`.
        log_attenuation : array-like, optional
            See :meth:`flux_band_log_cgs`.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        numpy.ndarray or float
            The natural log of :math:`\\bar{F}_\\nu`, in erg/s/cm^2/Hz.

        Raises
        ------
        TypeError
            If ``nu``/``t`` are not Quantities with frequency/time units.
        """
        if not isinstance(nu, Quantity) or nu.unit.physical_type != "frequency":
            raise TypeError("`nu` must be an astropy Quantity with frequency units.")
        if not isinstance(t, Quantity) or t.unit.physical_type != "time":
            raise TypeError("`t` must be an astropy Quantity with time units.")

        distances = resolve_cosmological_distances(
            redshift=redshift,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            cosmology=cosmology,
        )
        cgs_parameters: dict[str, CGSParameterValue] = {name: to_cgs_value(value) for name, value in parameters.items()}

        return cls.flux_band_log_cgs(
            nu.cgs.value,
            throughput,
            t.cgs.value,
            np.asarray(distances["redshift"], dtype=np.float64),
            distances["luminosity_distance"].cgs.value,
            log_attenuation=log_attenuation,
            **cgs_parameters,
        )

    @classmethod
    def flux_band_cgs(
        cls,
        nu: NumericalInput,
        throughput: NumericalInput,
        t: NumericalInput,
        redshift: NumericalInput,
        luminosity_distance: NumericalInput,
        *,
        log_attenuation: NumericalInput | None = None,
        **parameters: CGSParameterValue,
    ) -> FloatResult:
        r"""
        Band-averaged observed flux density as plain cgs numbers; see :meth:`flux_band_log_cgs`.

        Parameters
        ----------
        nu : array-like
            Observed frequency grid to integrate over, in Hz. Need not be sorted.
        throughput : array-like
            Dimensionless bandpass response at each ``nu`` sample.
        t : array-like
            Observed time since explosion, in seconds.
        redshift : array-like
            Cosmological redshift, dimensionless.
        luminosity_distance : array-like
            Luminosity distance, in cm.
        log_attenuation : array-like, optional
            See :meth:`flux_band_log_cgs`.
        **parameters
            This model's parameter values, in cgs units. See :meth:`flux_band_log_cgs`.

        Returns
        -------
        numpy.ndarray or float
            :math:`\bar{F}_\nu`, in erg/s/cm^2/Hz. A scalar is returned if
            the result is 0-dimensional.
        """
        return np.exp(
            cls.flux_band_log_cgs(
                nu,
                throughput,
                t,
                redshift,
                luminosity_distance,
                log_attenuation=log_attenuation,
                **parameters,
            )
        )

    @classmethod
    def flux_band(
        cls,
        nu: Quantity,
        throughput: NumericalInput,
        t: Quantity,
        *,
        redshift: PhysicalInput | None = None,
        luminosity_distance: Quantity | None = None,
        angular_diameter_distance: Quantity | None = None,
        proper_distance: Quantity | None = None,
        cosmology: FLRW | None = None,
        log_attenuation: NumericalInput | None = None,
        **parameters: ParameterValue,
    ) -> Quantity:
        r"""
        Evaluate the throughput-weighted mean observed flux density over a band.

        Parameters
        ----------
        nu : ~astropy.units.Quantity
            Observed frequency grid to integrate over. Need not be sorted.
        throughput : array-like
            Dimensionless bandpass response at each ``nu`` sample.
        t : ~astropy.units.Quantity
            Observed time since explosion.
        redshift : array-like or ~astropy.units.Quantity, optional
            See :meth:`flux_log`.
        luminosity_distance : ~astropy.units.Quantity, optional
            See :meth:`flux_log`.
        angular_diameter_distance : ~astropy.units.Quantity, optional
            See :meth:`flux_log`.
        proper_distance : ~astropy.units.Quantity, optional
            See :meth:`flux_log`.
        cosmology : ~astropy.cosmology.FLRW, optional
            See :meth:`flux_log`.
        log_attenuation : array-like, optional
            See :meth:`flux_band_log_cgs`.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`\bar{F}_\nu`, in erg/s/cm^2/Hz.
        """
        return (
            np.exp(
                cls.flux_band_log(
                    nu,
                    throughput,
                    t,
                    redshift=redshift,
                    luminosity_distance=luminosity_distance,
                    angular_diameter_distance=angular_diameter_distance,
                    proper_distance=proper_distance,
                    cosmology=cosmology,
                    log_attenuation=log_attenuation,
                    **parameters,
                )
            )
            * _SPEC_FLUX_UNIT
        )

    # -------------------------------------- #
    # Apparent AB Magnitudes                  #
    # -------------------------------------- #
    # A magnitude already is a logarithmic quantity, so there is no separate
    # "log" form here -- just a bare-float `*_cgs` form and a unitful form
    # returning an `astropy.units.Magnitude` (`u.ABmag`).
    @classmethod
    def mag_cgs(
        cls,
        nu: NumericalInput,
        t: NumericalInput,
        redshift: NumericalInput,
        luminosity_distance: NumericalInput,
        *,
        log_attenuation: NumericalInput | None = None,
        **parameters: CGSParameterValue,
    ) -> FloatResult:
        r"""
        Apparent AB magnitude: :math:`m_\mathrm{AB} = -2.5 \log_{10}(F_\nu / F_{\mathrm{AB},0})`.

        :math:`F_\nu` is :meth:`flux_cgs`'s observed flux density;
        :math:`F_{\mathrm{AB},0} = 3631` Jy. See :meth:`flux_log_cgs` for the
        meaning of ``log_attenuation``.

        Parameters
        ----------
        nu : array-like
            Observed frequency, in Hz.
        t : array-like
            Observed time since explosion, in seconds.
        redshift : array-like
            Cosmological redshift, dimensionless.
        luminosity_distance : array-like
            Luminosity distance, in cm.
        log_attenuation : array-like, optional
            See :meth:`flux_log_cgs`.
        **parameters
            This model's parameter values, in cgs units. See :meth:`flux_log_cgs`.

        Returns
        -------
        numpy.ndarray or float
            The apparent AB magnitude. A scalar is returned if the result is
            0-dimensional.
        """
        F_nu = cls.flux_cgs(
            nu,
            t,
            redshift,
            luminosity_distance,
            log_attenuation=log_attenuation,
            **parameters,
        )

        return -2.5 * np.log10(F_nu / AB_MAG_ZERO_POINT)

    @classmethod
    def mag(
        cls,
        nu: Quantity,
        t: Quantity,
        *,
        redshift: PhysicalInput | None = None,
        luminosity_distance: Quantity | None = None,
        angular_diameter_distance: Quantity | None = None,
        proper_distance: Quantity | None = None,
        cosmology: FLRW | None = None,
        log_attenuation: NumericalInput | None = None,
        **parameters: ParameterValue,
    ) -> Quantity:
        """
        Evaluate the apparent AB magnitude at the given frequency and time.

        Parameters
        ----------
        nu : ~astropy.units.Quantity
            Observed frequency.
        t : ~astropy.units.Quantity
            Observed time since explosion.
        redshift : array-like or ~astropy.units.Quantity, optional
            See :meth:`flux_log`.
        luminosity_distance : ~astropy.units.Quantity, optional
            See :meth:`flux_log`.
        angular_diameter_distance : ~astropy.units.Quantity, optional
            See :meth:`flux_log`.
        proper_distance : ~astropy.units.Quantity, optional
            See :meth:`flux_log`.
        cosmology : ~astropy.cosmology.FLRW, optional
            See :meth:`flux_log`.
        log_attenuation : array-like, optional
            See :meth:`flux_log_cgs`.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        ~astropy.units.Quantity
            The apparent AB magnitude, as an :attr:`~astropy.units.ABmag` Quantity.
        """
        if not isinstance(nu, Quantity) or nu.unit.physical_type != "frequency":
            raise TypeError("`nu` must be an astropy Quantity with frequency units.")
        if not isinstance(t, Quantity) or t.unit.physical_type != "time":
            raise TypeError("`t` must be an astropy Quantity with time units.")

        distances = resolve_cosmological_distances(
            redshift=redshift,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            cosmology=cosmology,
        )
        cgs_parameters: dict[str, CGSParameterValue] = {name: to_cgs_value(value) for name, value in parameters.items()}

        return (
            cls.mag_cgs(
                nu.cgs.value,
                t.cgs.value,
                np.asarray(distances["redshift"], dtype=np.float64),
                distances["luminosity_distance"].cgs.value,
                log_attenuation=log_attenuation,
                **cgs_parameters,
            )
            * u.ABmag
        )

    @classmethod
    def mag_band_cgs(
        cls,
        nu: NumericalInput,
        throughput: NumericalInput,
        t: NumericalInput,
        redshift: NumericalInput,
        luminosity_distance: NumericalInput,
        *,
        log_attenuation: NumericalInput | None = None,
        **parameters: CGSParameterValue,
    ) -> FloatResult:
        """
        Apparent AB magnitude of the band-averaged flux density. See :meth:`flux_band_cgs`/:meth:`mag_cgs`.

        Broadcasting (including ``log_attenuation``'s) follows :meth:`flux_band_cgs`'s rules exactly.

        Parameters
        ----------
        nu : array-like
            Observed frequency grid to integrate over, in Hz. Need not be sorted.
        throughput : array-like
            Dimensionless bandpass response at each ``nu`` sample.
        t : array-like
            Observed time since explosion, in seconds.
        redshift : array-like
            Cosmological redshift, dimensionless.
        luminosity_distance : array-like
            Luminosity distance, in cm.
        log_attenuation : array-like, optional
            See :meth:`flux_band_log_cgs`.
        **parameters
            This model's parameter values, in cgs units. See :meth:`flux_band_log_cgs`.

        Returns
        -------
        numpy.ndarray or float
            The apparent AB magnitude of the band-averaged flux density. A
            scalar is returned if the result is 0-dimensional.
        """
        F_nu = cls.flux_band_cgs(
            nu,
            throughput,
            t,
            redshift,
            luminosity_distance,
            log_attenuation=log_attenuation,
            **parameters,
        )

        return -2.5 * np.log10(F_nu / AB_MAG_ZERO_POINT)

    @classmethod
    def mag_band(
        cls,
        nu: Quantity,
        throughput: NumericalInput,
        t: Quantity,
        *,
        redshift: PhysicalInput | None = None,
        luminosity_distance: Quantity | None = None,
        angular_diameter_distance: Quantity | None = None,
        proper_distance: Quantity | None = None,
        cosmology: FLRW | None = None,
        log_attenuation: NumericalInput | None = None,
        **parameters: ParameterValue,
    ) -> Quantity:
        """
        Evaluate the apparent AB magnitude of the band-averaged flux density.

        Parameters
        ----------
        nu : ~astropy.units.Quantity
            Observed frequency grid to integrate over. Need not be sorted.
        throughput : array-like
            Dimensionless bandpass response at each ``nu`` sample.
        t : ~astropy.units.Quantity
            Observed time since explosion.
        redshift : array-like or ~astropy.units.Quantity, optional
            See :meth:`flux_log`.
        luminosity_distance : ~astropy.units.Quantity, optional
            See :meth:`flux_log`.
        angular_diameter_distance : ~astropy.units.Quantity, optional
            See :meth:`flux_log`.
        proper_distance : ~astropy.units.Quantity, optional
            See :meth:`flux_log`.
        cosmology : ~astropy.cosmology.FLRW, optional
            See :meth:`flux_log`.
        log_attenuation : array-like, optional
            See :meth:`flux_band_log_cgs`.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        ~astropy.units.Quantity
            The apparent AB magnitude, as an :attr:`~astropy.units.ABmag` Quantity.
        """
        if not isinstance(nu, Quantity) or nu.unit.physical_type != "frequency":
            raise TypeError("`nu` must be an astropy Quantity with frequency units.")
        if not isinstance(t, Quantity) or t.unit.physical_type != "time":
            raise TypeError("`t` must be an astropy Quantity with time units.")

        distances = resolve_cosmological_distances(
            redshift=redshift,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            cosmology=cosmology,
        )
        cgs_parameters: dict[str, CGSParameterValue] = {name: to_cgs_value(value) for name, value in parameters.items()}

        return (
            cls.mag_band_cgs(
                nu.cgs.value,
                throughput,
                t.cgs.value,
                np.asarray(distances["redshift"], dtype=np.float64),
                distances["luminosity_distance"].cgs.value,
                log_attenuation=log_attenuation,
                **cgs_parameters,
            )
            * u.ABmag
        )

    @classmethod
    def mag_bandpass(
        cls,
        bandpass: SpectralElement,
        t: Quantity,
        *,
        redshift: PhysicalInput | None = None,
        luminosity_distance: Quantity | None = None,
        angular_diameter_distance: Quantity | None = None,
        proper_distance: Quantity | None = None,
        cosmology: FLRW | None = None,
        log_attenuation: NumericalInput | None = None,
        **parameters: ParameterValue,
    ) -> Quantity:
        """
        Evaluate the apparent AB magnitude of the flux averaged over `bandpass`.

        Convenience wrapper around :meth:`mag_band` that reads its ``nu`` and
        ``throughput`` straight off `bandpass` instead of requiring them as
        separate arguments -- see :meth:`mag_bandpass_cgs`.

        Parameters
        ----------
        bandpass : ~synphot.SpectralElement
            The bandpass to average over, e.g. one of an
            `~m4opt.synphot.Detector`'s own
            `~m4opt.synphot.Detector.bandpasses`.
        t : ~astropy.units.Quantity
            Observed time since explosion.
        redshift : array-like or ~astropy.units.Quantity, optional
            See :meth:`flux_log`.
        luminosity_distance : ~astropy.units.Quantity, optional
            See :meth:`flux_log`.
        angular_diameter_distance : ~astropy.units.Quantity, optional
            See :meth:`flux_log`.
        proper_distance : ~astropy.units.Quantity, optional
            See :meth:`flux_log`.
        cosmology : ~astropy.cosmology.FLRW, optional
            See :meth:`flux_log`.
        log_attenuation : array-like, optional
            See :meth:`flux_band_log_cgs`.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        ~astropy.units.Quantity
            The apparent AB magnitude, as an :attr:`~astropy.units.ABmag` Quantity.
        """
        wave = bandpass.waveset
        nu = wave.to(u.Hz, equivalencies=u.spectral())
        throughput = bandpass(wave).to_value(u.dimensionless_unscaled)

        return cls.mag_band(
            nu,
            throughput,
            t,
            redshift=redshift,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            cosmology=cosmology,
            log_attenuation=log_attenuation,
            **parameters,
        )

    # -------------------------------------- #
    # Detector-Aware (Noisy) Photometry       #
    # -------------------------------------- #
    def measure_photometry(
        self,
        t: PhysicalInput,
        exptime: Quantity,
        detector: Detector,
        coord: SkyCoord,
        *,
        background: SourceSpectrum | None = None,
        bands: list | None = None,
        observer_location: EarthLocation | None = None,
        obstime: Time | None = None,
        redshift: PhysicalInput | None = None,
        luminosity_distance: Quantity | None = None,
        angular_diameter_distance: Quantity | None = None,
        proper_distance: Quantity | None = None,
        cosmology: FLRW | None = None,
        ebv: float | Quantity | FloatArray | None = None,
        dust_law: str | None = None,
        in_model: np.ndarray | None = None,
        sys_err: float | Mapping[str, float] | None = None,
        rng: RNGInput = None,
        noise_seed: int | FloatArray | None = None,
        noise_observation_keys: FloatArray | None = None,
        **parameters: ParameterValue,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """
        Measure this model with a real detector at given times: one noisy measurement per epoch and band.

        The array form of :meth:`simulate_photometry`, with no table around it. Every epoch can
        belong to a different event: `coord`, `observer_location`, `obstime`, `redshift`, the
        distances, `ebv`, `noise_seed`, and the model parameters can each be a scalar (shared by
        every epoch) or an array with one entry per epoch. That makes it the one measurement
        routine for a single event's photometry and for a survey-wide screening of many events.

        The model is only valid from the explosion onward (some light curve shapes diverge at
        negative times). Epochs not `in_model`, by default those with ``t < 0``, are therefore
        never evaluated against this model: they are measured against a flat, effectively
        zero-flux spectrum instead, i.e. pure background at that observation's own depth.

        Parameters
        ----------
        t : array-like or Quantity
            Time(s) since explosion, shape ``(N,)`` (or scalar, promoted to shape ``(1,)``),
            one entry per epoch.
        exptime : ~astropy.units.Quantity
            Exposure duration(s), scalar or shape matching `t`.
        detector : ~m4opt.synphot.Detector
            Supplies bandpasses, collecting area, plate scale, and noise terms (dark/read
            noise, gain, ...). Its own ``background`` is used unless `background` is given.
        coord : ~astropy.coordinates.SkyCoord
            Sky position of the target, scalar or shape ``(N,)``.
        background : ~synphot.SourceSpectrum, optional
            Sky background to simulate against for this call, replacing `detector`'s own
            (which is never mutated), e.g. `~m4opt.synphot.background.GalacticBackground()`
            alone or combined with others via ``+``.
        bands : list, optional
            Which of `detector`'s bandpasses to evaluate. Defaults to every bandpass.
        observer_location : ~astropy.coordinates.EarthLocation, optional
            Observer location, scalar or shape ``(N,)``. Matters only insofar as `background`
            depends on it (for example `~m4opt.synphot.background.EarthshineBackground`);
            defaults to a fixed placeholder.
        obstime : ~astropy.time.Time, optional
            Observation time, scalar or shape ``(N,)``. Matters only insofar as `background`
            depends on it (for example `~m4opt.synphot.background.ZodiacalBackground`);
            defaults to a fixed placeholder.
        redshift : array-like or ~astropy.units.Quantity, optional
            See :meth:`as_source_spectrum`.
        luminosity_distance : ~astropy.units.Quantity, optional
            See :meth:`as_source_spectrum`.
        angular_diameter_distance : ~astropy.units.Quantity, optional
            See :meth:`as_source_spectrum`.
        proper_distance : ~astropy.units.Quantity, optional
            See :meth:`as_source_spectrum`.
        cosmology : ~astropy.cosmology.FLRW, optional
            See :meth:`as_source_spectrum`.
        ebv, dust_law : float, ~astropy.units.Quantity, numpy.ndarray, or str, optional
            See :meth:`as_source_spectrum`.
        in_model : numpy.ndarray, optional
            Boolean, shape ``(N,)``: which epochs are evaluated against this model. By default
            ``t >= 0``. A caller can mark more epochs as background, e.g. an exposure that runs
            past the end of the window it cares about.
        sys_err : float or Mapping[str, float], optional
            See `~uvex_transients.models._utils.measure_bands`.
        rng : numpy.random.Generator, int, or None
            See `~uvex_transients.models._utils.measure_bands`.
        noise_seed, noise_observation_keys : optional
            See `~uvex_transients.models._utils.measure_bands`.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        flux : numpy.ndarray
            Measured flux density in Jy, shape ``(n_bands, N)``.
        flux_err : numpy.ndarray
            One-sigma uncertainty in Jy, same shape.
        snr : numpy.ndarray
            Measured SNR, ``flux / flux_err``, same shape.
        snr_expected : numpy.ndarray
            Noiseless SNR, the true flux over ``flux_err``, same shape.

        Raises
        ------
        ValueError
            If `bands` or `sys_err` doesn't match `detector`, `exptime`, `in_model`, `coord`,
            `observer_location` or `obstime` is neither scalar nor shaped like `t`, or
            `noise_seed` is given without matching `noise_observation_keys`.
        """
        if background is not None:
            detector = replace(detector, background=background)
        band_names = resolve_bands(detector, bands, sys_err)
        t, exptime = normalize_times(t, exptime)
        if in_model is None:
            in_model = t.to_value(u.day) >= 0
        in_model = np.asarray(in_model, dtype=bool)
        if in_model.shape != t.shape:
            raise ValueError(f"'in_model' must have shape {t.shape}, got {in_model.shape}.")
        if observer_location is None:
            observer_location = _PLACEHOLDER_OBSERVER_LOCATION
        if obstime is None:
            obstime = _PLACEHOLDER_OBSTIME
        for name, value in (("coord", coord), ("observer_location", observer_location), ("obstime", obstime)):
            if not value.isscalar and value.shape != t.shape:
                raise ValueError(f"'{name}' must be scalar or match 't' shape {t.shape}, got {value.shape}.")

        # Give every per-epoch input one entry per epoch, so that selecting an epoch's rows is the
        # same indexing whether the caller passed a scalar (shared by every epoch) or an array.
        broadcast = lambda value: np.broadcast_to(value, t.shape, subok=True)  # noqa: E731

        coord, observer_location, obstime = broadcast(coord), broadcast(observer_location), broadcast(obstime)
        sed_inputs = {
            name: broadcast(value)
            for name, value in dict(
                redshift=redshift,
                luminosity_distance=luminosity_distance,
                angular_diameter_distance=angular_diameter_distance,
                proper_distance=proper_distance,
                ebv=ebv,
                **parameters,
            ).items()
            if value is not None
        }
        if noise_seed is not None:
            noise_seed = broadcast(np.asarray(noise_seed, dtype=np.uint64))

        rng = get_rng(rng)
        measurements = [np.full((len(band_names), len(t)), np.nan) for _ in range(4)]
        for evaluate_model in (True, False):
            rows = np.flatnonzero(in_model == evaluate_model)
            if len(rows) == 0:
                continue
            if evaluate_model:
                # Everything per-epoch gets a reserved trailing axis for wavelength, except `ebv`,
                # which `attenuation_callable` reserves itself.
                inputs = {
                    name: value[rows] if name == "ebv" else value[rows][:, np.newaxis]
                    for name, value in sed_inputs.items()
                }
                spectra = self.as_source_spectrum(
                    t[rows][:, np.newaxis], cosmology=cosmology, dust_law=dust_law, **inputs
                )
            else:
                spectra = _flat_flux_source_spectrum(np.full(len(rows), _NEGLIGIBLE_FLUX_JY) * u.Jy)
            measured = measure_bands(
                detector,
                band_names,
                exptime[rows],
                spectra,
                coord[rows],
                observer_location[rows],
                obstime[rows],
                sys_err=sys_err,
                rng=rng,
                noise_seed=None if noise_seed is None else noise_seed[rows],
                noise_observation_keys=None if noise_observation_keys is None else noise_observation_keys[rows],
            )
            for out, values in zip(measurements, measured):
                out[:, rows] = values
        return tuple(measurements)

    def simulate_photometry(
        self,
        t: PhysicalInput,
        exptime: Quantity,
        detector: Detector,
        coord: SkyCoord,
        *,
        background: SourceSpectrum | None = None,
        bands: list | None = None,
        observer_location: EarthLocation | None = None,
        obstime: Time | None = None,
        redshift: PhysicalInput | None = None,
        luminosity_distance: Quantity | None = None,
        angular_diameter_distance: Quantity | None = None,
        proper_distance: Quantity | None = None,
        cosmology: FLRW | None = None,
        ebv: float | Quantity | FloatArray | None = None,
        dust_law: str | None = None,
        in_model: np.ndarray | None = None,
        n_sigma: float | None = None,
        sys_err: float | Mapping[str, float] | None = None,
        rng: RNGInput = None,
        noise_seed: int | FloatArray | None = None,
        noise_observation_keys: FloatArray | None = None,
        **parameters: ParameterValue,
    ) -> QTable:
        """
        Simulate noisy synthetic photometry of this model at given times, against a real detector.

        The noise-aware counterpart to :meth:`mag`/:meth:`flux`/:meth:`as_source_spectrum`:
        those give the noiseless truth; this reports a synthetic *measurement*, one Gaussian
        realization of the true flux at each requested time and band at that time and band's
        own implied uncertainty, with the magnitudes and bounds worked out from it. The
        measurement itself is :meth:`measure_photometry`; this lays it out as a table.

        Deliberately independent of any :class:`~uvex_transients.surveys.base.SurveySchedule`
        or :class:`~uvex_transients.simulation.event.Event`: `t`/`exptime` are whatever
        times and exposures the caller wants evaluated (for example a target of opportunity's
        own cadence), not ones a schedule was queried for.

        Parameters
        ----------
        t : array-like or Quantity
            Time(s) since explosion, shape ``(N,)`` (or scalar, promoted to shape ``(1,)``).
        exptime : ~astropy.units.Quantity
            Exposure duration(s), scalar or shape matching `t`.
        detector : ~m4opt.synphot.Detector
            See :meth:`measure_photometry`.
        coord : ~astropy.coordinates.SkyCoord
            Sky position of the target, scalar or shape ``(N,)``.
        background : ~synphot.SourceSpectrum, optional
            See :meth:`measure_photometry`.
        bands : list, optional
            See :meth:`measure_photometry`.
        observer_location : ~astropy.coordinates.EarthLocation, optional
            See :meth:`measure_photometry`.
        obstime : ~astropy.time.Time, optional
            See :meth:`measure_photometry`.
        redshift : array-like or ~astropy.units.Quantity, optional
            See :meth:`as_source_spectrum`.
        luminosity_distance : ~astropy.units.Quantity, optional
            See :meth:`as_source_spectrum`.
        angular_diameter_distance : ~astropy.units.Quantity, optional
            See :meth:`as_source_spectrum`.
        proper_distance : ~astropy.units.Quantity, optional
            See :meth:`as_source_spectrum`.
        cosmology : ~astropy.cosmology.FLRW, optional
            See :meth:`as_source_spectrum`.
        ebv, dust_law : float, ~astropy.units.Quantity, numpy.ndarray, or str, optional
            See :meth:`as_source_spectrum`.
        in_model : numpy.ndarray, optional
            See :meth:`measure_photometry`. By default ``t >= 0``.
        n_sigma : float, optional
            Width, in multiples of ``flux_err``, of the ``flux_upper``/``flux_lower``/
            ``mag_upper``/``mag_lower`` interval. If `None` (the default), uses
            ``config["simulation.detection_n_sigma"]`` (5 out of the box).
        sys_err : float or Mapping[str, float], optional
            See :meth:`measure_photometry`.
        rng : numpy.random.Generator, int, or None
            See :meth:`measure_photometry`.
        noise_seed, noise_observation_keys : optional
            See :meth:`measure_photometry`.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        astropy.table.QTable
            One row per (time, band), sorted by ``t`` then ``band``, with columns ``t``,
            ``exptime``, ``band``, ``snr`` (the measured SNR, ``flux / flux_err``),
            ``flux``/``flux_err`` (Jy), ``flux_upper``/``flux_lower`` (Jy, ``flux ±
            n_sigma*flux_err``), ``ab_mag``/``mag_err``, ``mag_upper``/``mag_lower`` -- the
            ``n_sigma`` interval transformed to magnitude, brighter bound first -- and
            ``in_model``. See :meth:`~uvex_transients.simulation.event.Event.simulate_photometry`
            for the exact semantics of every column. If `sys_err` is given,
            ``snr``/``flux_err``/``mag_err`` (and everything derived from them) reflect the
            combined shot-noise-plus-systematic uncertainty.

        Raises
        ------
        ValueError
            As for :meth:`measure_photometry`.
        """
        t, exptime = normalize_times(t, exptime)
        if in_model is None:
            in_model = t.to_value(u.day) >= 0
        in_model = np.asarray(in_model, dtype=bool)

        measurements = self.measure_photometry(
            t,
            exptime,
            detector,
            coord,
            background=background,
            bands=bands,
            observer_location=observer_location,
            obstime=obstime,
            redshift=redshift,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            cosmology=cosmology,
            ebv=ebv,
            dust_law=dust_law,
            in_model=in_model,
            sys_err=sys_err,
            rng=rng,
            noise_seed=noise_seed,
            noise_observation_keys=noise_observation_keys,
            **parameters,
        )
        band_names = resolve_bands(detector, bands, sys_err)
        return photometry_table(t, exptime, band_names, *measurements, n_sigma=n_sigma, in_model=in_model)

    # -------------------------------------- #
    # Simulation                              #
    # -------------------------------------- #
    def simulate(self, nu: Quantity, t: Quantity, size: int = 1, *, rng: RNGInput = None) -> Quantity:
        r"""
        Draw random parameter realizations and evaluate the model at the given frequency and time.

        Equivalent to ``self.eval(nu, t, **self.sample_parameters(size=size, rng=rng))``.

        Parameters
        ----------
        nu : ~astropy.units.Quantity
            Frequency at which to evaluate the model. See :meth:`eval`.
        t : ~astropy.units.Quantity
            Time since explosion. See :meth:`eval`.
        size : int
            Number of realizations to draw.
        rng : numpy.random.Generator, int, or None
            Random-number source, forwarded to :meth:`sample_parameters`.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`L_\nu(\nu, t)`, in erg/s/Hz. :meth:`sample_parameters`
            always returns array-valued parameters, even for ``size=1``, so
            the batch axis is never squeezed away here.

        See Also
        --------
        eval : The underlying evaluation.
        sample_parameters : The underlying sampling.
        """
        return self.eval(nu, t, **self.sample_parameters(size=size, rng=rng))

    # -------------------------------------- #
    # Observability (Detection Horizon)       #
    # -------------------------------------- #
    def get_effective_peak_magnitudes(
        self,
        z_grid: NumericalInput,
        bandpasses: Mapping[str, SpectralElement] | Sequence[SpectralElement],
        *,
        t_rest: Quantity | None = None,
        t_min: Quantity | None = None,
        t_max: Quantity | None = None,
        n_time: int = 200,
        criterion: Criterion = "any",
        k: int | None = None,
        n_samples: int = 1000,
        cosmology: FLRW | None = None,
        chunk_size: int | None = None,
        progress: bool = True,
        rng: RNGInput = None,
    ) -> EffectivePeakGrid:
        r"""
        Evaluate a sampled population's peak magnitude at every redshift of a grid.

        This is the expensive half of the detection-horizon calculation. It does not depend on any
        magnitude limit, so its result can be reused for many limits with
        :meth:`get_observability_curve`.

        Draw `n_samples` parameter sets :math:`\Theta_i` from this model's priors. Evaluate every
        one of them at every redshift of `z_grid`: the light curve is computed on a rest-frame
        time grid in each band, reduced to its brightest point (the minimum magnitude over time),
        and the bands are combined as `criterion` says. The result has one number per
        :math:`(\Theta_i, z)` pair.

        Parameters
        ----------
        z_grid : array-like
            Redshifts to evaluate, shape ``(Z,)``, strictly increasing and positive. Zero is not
            allowed because the luminosity distance vanishes there. The grid must reach past the
            horizon you care about, or :meth:`get_observability_curve` cannot bracket it.
        bandpasses : Mapping[str, ~synphot.SpectralElement] or sequence of ~synphot.SpectralElement
            The bands to evaluate, for example ``detector.bandpasses``. A sequence is named
            ``band0``, ``band1``, and so on. A band with fewer frequency samples is much
            cheaper: see :func:`~uvex_transients.missions.downsample_mission`.
        t_rest : ~astropy.units.Quantity, optional
            Rest-frame times since explosion at which to evaluate the light curve. Rest-frame
            times are the right choice here, since time dilation moves the peak in observed time
            but does not change its magnitude, so any observing cadence is irrelevant.
        t_min, t_max : ~astropy.units.Quantity, optional
            If `t_rest` is omitted, the ends of a logarithmically spaced grid of `n_time` points
            (``0 < t_min < t_max``). One of the two ways of giving a time grid is required,
            because a model has no notion of its own duration.
        n_time : int, optional
            Number of points of the logarithmic grid. Make it dense enough to resolve the peak:
            a grid that misses the peak makes every event look fainter than it is, and so the
            horizon too close, which is not the conservative direction.
        criterion : {"any", "all", "k_of_n"}, optional
            How bands combine into one effective peak magnitude. ``"any"``: detected in at least
            one band (the minimum over bands). ``"all"``: detected in every band (the maximum of
            the per-band peaks). ``"k_of_n"``: detected in at least `k` bands (the `k`-th smallest).
        k : int, optional
            The number of bands required for ``"k_of_n"``, from 1 to the number of bands.
        n_samples : int, optional
            Number of parameter draws, ``M``. :meth:`get_observability_curve` needs at least
            299 for 95% confidence at 1% tolerance.
        cosmology : ~astropy.cosmology.FLRW, optional
            Cosmology for the luminosity distances. Defaults to the configured cosmology.
        chunk_size : int, optional
            Number of parameter draws evaluated per call. If `None` (the default), it is chosen
            so the largest intermediate array stays near 8 MB. That size fits in cache and
            benchmarks fastest, so raising it only costs speed and memory.
        progress : bool, optional
            Whether to show a progress bar. The default is `True`.
        rng : numpy.random.Generator, int, or None
            Random-number source for the parameter draws. Pass an int for reproducible output.

        Returns
        -------
        ~uvex_transients.models.core.EffectivePeakGrid
            The ``(n_samples, Z)`` array of effective peak magnitudes, with the grids and the
            sampled parameters that produced it.

        Raises
        ------
        ValueError
            If `z_grid` is not a 1-d, strictly increasing array of positive redshifts, if no
            time grid or bandpass is given, or if `criterion` and `k` are inconsistent.

        See Also
        --------
        get_observability_curve : Turn the grid into a horizon for each magnitude limit.

        Notes
        -----
        The redshift is a grid every :math:`\Theta_i` is evaluated on, not a random draw. A
        single random redshift per draw would give each transient one magnitude at one distance,
        and so no horizon to speak of. The horizon of a draw, the redshift at which it drops
        below a limit, only exists if the draw is followed across redshift.

        The time minimum is taken separately at every redshift, not once for the whole grid,
        because redshift shifts the bands across the spectrum (the K-correction) and so changes
        which epoch is brightest in a band.

        The full ``(n_samples, Z, T, B)`` array is never built. Time and bands are reduced as soon
        as they are computed, one chunk of draws at a time, and only the ``(n_samples, Z)`` result
        is kept. The cost is linear in each of ``n_samples``, ``Z``, ``n_time`` and the number
        of frequency samples across all bands.

        Milky Way extinction and sky position are not included. The result covers the SED's
        parameter priors only.

        Examples
        --------
        >>> import numpy as np
        >>> from astropy import units as u
        >>> from m4opt.missions import uvex
        >>> from uvex_transients.missions import (
        ...     downsample_mission,
        ... )
        >>> from uvex_transients.models.kilonovae import (
        ...     KilonovaCoolingBlackbodySED,
        ... )
        >>> grid = KilonovaCoolingBlackbodySED().get_effective_peak_magnitudes(
        ...     np.geomspace(0.003, 1.5, 30),
        ...     downsample_mission(uvex).detector.bandpasses,
        ...     t_min=0.01 * u.day,
        ...     t_max=30 * u.day,
        ...     n_samples=500,
        ...     rng=0,
        ... )  # doctest: +SKIP
        """
        z_grid = np.asarray(z_grid, dtype=np.float64)
        if z_grid.ndim != 1 or z_grid.size == 0 or z_grid[0] <= 0 or np.any(np.diff(z_grid) <= 0):
            raise ValueError("'z_grid' must be a 1-d, strictly increasing array of positive redshifts.")
        t_rest_s = resolve_time_grid(t_rest, t_min, t_max, n_time)
        band_names, band_grids = resolve_bandpasses(bandpasses)
        reduce_bands(np.zeros(len(band_grids)), criterion, k)  # fail now on a bad criterion or k

        # Everything the loop below needs, converted once: distances for every grid redshift,
        # and the sampled parameters as plain cgs numbers.
        luminosity_distance = resolve_cosmological_distances(redshift=z_grid, cosmology=cosmology)[
            "luminosity_distance"
        ].cgs.value
        samples = self.sample_parameters(size=n_samples, rng=rng)
        cgs_samples = {name: to_cgs_value(value) for name, value in samples.items()}

        if chunk_size is None:
            widest_band = max(len(nu) for nu, _ in band_grids)
            chunk_size = int(np.clip(_CHUNK_BYTES // (8 * t_rest_s.size * widest_band), 1, n_samples))
        n_chunks = -(-n_samples // chunk_size)

        m_eff = np.empty((n_samples, z_grid.size))
        peak_by_band = np.empty((chunk_size, len(band_grids)))  # reused for every (chunk, z)
        with (
            tqdm(
                total=n_chunks * z_grid.size, desc="Evaluating peak magnitudes", unit="step", disable=not progress
            ) as pbar,
            logging_redirect_tqdm(loggers=[logger]),
        ):
            for start in range(0, n_samples, chunk_size):
                stop = min(start + chunk_size, n_samples)
                n_rows = stop - start
                # Parameters get a trailing axis so they broadcast against the time axis: (n_rows, 1).
                chunk = {name: value[start:stop, np.newaxis] for name, value in cgs_samples.items()}

                for j, (z, d_l) in enumerate(zip(z_grid, luminosity_distance)):
                    # The model takes observed times, which are the rest-frame ones stretched by (1 + z).
                    t_obs = (t_rest_s * (1.0 + z))[np.newaxis, :]
                    for b, (nu, throughput) in enumerate(band_grids):
                        with np.errstate(divide="ignore"):  # zero flux is an infinitely faint magnitude
                            mags = self.mag_band_cgs(nu, throughput, t_obs, z, d_l, **chunk)  # (n_rows, T)
                        # Some models return NaN outside their validity window. Those epochs are
                        # skipped; a draw with no valid epoch at all stays NaN (handled below).
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)  # "All-NaN slice"
                            peak_by_band[:n_rows, b] = np.nanmin(mags, axis=1)  # brightest valid epoch
                    m_eff[start:stop, j] = reduce_bands(peak_by_band[:n_rows], criterion, k)
                    pbar.update(1)

        if np.isnan(m_eff).any():
            logger.warning(
                "%d of %d effective peak magnitudes are NaN because the model has no valid epoch on the time "
                "grid for that draw, redshift and band. They are treated as undetectable.",
                int(np.isnan(m_eff).sum()),
                m_eff.size,
            )

        return EffectivePeakGrid(
            m_eff=m_eff,
            z_grid=z_grid,
            criterion=criterion,
            k=k,
            bands=band_names,
            t_rest=t_rest_s,
            parameters=samples,
        )

    def get_observability_curve(
        self,
        mag_limits: NumericalInput,
        *,
        grid: EffectivePeakGrid | None = None,
        confidence: float = 0.95,
        tolerance: float = 0.01,
        interpolate: bool = False,
        **grid_kwargs,
    ) -> QTable:
        r"""
        Bound the redshift beyond which a population is undetectable, for each magnitude limit.

        For each parameter draw :math:`\Theta_i` in the grid, its detection redshift
        :math:`z_{\mathrm{det},i}` is the first grid redshift where its effective peak magnitude
        is fainter than the limit. Sorting those values and taking the :math:`r`-th largest gives
        a limit with a distribution-free guarantee (Wilks 1941): with probability `confidence`,
        at most a fraction `tolerance` of the population has a detection redshift above it. The
        rank :math:`r` is the largest one that still meets `confidence`, which gives the
        tightest valid limit, and grows with the number of draws.

        This is the number to use as a population's redshift limit: sampling beyond it wastes
        effort on events that are, with the stated confidence, not detectable.

        Parameters
        ----------
        mag_limits : float or array-like
            AB magnitude limit(s). Tabulating many is almost free once the grid exists.
        grid : ~uvex_transients.models.core.EffectivePeakGrid, optional
            A grid from :meth:`get_effective_peak_magnitudes`. If omitted, one is built from
            `grid_kwargs`.
        confidence : float, optional
            Probability that the guarantee holds, in (0, 1). The default is 0.95.
        tolerance : float, optional
            Fraction of the population allowed to lie beyond the limit, in (0, 1). The default
            is 0.01.
        interpolate : bool, optional
            If `False` (the default), each detection redshift is rounded *up* to the first grid
            redshift where the draw is too faint. That can only overestimate the horizon, so the
            guarantee stays valid, and it is coarse by up to one grid step. If `True`, the
            crossing is interpolated linearly in magnitude against :math:`\log(1+z)`, which is
            tighter but no longer guaranteed to be an overestimate.
        **grid_kwargs
            Arguments for :meth:`get_effective_peak_magnitudes`, required (`z_grid`, `bandpasses`
            and a time grid) when `grid` is omitted, and not allowed otherwise.

        Returns
        -------
        ~astropy.table.QTable
            One row per magnitude limit, with columns

            ``mag_limit``
                The AB magnitude limit.
            ``z_limit``
                The upper limit on the detection redshift. It is ``inf``, with a warning, if the
                grid does not reach the limit, in which case `z_grid` must be extended.
            ``n_censored``
                How many draws never dropped below the limit inside the grid.

            ``meta`` records ``confidence``, ``tolerance``, ``n_samples``, ``rank``,
            ``criterion`` and ``bands``.

        Raises
        ------
        ValueError
            If there are too few draws for the requested `confidence` and `tolerance`. The
            sample maximum is only a valid limit from ``ln(1 - confidence) / ln(1 - tolerance)``
            draws on, 299 for the defaults. This is checked before the expensive grid is built.
        TypeError
            If `grid` and `grid_kwargs` are both given.

        See Also
        --------
        get_effective_peak_magnitudes : The expensive grid this reads.

        Notes
        -----
        The statement covers the SED's parameter priors only. Milky Way extinction and sky
        position are not included, so leave a margin for dusty fields.

        Examples
        --------
        >>> table = sed.get_observability_curve(
        ...     [24.0, 25.0, 26.0],
        ...     grid=grid,
        ...     confidence=0.95,
        ...     tolerance=0.01,
        ... )  # doctest: +SKIP
        >>> table["z_limit"]  # doctest: +SKIP
        """
        if grid is None:
            n_samples = grid_kwargs.get("n_samples", 1000)
            if n_samples < min_samples(confidence, tolerance):
                wilks_rank(n_samples, confidence, tolerance)  # raises, naming the required minimum
            grid = self.get_effective_peak_magnitudes(**grid_kwargs)
        elif grid_kwargs:
            raise TypeError(f"Unexpected arguments with a precomputed 'grid': {sorted(grid_kwargs)}.")

        limits = np.atleast_1d(np.asarray(getattr(mag_limits, "value", mag_limits), dtype=np.float64))
        z_limit = np.empty(limits.size)
        n_censored = np.empty(limits.size, dtype=int)
        for i, mag_limit in enumerate(limits):
            z_det = first_crossing(grid.m_eff, grid.z_grid, mag_limit, interpolate=interpolate)
            z_limit[i], rank, n_censored[i] = tolerance_limit(z_det, confidence, tolerance)

        if np.any(np.isinf(z_limit)):
            logger.warning(
                "The redshift grid (z <= %.3g) is too short to bracket the limit for mag_limit = %s; extend 'z_grid'.",
                grid.z_grid[-1],
                limits[np.isinf(z_limit)],
            )

        return QTable(
            {"mag_limit": limits * u.ABmag, "z_limit": z_limit, "n_censored": n_censored},
            meta={
                "confidence": confidence,
                "tolerance": tolerance,
                "n_samples": grid.n_samples,
                "rank": rank,
                "criterion": grid.criterion,
                "bands": grid.bands,
            },
        )


class ComposedSpectralModel(SpectralModel):
    r"""
    A :class:`SpectralModel` built by pairing a :class:`Lightcurve` with a :class:`Spectrum`.

    .. math::

        L_\nu(\nu, t) = L_\mathrm{bol}(t) \cdot \frac{S(\nu)}{\int S(\nu')\,d\nu'}

    where :math:`L_\mathrm{bol}(t)` is :attr:`_LIGHTCURVE_CLASS`'s bolometric
    luminosity and the fraction is :attr:`_SPECTRUM_CLASS`'s shape,
    normalized (via :meth:`Spectrum.eval_normalization`) to integrate to 1
    over :math:`\nu`. Because both halves are already exact on their own,
    :meth:`_eval_bolometric` and :meth:`_eval_spectrum` are closed-form
    combinations of the two components' own primitives -- unlike
    :class:`SpectralModel`'s generic defaults, neither ever falls back to
    numerical integration over frequency here.

    A concrete SED is defined just by naming the two component classes:

    .. code-block:: python

        class MySED(ComposedSpectralModel):
            _LIGHTCURVE_CLASS = FREDLightcurve
            _SPECTRUM_CLASS = BlackbodySpectrum

    At class-definition time, :meth:`__init_subclass__` merges
    ``_LIGHTCURVE_CLASS._DEFAULT_PARAMETERS`` and
    ``_SPECTRUM_CLASS._DEFAULT_PARAMETERS`` into this subclass's own
    :attr:`~SpectralModel._DEFAULT_PARAMETERS` (any entries the subclass
    declares directly itself win, as overrides on top of that merge) -- so
    from then on, ``MySED`` behaves exactly like any other
    :class:`SpectralModel` subclass: ``MySED()`` or ``MySED(amplitude=...,
    temperature=...)``, one flat parameter namespace. Every parameter name
    must be unique across the two component classes, checked at
    class-definition time.

    Because every :class:`SpectralModel` method is a classmethod operating
    purely on ``**parameters`` (see the module docstring), no instance-level
    wiring of components is needed: :meth:`_eval_bolometric`,
    :meth:`_eval_spectrum`, and :meth:`_eval` simply split the incoming
    ``parameters`` dict by name and call :attr:`_LIGHTCURVE_CLASS`'s/
    :attr:`_SPECTRUM_CLASS`'s own classmethods directly.

    See Also
    --------
    Lightcurve : The time-only half of this composition.
    Spectrum : The frequency-only half of this composition.
    """

    _LIGHTCURVE_CLASS: ClassVar[type[Lightcurve] | None] = None
    """type of Lightcurve, or None: The bolometric lightcurve class driving this SED's time dependence.

    ``None`` on :class:`ComposedSpectralModel` itself; every concrete
    subclass must set this (and :attr:`_SPECTRUM_CLASS`) directly in its own
    class body.
    """

    _SPECTRUM_CLASS: ClassVar[type[Spectrum] | None] = None
    """type of Spectrum, or None: The spectral shape class driving this SED's frequency dependence."""

    # -------------------------------------- #
    # Subclass Validation                    #
    # -------------------------------------- #
    def __init_subclass__(cls, **kwargs) -> None:
        """
        Merge the component classes' parameters into :attr:`_DEFAULT_PARAMETERS`.

        Parameters
        ----------
        **kwargs
            Forwarded to :meth:`object.__init_subclass__` unchanged; this
            class declares no class-keyword-argument options of its own.
        """
        super().__init_subclass__(**kwargs)

        if cls._LIGHTCURVE_CLASS is None or cls._SPECTRUM_CLASS is None:
            return

        if not (isinstance(cls._LIGHTCURVE_CLASS, type) and issubclass(cls._LIGHTCURVE_CLASS, Lightcurve)):
            raise TypeError(f"{cls.__name__}._LIGHTCURVE_CLASS must be a Lightcurve subclass.")
        if not (isinstance(cls._SPECTRUM_CLASS, type) and issubclass(cls._SPECTRUM_CLASS, Spectrum)):
            raise TypeError(f"{cls.__name__}._SPECTRUM_CLASS must be a Spectrum subclass.")

        overlap = set(cls._LIGHTCURVE_CLASS._DEFAULT_PARAMETERS) & set(cls._SPECTRUM_CLASS._DEFAULT_PARAMETERS)
        if overlap:
            raise TypeError(
                f"{cls.__name__}: _LIGHTCURVE_CLASS and _SPECTRUM_CLASS share "
                f"parameter name(s) {sorted(overlap)}; every parameter name must "
                "be unique across the two."
            )

        cls._DEFAULT_PARAMETERS = {
            **cls._LIGHTCURVE_CLASS._DEFAULT_PARAMETERS,
            **cls._SPECTRUM_CLASS._DEFAULT_PARAMETERS,
            **cls.__dict__.get("_DEFAULT_PARAMETERS", {}),
        }

    # -------------------------------------- #
    # Parameter Splitting                    #
    # -------------------------------------- #
    @classmethod
    def _split_parameters(
        cls, parameters: Mapping[str, CGSParameterValue]
    ) -> tuple[dict[str, CGSParameterValue], dict[str, CGSParameterValue]]:
        """
        Split a flat ``parameters`` dict into ``(lightcurve_parameters, spectrum_parameters)``.

        Parameters
        ----------
        parameters : mapping of str to (float or numpy.ndarray)
            This model's full, flat parameter dict, keyed by name.

        Returns
        -------
        tuple of (dict of str to (float or numpy.ndarray))
            ``(lightcurve_parameters, spectrum_parameters)``, each holding
            only the entries of `parameters` belonging to that component.

        Raises
        ------
        TypeError
            If :attr:`_LIGHTCURVE_CLASS` or :attr:`_SPECTRUM_CLASS` is unset.
        """
        if cls._LIGHTCURVE_CLASS is None or cls._SPECTRUM_CLASS is None:
            raise TypeError(
                f"{cls.__name__} must be subclassed with both _LIGHTCURVE_CLASS "
                "and _SPECTRUM_CLASS set before it can be evaluated."
            )

        return (
            {name: parameters[name] for name in cls._LIGHTCURVE_CLASS._DEFAULT_PARAMETERS},
            {name: parameters[name] for name in cls._SPECTRUM_CLASS._DEFAULT_PARAMETERS},
        )

    # -------------------------------------- #
    # Bolometric Luminosity: L_bol(t)         #
    # -------------------------------------- #
    @classmethod
    def _eval_bolometric(cls, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        :math:`\log L_\mathrm{bol}(t)`, delegated directly to :attr:`_LIGHTCURVE_CLASS`.

        Exact, not an approximation: unlike
        :meth:`SpectralModel._eval_bolometric`'s generic numerical-quadrature
        fallback, no integration is needed here -- the lightcurve's own
        :meth:`Lightcurve._eval` already *is* the bolometric luminosity, by
        construction.

        Parameters
        ----------
        t : numpy.ndarray
            Time since explosion, in seconds.
        **parameters
            This model's parameter values, in cgs units, broadcastable
            against ``t``.

        Returns
        -------
        numpy.ndarray
            The natural log of :math:`L_\mathrm{bol}(t)`, in erg/s.
        """
        lightcurve_parameters, _ = cls._split_parameters(parameters)

        # `_split_parameters` already raises if either class is unset.
        assert cls._LIGHTCURVE_CLASS is not None
        return cls._LIGHTCURVE_CLASS._eval(t, **lightcurve_parameters)

    # -------------------------------------- #
    # Normalized Spectral Shape: S(nu, t)    #
    # -------------------------------------- #
    @classmethod
    def _eval_spectrum(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        :math:`\log S(\nu)`, delegated to :attr:`_SPECTRUM_CLASS` and normalized to integrate to 1 (``t`` is unused).

        Exact, not an approximation: divides out the spectrum's own
        :meth:`Spectrum._eval_normalization` directly, rather than falling
        back to :meth:`SpectralModel._eval_spectrum`'s
        bolometric-integral-subtraction default.

        Parameters
        ----------
        nu : numpy.ndarray
            Frequency, in Hz.
        t : numpy.ndarray
            Time since explosion, in seconds. Unused (the shape is already
            normalized independently of time); accepted only to match
            :meth:`SpectralModel._eval_spectrum`'s signature.
        **parameters
            This model's parameter values, in cgs units, broadcastable
            against ``nu``.

        Returns
        -------
        numpy.ndarray
            The natural log of :math:`S(\nu)`, in 1/Hz.
        """
        _, spectrum_parameters = cls._split_parameters(parameters)

        # `_split_parameters` already raises if either class is unset.
        assert cls._SPECTRUM_CLASS is not None
        return cls._SPECTRUM_CLASS._eval(nu, **spectrum_parameters) - cls._SPECTRUM_CLASS._eval_normalization(
            **spectrum_parameters
        )

    # -------------------------------------- #
    # Spectral Luminosity: L_nu(nu, t)        #
    # -------------------------------------- #
    @classmethod
    def _eval(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        :math:`\log L_\nu(\nu, t) = \log L_\mathrm{bol}(t) + \log S(\nu)`.

        Since this sums :meth:`_eval_bolometric`'s (``t``-shaped) and
        :meth:`_eval_spectrum`'s (``nu``-shaped) results, ``nu``/``t`` must
        already broadcast against each other the way the caller wants --
        see :meth:`SpectralModel._eval`'s broadcasting contract.

        Parameters
        ----------
        nu : numpy.ndarray
            Frequency, in Hz.
        t : numpy.ndarray
            Time since explosion, in seconds.
        **parameters
            This model's parameter values, in cgs units, broadcastable
            against ``nu``/``t``.

        Returns
        -------
        numpy.ndarray
            The natural log of :math:`L_\nu(\nu, t)`, in erg/s/Hz.
        """
        return cls._eval_bolometric(t, **parameters) + cls._eval_spectrum(nu, t, **parameters)
