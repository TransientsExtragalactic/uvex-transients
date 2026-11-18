r"""
The core of the sampling infrastructure: the sampler contract, its products and the stopping rule.

This module knows nothing about surveys. The samplers that simulate one are in
`~uvex_transients.phase_space.sampling.core` and `~uvex_transients.phase_space.sampling.breakout`.

Products: the quantities of a map, each of which owns how it is made.

A fitted map is of one thing, the success fraction :math:`f` over the space. Everything else a figure or a table shows
is made from it and from the point: the fraction times a volume-time is an expected yield, the fraction times a rate on
top of that is an expected count, and a redshift limit does not involve the fraction at all. A `Product` is one of those
quantities. It is a callable that takes the points and the map"s `Estimate` of the success fraction there and returns
the columns it creates, so it is free to combine them however the quantity needs. Nothing else has to know what a
volume-time or a rate is.

A sampler supplies its own products (`~uvex_transients.phase_space.sampling.base.Sampler.products`), which are built
around whatever machinery of the sampler they need. `Scale` and `Weighted` cover the two common cases, a quantity of
the point alone and the success fraction times one.

The sampler: how many detections in n tries at a point of the space.

A phase-space map asks one question of the world, again and again: at this point `x`, of `n` independent tries, how
many succeed? A `Sampler` answers it, and nothing else. It does not decide how many tries to make (that is the
`~uvex_transients.phase_space.sampling.base.StoppingRule`) and it does not know about the Gaussian process that
models the answers (that is the map). What it does know is the physics of one kind of experiment:

* `Sampler.space` is the `~uvex_transients.phase_space.space.Space` the experiment is over, with its axes and their
  default ranges;
* `Sampler.draw` makes `n` tries at a point and returns the number of successes;
* `Sampler.products` are the quantities of its map beyond the success fraction (a redshift limit, a recovered
  volume-time), each a `~uvex_transients.phase_space.sampling.base.Product` that owns how it is made.

Every sampler has a `kind`, a name under which it is registered, so that a file can record which sampler made it and
`get_sampler` can find the class again without importing it by path. A subclass registers itself by naming its kind in
the class statement,

.. code-block:: python

    class MySampler(Sampler, kind="my_sampler"): ...

and an abstract intermediate class leaves it out.

Everything beyond those four members is an optional hook with a default, so that the code that consumes a sampler (the
experiment, the stages that write and draw a map, the CLI) never has to ask what kind it is: `Sampler.rate_products`
and `Sampler.table_meta` extend the table of the map, `Sampler.describe` titles its figure, `Sampler.draw_map` draws it
and `Sampler.annotate_map` decorates it, `Sampler.HAS_OVERLAY` and `Sampler.measure_overlay` supply a population to
shade on it, and `Sampler.from_run_config` builds the sampler from a run-config. A sampler that has something to say
overrides the hook; one that does not inherits an answer that says nothing. The default figure is of a plane, so a map
over more than two axes is not drawn.

The stopping rule: how many tries a point gets.

A sampler makes however many tries it is asked for. How many to ask for is a question of statistics, not of physics: a
point where the success fraction is high needs few tries to pin it down, and one where it is small needs many, so a
point is sampled in chunks until it has enough successes or has used its budget. `StoppingRule` is that rule, and it is
a separate object because two parties need it: the campaign applies it, and the map"s acquisition predicts from it the
noise of a point it has not measured yet.
"""

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from astropy.table import Table
from astropy.units import Quantity
from numpy.typing import ArrayLike, NDArray

from uvex_transients.phase_space._config import default
from uvex_transients.phase_space.space import Space
from uvex_transients.utils import logger

__all__ = [
    "Estimate",
    "Outcome",
    "Product",
    "Sampler",
    "Scale",
    "StoppingRule",
    "Weighted",
    "available_samplers",
    "get_sampler",
]


# ============================================ #
# Products                                     #
# ============================================ #

#: Suffixes of the columns of an interval, by convention: a product that carries one names them ``<name>_lo`` and
#: ``<name>_hi``.
LOWER = "_lo"
UPPER = "_hi"


# ============================================ #
# Estimate                                     #
# ============================================ #
@dataclass(frozen=True)
class Estimate:
    """
    The map's estimate of the success fraction at points: a value with an interval.

    Raises
    ------
    ValueError
        If the three arrays do not have the same shape.
    """

    p: NDArray[np.float64]
    """numpy.ndarray: The estimated success fraction, of shape ``(...)``."""

    p_lo: NDArray[np.float64]
    """numpy.ndarray: The lower end of its interval, of the shape of `p`."""

    p_hi: NDArray[np.float64]
    """numpy.ndarray: The upper end of its interval, of the shape of `p`."""

    def __post_init__(self):
        p, p_lo, p_hi = (np.asarray(value, dtype=np.float64) for value in (self.p, self.p_lo, self.p_hi))
        if not p.shape == p_lo.shape == p_hi.shape:
            raise ValueError(
                f"The estimate and its interval need one shape, got {p.shape}, {p_lo.shape}, {p_hi.shape}."
            )
        object.__setattr__(self, "p", p)
        object.__setattr__(self, "p_lo", p_lo)
        object.__setattr__(self, "p_hi", p_hi)

    @property
    def shape(self) -> tuple[int, ...]:
        """Tuple of int: The shape of the estimate, which is the shape of the points without their last axis."""
        return self.p.shape


# ============================================ #
# Product                                      #
# ============================================ #
class Product(ABC):
    r"""
    A quantity of the map, and how it is made from the points and the estimate of the success fraction.

    Parameters
    ----------
    name : str
        The name of the product, which is the name of its main column. A valid Python identifier, so that it can be
        used in an expression.
    label : str, optional
        What a figure calls the quantity, in mathtext if it has symbols.
    unit : str, optional
        The unit of the quantity.
    draw : bool, optional
        Whether the default figure of the map has a panel for it. Off by default; a `Weighted` product, which is what a
        map is for, is drawn unless it says otherwise. Any column can be drawn by asking for it by name.

    Raises
    ------
    ValueError
        If `name` is not an identifier.
    """

    def __init__(self, name: str, label: str = "", unit: str = "", draw: bool = False):
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"The name of a product must be a valid identifier, got {name!r}.")
        self.draw = bool(draw)
        """bool: Whether the default figure has a panel for the quantity."""
        self.name = name
        """str: The name of the product, and of its main column."""
        self.label = label
        """str: What a figure calls the quantity."""
        self.unit = unit
        """str: The unit of the quantity."""

    @abstractmethod
    def __call__(self, x: ArrayLike, estimate: Estimate) -> dict[str, NDArray[np.float64]]:
        """
        Make the columns of the product at points of the space.

        Parameters
        ----------
        x : array-like
            Points in the coordinates of the sampler's space, shape ``(..., ndim)``.
        estimate : Estimate
            The map's estimate of the success fraction at those points, of shape ``(...)``.

        Returns
        -------
        dict of str to numpy.ndarray
            The columns it creates, each of shape ``(...)``. The one called `name` is the main one; an interval is
            carried by convention in ``<name>_lo`` and ``<name>_hi``.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


def _points(x: ArrayLike, estimate: Estimate) -> NDArray[np.float64]:
    """Return `x` as floats, after checking that it has the points that `estimate` is of."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0 or x.shape[:-1] != estimate.shape:
        raise ValueError(f"The points {x.shape} and the estimate {estimate.shape} do not describe the same points.")
    return x


class Scale(Product):
    r"""
    A quantity of the point alone, which does not involve the success fraction.

    A redshift limit, a coverage or a volume-time is a `Scale`: it follows from the physics at the point, with no Monte
    Carlo, and so has no interval.

    Parameters
    ----------
    name : str
        The name of the product, and of its one column.
    function : callable
        ``function(x)`` takes points of shape ``(..., ndim)`` and returns the quantity, of shape ``(...)``.
    label, unit, draw : optional
        As for `Product`.
    """

    def __init__(
        self,
        name: str,
        function: Callable[[NDArray], ArrayLike],
        label: str = "",
        unit: str = "",
        draw: bool = False,
    ):
        super().__init__(name, label, unit, draw)
        self._function = function

    def __call__(self, x: ArrayLike, estimate: Estimate) -> dict[str, NDArray[np.float64]]:
        """
        Make the column of the quantity at points of the space.

        Parameters
        ----------
        x : array-like
            Points, shape ``(..., ndim)``.
        estimate : Estimate
            The estimate at those points; it only supplies their shape.

        Returns
        -------
        dict of str to numpy.ndarray
            ``{name: values}``, of shape ``(...)``.
        """
        x = _points(x, estimate)
        return {self.name: np.broadcast_to(np.asarray(self._function(x), dtype=np.float64), estimate.shape).copy()}


class Weighted(Product):
    r"""
    The success fraction times a quantity of the point, with its interval.

    A recovered volume-time is the success fraction times the volume-time of the survey for an event there, and an
    efficiency is the fraction times the coverage. The interval of the fraction carries through, since the weight is
    positive.

    Parameters
    ----------
    name : str
        The name of the product. Its columns are `name`, ``<name>_lo`` and ``<name>_hi``.
    weight : callable
        ``weight(x)`` takes points of shape ``(..., ndim)`` and returns the weight, of shape ``(...)``, which must
        not be negative.
    label, unit : str, optional
        As for `Product`.
    draw : bool, optional
        As for `Product`, but on by default: a quantity with an interval is what a figure of the map shows.
    """

    def __init__(
        self,
        name: str,
        weight: Callable[[NDArray], ArrayLike],
        label: str = "",
        unit: str = "",
        draw: bool = True,
    ):
        super().__init__(name, label, unit, draw)
        self._weight = weight

    def __call__(self, x: ArrayLike, estimate: Estimate) -> dict[str, NDArray[np.float64]]:
        """
        Make the columns of the weighted estimate at points of the space.

        Parameters
        ----------
        x : array-like
            Points, shape ``(..., ndim)``.
        estimate : Estimate
            The estimate of the success fraction at those points.

        Returns
        -------
        dict of str to numpy.ndarray
            ``name``, ``<name>_lo`` and ``<name>_hi``, each of shape ``(...)``.

        Raises
        ------
        ValueError
            If a weight is negative, which would swap the ends of the interval.
        """
        x = _points(x, estimate)
        weight = np.broadcast_to(np.asarray(self._weight(x), dtype=np.float64), estimate.shape)
        if np.any(weight < 0.0):
            raise ValueError(f"The weight of {self.name!r} must not be negative.")
        return {
            self.name: estimate.p * weight,
            self.name + LOWER: estimate.p_lo * weight,
            self.name + UPPER: estimate.p_hi * weight,
        }


# ============================================ #
# Outcome                                      #
# ============================================ #
@dataclass(frozen=True)
class Outcome:
    """
    The result of sampling one point: `k` successes of `n` tries, and whatever else was worth recording.

    Raises
    ------
    ValueError
        If the counts do not satisfy ``0 <= k <= n`` with ``n >= 1``.
    """

    k: int
    """int: The number of successes."""

    n: int
    """int: The number of tries, at least 1."""

    extras: Mapping[str, Any] = field(default_factory=dict)
    """Mapping: Other entries of the point, by name, which become columns of the table of points."""

    def __post_init__(self):
        k, n = int(self.k), int(self.n)
        if n < 1 or not 0 <= k <= n:
            raise ValueError(f"An outcome needs 0 <= k <= n and n >= 1, got k={self.k!r} and n={self.n!r}.")
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "extras", dict(self.extras))

    @property
    def fraction(self) -> float:
        """float: The observed success fraction, ``k / n``."""
        return self.k / self.n


# ============================================ #
# Registry                                     #
# ============================================ #
_REGISTRY: dict[str, type["Sampler"]] = {}


def available_samplers() -> tuple[str, ...]:
    """
    Return the kinds of every registered sampler.

    Returns
    -------
    tuple of str
        The kinds, sorted.
    """
    return tuple(sorted(_REGISTRY))


def get_sampler(kind: str) -> type["Sampler"]:
    """
    Return the registered sampler class of a kind.

    Parameters
    ----------
    kind : str
        The kind the class was registered under.

    Returns
    -------
    type of Sampler
        The class.

    Raises
    ------
    KeyError
        If no sampler is registered under `kind`.
    """
    try:
        return _REGISTRY[kind]
    except KeyError:
        raise KeyError(f"No sampler of kind {kind!r}; the registered kinds are {list(available_samplers())}.") from None


# ============================================ #
# Sampler                                      #
# ============================================ #
class Sampler(ABC):
    r"""
    Answer "how many detections in n tries at a point?" for one kind of experiment.

    A subclass supplies `space`, `draw`, `settings` and `from_settings`. Name a `kind` in the class statement to
    register it; see the module documentation. Everything else is an optional hook, with a default that a sampler
    overrides only if it has something to say, so that nothing that consumes a sampler has to ask what kind it is:

    * `products`, `rate_products` and `yield_weight`: the quantities of its map beyond the success fraction, and the one
      that two maps are compared on;
    * `table_meta`: what a figure or a comparison needs to know about the plane the axes are measured on;
    * `describe`: what the map is, in a few words, for the title of its figure;
    * `draw_map` and `annotate_map`: the figure of its map, and what is drawn on it;
    * `HAS_OVERLAY` and `measure_overlay`: a population to shade on the figure, if there is one;
    * `from_run_config`: how a run-config builds it.

    Raises
    ------
    TypeError
        If a subclass names a kind that is already registered, or one that is not a non-empty string.
    """

    kind: ClassVar[str | None] = None
    """str or None: The name the class is registered under, which a file records. `None` for an unregistered class."""

    def __init_subclass__(cls, kind: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)

        # A subclass that does not name a kind is not registered, and does not claim the kind of its parent
        cls.kind = None
        if kind is None:
            return
        if not isinstance(kind, str) or not kind:
            raise TypeError(f"The kind of {cls.__name__} must be a non-empty string, got {kind!r}.")
        if kind in _REGISTRY:
            raise TypeError(f"A sampler of kind {kind!r} is already registered: {_REGISTRY[kind].__name__}.")
        cls.kind = kind
        _REGISTRY[kind] = cls

    @classmethod
    def kinds(cls) -> tuple[str, ...]:
        """
        Return the registered kinds this class is, the most specific first.

        A sampler built on another is also of the other's kind: the bracketed breakout is of ``breakout_bracketed`` and
        of ``breakout``. This is the order in which its configured defaults are looked up (see
        `~uvex_transients.phase_space._config.default`).

        Returns
        -------
        tuple of str
            The kinds of the class and of its registered parents, `()` for an unregistered class.
        """
        return tuple(klass.kind for klass in cls.__mro__ if vars(klass).get("kind"))

    # -------------------------------------- #
    # What a subclass supplies               #
    # -------------------------------------- #
    @property
    @abstractmethod
    def space(self) -> Space:
        """Space: The axes the experiment is over, with their default ranges."""

    @abstractmethod
    def draw(self, x: ArrayLike, n: int, seed: np.random.SeedSequence) -> int:
        """
        Make `n` independent tries at a point and return how many succeed.

        Parameters
        ----------
        x : array-like
            The point, in the coordinates of `space`, shape ``(ndim,)``.
        n : int
            The number of tries.
        seed : numpy.random.SeedSequence
            The source of all the randomness of the tries. The same point, count and seed give the same number of
            successes.

        Returns
        -------
        int
            The number of successes, between 0 and `n`.
        """

    @property
    @abstractmethod
    def settings(self) -> dict:
        """dict: What `from_settings` needs to rebuild this sampler, as plain types that an ECSV header can hold."""

    @classmethod
    @abstractmethod
    def from_settings(cls, settings: Mapping, **context) -> "Sampler":
        """
        Rebuild a sampler from the `settings` of another one.

        Parameters
        ----------
        settings : Mapping
            The `settings` of the sampler to rebuild.
        **context
            What a file cannot record, such as a survey schedule that is not in the registry, for the samplers that
            need it.

        Returns
        -------
        Sampler
            A sampler that makes the same experiment.
        """

    # -------------------------------------- #
    # What a subclass may supply             #
    # -------------------------------------- #
    @property
    def products(self) -> tuple[Product, ...]:
        """
        Tuple of Product: The quantities of a map of this sampler beyond the success fraction.

        Each product owns how it is made from the points and the map's estimate, so it can be built around whatever
        machinery of the sampler it needs: a redshift limit, a recovered volume-time. Empty by default.
        """
        return ()

    def rate_products(self, rate: Quantity) -> tuple[Product, ...]:
        """
        Return the quantities of a map that need a volumetric rate, such as the expected number of recoveries.

        Parameters
        ----------
        rate : ~astropy.units.Quantity
            The comoving volumetric rate, with units of inverse volume per time.

        Returns
        -------
        tuple of Product
            The products, which are added to `products` when a map is made with a rate. Empty by default, for a sampler
            that has no volume to apply a rate to.
        """
        return ()

    @property
    def table_meta(self) -> dict:
        """
        dict: What a figure of the map needs to label its axes, and what a comparison checks before it differences two.

        These are entries of the ``meta`` of the fitted map's table, for plain types: the band, the shape of the
        transient that the axes are measured on. Two maps are only comparable if their entries agree. Empty by default.
        """
        return {}

    @classmethod
    def describe(cls, settings: Mapping) -> str:
        """
        Describe the map of a sampler in a few words, for the title of its figure.

        It is a class method of the `settings` that a file records, so that a map can be titled from its files alone,
        without rebuilding the sampler (which can need a survey schedule).

        Parameters
        ----------
        settings : Mapping
            The `settings` of a sampler of this class.

        Returns
        -------
        str
            What the map is. By default the kind.
        """
        return cls.kind or cls.__name__

    @classmethod
    def draw_map(
        cls,
        grid: Table,
        points: Table,
        path: str | Path,
        title: str,
        *,
        overlay: Table | None = None,
        show_points: bool = False,
        fields: Sequence[str] | None = None,
    ):
        """
        Draw the fitted map of a sampler of this class and save the figure.

        It is a class method of the files of a map, so that a map can be drawn without rebuilding the sampler. The
        default is `~uvex_transients.phase_space.plotting.plot_grid`, which draws a map over two axes with the products
        it carries and decorates it with `annotate_map`. A map over any other number of axes is not drawn: the figure is
        of a plane, so the default logs a warning and returns `None`. A sampler whose map has a figure of its own
        overrides this, and is free to draw more axes in its own way.

        Parameters
        ----------
        grid : astropy.table.Table
            The fitted map (``map.ecsv``).
        points : astropy.table.Table
            The measured points (``points.ecsv``).
        path : str or pathlib.Path
            Where to save the figure.
        title : str
            The suptitle.
        overlay : astropy.table.Table, optional
            The measurements of `measure_overlay` (``overlay.ecsv``), for a sampler that has them.
        show_points : bool, optional
            Whether to mark the measured points.
        fields : sequence of str, optional
            The columns of the map to draw, one panel each (any column, including a quantity that carries no
            interval). By default the success fraction, every quantity with an interval, and the uncertainty.

        Returns
        -------
        matplotlib.figure.Figure or None
            The figure, which the caller closes, or `None` if the map is not drawn.
        """
        from uvex_transients.phase_space.plotting import plot_grid

        ndim = len(grid.meta["space"]["axes"]) if "space" in grid.meta else 2
        if ndim != 2:
            logger.warning("The map is over %d axes and a figure is of two, so none is drawn.", ndim)
            return None

        annotate = None if overlay is None else lambda ax: cls.annotate_map(ax, overlay)
        return plot_grid(grid, points, path, title, fields=fields, show_points=show_points, annotate=annotate)

    @classmethod
    def annotate_map(cls, ax, overlay: Table) -> Sequence | None:
        """
        Draw the overlay of a sampler on the axes of a figure of its map.

        This is how a map of a sampler with an overlay is decorated in every figure that has one (the map and the
        difference of two maps), so that no figure has to know what the overlay is.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes of a panel, in the plotted coordinates (the values of a logarithmic axis, not their logarithms).
        overlay : astropy.table.Table
            The measurements of `measure_overlay` (``overlay.ecsv``).

        Returns
        -------
        sequence of matplotlib.artist.Artist or None
            Legend handles for what was drawn, if it is to be named in the legend. `None` by default, which draws
            nothing.
        """
        return None

    yield_weight: ClassVar[str | None] = None
    """
    str or None: The name of the `Scale` product that turns the success fraction into the quantity two maps are
    compared on (the volume-time, for a survey sampler: the recovered volume-time is its product with the fraction).
    `None` compares the success fractions themselves. It is recorded in the table of the map, so a comparison reads it
    from the files alone.
    """

    HAS_OVERLAY: ClassVar[bool] = False
    """bool: Whether `measure_overlay` is implemented, for the callers that decide whether to ask for one."""

    def measure_overlay(self, n_draws: int | None = None, seed: int = 0) -> Table:
        """
        Measure a population in the plane of the map, to be shaded on its figure.

        Only a sampler with ``HAS_OVERLAY`` set implements this.

        Parameters
        ----------
        n_draws : int, optional
            How many draws of each member of the population. By default the configured number.
        seed : int, optional
            The seed of the draws.

        Returns
        -------
        astropy.table.Table
            The measurements, which `draw_map` takes as its ``overlay``.

        Raises
        ------
        NotImplementedError
            If this class has no overlay.
        """
        raise NotImplementedError(f"A sampler of kind {self.kind!r} has no overlay.")

    @classmethod
    def from_run_config(cls, run: Any, **options) -> "Sampler":
        """
        Build a sampler from the arguments of a map in a run-config.

        A sampler that needs more than its own arguments (the survey, say) takes it from `run`.

        Parameters
        ----------
        run : object
            The parsed run-config, with the attributes ``schedule`` (the resolved survey schedule), ``mission_name`` and
            ``schedule_name``, which it resolves only when they are read.
        **options
            The arguments of the sampler: the detection, simulation and ``settings:`` sections of the map, merged.

        Returns
        -------
        Sampler
            The sampler. By default ``cls(**options)``.

        Raises
        ------
        ValueError
            If the sampler does not take one of the `options`.
        """
        cls._check_options(**options)
        return cls(**options)

    @classmethod
    def _check_options(cls, *positional, **options) -> None:
        """
        Check that the constructor takes `options`, before it is called.

        This catches an unknown or missing argument without wrapping the constructor in a handler that would also
        report a ``TypeError`` raised inside it as a mistake in the settings.

        Parameters
        ----------
        *positional
            Placeholders for the positional arguments the caller will pass.
        **options
            The keyword arguments the caller will pass.

        Raises
        ------
        ValueError
            If the arguments do not fit the signature of the constructor.
        """
        signature = inspect.signature(cls)
        try:
            signature.bind(*positional, **options)
        except TypeError as error:
            raise ValueError(f"The {cls.kind!r} sampler does not take these settings: {error}") from error

        # A constructor that forwards ``**kwargs`` to its parent accepts anything, so check the names of the parents'
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
            named = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            accepted = {
                name
                for klass in cls.__mro__
                if "__init__" in vars(klass)
                for name, parameter in inspect.signature(vars(klass)["__init__"]).parameters.items()
                if parameter.kind in named
            }
            unknown = sorted(set(options) - accepted)
            if unknown:
                raise ValueError(f"The {cls.kind!r} sampler does not take the setting(s) {unknown}.")


# ============================================ #
# The stopping rule                            #
# ============================================ #


@dataclass(frozen=True)
class StoppingRule:
    """
    Sample a point in chunks of tries until it has `k_target` successes or has used `n_max` tries.

    Raises
    ------
    ValueError
        If `chunk_size` or `k_target` is below 1, or `n_max` is below `chunk_size`.
    """

    chunk_size: int
    """int: The tries made at a time, and so the fewest a point gets."""

    k_target: int
    """int: Stop adding chunks to a point once it has this many successes."""

    n_max: int
    """int: Never make more than this many tries at a point (the last chunk may take it past)."""

    def __post_init__(self):
        chunk_size, k_target, n_max = int(self.chunk_size), int(self.k_target), int(self.n_max)
        if chunk_size < 1 or k_target < 1 or n_max < chunk_size:
            raise ValueError(
                f"Need chunk_size >= 1, k_target >= 1 and n_max >= chunk_size, got {chunk_size}, {k_target}, {n_max}."
            )
        object.__setattr__(self, "chunk_size", chunk_size)
        object.__setattr__(self, "k_target", k_target)
        object.__setattr__(self, "n_max", n_max)

    @classmethod
    def from_config(
        cls,
        *,
        chunk_size: int | None = None,
        k_target: int | None = None,
        n_max: int | None = None,
        kinds: Sequence[str] = (),
    ) -> "StoppingRule":
        """
        Build the rule, taking what is not given from ``config["phase_space.stopping"]``.

        Parameters
        ----------
        chunk_size, k_target, n_max : int, optional
            As for the class. Any that is `None` (the default) is read from the configuration.
        kinds : sequence of str, optional
            The kinds of the sampler the rule is for, whose own sections of the configuration take precedence (see
            `~uvex_transients.phase_space._config.default`).

        Returns
        -------
        StoppingRule
            The rule.
        """
        given = {"chunk_size": chunk_size, "k_target": k_target, "n_max": n_max}
        return cls(
            **{name: default("stopping", name, kinds) if value is None else value for name, value in given.items()}
        )

    def run(self, sampler: Sampler, x: ArrayLike, seed: np.random.SeedSequence) -> Outcome:
        """
        Sample a point until the rule says to stop.

        Each chunk draws from its own child of `seed`, so a chunk does not depend on the others, and `seed` is
        advanced by one child per chunk. Give every point a seed of its own.

        Parameters
        ----------
        sampler : Sampler
            What makes the tries.
        x : array-like
            The point, in the coordinates of the sampler's space.
        seed : numpy.random.SeedSequence
            The root seed of the point, whose children seed the chunks.

        Returns
        -------
        Outcome
            The successes and tries, with the number of chunks as ``extras["n_chunks"]``.
        """
        x = np.asarray(x, dtype=np.float64)
        k = n = chunks = 0

        # Top the point up until it has enough successes, or has used all the tries it is allowed
        while n < self.n_max and k < self.k_target:
            k += sampler.draw(x, self.chunk_size, seed.spawn(1)[0])
            n += self.chunk_size
            chunks += 1
        return Outcome(k, n, {"n_chunks": chunks})

    def expected_n(self, p: ArrayLike) -> NDArray[np.float64]:
        """
        Predict how many tries a point of success fraction `p` will get.

        Enough to expect `k_target` successes, bounded by one chunk below and `n_max` above. This is what the map's
        acquisition uses to predict the noise of a point before it is measured.

        Parameters
        ----------
        p : array-like
            The success fraction, which must be positive.

        Returns
        -------
        numpy.ndarray
            The expected number of tries, of the shape of `p`.
        """
        return np.clip(self.k_target / np.asarray(p, dtype=np.float64), self.chunk_size, self.n_max)
