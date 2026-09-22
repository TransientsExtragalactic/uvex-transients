"""
Milky Way foreground dust extinction.

This module provides access to the PlanckGNILC E(B-V) map (`dust_map`) and a
reddening law (`get_dust_law`, Gordon+2023's `G23` by default -- see
``config["physics.default_dust_law"]``), combined by a single vectorized entry point
(`log_attenuation`) into a natural-log attenuation array ready to add directly into a
`~uvex_transients.models.core.base.SpectralModel` flux calculation (its
``log_attenuation`` keyword argument). `attenuation_callable` gives the same thing as
a callable; `~uvex_transients.models.core.base.SpectralModel.as_source_spectrum`/
`~uvex_transients.models.core.base.SpectralModel.simulate_photometry` call it
internally whenever their own ``ebv`` keyword is given, so ordinary callers of this
package never touch this module, a callable, or `functools.partial` at all.
"""

from collections.abc import Callable
from functools import cache, partial
from typing import Protocol, Union, runtime_checkable

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from astropy.utils.data import download_file
from dust_extinction.parameter_averages import G23
from dustmaps.planck import PlanckGNILCQuery
from numpy.typing import NDArray

from uvex_transients.utils import config, logger


# =========================================================================== #
# TYPING MANAGEMENT                                                           #
# =========================================================================== #
@runtime_checkable
class DustMapLike(Protocol):
    """Structural type for anything queryable for E(B-V) at a sky position, like `PlanckGNILCQuery`."""

    def query(self, coord: SkyCoord) -> float:
        """
        Query the dust map at a sky coordinate.

        Parameters
        ----------
        coord : astropy.coordinates.SkyCoord
            Sky coordinate at which to evaluate the dust map.

        Returns
        -------
        float
            Dust-map value at the specified coordinate.
        """
        ...


Reddening = Union[float, Quantity, NDArray[np.float64], DustMapLike]
"""An explicit E(B-V) value/array (`float`/dimensionless `Quantity`/`numpy.ndarray`), or a
`DustMapLike` to query at a given `~astropy.coordinates.SkyCoord` -- see `resolve_ebv`."""


# =========================================================================== #
# DUST MAPS                                                                    #
# =========================================================================== #

# Same two mirrors (IRSA, then ESA as fallback) and the same `astropy` download
# cache that m4opt's own `m4opt.synphot.extinction._dust.dust_map` uses. Reusing
# the exact URLs means that if `m4opt prime` (or any other m4opt code path) has
# already warmed the astropy download cache for this file, we get a cache hit
# instead of a second multi-hundred-megabyte download -- even though this
# loader is otherwise fully independent of m4opt. Configurable via
# ``config["physics.dust_map_sources"]`` (see `uvex_transients.utils.config`),
# in case a mirror goes stale.


@cache
def dust_map() -> PlanckGNILCQuery:
    """
    Return the Planck GNILC E(B-V) sky map, downloading and caching it on first use.

    The map is fetched via :func:`astropy.utils.data.download_file` (cached to disk,
    with the ESA mirror as a fallback source), and the resulting :class:`PlanckGNILCQuery`
    is itself process-cached via :func:`functools.cache`, so repeated calls -- even across
    millions of per-event reddening lookups -- pay the download/load cost only once.

    Returns
    -------
    dustmaps.planck.PlanckGNILCQuery
        Queryable dust map: ``dust_map().query(coord)`` returns E(B-V) at ``coord``,
        vectorized over an array-valued ``coord``.
    """
    sources = list(config["physics.dust_map_sources"])
    logger.info(
        "Fetching Planck GNILC dust map (first call only; cached to disk on success) from %s.",
        sources[0],
    )
    path = download_file(sources[-1], cache=True, sources=sources)
    logger.info("Planck GNILC dust map ready at %s.", path)
    return PlanckGNILCQuery(path)


# =========================================================================== #
# REDDENING LAW                                                               #
# =========================================================================== #
# To model the extinction for UVEX, we adopt the Gordon+2023 dust model, which covers
# IR through FUV and is suitable for our needs -- kept as a name in this registry,
# resolved through `config["physics.default_dust_law"]`, rather than hardcoded, purely
# so it's a config knob and not a code change. In practice nobody is expected to reach
# for anything but the default.
_DUST_LAWS = {"g23": G23}


@cache
def _instantiate_dust_law(name: str):
    """
    Build (and cache) the named law.

    See `get_dust_law`; split out so the cache key is always the resolved name, never `None`.
    """
    try:
        return _DUST_LAWS[name]()
    except KeyError:
        raise KeyError(f"Unknown dust law {name!r}; known laws are {sorted(_DUST_LAWS)}.") from None


def get_dust_law(dust_law: str | None = None):
    """
    Return the (cached, instantiated) reddening law to use.

    Parameters
    ----------
    dust_law : str, optional
        Name of a registered law (case-insensitive), currently just ``"g23"``
        (`dust_extinction.parameter_averages.G23`). If ``None`` (the default), the
        configured default (``config["physics.default_dust_law"]``, ``"g23"`` out of
        the box) is used.

    Returns
    -------
    dust_extinction.baseclasses.BaseExtModel
        Instantiated reddening law, exposing ``.Rv``, ``.x_range``, and
        ``law(x) -> A(x)/A(V)`` -- see `log_attenuation`.

    Raises
    ------
    KeyError
        If ``dust_law`` (or the configured default) doesn't name a registered law.
    """
    name = (dust_law or config["physics.default_dust_law"]).lower()
    return _instantiate_dust_law(name)


# =========================================================================== #
# REDDENING RESOLUTION                                                       #
# =========================================================================== #
def resolve_ebv(reddening: Reddening, coord: SkyCoord | None = None) -> NDArray[np.float64]:
    """
    Resolve a `Reddening` value into a plain E(B-V) array.

    Parameters
    ----------
    reddening : float, ~astropy.units.Quantity, numpy.ndarray, or DustMapLike
        Either an explicit :math:`E(B-V)` value/array, or an object exposing
        ``.query(coord) -> E(B-V)`` (e.g. `dust_map`'s return value).
    coord : ~astropy.coordinates.SkyCoord, optional
        Sky position(s) to query, any shape -- required if ``reddening`` is a
        `DustMapLike`; ignored (and may be omitted) otherwise.

    Returns
    -------
    numpy.ndarray
        Dimensionless E(B-V), shape matching ``reddening`` (if already a value) or
        ``coord`` (if queried from a map).

    Raises
    ------
    TypeError
        If ``reddening`` is a `DustMapLike` and ``coord`` isn't given.
    """
    if isinstance(reddening, DustMapLike):
        if coord is None:
            raise TypeError("`coord` is required to resolve a `DustMapLike` `reddening`.")
        return np.asarray(reddening.query(coord), dtype=np.float64)

    if isinstance(reddening, Quantity):
        return np.asarray(reddening.to_value(u.dimensionless_unscaled), dtype=np.float64)

    return np.asarray(reddening, dtype=np.float64)


def log_attenuation(
    nu: Quantity,
    Ebv: float | Quantity | NDArray[np.float64],
    dust_law: str | None = None,
) -> NDArray[np.float64]:
    r"""
    Natural log of the Milky Way foreground attenuation, :math:`\ln(10^{-0.4\,A(\nu)})`, at ``nu``.

    Takes an already-resolved E(B-V) -- resolving one from a dust map and a sky
    position is a separate, prior step (`resolve_ebv`), deliberately kept out of
    this function: fixing ``Ebv``'s shape once, up front, is what lets this stay a
    single, fully vectorized computation (no per-position or per-frequency Python
    loop, and, unlike building a `~synphot.SpectralElement` per position, no
    repeated `~astropy.modeling.Model` construction either) -- the reddening law's
    dimensionless shape :math:`A(\nu)/A(V)` is evaluated once at every ``nu``
    sample (shape ``(K,)``), and combined with ``Ebv`` (shape ``S``) by plain numpy
    broadcasting into a result of shape ``S + (K,)``.

    Add this directly to a `~uvex_transients.models.core.base.SpectralModel` flux
    calculation's log flux (its ``log_attenuation`` keyword) -- e.g., having already
    called ``ebv = resolve_ebv(dust_map(), coord)`` once,
    ``model.flux_log(nu, t, ..., log_attenuation=log_attenuation(nu, ebv))`` --
    rather than multiplying a linear transmission fraction in, since everything on
    that side is already computed in log space.

    `SpectralModel.as_source_spectrum`/`as_astropy_model` take the same shape one
    level removed, as a callable (their wavelength grid isn't known until the
    resulting `~synphot.SourceSpectrum` is actually called): use `attenuation_callable`
    rather than binding this function yourself, e.g.
    ``model.as_source_spectrum(t, ..., log_attenuation=attenuation_callable(ebv))``.

    Parameters
    ----------
    nu : ~astropy.units.Quantity
        Frequency(ies) (or frequency-equivalent, e.g. wavelength) to evaluate the
        reddening law at, any shape ``(K,)`` (or scalar, in which case the trailing
        frequency axis below is squeezed away).
    Ebv : float, ~astropy.units.Quantity, or numpy.ndarray
        Already-resolved, dimensionless :math:`E(B-V)`, any shape ``S`` -- see
        `resolve_ebv` for turning a dust map + sky position into this.
    dust_law : str, optional
        Passed through to `get_dust_law`; ``None`` (the default) uses the configured
        default reddening law.

    Returns
    -------
    numpy.ndarray
        Natural log of the dimensionless attenuation, ``NaN`` wherever ``nu`` falls
        outside the reddening law's native range (roughly 900 Angstrom to 32 microns
        for the default, `dust_extinction.parameter_averages.G23`). Shape
        ``S + (K,)``, or plain ``S`` if ``nu`` was scalar.
    """
    law = get_dust_law(dust_law)

    Ebv = (
        np.asarray(Ebv.to_value(u.dimensionless_unscaled), dtype=np.float64)
        if isinstance(Ebv, Quantity)
        else np.asarray(Ebv, dtype=np.float64)
    )

    x = np.atleast_1d(np.asarray(nu.to_value(1 / u.micron, equivalencies=u.spectral()), dtype=np.float64))

    # `G23.__call__` (inherited from `dust_extinction.baseclasses.BaseExtModel`) raises
    # `ValueError` if *any* input falls outside its native range, rather than returning
    # `NaN` for just the offending samples -- so out-of-range `x` is masked out *before*
    # the call. It's tempting to pass `NaN` through for those samples and let the range
    # check's own `<=`/`>=` comparisons (always `False` against `NaN`) silently wave them
    # through -- the previous version of this module did exactly that -- but verify before
    # relying on it: empirically, G23's spline evaluates `NaN` in as ``0`` (zero
    # extinction) out, not `NaN` out, so that trick only avoids the exception, it doesn't
    # produce the documented `NaN`. This masks the *output* explicitly instead, using
    # `valid` computed from `x` directly, which is correct regardless of what G23 happens
    # to do internally with a `NaN` input.
    lo, hi = law.x_range
    delta = 1e-6
    valid = (x > lo - delta) & (x < hi + delta)
    x_safe = np.where(valid, x, lo)

    axav = np.where(valid, np.asarray(law(x_safe / u.micron), dtype=np.float64), np.nan)
    Av = law.Rv.value * Ebv

    result = -0.4 * np.log(10.0) * Av[..., np.newaxis] * axav

    return result if nu.shape else result[..., 0]


def attenuation_callable(
    Ebv: float | Quantity | NDArray[np.float64],
    dust_law: str | None = None,
) -> Callable[[Quantity], NDArray[np.float64]]:
    """
    Bind `log_attenuation` to a fixed E(B-V)/dust law, as a plain ``nu -> ln(transmission)`` callable.

    This is what `~uvex_transients.models.core.base.SpectralModel.as_source_spectrum`
    and `~uvex_transients.models.core.base.SpectralModel.simulate_photometry` call
    internally, themselves, whenever their own ``ebv`` keyword is given -- callers of
    *those* methods just pass ``ebv=...`` directly and never see this function, a
    callable, or `functools.partial` at all. It's exposed here only for the rare case
    of building a bespoke :class:`~astropy.modeling.Model`/:class:`~synphot.SourceSpectrum`
    by hand, outside of `as_source_spectrum`, that still wants this module's
    already-resolved-:math:`E(B-V)`-to-callable contract (`log_attenuation(wave) ->
    ln(transmission)`, evaluated against a wavelength grid unknown until
    :mod:`synphot` samples it).

    Parameters
    ----------
    Ebv : float, ~astropy.units.Quantity, or numpy.ndarray
        Already-resolved E(B-V) -- see `resolve_ebv`.
    dust_law : str, optional
        Passed through to `log_attenuation`/`get_dust_law`. The configured default is
        almost always the right choice; this is here mainly so the resulting callable
        stays a pure function of its inputs.

    Returns
    -------
    Callable[[~astropy.units.Quantity], numpy.ndarray]
        ``nu -> log_attenuation(nu, Ebv, dust_law)``.
    """
    return partial(log_attenuation, Ebv=Ebv, dust_law=dust_law)
