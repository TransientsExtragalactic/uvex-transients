"""Cosmological utility functions."""

from typing import Union

import numpy as np
from astropy import units as u
from astropy.cosmology import Cosmology, z_at_value
from astropy.units import Quantity
from numpy.typing import NDArray

from .config import config


def get_cosmology(cosmology=None):
    """
    Return the cosmology to be used for calculations.

    If no cosmology is provided, the configured default
    (``config["physics.default_cosmology"]``, ``Planck18`` out of the box) is
    returned.

    Parameters
    ----------
    cosmology : astropy.cosmology.FLRW, optional
        Cosmology to use. If ``None``, the configured default is returned.

    Returns
    -------
    astropy.cosmology.FLRW
        Cosmology object used for subsequent calculations.

    Notes
    -----
    This helper ensures that all parts of the framework consistently
    use the same default cosmology unless explicitly overridden. The default
    itself is set via ``config["physics.default_cosmology"]``; see
    `uvex_transients.utils.config`.
    """
    if cosmology is None:
        return config["physics.default_cosmology"]
    return cosmology


def resolve_cosmological_distances(
    redshift=None,
    luminosity_distance=None,
    angular_diameter_distance=None,
    proper_distance=None,
    cosmology=None,
):
    """
    Resolve cosmological distance measures and redshift.

    Given **exactly one** of redshift or a cosmological distance,
    compute all other distance measures consistently using the
    specified cosmology.

    Parameters
    ----------
    redshift : float or array-like, optional
        Cosmological redshift.

    luminosity_distance : `~astropy.units.Quantity`, optional
        Luminosity distance :math:`D_L`.

    angular_diameter_distance : `~astropy.units.Quantity`, optional
        Angular diameter distance :math:`D_A`.

    proper_distance : `~astropy.units.Quantity`, optional
        Proper (comoving line-of-sight) distance :math:`D`.

    cosmology : `~astropy.cosmology.FLRW`, optional
        Cosmology used to compute the relations between redshift
        and distances. If ``None``, the configured default cosmology
        is used.

    Returns
    -------
    dict
        Dictionary containing

        - ``redshift`` : float or ndarray
        - ``luminosity_distance`` : `~astropy.units.Quantity`
        - ``angular_diameter_distance`` : `~astropy.units.Quantity`
        - ``proper_distance`` : `~astropy.units.Quantity`

    Raises
    ------
    ValueError
        If neither ``redshift`` nor exactly one distance is provided, unless the
        ``redshift`` + ``luminosity_distance`` shortcut below applies.

    Notes
    -----
    The relations between cosmological distances are

    .. math::

        D_L = (1+z)^2 D_A

    and

    .. math::

        D = D_C

    where :math:`D_C` is the line-of-sight comoving distance.

    Astropy internally handles these relations via the cosmology object.

    As a shortcut, ``redshift`` and ``luminosity_distance`` may be given *together*,
    as an already-consistent pair (e.g. a per-event redshift and a luminosity
    distance already interpolated off a cached grid, rather than looked up fresh
    here). In that case ``angular_diameter_distance``/
    ``proper_distance`` are derived from the two directly (:math:`D_A = D_L/(1+z)^2`,
    :math:`D_C = D_L/(1+z)`), with no cosmology lookup at all -- ``cosmology`` is
    ignored in this branch. This is what lets a `SpectralModel.flux`/`flux_band`
    call reuse an already-known distance instead of re-deriving it from ``redshift``
    on every call.

    Examples
    --------
    Resolve distances from redshift

    >>> distances = resolve_cosmological_distances(
    ...     redshift=0.5
    ... )
    >>> distances["luminosity_distance"]
    <Quantity 2919.62495218 Mpc>

    Resolve redshift from luminosity distance

    >>> import astropy.units as u
    >>> distances = resolve_cosmological_distances(
    ...     luminosity_distance=3 * u.Gpc
    ... )
    >>> distances["redshift"]
    <Quantity 0.51147033 redshift>
    """
    cosmo = get_cosmology(cosmology)

    if (
        redshift is not None
        and luminosity_distance is not None
        and angular_diameter_distance is None
        and proper_distance is None
    ):
        z = np.asarray(redshift) if not isinstance(redshift, Quantity) else redshift
        return {
            "redshift": redshift,
            "luminosity_distance": luminosity_distance,
            "angular_diameter_distance": luminosity_distance / (1 + z) ** 2,
            "proper_distance": luminosity_distance / (1 + z),
        }

    provided = [
        redshift is not None,
        luminosity_distance is not None,
        angular_diameter_distance is not None,
        proper_distance is not None,
    ]

    if sum(provided) != 1:
        raise ValueError(
            "Exactly one of redshift, luminosity_distance, angular_diameter_distance, or "
            "proper_distance must be provided (or redshift and luminosity_distance together)."
        )

    # ---------------------------------------------------
    # Determine redshift
    # ---------------------------------------------------
    if redshift is None:
        if luminosity_distance is not None:
            redshift = z_at_value(cosmo.luminosity_distance, luminosity_distance)

        elif angular_diameter_distance is not None:
            redshift = z_at_value(cosmo.angular_diameter_distance, angular_diameter_distance)

        elif proper_distance is not None:
            redshift = z_at_value(cosmo.comoving_distance, proper_distance)

    # ---------------------------------------------------
    # Compute distances from redshift
    # ---------------------------------------------------
    luminosity_distance = cosmo.luminosity_distance(redshift)
    angular_diameter_distance = cosmo.angular_diameter_distance(redshift)
    proper_distance = cosmo.comoving_distance(redshift)

    return {
        "redshift": redshift,
        "luminosity_distance": luminosity_distance,
        "angular_diameter_distance": angular_diameter_distance,
        "proper_distance": proper_distance,
    }


def redshift_to_age(z, cosmology=None):
    """
    Compute the age of the universe at a given redshift.

    Parameters
    ----------
    z : float or array-like
        Cosmological redshift.

    cosmology : `~astropy.cosmology.FLRW`, optional
        Cosmology used to evaluate the relation between redshift and cosmic
        age. If ``None``, the configured default cosmology is used.

    Returns
    -------
    `~astropy.units.Quantity`
        Age of the universe at redshift ``z`` with units of time.

    Notes
    -----
    This is equivalent to calling

    .. code-block:: python

        cosmology.age(z)

    from the Astropy cosmology API.

    Examples
    --------
    >>> redshift_to_age(1.0)
    <Quantity 5.8513433 Gyr>
    """
    cosmo = get_cosmology(cosmology)
    return cosmo.age(z)


def redshift_to_lookback_time(z, cosmology=None):
    r"""
    Compute the lookback time corresponding to a given redshift.

    Parameters
    ----------
    z : float or array-like
        Cosmological redshift.

    cosmology : `~astropy.cosmology.FLRW`, optional
        Cosmology used to compute the lookback time. If ``None``, the
        configured default cosmology is used.

    Returns
    -------
    `~astropy.units.Quantity`
        Lookback time to redshift ``z`` with units of time.

    Notes
    -----
    The lookback time is the difference between the current age of the
    universe and the age of the universe at redshift ``z``:

    .. math::

        t_{\mathrm{lookback}} = t_0 - t(z)

    Examples
    --------
    >>> redshift_to_lookback_time(0.5)
    <Quantity 5.19623953 Gyr>
    """
    cosmo = get_cosmology(cosmology)
    return cosmo.lookback_time(z)


def age_to_redshift(age, cosmology=None):
    """
    Compute the redshift corresponding to a given cosmic age.

    Parameters
    ----------
    age : `~astropy.units.Quantity`
        Age of the universe.

    cosmology : `~astropy.cosmology.FLRW`, optional
        Cosmology used to perform the inversion. If ``None``, the
        configured default cosmology is used.

    Returns
    -------
    float or ndarray
        Redshift corresponding to the provided cosmic age.

    Notes
    -----
    This function numerically inverts the cosmological age relation using
    :func:`astropy.cosmology.z_at_value`.

    Examples
    --------
    >>> import astropy.units as u
    >>> age_to_redshift(5 * u.Gyr)
    <Quantity 1.23764714 redshift>
    """
    cosmo = get_cosmology(cosmology)
    return z_at_value(cosmo.age, age)


def angular_to_physical(theta, redshift=None, cosmology=None):
    r"""
    Convert an angular size to a physical transverse size.

    Parameters
    ----------
    theta : `~astropy.units.Quantity`
        Angular size (e.g., arcsec or radians).

    redshift : float
        Redshift of the object.

    cosmology : `~astropy.cosmology.FLRW`, optional
        Cosmology used to compute the angular diameter distance.
        If ``None``, the configured default cosmology is used.

    Returns
    -------
    `~astropy.units.Quantity`
        Physical transverse size corresponding to the angular extent.

    Notes
    -----
    The physical transverse size is given by

    .. math::

        s = \theta \, D_A

    where :math:`D_A` is the angular diameter distance.

    Examples
    --------
    >>> import astropy.units as u
    >>> angular_to_physical(1 * u.arcsec, redshift=1.0)
    <Quantity 8.23125024 kpc>
    """
    distances = resolve_cosmological_distances(redshift=redshift, cosmology=cosmology)
    DA = distances["angular_diameter_distance"]
    return (theta * DA).to(u.kpc, equivalencies=u.dimensionless_angles())


def physical_to_angular(size, redshift=None, cosmology=None):
    r"""
    Convert a physical transverse size to an angular size.

    Parameters
    ----------
    size : `~astropy.units.Quantity`
        Physical transverse size.

    redshift : float
        Redshift of the object.

    cosmology : `~astropy.cosmology.FLRW`, optional
        Cosmology used to compute the angular diameter distance.
        If ``None``, the configured default cosmology is used.

    Returns
    -------
    `~astropy.units.Quantity`
        Angular size corresponding to the physical extent.

    Notes
    -----
    The angular size is given by

    .. math::

        \theta = \frac{s}{D_A}

    where :math:`D_A` is the angular diameter distance.

    Examples
    --------
    >>> import astropy.units as u
    >>> physical_to_angular(10 * u.kpc, redshift=0.5)
    <Quantity 1.58957339 arcsec>
    """
    distances = resolve_cosmological_distances(redshift=redshift, cosmology=cosmology)
    DA = distances["angular_diameter_distance"]
    return (size / DA).to(u.arcsec, equivalencies=u.dimensionless_angles())


def core_collapse_rate_coefficient(cosmology: Union[Cosmology, None] = None) -> Quantity:
    """
    Return the (redshift-independent) normalization of the total volumetric core-collapse SNe rate.

    This is the amplitude `A` in ``core_collapse_rate(z) = core_collapse_rate_coefficient()
    * core_collapse_rate_shape(z)``, i.e. the class-level ``rate`` each core-collapse subtype
    in `uvex_transients.transients.supernovae` reports (scaled by its own subtype fraction) --
    see that module.

    Parameters
    ----------
    cosmology : `~astropy.cosmology.FLRW`, optional
        Cosmology whose little Hubble parameter `h` rescales the rate (see Notes).
        If ``None``, the configured default cosmology is used; see `get_cosmology`.

    Returns
    -------
    ~astropy.units.Quantity
        The rate normalization, in events per cubic megaparsec per year.

    Notes
    -----
    The Madau & Dickinson (2014) coefficient this is built from is quoted for
    :math:`h = 0.7`; its :math:`h^2` scaling is applied here using the actual
    little Hubble parameter of ``cosmology`` rather than a hardcoded value, so the
    rate stays consistent with whatever cosmology the caller is using elsewhere.
    """
    cosmo = get_cosmology(cosmology)

    # Madau & Dickinson (2014) coefficient * CC rate from LGS 2015, rescaled by h^2
    # from the cosmology's actual little Hubble parameter rather than the h=0.7
    # value the coefficient was originally quoted at.
    return 0.0001365 * cosmo.h**2 / (u.Mpc**3 * u.yr)


def core_collapse_rate_shape(z: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
    """
    Return the dimensionless Madau & Dickinson (2014) redshift shape of the core-collapse SNe rate.

    This is the shape `f(z)` in ``core_collapse_rate(z) = core_collapse_rate_coefficient()
    * core_collapse_rate_shape(z)``; see `core_collapse_rate_coefficient`.

    Parameters
    ----------
    z : float or array-like
        Redshift(s) at which to evaluate the shape.

    Returns
    -------
    float or array-like
        The dimensionless rate shape at the specified redshift(s).
    """
    z = np.asarray(z)
    shape = (1 + z) ** 2.7 / (1 + ((1 + z) / 2.9) ** 5.6)
    return shape if z.ndim > 0 else shape.item()


def core_collapse_rate(
    z: Union[float, NDArray[np.float64]],
    cosmology: Union[Cosmology, None] = None,
) -> Union[float, NDArray[np.float64]]:
    """
    Return the total (all-subtype) volumetric core-collapse SNe rate at redshift(s) `z`.

    Shared by each core-collapse subtype class's own ``rate``/``rate_shape`` (see
    `uvex_transients.transients.supernovae`), so the per-subtype rates always stay a
    fixed fraction of the same underlying total rather than risking independent drift.
    A thin convenience wrapper around `core_collapse_rate_coefficient` *
    `core_collapse_rate_shape`.

    Parameters
    ----------
    z : float or array-like
        Redshift(s) at which to evaluate the event rate.
    cosmology : `~astropy.cosmology.FLRW`, optional
        Cosmology whose little Hubble parameter `h` rescales the rate (see
        `core_collapse_rate_coefficient`). If ``None``, the configured default
        cosmology is used; see `get_cosmology`.

    Returns
    -------
    float or array-like
        The volumetric event rate at the specified redshift(s), in events per cubic megaparsec per year.
    """
    z = np.asarray(z)
    coefficient = core_collapse_rate_coefficient(cosmology).to_value(u.Mpc**-3 * u.yr**-1)
    rate = coefficient * core_collapse_rate_shape(z)

    return rate if z.ndim > 0 else rate.item()  # Return scalar if input was scalar.
