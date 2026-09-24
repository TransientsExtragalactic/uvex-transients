"""Cosmological utility functions."""

from typing import Literal

import numpy as np
from astropy import units as u
from astropy.cosmology import Cosmology, z_at_value
from astropy.units import Quantity
from numpy.typing import NDArray

from .config import config


def get_cosmology(cosmology: Cosmology | None = None) -> Cosmology:
    """
    Return the cosmology to be used for calculations.

    If no cosmology is provided, the configured default
    (``config["physics.default_cosmology"]``, ``Planck18`` out of the box) is
    returned.

    Parameters
    ----------
    cosmology : ~astropy.cosmology.FLRW, optional
        Cosmology to use. If ``None``, the configured default is returned.

    Returns
    -------
    ~astropy.cosmology.FLRW
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
    redshift: float | NDArray[np.float64] | Quantity | None = None,
    luminosity_distance: Quantity | None = None,
    angular_diameter_distance: Quantity | None = None,
    proper_distance: Quantity | None = None,
    cosmology: Cosmology | None = None,
) -> dict[str, float | NDArray[np.float64] | Quantity]:
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


def redshift_to_age(z: float | NDArray[np.float64], cosmology: Cosmology | None = None) -> Quantity:
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


def redshift_to_lookback_time(z: float | NDArray[np.float64], cosmology: Cosmology | None = None) -> Quantity:
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


def age_to_redshift(age: Quantity, cosmology: Cosmology | None = None) -> Quantity:
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
    ~astropy.units.Quantity
        Redshift corresponding to the provided cosmic age, as a dimensionless
        `~astropy.units.Quantity` (see :func:`astropy.cosmology.z_at_value`).

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


def angular_to_physical(
    theta: Quantity, redshift: float | NDArray[np.float64] | None = None, cosmology: Cosmology | None = None
) -> Quantity:
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


def physical_to_angular(
    size: Quantity, redshift: float | NDArray[np.float64] | None = None, cosmology: Cosmology | None = None
) -> Quantity:
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


# =========================================== #
# Star formation utilities                    #
# =========================================== #
_SFR_NORMALIZATION = {
    "md14": {
        "salpeter": 0.015,
        "kroupa": 0.015 * 0.67,
        "chabrier": 0.015 * 0.63,
    },
    "mf17": {
        "salpeter": 0.010 / 0.66,
        "kroupa": 0.010,
        "chabrier": 0.010 / 0.66 * 0.63,
    },
}
"""dict: Prefactor of the cosmic star-formation-rate density (`u.Msun / (u.yr * u.Mpc**3)`), keyed by
`shape` then `imf`. Both rows are quoted for a Salpeter IMF in the literature
(:footcite:t:`madau2014`, :footcite:t:`2017ApJ...840...39M`); the Kroupa/Chabrier entries
rescale that value by the same IMF-to-IMF stellar-mass conversion factors used throughout this
module (0.67 for Kroupa, 0.63 for Chabrier, relative to Salpeter; 0.66 converts the ``"mf17"`` row's
native Chabrier-like calibration back to Salpeter before the same factors are applied).

References
----------
.. footbibliography::
"""


def _madau_dickinson_2014_shape(z: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    r"""
    Return the redshift dependence of the Madau & Dickinson (2014) cosmic star formation rate density.

    .. math::

        \psi(z) \propto
        \frac{(1+z)^{2.7}}
        {1 + [(1+z)/2.9]^{5.6}}.

    Parameters
    ----------
    z : float or numpy.ndarray
        Redshift.

    Returns
    -------
    float or numpy.ndarray
        The dimensionless shape :math:`\psi(z)`, unnormalized (see :data:`_SFR_NORMALIZATION`
        for the accompanying prefactor).

    References
    ----------
    .. footbibliography::
    """
    z = np.asarray(z, dtype=np.float64)

    shape = (1 + z) ** 2.7 / (1 + ((1 + z) / 2.9) ** 5.6)
    return shape.item() if shape.ndim == 0 else shape


def _madau_fragos_2017_shape(z: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    r"""
    Return the redshift dependence of the Madau & Fragos (2017) cosmic star formation rate density.

    .. math::

        \psi(z) \propto
        \frac{(1+z)^{2.6}}
        {1 + [(1+z)/3.2]^{6.2}}.

    Parameters
    ----------
    z : float or numpy.ndarray
        Redshift.

    Returns
    -------
    float or numpy.ndarray
        The dimensionless shape :math:`\psi(z)`, unnormalized (see :data:`_SFR_NORMALIZATION`
        for the accompanying prefactor).

    References
    ----------
    .. footbibliography::
    """
    z = np.asarray(z, dtype=np.float64)

    shape = (1 + z) ** 2.6 / (1 + ((1 + z) / 3.2) ** 6.2)
    return shape.item() if shape.ndim == 0 else shape


_SFR_SHAPES = {
    "md14": _madau_dickinson_2014_shape,
    "mf17": _madau_fragos_2017_shape,
}


def star_formation_rate(
    z: float | NDArray[np.float64],
    shape: Literal["md14", "mf17"] = "md14",
    imf: Literal["salpeter", "kroupa", "chabrier"] = "salpeter",
) -> Quantity:
    r"""
    Compute the cosmic star formation rate density.

    .. math::

        \dot{\rho}_\star(z) = \rho_0\,\psi(z),

    where :math:`\psi(z)` is one of the two dimensionless redshift shapes below and
    :math:`\rho_0` is the IMF-dependent prefactor in :data:`_SFR_NORMALIZATION`.

    Parameters
    ----------
    z : float or ~numpy.ndarray
        Redshift.
    shape : {"md14", "mf17"}, optional
        Cosmic star formation history prescription. ``"md14"`` uses
        Madau & Dickinson (2014, :footcite:t:`madau2014`), while ``"mf17"`` uses
        Madau & Fragos (2017, :footcite:t:`2017ApJ...840...39M`). Default is ``"md14"``.
    imf : {"salpeter", "kroupa", "chabrier"}, optional
        Initial mass function used to calibrate the star formation rate.
        Default is ``"salpeter"``.

    Returns
    -------
    ~astropy.units.Quantity
        Cosmic star formation rate density in
        :math:`{\rm M_\odot\,yr^{-1}\,Mpc^{-3}}`.

    Raises
    ------
    ValueError
        If `shape` or `imf` is not one of the supported options.

    References
    ----------
    .. footbibliography::
    """
    try:
        shape_function = _SFR_SHAPES[shape]
    except KeyError:
        raise ValueError(f"Unknown SFR shape {shape!r}; expected one of {tuple(_SFR_SHAPES)}.") from None

    try:
        normalization = _SFR_NORMALIZATION[shape][imf]
    except KeyError:
        raise ValueError(f"Unknown IMF {imf!r}; expected one of {tuple(_SFR_NORMALIZATION[shape])}.") from None

    return normalization * shape_function(z) * u.Msun / (u.yr * u.Mpc**3)


# =========================================== #
# Core-Collapse Rate                          #
# =========================================== #
_CORE_COLLAPSE_EFFICIENCY = {
    "salpeter": 0.0070 / u.Msun,
    "kroupa": 0.0104 / u.Msun,
}
"""dict: Number of core-collapse SNe per unit stellar mass formed, :math:`k_{\\rm CC}`, keyed by
`imf`. The Salpeter value is from Strolger et al. (2015, :footcite:t:`strolger2015`); the Kroupa
value rescales it by the same Salpeter-to-Kroupa stellar-mass conversion factor (0.67) used in
:data:`_SFR_NORMALIZATION`, so that the two IMFs predict the same total number of core-collapse SNe
per unit *observed* UV/IR luminosity.

References
----------
.. footbibliography::
"""


def core_collapse_rate(
    z: float | NDArray[np.float64],
    imf: Literal["salpeter", "kroupa"] = "salpeter",
) -> Quantity:
    r"""
    Compute the volumetric core-collapse supernova rate.

    The core-collapse rate is assumed to trace the cosmic star formation
    rate instantaneously,

    .. math::

        \mathcal{R}_{\rm CC}(z)
        =
        k_{\rm CC}\,\dot{\rho}_\star(z),

    where :math:`k_{\rm CC}` is the number of core-collapse supernovae per
    unit stellar mass formed (:data:`_CORE_COLLAPSE_EFFICIENCY`) and
    :math:`\dot{\rho}_\star(z)` is the Madau & Dickinson (2014) cosmic star formation
    rate density (`star_formation_rate`, ``shape="md14"``).

    Parameters
    ----------
    z : float or ~numpy.ndarray
        Redshift.
    imf : {"salpeter", "kroupa"}, optional
        Initial mass function used for both the star formation rate and
        core-collapse conversion. Default is ``"salpeter"``.

    Returns
    -------
    ~astropy.units.Quantity
        Volumetric core-collapse supernova rate.

    Raises
    ------
    ValueError
        If `imf` is not one of the supported options.

    References
    ----------
    .. footbibliography::
    """
    z = np.asarray(z, dtype=np.float64)

    try:
        efficiency = _CORE_COLLAPSE_EFFICIENCY[imf]
    except KeyError:
        raise ValueError(f"Unknown IMF {imf!r}; expected one of {tuple(_CORE_COLLAPSE_EFFICIENCY)}.") from None

    rate = efficiency * star_formation_rate(z, imf=imf)

    return rate.item() if rate.ndim == 0 else rate


def core_collapse_rate_coefficient(
    cosmology: Cosmology | None = None, imf: Literal["salpeter", "kroupa"] = "salpeter"
) -> Quantity:
    """
    Return the (redshift-independent) normalization of the total volumetric core-collapse SNe rate.

    This is the amplitude `A` in ``core_collapse_rate(z) = core_collapse_rate_coefficient() *
    core_collapse_rate_shape(z)``, i.e. the class-level ``rate`` each core-collapse subtype in
    `uvex_transients.transients.supernovae` reports (scaled by its own subtype fraction) -- see
    that module.

    Parameters
    ----------
    cosmology : `~astropy.cosmology.FLRW`, optional
        Accepted for interface symmetry with `get_cosmology`/`core_collapse_rate_shape`'s siblings,
        but currently unused: `core_collapse_rate` has no cosmology dependence (unlike an earlier
        version of this function, which rescaled by the little Hubble parameter :math:`h^2`). If
        ``None``, the configured default cosmology is resolved and validated but otherwise ignored.
    imf : {"salpeter", "kroupa"}, optional
        Initial mass function; see `core_collapse_rate`. Default is ``"salpeter"``.

    Returns
    -------
    ~astropy.units.Quantity
        The rate normalization, in events per cubic megaparsec per year.

    See Also
    --------
    core_collapse_rate_shape : The accompanying dimensionless redshift shape.
    core_collapse_rate : The full rate, ``coefficient() * shape(z)``.
    """
    get_cosmology(cosmology)
    return core_collapse_rate(0.0, imf=imf)


def core_collapse_rate_shape(
    z: float | NDArray[np.float64], shape: Literal["md14", "mf17"] = "md14"
) -> float | NDArray[np.float64]:
    """
    Return the dimensionless redshift shape of the core-collapse SNe rate, normalized to 1 at z=0.

    This is the shape `f(z)` in ``core_collapse_rate(z) = core_collapse_rate_coefficient() *
    core_collapse_rate_shape(z)``; see `core_collapse_rate_coefficient`.

    Parameters
    ----------
    z : float or array-like
        Redshift(s) at which to evaluate the shape.
    shape : {"md14", "mf17"}, optional
        Cosmic star formation history prescription; see `star_formation_rate`. Must match whatever
        `core_collapse_rate` is implicitly using (currently always ``"md14"``) for the product with
        `core_collapse_rate_coefficient` to reproduce `core_collapse_rate` exactly. Default is
        ``"md14"``.

    Returns
    -------
    float or array-like
        The dimensionless rate shape at the specified redshift(s), equal to 1 at ``z=0``.

    Raises
    ------
    ValueError
        If `shape` is not one of the supported options.

    See Also
    --------
    core_collapse_rate_coefficient : The accompanying redshift-independent normalization.
    core_collapse_rate : The full rate, ``coefficient() * shape(z)``.
    """
    try:
        shape_function = _SFR_SHAPES[shape]
    except KeyError:
        raise ValueError(f"Unknown SFR shape {shape!r}; expected one of {tuple(_SFR_SHAPES)}.") from None

    return shape_function(z) / shape_function(0.0)


# =========================================== #
# Type 1a Rates                               #
# =========================================== #
# The type 1a rates require a convolution over a DTD for the supernova delay and
# the SFR.
#
# Maoz & Graur (2017, :footcite:t:`2017ApJ...848...25M`): DTD ~ t^(-1.1 +/- 0.1),
# N_Ia/M = 1.3 +/- 0.1 per 1000 Msun formed, integrated from 40 Myr to 13.7 Gyr, for an
# SFH with a Kroupa-like (low-mass turnover) IMF.
_MAOZ_GRAUR_CUTOFF = 0.04  # Gyr
_MAOZ_GRAUR_SLOPE = -1.10
_MAOZ_GRAUR_NIA_PER_M = 1.3e-3  # per Msun
_MAOZ_GRAUR_T_NORM = 13.7  # Gyr; upper limit of the N_Ia/M normalization integral

_trapezoid = getattr(np, "trapezoid", None) or np.trapz
"""callable: `numpy.trapezoid` (numpy >= 2.0) or its removed predecessor `numpy.trapz`
(numpy < 2.0), whichever is available -- this package supports both (see ``numpy>=1.26``
in ``pyproject.toml``)."""


def _maoz_graur_norm(
    t_min: float = _MAOZ_GRAUR_CUTOFF,
    t_max: float = _MAOZ_GRAUR_T_NORM,
    alpha: float = _MAOZ_GRAUR_SLOPE,
    n_ia_per_m: float = _MAOZ_GRAUR_NIA_PER_M,
) -> Quantity:
    r"""
    Return the DTD amplitude :math:`A` (at 1 Gyr) for :math:`\Psi(\tau) = A\,(\tau/{\rm Gyr})^\alpha`.

    :math:`A` is fixed by requiring :math:`\int_{t_{\min}}^{t_{\max}} \Psi\,d\tau = N_{\rm Ia}/M_\star`.

    Parameters
    ----------
    t_min : float, optional
        Minimum delay time, in Gyr.
    t_max : float, optional
        Upper limit of the normalization integral, in Gyr.
    alpha : float, optional
        DTD power-law index :math:`\alpha`.
    n_ia_per_m : float, optional
        Number of SNe Ia per unit stellar mass formed, :math:`N_{\rm Ia}/M_\star`, in
        :math:`{\rm M_\odot^{-1}}`.

    Returns
    -------
    ~astropy.units.Quantity
        The DTD amplitude :math:`A`, in :math:`{\rm M_\odot^{-1}\,Gyr^{-1}}`.

    References
    ----------
    .. footbibliography::
    """
    if np.isclose(alpha, -1.0):
        integral = np.log(t_max / t_min)
    else:
        integral = (t_max ** (alpha + 1) - t_min ** (alpha + 1)) / (alpha + 1)
    return n_ia_per_m / integral / (u.Msun * u.Gyr)


def _maoz_graur_dtd(
    tau: float | NDArray[np.float64],
    t_min: float = _MAOZ_GRAUR_CUTOFF,
    t_max: float = _MAOZ_GRAUR_T_NORM,
    alpha: float = _MAOZ_GRAUR_SLOPE,
    n_ia_per_m: float = _MAOZ_GRAUR_NIA_PER_M,
) -> Quantity:
    r"""
    Evaluate the power-law SN Ia delay-time distribution.

    .. math::

        \Psi(\tau) = A \left(\frac{\tau}{\rm Gyr}\right)^{\alpha}\,\Theta(\tau - t_{\min}),

    The DTD is not truncated at ``t_max``; that only sets the normalization convention (see
    `_maoz_graur_norm`).

    Parameters
    ----------
    tau : float or numpy.ndarray
        Delay time since star formation, in Gyr.
    t_min : float, optional
        Minimum delay time, in Gyr; the DTD is zero below this.
    t_max : float, optional
        Upper limit of the normalization integral, in Gyr; see `_maoz_graur_norm`.
    alpha : float, optional
        DTD power-law index :math:`\alpha`.
    n_ia_per_m : float, optional
        Number of SNe Ia per unit stellar mass formed, :math:`N_{\rm Ia}/M_\star`, in
        :math:`{\rm M_\odot^{-1}}`.

    Returns
    -------
    ~astropy.units.Quantity
        :math:`\Psi(\tau)`, in :math:`{\rm M_\odot^{-1}\,Gyr^{-1}}`.

    References
    ----------
    .. footbibliography::
    """
    tau = np.asarray(tau, dtype=np.float64)
    amplitude = _maoz_graur_norm(t_min, t_max, alpha, n_ia_per_m)
    with np.errstate(divide="ignore", invalid="ignore"):
        shape = np.where(tau >= t_min, tau**alpha, 0.0)
    return amplitude * shape


def supernovae_Ia_rate(
    z: float | NDArray[np.float64],
    cosmology: Cosmology | None = None,
    dtd_slope: float = _MAOZ_GRAUR_SLOPE,
    t_min: float | Quantity = _MAOZ_GRAUR_CUTOFF * u.Gyr,
    n_ia_per_m: float | Quantity = _MAOZ_GRAUR_NIA_PER_M,
    t_norm: float | Quantity = _MAOZ_GRAUR_T_NORM * u.Gyr,
    sfr_shape: Literal["md14", "mf17"] = "md14",
    imf: Literal["salpeter", "kroupa", "chabrier"] = "kroupa",
    z_form: float = 20.0,
    n_tau: int = 2000,
) -> Quantity:
    r"""
    Compute the volumetric Type Ia supernova rate.

    The rate is the cosmic star formation history convolved with a power-law
    delay-time distribution (DTD),

    .. math::

        \mathcal{R}_{\rm Ia}(z)
        =
        \int_{t_{\min}}^{t(z) - t(z_{\rm form})}
        \dot{\rho}_\star\big(t(z) - \tau\big)\,\Psi(\tau)\,d\tau,
        \qquad
        \Psi(\tau) = A\left(\frac{\tau}{\rm Gyr}\right)^{\alpha}\ (\tau > t_{\min}),

    with :math:`A` set so that :math:`\int_{t_{\min}}^{t_{\rm norm}}\Psi\,d\tau
    = N_{\rm Ia}/M_\star`. Values are taken from :footcite:t:`2017ApJ...848...25M`.

    Parameters
    ----------
    z : float or ~numpy.ndarray
        Redshift.
    cosmology : `~astropy.cosmology.FLRW`, optional
        Cosmology used for the redshift--cosmic-time relation. If ``None``, the
        configured default cosmology is used; see `get_cosmology`.
    dtd_slope : float, optional
        DTD power-law index :math:`\alpha`. Default -1.1 (Maoz & Graur 2017).
    t_min : float or ~astropy.units.Quantity, optional
        Minimum delay time. Bare numbers are in Gyr. Default 40 Myr.
    n_ia_per_m : float or ~astropy.units.Quantity, optional
        Number of SNe Ia per unit stellar mass formed, :math:`N_{\rm Ia}/M_\star`,
        integrated from ``t_min`` to ``t_norm``. Bare numbers are per
        :math:`{\rm M_\odot}`. Default :math:`1.3\times10^{-3}` (Maoz & Graur 2017).
    t_norm : float or ~astropy.units.Quantity, optional
        Upper limit of the normalization integral. Bare numbers are in Gyr.
        Default 13.7 Gyr, the convention used for literature :math:`N_{\rm Ia}/M_\star`.
    sfr_shape : {"md14", "mf17"}, optional
        Cosmic star formation history; see `star_formation_rate`. Default ``"md14"``.
    imf : {"salpeter", "kroupa", "chabrier"}, optional
        IMF used to normalize the star formation rate. It must match the IMF that
        ``n_ia_per_m`` is defined for. Default ``"kroupa"``, matching the
        Maoz & Graur (2017) efficiency.
    z_form : float, optional
        Redshift at which star formation begins. Default 20.
    n_tau : int, optional
        Number of log-spaced delay-time samples used in the convolution.

    Returns
    -------
    ~astropy.units.Quantity
        Volumetric Type Ia supernova rate in :math:`{\rm yr^{-1}\,Mpc^{-3}}`.

    Raises
    ------
    ValueError
        If `t_min` is not strictly between 0 and `t_norm`.

    References
    ----------
    .. footbibliography::
    """
    cosmo = get_cosmology(cosmology)
    z = np.asarray(z, dtype=np.float64)

    t_min_gyr = u.Quantity(t_min, u.Gyr).to_value(u.Gyr)
    t_norm_gyr = u.Quantity(t_norm, u.Gyr).to_value(u.Gyr)
    n_per_m = u.Quantity(n_ia_per_m, 1 / u.Msun).to_value(1 / u.Msun)
    if not 0 < t_min_gyr < t_norm_gyr:
        raise ValueError("Require 0 < t_min < t_norm.")

    # Cosmic time <-> redshift lookup, from z = 0 to z_form (dense at low z).
    z_grid = np.concatenate([[0.0], np.geomspace(1e-4, z_form, 4000)])
    t_grid = cosmo.age(z_grid).to_value(u.Gyr)  # decreasing with z
    sfr_grid = star_formation_rate(z_grid, shape=sfr_shape, imf=imf).to_value(u.Msun / (u.yr * u.Mpc**3))
    t_inc, sfr_inc = t_grid[::-1], sfr_grid[::-1]  # increasing time for np.interp
    t_form = t_grid[-1]

    t_obs = np.atleast_1d(cosmo.age(z.ravel()).to_value(u.Gyr))
    tau_max = t_obs - t_form

    # Log-spaced delays in [t_min, tau_max] for each z: resolves the steep DTD near t_min.
    frac = np.linspace(0.0, 1.0, n_tau)
    tau = t_min_gyr * (np.maximum(tau_max, t_min_gyr)[:, None] / t_min_gyr) ** frac

    sfr_at_birth = np.interp(t_obs[:, None] - tau, t_inc, sfr_inc, left=0.0, right=0.0)
    dtd = _maoz_graur_dtd(tau, t_min=t_min_gyr, t_max=t_norm_gyr, alpha=dtd_slope, n_ia_per_m=n_per_m).to_value(
        1 / (u.Msun * u.Gyr)
    )

    rate = _trapezoid(sfr_at_birth * dtd, tau, axis=1)  # [Msun/yr/Mpc^3] * [1/(Msun Gyr)] * [Gyr]
    rate = np.where(tau_max > t_min_gyr, rate, 0.0).reshape(z.shape)

    rate = rate / (u.yr * u.Mpc**3)
    return rate.item() if rate.ndim == 0 else rate
