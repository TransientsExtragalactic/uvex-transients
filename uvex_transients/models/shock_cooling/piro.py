r"""
Analytic shock cooling emission from extended, low-mass envelopes (Piro+21).

Implements the model of :footcite:t:`2021ApJ...909..209P`. A low-mass envelope (mass :math:`M_e`, initial radius
:math:`R_e`) is given an energy :math:`E_e` by a passing shock and then expands homologously with a broken power law
density profile (outer index :math:`n`, inner index :math:`\delta`) and a constant electron scattering opacity
:math:`\kappa`. The result is a closed-form bolometric luminosity, photospheric radius, and blackbody temperature
as functions of time. It describes the first days to weeks of emission from explosions with extended material, such
as Type IIb supernovae, and is valid for :math:`n > 5` and :math:`1 < \delta < 3`. It neglects recombination and
interior heating, so the early power-law scalings are the most robust predictions. Equation numbers in this
module refer to Piro+21.

Private functions work on bare CGS floats and arrays. The public ``compute_*_piro21`` functions accept
:class:`~astropy.units.Quantity` inputs (bare numbers are assumed to be CGS), validate them, and return
:class:`~astropy.units.Quantity` outputs. :class:`PiroShockCoolingSED` turns the luminosity and temperature into a
blackbody :class:`~uvex_transients.models.core.base.SpectralModel`.

References
----------
.. footbibliography::
"""

from typing import ClassVar

import numpy as np
from astropy import units as u

from uvex_transients.models._constants import C_CGS
from uvex_transients.models._typing import CGSParameterValue, FloatArray, NumericalInput, PhysicalInput
from uvex_transients.models._util_functions import blackbody_temperature_cgs, planck_shape_log_cgs
from uvex_transients.models._utils import ensure_in_units, to_cgs_value
from uvex_transients.models.core.base import SpectralModel
from uvex_transients.models.core.parameters import Parameter
from uvex_transients.models.core.priors import ConstantPrior, LogNormalPrior

__all__ = [
    "PiroShockCoolingSED",
    "compute_ejecta_normalization_piro21",
    "compute_transition_velocity_piro21",
    "compute_diffusion_time_piro21",
    "compute_photospheric_time_piro21",
    "compute_luminosity_piro21",
    "compute_photospheric_radius_piro21",
    "compute_temperature_piro21",
]

DEFAULT_N: float = 10.0
"""float: Typical outer density power-law index (Chevalier & Soker 1989; see Appendix A of Piro+21)."""

DEFAULT_DELTA: float = 1.1
"""float: Typical inner density power-law index (Chevalier & Soker 1989)."""

_UNITS = {"t": u.s, "E_envelope": u.erg, "R_envelope": u.cm, "M_envelope": u.g, "kappa": u.cm**2 / u.g}
"""dict: The CGS unit each physical input is converted to."""


# ================================================= #
# Physics (CGS)                                     #
# ================================================= #
def _ejecta_normalization_piro21(n: FloatArray, delta: FloatArray) -> FloatArray:
    """Density normalization K from mass conservation (Eq. 3 of Piro+21)."""
    return (n - 3.0) * (3.0 - delta) / (4.0 * np.pi * (n - delta))


def _transition_velocity_cgs_piro21(
    E_envelope: FloatArray, M_envelope: FloatArray, n: FloatArray, delta: FloatArray
) -> FloatArray:
    """Transition velocity v_t between the outer and inner envelope, in cm/s (Eq. 4 of Piro+21)."""
    coefficient = ((n - 5.0) * (5.0 - delta)) / ((n - 3.0) * (3.0 - delta))
    return np.sqrt(coefficient) * np.sqrt(2.0 * E_envelope / M_envelope)


def _diffusion_time_cgs_piro21(
    E_envelope: FloatArray, M_envelope: FloatArray, kappa: FloatArray, n: FloatArray, delta: FloatArray
) -> FloatArray:
    """Time at which the diffusion depth reaches v_t, in s (Eq. 14 of Piro+21)."""
    K = _ejecta_normalization_piro21(n, delta)
    v_t = _transition_velocity_cgs_piro21(E_envelope, M_envelope, n, delta)
    return np.sqrt(3.0 * kappa * K * M_envelope / ((n - 1.0) * v_t * C_CGS))


def _photospheric_time_cgs_piro21(
    E_envelope: FloatArray, M_envelope: FloatArray, kappa: FloatArray, n: FloatArray, delta: FloatArray
) -> FloatArray:
    """Time at which the photosphere reaches v_t, in s (Eq. 8 of Piro+21)."""
    K = _ejecta_normalization_piro21(n, delta)
    v_t = _transition_velocity_cgs_piro21(E_envelope, M_envelope, n, delta)
    return np.sqrt(3.0 * kappa * K * M_envelope / (2.0 * (n - 1.0) * v_t * v_t))


def _luminosity_cgs_piro21(
    t: FloatArray,
    E_envelope: FloatArray,
    R_envelope: FloatArray,
    M_envelope: FloatArray,
    kappa: FloatArray,
    n: FloatArray,
    delta: FloatArray,
) -> FloatArray:
    """Bolometric luminosity in erg/s (Eqs. 17, 20, and 21 of Piro+21)."""
    v_t = _transition_velocity_cgs_piro21(E_envelope, M_envelope, n, delta)
    t_d = _diffusion_time_cgs_piro21(E_envelope, M_envelope, kappa, n, delta)

    prefactor = np.pi * (n - 1.0) / (3.0 * (n - 5.0)) * C_CGS * R_envelope * v_t**2 / kappa
    x = t / t_d

    # Both branches are finite for every t > 0 (the unused one at worst under or overflows harmlessly), so
    # np.where is safe and broadcasts over arbitrary parameter and time shapes.
    shape = np.where(x <= 1.0, x ** (-4.0 / (n - 2.0)), np.exp(-0.5 * (x**2 - 1.0)))
    return prefactor * shape


def _photospheric_radius_cgs_piro21(
    t: FloatArray,
    E_envelope: FloatArray,
    M_envelope: FloatArray,
    kappa: FloatArray,
    n: FloatArray,
    delta: FloatArray,
) -> FloatArray:
    """Photospheric radius in cm (Eqs. 7 and 9 of Piro+21)."""
    v_t = _transition_velocity_cgs_piro21(E_envelope, M_envelope, n, delta)
    t_ph = _photospheric_time_cgs_piro21(E_envelope, M_envelope, kappa, n, delta)
    x = t / t_ph

    # As above, both branches are finite for every t > 0: the late-time base is at least
    # 1 - (delta - 1) / (n - 1) > 0 for x < 1.
    shape = np.where(
        x <= 1.0,
        x ** (-2.0 / (n - 1.0)),
        ((delta - 1.0) / (n - 1.0) * (x**2 - 1.0) + 1.0) ** (-1.0 / (delta - 1.0)),
    )
    return shape * v_t * t


def _temperature_cgs_piro21(
    t: FloatArray,
    E_envelope: FloatArray,
    R_envelope: FloatArray,
    M_envelope: FloatArray,
    kappa: FloatArray,
    n: FloatArray,
    delta: FloatArray,
) -> FloatArray:
    """Blackbody temperature in K at the photospheric radius (Eq. 22 of Piro+21)."""
    L = _luminosity_cgs_piro21(t, E_envelope, R_envelope, M_envelope, kappa, n, delta)
    r_ph = _photospheric_radius_cgs_piro21(t, E_envelope, M_envelope, kappa, n, delta)
    return blackbody_temperature_cgs(L, r_ph)


# ================================================= #
# Input handling                                    #
# ================================================= #
def _coerce_piro21(n: NumericalInput, delta: NumericalInput, **physical: PhysicalInput) -> dict[str, FloatArray]:
    """
    Convert the inputs of a public function to CGS arrays and validate them.

    Every ``physical`` input (named as in :data:`_UNITS`) must be finite and positive, ``n`` must exceed 5, and
    ``delta`` must lie in (1, 3). The result is keyed by argument name, so it can be unpacked into a private function.
    """
    cgs = {name: ensure_in_units(value, _UNITS[name]) for name, value in physical.items()}
    for name, value in cgs.items():
        if not (np.all(np.isfinite(value)) and np.all(value > 0.0)):
            raise ValueError(f"{name} must be finite and positive")

    n = ensure_in_units(n, None)
    delta = ensure_in_units(delta, None)
    if not (np.all(np.isfinite(n)) and np.all(n > 5.0)):
        raise ValueError("n must be finite and greater than 5")
    if not (np.all(np.isfinite(delta)) and np.all(delta > 1.0) and np.all(delta < 3.0)):
        raise ValueError("delta must be finite and in the open interval (1, 3)")
    return {**cgs, "n": n, "delta": delta}


# ================================================= #
# Public API                                        #
# ================================================= #
def compute_ejecta_normalization_piro21(
    n: NumericalInput = DEFAULT_N, delta: NumericalInput = DEFAULT_DELTA
) -> np.ndarray:
    r"""
    Density normalization :math:`K` of the two-component envelope.

    Set by mass conservation (Eq. 3 of :footcite:t:`2021ApJ...909..209P`),

    .. math::

        K = \frac{(n - 3)(3 - \delta)}{4\pi (n - \delta)}.

    Parameters
    ----------
    n : array_like, optional
        Outer density power-law index, :math:`\rho \propto r^{-n}`. Must exceed 5. Default 10.
    delta : array_like, optional
        Inner density power-law index, :math:`\rho \propto r^{-\delta}`. Must lie in (1, 3). Default 1.1.

    Returns
    -------
    numpy.ndarray
        Dimensionless :math:`K` (about 0.119 for the defaults).

    References
    ----------
    .. footbibliography::
    """
    return _ejecta_normalization_piro21(**_coerce_piro21(n, delta))


def compute_transition_velocity_piro21(
    E_envelope: PhysicalInput,
    M_envelope: PhysicalInput,
    n: NumericalInput = DEFAULT_N,
    delta: NumericalInput = DEFAULT_DELTA,
) -> u.Quantity:
    r"""
    Transition velocity :math:`v_t` between the outer and inner envelope.

    From energy conservation (Eq. 4 of :footcite:t:`2021ApJ...909..209P`),

    .. math::

        v_t = \left[\frac{(n - 5)(5 - \delta)}{(n - 3)(3 - \delta)}\right]^{1/2}
              \left(\frac{2 E_e}{M_e}\right)^{1/2}.

    Parameters
    ----------
    E_envelope : ~astropy.units.Quantity or array_like
        Energy :math:`E_e` imparted to the envelope (floats are taken to be in erg).
    M_envelope : ~astropy.units.Quantity or array_like
        Envelope mass :math:`M_e` (floats are taken to be in g).
    n, delta : array_like, optional
        Density power-law indices; see :func:`compute_ejecta_normalization_piro21`.

    Returns
    -------
    ~astropy.units.Quantity
        :math:`v_t` in cm/s.

    References
    ----------
    .. footbibliography::
    """
    inputs = _coerce_piro21(n, delta, E_envelope=E_envelope, M_envelope=M_envelope)
    return _transition_velocity_cgs_piro21(**inputs) * (u.cm / u.s)


def compute_diffusion_time_piro21(
    E_envelope: PhysicalInput,
    M_envelope: PhysicalInput,
    kappa: PhysicalInput,
    n: NumericalInput = DEFAULT_N,
    delta: NumericalInput = DEFAULT_DELTA,
) -> u.Quantity:
    r"""
    Diffusion time :math:`t_d`, when the diffusion depth reaches the transition velocity.

    Eq. 14 of :footcite:t:`2021ApJ...909..209P`,

    .. math::

        t_d = \left[\frac{3\kappa K M_e}{(n - 1) v_t c}\right]^{1/2}.

    The luminosity switches from a power law to an exponential decay here.

    Parameters
    ----------
    E_envelope : ~astropy.units.Quantity or array_like
        Energy :math:`E_e` imparted to the envelope (floats are taken to be in erg).
    M_envelope : ~astropy.units.Quantity or array_like
        Envelope mass :math:`M_e` (floats are taken to be in g).
    kappa : ~astropy.units.Quantity or array_like
        Constant opacity (floats are taken to be in cm^2/g).
    n, delta : array_like, optional
        Density power-law indices; see :func:`compute_ejecta_normalization_piro21`.

    Returns
    -------
    ~astropy.units.Quantity
        :math:`t_d` in s.

    References
    ----------
    .. footbibliography::
    """
    inputs = _coerce_piro21(n, delta, E_envelope=E_envelope, M_envelope=M_envelope, kappa=kappa)
    return _diffusion_time_cgs_piro21(**inputs) * u.s


def compute_photospheric_time_piro21(
    E_envelope: PhysicalInput,
    M_envelope: PhysicalInput,
    kappa: PhysicalInput,
    n: NumericalInput = DEFAULT_N,
    delta: NumericalInput = DEFAULT_DELTA,
) -> u.Quantity:
    r"""
    Photospheric time :math:`t_\mathrm{ph}`, when the photosphere reaches the transition velocity.

    Eq. 8 of :footcite:t:`2021ApJ...909..209P`,

    .. math::

        t_\mathrm{ph} = \left[\frac{3\kappa K M_e}{2 (n - 1) v_t^2}\right]^{1/2}
                      = \left(\frac{c}{2 v_t}\right)^{1/2} t_d.

    The photospheric radius shows a break here.

    Parameters
    ----------
    E_envelope : ~astropy.units.Quantity or array_like
        Energy :math:`E_e` imparted to the envelope (floats are taken to be in erg).
    M_envelope : ~astropy.units.Quantity or array_like
        Envelope mass :math:`M_e` (floats are taken to be in g).
    kappa : ~astropy.units.Quantity or array_like
        Constant opacity (floats are taken to be in cm^2/g).
    n, delta : array_like, optional
        Density power-law indices; see :func:`compute_ejecta_normalization_piro21`.

    Returns
    -------
    ~astropy.units.Quantity
        :math:`t_\mathrm{ph}` in s.

    References
    ----------
    .. footbibliography::
    """
    inputs = _coerce_piro21(n, delta, E_envelope=E_envelope, M_envelope=M_envelope, kappa=kappa)
    return _photospheric_time_cgs_piro21(**inputs) * u.s


def compute_luminosity_piro21(
    t: PhysicalInput,
    E_envelope: PhysicalInput,
    R_envelope: PhysicalInput,
    M_envelope: PhysicalInput,
    kappa: PhysicalInput,
    n: NumericalInput = DEFAULT_N,
    delta: NumericalInput = DEFAULT_DELTA,
) -> u.Quantity:
    r"""
    Bolometric shock cooling luminosity :math:`L(t)`.

    Eqs. 17, 20, and 21 of :footcite:t:`2021ApJ...909..209P`,

    .. math::

        L(t) = \frac{\pi (n - 1)}{3 (n - 5)} \frac{c R_e v_t^2}{\kappa}
        \begin{cases}
            (t_d / t)^{4/(n - 2)} & t \le t_d \\
            \exp\left[-\frac{1}{2}\left(t^2 / t_d^2 - 1\right)\right] & t \ge t_d.
        \end{cases}

    Parameters
    ----------
    t : ~astropy.units.Quantity or array_like
        Times since explosion (floats are taken to be in s). Must be positive.
    E_envelope : ~astropy.units.Quantity or array_like
        Energy :math:`E_e` imparted to the envelope (floats are taken to be in erg).
    R_envelope : ~astropy.units.Quantity or array_like
        Initial envelope radius :math:`R_e` (floats are taken to be in cm).
    M_envelope : ~astropy.units.Quantity or array_like
        Envelope mass :math:`M_e` (floats are taken to be in g).
    kappa : ~astropy.units.Quantity or array_like
        Constant opacity (floats are taken to be in cm^2/g).
    n, delta : array_like, optional
        Density power-law indices; see :func:`compute_ejecta_normalization_piro21`.

    Returns
    -------
    ~astropy.units.Quantity
        Luminosity in erg/s. Inputs are broadcast against one another, so e.g. a time grid of shape ``(T,)``
        against parameters of shape ``(N, 1)`` gives an ``(N, T)`` result.

    Raises
    ------
    ValueError
        If ``t`` or a physical parameter is non-finite or not positive, if ``n <= 5``, or if ``delta`` is outside
        (1, 3).

    References
    ----------
    .. footbibliography::
    """
    inputs = _coerce_piro21(
        n, delta, t=t, E_envelope=E_envelope, R_envelope=R_envelope, M_envelope=M_envelope, kappa=kappa
    )
    return _luminosity_cgs_piro21(**inputs) * (u.erg / u.s)


def compute_photospheric_radius_piro21(
    t: PhysicalInput,
    E_envelope: PhysicalInput,
    M_envelope: PhysicalInput,
    kappa: PhysicalInput,
    n: NumericalInput = DEFAULT_N,
    delta: NumericalInput = DEFAULT_DELTA,
) -> u.Quantity:
    r"""
    Photospheric radius :math:`r_\mathrm{ph}(t)`, where the optical depth is 2/3.

    Eqs. 7 and 9 of :footcite:t:`2021ApJ...909..209P`,

    .. math::

        r_\mathrm{ph}(t) = v_t t
        \begin{cases}
            (t_\mathrm{ph} / t)^{2/(n - 1)} & t \le t_\mathrm{ph} \\
            \left[\frac{\delta - 1}{n - 1}\left(t^2 / t_\mathrm{ph}^2 - 1\right) + 1\right]^{-1/(\delta - 1)}
            & t \ge t_\mathrm{ph}.
        \end{cases}

    Unlike the luminosity, this does not depend on the initial radius :math:`R_e`. Since :math:`\delta` is close to
    1, the late-time branch is very sensitive to :math:`\delta`, which the paper cautions is hard to predict
    without simulations.

    Parameters
    ----------
    t : ~astropy.units.Quantity or array_like
        Times since explosion (floats are taken to be in s). Must be positive.
    E_envelope : ~astropy.units.Quantity or array_like
        Energy :math:`E_e` imparted to the envelope (floats are taken to be in erg).
    M_envelope : ~astropy.units.Quantity or array_like
        Envelope mass :math:`M_e` (floats are taken to be in g).
    kappa : ~astropy.units.Quantity or array_like
        Constant opacity (floats are taken to be in cm^2/g).
    n, delta : array_like, optional
        Density power-law indices; see :func:`compute_ejecta_normalization_piro21`.

    Returns
    -------
    ~astropy.units.Quantity
        Photospheric radius in cm, broadcast as in :func:`compute_luminosity_piro21`.

    Raises
    ------
    ValueError
        If ``t`` or a physical parameter is non-finite or not positive, if ``n <= 5``, or if ``delta`` is outside
        (1, 3).

    References
    ----------
    .. footbibliography::
    """
    inputs = _coerce_piro21(n, delta, t=t, E_envelope=E_envelope, M_envelope=M_envelope, kappa=kappa)
    return _photospheric_radius_cgs_piro21(**inputs) * u.cm


def compute_temperature_piro21(
    t: PhysicalInput,
    E_envelope: PhysicalInput,
    R_envelope: PhysicalInput,
    M_envelope: PhysicalInput,
    kappa: PhysicalInput,
    n: NumericalInput = DEFAULT_N,
    delta: NumericalInput = DEFAULT_DELTA,
) -> u.Quantity:
    r"""
    Blackbody temperature :math:`T(t)` of the photosphere.

    Eq. 22 of :footcite:t:`2021ApJ...909..209P`,

    .. math::

        T(t) = \left[\frac{L(t)}{4\pi r_\mathrm{ph}^2(t)\,\sigma_\mathrm{SB}}\right]^{1/4},

    with :math:`L` from :func:`compute_luminosity_piro21` and :math:`r_\mathrm{ph}` from
    :func:`compute_photospheric_radius_piro21`.

    Parameters
    ----------
    t : ~astropy.units.Quantity or array_like
        Times since explosion (floats are taken to be in s). Must be positive.
    E_envelope : ~astropy.units.Quantity or array_like
        Energy :math:`E_e` imparted to the envelope (floats are taken to be in erg).
    R_envelope : ~astropy.units.Quantity or array_like
        Initial envelope radius :math:`R_e` (floats are taken to be in cm).
    M_envelope : ~astropy.units.Quantity or array_like
        Envelope mass :math:`M_e` (floats are taken to be in g).
    kappa : ~astropy.units.Quantity or array_like
        Constant opacity (floats are taken to be in cm^2/g).
    n, delta : array_like, optional
        Density power-law indices; see :func:`compute_ejecta_normalization_piro21`.

    Returns
    -------
    ~astropy.units.Quantity
        Temperature in K, broadcast as in :func:`compute_luminosity_piro21`.

    Raises
    ------
    ValueError
        If ``t`` or a physical parameter is non-finite or not positive, if ``n <= 5``, or if ``delta`` is outside
        (1, 3).

    References
    ----------
    .. footbibliography::
    """
    inputs = _coerce_piro21(
        n, delta, t=t, E_envelope=E_envelope, R_envelope=R_envelope, M_envelope=M_envelope, kappa=kappa
    )
    return _temperature_cgs_piro21(**inputs) * u.K


# ================================================= #
# Model                                             #
# ================================================= #
class PiroShockCoolingSED(SpectralModel):
    r"""
    Blackbody shock cooling SED from an extended, low-mass envelope.

    Based on :footcite:t:`2021ApJ...909..209P`. A blackbody at the temperature of Eq. 22 of Piro+21, normalized to
    the bolometric luminosity of Eq. 17, so that the spectrum integrates exactly to :math:`L(t)`:

    .. math::

        L_\nu(\nu, t) = L(t)\, S\big(\nu, T(t)\big).

    See :func:`compute_luminosity_piro21` and :func:`compute_temperature_piro21` for the two ingredients. The model
    has no regime of validity of its own, but it neglects recombination and interior heating, so it describes only the
    early, shock cooling phase and falls off exponentially after the diffusion time.

    .. rubric:: Parameters

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``E_envelope``
         - :math:`E_e`
         - Energy imparted to the envelope by the shock.
       * - ``M_envelope``
         - :math:`M_e`
         - Envelope mass.
       * - ``R_envelope``
         - :math:`R_e`
         - Initial envelope radius.
       * - ``kappa``
         - :math:`\kappa`
         - Constant opacity. Fixed at the electron-scattering value.
       * - ``n``
         - :math:`n`
         - Outer density power-law index. Fixed at the typical value.
       * - ``delta``
         - :math:`\delta`
         - Inner density power-law index. Fixed at the typical value.

    Notes
    -----
    The default priors are broad log-normals around a Type IIb-like envelope, not a fit to any specific event.

    References
    ----------
    .. footbibliography::
    """

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "E_envelope": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e50 * u.erg,
            description="Energy imparted to the envelope by the shock.",
            latex=r"E_e",
        ),
        "M_envelope": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=1.0),
            scale=0.05 * u.Msun,
            description="Envelope mass.",
            latex=r"M_e",
        ),
        "R_envelope": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.85),
            scale=3e13 * u.cm,
            description="Initial envelope radius.",
            latex=r"R_e",
        ),
        "kappa": Parameter(
            prior=ConstantPrior(value=1.0),
            scale=0.34 * u.cm**2 / u.g,
            description="Constant opacity. Fixed at the electron-scattering value.",
            latex=r"\kappa",
        ),
        "n": Parameter(
            prior=ConstantPrior(value=1.0),
            scale=DEFAULT_N,
            description="Outer density power-law index. Fixed at the typical value.",
            latex=r"n",
        ),
        "delta": Parameter(
            prior=ConstantPrior(value=1.0),
            scale=DEFAULT_DELTA,
            description="Inner density power-law index. Fixed at the typical value.",
            latex=r"\delta",
        ),
    }

    @classmethod
    def temperature(cls, t: u.Quantity, **parameters: CGSParameterValue) -> u.Quantity:
        r"""
        Photospheric temperature :math:`T(t)` (Eq. 22 of Piro+21).

        Parameters
        ----------
        t : ~astropy.units.Quantity
            Time since explosion.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`T(t)`, in Kelvin.
        """
        cgs_parameters = {name: to_cgs_value(value) for name, value in parameters.items()}
        return _temperature_cgs_piro21(t.cgs.value, **cgs_parameters) * u.K

    @classmethod
    def _eval_bolometric(cls, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""Natural log of :math:`L_\mathrm{bol}(t)` in erg/s (Eqs. 17, 20, and 21 of Piro+21)."""
        return np.log(_luminosity_cgs_piro21(t, **parameters))

    @classmethod
    def _eval_spectrum(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""Natural log of the normalized blackbody shape :math:`S(\nu, T(t))` in 1/Hz."""
        # The exponential decay underflows to L = T = 0 at late times. The 1 K floor keeps the shape finite there
        # (not nan), and changes nothing since L_bol is already exactly zero wherever it applies.
        temperature = np.maximum(_temperature_cgs_piro21(t, **parameters), 1.0)
        return planck_shape_log_cgs(nu, temperature)

    @classmethod
    def _eval(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""Natural log of :math:`L_\nu(\nu, t) = L_\mathrm{bol}(t)\,S(\nu, T(t))` in erg/s/Hz."""
        return cls._eval_bolometric(t, **parameters) + cls._eval_spectrum(nu, t, **parameters)
