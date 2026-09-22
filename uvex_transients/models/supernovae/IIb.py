"""Models of SEDs from type IIb supernovae and associated scenarios."""

from collections import namedtuple
from typing import ClassVar

import numpy as np
from astropy import units as u

from uvex_transients.models._constants import (
    C_CGS,
    LOG_EV_TO_ERG,
    LOG_H_CGS,
    LOG_KELVIN_PER_EV,
    MSUN_G,
    SECONDS_PER_DAY,
    SECONDS_PER_HOUR,
)
from uvex_transients.models._typing import CGSParameterValue, FloatArray
from uvex_transients.models._util_functions import (
    cooling_temperature_cgs,
    planck_Bnu_log_cgs,
    planck_shape_log_cgs,
)
from uvex_transients.models._utils import to_cgs_value
from uvex_transients.models.core.base import SpectralModel
from uvex_transients.models.core.parameters import Parameter
from uvex_transients.models.core.priors import ConstantPrior, LogNormalPrior, NormalPrior, UniformPrior
from uvex_transients.models.lightcurves.generic import TwoComponentBazinLightcurve
from uvex_transients.models.spectra.thermal import BlackbodySpectrum

__all__ = ["MoragShockCoolingBlackbodySED", "MoragShockCoolingSED", "TypeIIbSED"]

# ================================================ #
# Types / Containers                               #
# ================================================ #
_MoragComponents = namedtuple(
    "MoragComponents",
    [
        "log_v",
        "log_R",
        "log_kappa",
        "log_M_E",
        "log_M_C",
        "log_f_rho_M",
        "log_t_break",
        "log_t_rel",
        "log_L_break",
        "log_T_break",
        "log_t_tr",
        "log_t_min",
        "log_t_max",
        "valid",
    ],
)
# ================================================ #
# Constants / Scalings                             #
# ================================================ #
# Geometric prefactor (4*pi)**2 / sqrt(3) in Eq. A9 of Morag+24.
_LOG_A9_PREFACTOR = np.log((4.0 * np.pi) ** 2 / np.sqrt(3.0))


# ================================================ #
# Utility / Component Functions                    #
# ================================================ #
def _log_morag_components(
    log_time: FloatArray,
    log_v_star: CGSParameterValue,
    log_radius: CGSParameterValue,
    log_opacity: CGSParameterValue,
    log_envelope_mass: CGSParameterValue,
    log_core_mass: CGSParameterValue,
) -> _MoragComponents:
    """
    Precompute the intermediate scalings/break points shared by Morag+24's shock-cooling relations.

    Parameters
    ----------
    log_time : ~numpy.ndarray or float
        The logarithm of the time, in seconds, since the epoch of explosion.
    log_v_star : ~numpy.ndarray or float
        The logarithm of the scale velocity of the shock near the surface, in cm/s.
    log_radius : ~numpy.ndarray or float
        The logarithm of the stellar radius, in cm.
    log_opacity : ~numpy.ndarray or float
        The logarithm of the Rosseland mean opacity, in cm^2/g.
    log_envelope_mass : ~numpy.ndarray or float
        The logarithm of the envelope mass, in g.
    log_core_mass : ~numpy.ndarray or float
        The logarithm of the core mass, in g.

    Returns
    -------
    _MoragComponents
        The precomputed scalings, break time/temperature/luminosity, validity
        time bounds, and a boolean ``valid`` mask, shared by
        :func:`_log_morag_temperature_log_K` and :func:`_log_morag_bolometric`.
    """
    # Coerce inputs. Equations in the paper are normalized against standard scalings for
    # v, R, and kappa.
    log_time = np.asarray(log_time, dtype=np.float64)
    log_v, log_R, log_kappa = (log_v_star - np.log(10**8.5), log_radius - np.log(10**13), log_opacity - np.log(0.34))
    log_M_C, log_M_E = (
        log_core_mass - np.log(MSUN_G),
        log_envelope_mass - np.log(MSUN_G),
    )

    # Compute the polytropic coefficient, which is f_rho * M in solar masses.
    log_f_rho_M = 0.5 * (log_M_E - log_M_C) + np.logaddexp(log_M_C, log_M_E)

    # Determine the break from the planar to spherical evolution phase. We maintain our
    # CGS convention for all parameters.
    log_t_break = (
        np.log(0.86) + 1.26 * log_R - 1.13 * log_v - 0.13 * (log_f_rho_M + log_kappa) + np.log(SECONDS_PER_HOUR)
    )
    log_t_rel = log_time - log_t_break

    # Determine the scale temperature at the transition point. We do this also in K.
    log_T_break = np.log(8.19) - 0.32 * log_R + 0.58 * log_v + 0.03 * log_f_rho_M - 0.22 * log_kappa + LOG_KELVIN_PER_EV

    # Compute the bolometric luminosity at the break (erg/s).
    log_L_break = np.log(3.69e42) + 0.78 * log_R + 2.11 * log_v + 0.11 * log_f_rho_M - 0.89 * log_kappa

    # Compute the validity mask.
    # Compute the relevant time cuts. These are all in base CGS units.
    _log_lt_time = log_radius - np.log(C_CGS)
    _log_bo_time = np.log(30) + 2.16 * log_R - 1.58 * log_v - 0.58 * (log_f_rho_M + log_kappa)
    _log_07_time = (
        np.log(6.86) + 0.56 * log_R + 0.16 * log_v - 0.61 * log_kappa - 0.06 * log_f_rho_M + np.log(SECONDS_PER_DAY)
    )
    _log_tr_time = np.log(19.5) + 0.5 * (log_kappa + log_M_E - log_v) + np.log(SECONDS_PER_DAY)

    # Compute the regimes of validity.
    _log_t_min = np.maximum(_log_lt_time + np.log(3), _log_bo_time)
    _log_t_max = np.minimum(_log_07_time, _log_tr_time - np.log(2))

    mask = np.logical_and(log_time > _log_t_min, log_time < _log_t_max)

    return _MoragComponents(
        log_v=log_v,
        log_R=log_R,
        log_kappa=log_kappa,
        log_M_E=log_M_E,
        log_M_C=log_M_C,
        log_f_rho_M=log_f_rho_M,
        log_t_break=log_t_break,
        log_t_rel=log_t_rel,
        log_L_break=log_L_break,
        log_T_break=log_T_break,
        log_t_tr=_log_tr_time,
        log_t_min=_log_t_min,
        log_t_max=_log_t_max,
        valid=mask,
    )


def _log_morag_temperature_log_K(
    components: _MoragComponents,
    log_time: FloatArray,
    mask_invalid: bool = False,
) -> FloatArray:
    r"""
    Color temperature (natural log, Kelvin) from precomputed Morag+24 :attr:`components`.

    Shared by :func:`_log_morag_Tcolor` (which computes :attr:`components` itself, for
    standalone use) and any caller that already has :attr:`components` in hand, to avoid
    recomputing :func:`_log_morag_components`.

    Parameters
    ----------
    components : _MoragComponents
        Precomputed scalings/break points, from :func:`_log_morag_components`.
    log_time : ~numpy.ndarray or float
        The logarithm of the time, in seconds, since the epoch of explosion.
    mask_invalid : bool, optional
        If ``True``, mask out invalid parameter ranges (see
        :attr:`_MoragComponents.valid`) with ``nan``. Default ``False``.

    Returns
    -------
    ~numpy.ndarray or float
        The natural log of the color temperature, in Kelvin.
    """
    log_t_rel = log_time - components.log_t_break
    log_T = components.log_T_break + np.minimum(
        np.log(0.97) - (1 / 3) * log_t_rel,
        -0.45 * log_t_rel,
    )

    if mask_invalid:
        log_T = np.where(components.valid, log_T, np.nan)

    return log_T


def _log_morag_bolometric(
    components: _MoragComponents,
    log_time: FloatArray,
    mask_invalid: bool = False,
) -> FloatArray:
    r"""
    Bolometric luminosity (natural log, cgs), Eq. A1 of Morag+24, from precomputed :attr:`components`.

    :math:`L(t) = L_\mathrm{break} \left[ \tilde t^{-4/3} +
    0.9\,e^{-\sqrt{2t/t_\mathrm{tr}}}\,\tilde t^{-0.17} \right]`,
    where :math:`\tilde t = t / t_\mathrm{break}`.

    Parameters
    ----------
    components : _MoragComponents
        Precomputed scalings/break points, from :func:`_log_morag_components`.
    log_time : ~numpy.ndarray or float
        The logarithm of the time, in seconds, since the epoch of explosion.
    mask_invalid : bool, optional
        If ``True``, mask out invalid parameter ranges (see
        :attr:`_MoragComponents.valid`) with ``nan``. Default ``False``.

    Returns
    -------
    ~numpy.ndarray or float
        The natural log of the bolometric luminosity, in erg/s.
    """
    log_t_rel = components.log_t_rel
    log_sqrt_arg = 0.5 * (np.log(2.0) + log_time - components.log_t_tr)
    log_L = components.log_L_break + np.logaddexp(
        -(4.0 / 3.0) * log_t_rel,
        np.log(0.9) - np.exp(log_sqrt_arg) - 0.17 * log_t_rel,
    )

    if mask_invalid:
        log_L = np.where(components.valid, log_L, np.nan)

    return log_L


def _log_morag_Tcolor(
    log_time: FloatArray,
    log_v_star: CGSParameterValue,
    log_radius: CGSParameterValue,
    log_opacity: CGSParameterValue,
    log_envelope_mass: CGSParameterValue,
    log_core_mass: CGSParameterValue,
    mask_invalid: bool = False,
):
    r"""
    Color temperature from Morag+24\ :footcite:p:`2024MNRAS.528.7137M` for shock cooling IIb's.

    Parameters
    ----------
    log_time : ~numpy.ndarray or float
        The logarithm of the time, in seconds, since the epoch of explosion.
    log_v_star : ~numpy.ndarray or float
        The logarithm of the scale velocity of the shock near the surface in :math:`{\rm cm/s}`.
        See equations (2) and (3) of :footcite:t:`2024MNRAS.528.7137M`.
    log_radius : ~numpy.ndarray or float
        The logarithm of the stellar radius in :math:`{\rm cm}`.
    log_opacity : ~numpy.ndarray or float
        The logarithm of the Rosseland mean opacity in :math:`{\rm cm^2/g}`.
    log_envelope_mass : ~numpy.ndarray or float
        The logarithm of the envelope mass in :math:`{\rm g}`.
    log_core_mass : ~numpy.ndarray or float
        The logarithm of the core mass in :math:`{\rm g}`.
    mask_invalid : bool
        If ``True``, then invalid parameter ranges will be masked out of the final result. Otherwise
        they will be retained. By default, this is ``False``.

    Returns
    -------
    ~numpy.ndarray or float
        The logarithmic temperature produced by the model.
    """
    # Coerce inputs. Equations in the paper are normalized against standard scalings for
    # v, R, and kappa.
    log_time = np.asarray(log_time, dtype=np.float64)

    components = _log_morag_components(
        log_time=log_time,
        log_v_star=log_v_star,
        log_radius=log_radius,
        log_opacity=log_opacity,
        log_envelope_mass=log_envelope_mass,
        log_core_mass=log_core_mass,
    )

    return _log_morag_temperature_log_K(components, log_time, mask_invalid=mask_invalid)


def _log_morag_Lnu(
    log_time: FloatArray,
    log_frequency: FloatArray,
    log_v_star: CGSParameterValue,
    log_radius: CGSParameterValue,
    log_opacity: CGSParameterValue,
    log_envelope_mass: CGSParameterValue,
    log_core_mass: CGSParameterValue,
    mask_invalid: bool = False,
):
    r"""
    Specific luminosity :math:`L_\nu` from Morag+24\ :footcite:p:`2024MNRAS.528.7137M` for shock cooling IIb's.

    Frequency-dependent SED of Eq. A7 -- line suppression above :math:`3.5\,T_\mathrm{col}`, plus
    the free-free correction below it. See :class:`MoragShockCoolingBlackbodySED` for the simpler
    pure-blackbody form (Eq. A8), which is bolometrically exact by construction; this is not (the
    diffusive/free-free redistribution of Eq. A9 does not conserve bolometric energy relative to a
    blackbody).

    Parameters
    ----------
    log_time : ~numpy.ndarray or float
        The logarithm of the time, in seconds, since the epoch of explosion.
    log_frequency : ~numpy.ndarray or float
        The logarithm of the (rest-frame) frequency in :math:`{\rm Hz}`.
    log_v_star : ~numpy.ndarray or float
        The logarithm of the scale velocity of the shock near the surface in :math:`{\rm cm/s}`.
        See equations (2) and (3) of :footcite:t:`2024MNRAS.528.7137M`.
    log_radius : ~numpy.ndarray or float
        The logarithm of the stellar radius in :math:`{\rm cm}`.
    log_opacity : ~numpy.ndarray or float
        The logarithm of the Rosseland mean opacity in :math:`{\rm cm^2/g}`.
    log_envelope_mass : ~numpy.ndarray or float
        The logarithm of the envelope mass in :math:`{\rm g}`.
    log_core_mass : ~numpy.ndarray or float
        The logarithm of the core mass in :math:`{\rm g}`.
    mask_invalid : bool
        If ``True``, then invalid parameter ranges will be masked out of the final result. Otherwise
        they will be retained. By default, this is ``False``.

    Returns
    -------
    ~numpy.ndarray or float
        The logarithm of the specific luminosity, in :math:`{\rm erg\,s^{-1}\,Hz^{-1}}`.
    """
    # Coerce components.
    log_time = np.asarray(log_time, dtype=np.float64)
    log_frequency = np.asarray(log_frequency, dtype=np.float64)
    nu = np.exp(log_frequency)
    components = _log_morag_components(
        log_time=log_time,
        log_v_star=log_v_star,
        log_radius=log_radius,
        log_opacity=log_opacity,
        log_envelope_mass=log_envelope_mass,
        log_core_mass=log_core_mass,
    )

    # Calculate the color temperature using the components (Kelvin -- `LOG_KELVIN_PER_EV` is folded in).
    log_T_break = components.log_T_break
    log_t_rel = components.log_t_rel
    log_T = _log_morag_temperature_log_K(components, log_time)

    # Bolometric luminosity (A1).
    log_L = _log_morag_bolometric(components, log_time)

    # The power-law scalings below (Lb, Tb, T_nu, k_ff, hnu) follow Morag+24's equations
    # exactly, which are fit with T and h*nu in eV. Only the two blackbody evaluations
    # themselves need Kelvin, so that conversion (`+ LOG_KELVIN_PER_EV`) is applied inline,
    # right where each is evaluated, rather than eV in general.
    log_T_eV = log_T - LOG_KELVIN_PER_EV
    log_T_break_eV = log_T_break - LOG_KELVIN_PER_EV
    log_hnu = LOG_H_CGS + log_frequency - LOG_EV_TO_ERG

    log_Lb = components.log_L_break - 42.5 * np.log(10.0)
    log_Tb = log_T_break_eV - np.log(5.0)
    log_k = components.log_kappa
    log_R13 = components.log_R

    # --- frequency-dependent thermal depth, low-absorption regime (A9-A12) ---
    log_r_col = np.logaddexp(
        log_radius,
        np.log(2.18e13) + 0.48 * log_Lb - 1.97 * log_Tb - 0.07 * log_k + 0.80 * log_t_rel - 0.08 * log_hnu,
    )  # A10 [cm]
    log_T_nu = (
        np.log(5.47) + 0.05 * log_Lb + 0.92 * log_Tb + 0.22 * log_k - 0.42 * log_t_rel + 0.25 * log_hnu
    )  # A11 [eV]
    log_k_ff = (
        np.log(0.03) - 0.37 * log_Lb + 0.56 * log_Tb - 0.47 * log_k - 0.19 * log_t_rel - 1.66 * log_hnu
    )  # A12 [cm^2/g]
    log_eps = log_k_ff - np.logaddexp(log_k_ff, log_opacity)
    sqrt_eps = np.exp(0.5 * log_eps)

    # B_nu(T_nu), the un-normalized Planck function; T_nu converted from eV to Kelvin.
    log_T_nu_kelvin = log_T_nu + LOG_KELVIN_PER_EV
    log_B_nu_Tnu = planck_Bnu_log_cgs(nu, np.exp(log_T_nu_kelvin))
    log_L_eps = _LOG_A9_PREFACTOR + 2.0 * log_r_col + 0.5 * log_eps - np.log1p(sqrt_eps) + log_B_nu_Tnu  # A9

    m = 5.0
    log_T_low_kelvin = np.log(0.85) + log_T_eV + LOG_KELVIN_PER_EV
    log_L_BB_low = log_L + planck_shape_log_cgs(nu, np.exp(log_T_low_kelvin))
    log_low = -(1.0 / m) * np.logaddexp(-m * log_L_BB_low, -m * log_L_eps)  # A7, h nu < 3.5 T_col

    log_time_days = log_time - np.log(SECONDS_PER_DAY)
    log_T_high_kelvin = np.log(0.85) + 0.13 * log_R13 - 0.13 * log_time_days + log_T_eV + LOG_KELVIN_PER_EV
    log_high = np.log(1.2) + log_L + planck_shape_log_cgs(nu, np.exp(log_T_high_kelvin))  # A7, h nu > 3.5 T_col

    log_L_nu = np.where(log_hnu < np.log(3.5) + log_T_eV, log_low, log_high)

    if mask_invalid:
        log_L_nu = np.where(components.valid, log_L_nu, np.nan)

    return log_L_nu


# ======================================== #
# Model: Shared Parameters                 #
# ======================================== #
class _MoragShockCoolingBase(SpectralModel):
    r"""
    Shared parameters for the Morag+24\ :footcite:p:`2024MNRAS.528.7137M` shock-cooling SED family.

    Not instantiable directly (leaves ``_eval`` abstract) -- see
    :class:`MoragShockCoolingSED` (the full UV-suppressed/free-free SED, Eq. A7) and
    :class:`MoragShockCoolingBlackbodySED` (the simpler pure-blackbody SED, Eq. A8), which share
    this parameter set and differ only in their spectral shape.

    .. rubric:: Parameters

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``v_star``
         - :math:`v_*`
         - Scale velocity of the shock near the stellar surface.
       * - ``radius``
         - :math:`R`
         - Progenitor stellar radius.
       * - ``opacity``
         - :math:`\kappa`
         - Rosseland mean opacity. Fixed at the electron-scattering value.
       * - ``envelope_mass``
         - :math:`M_E`
         - Envelope mass.
       * - ``core_mass``
         - :math:`M_C`
         - Core mass.

    References
    ----------
    .. footbibliography::
    """

    _MASK_INVALID: bool = False
    """
    bool: If ``True``, invalid points are masked out of the analysis.
    """

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "v_star": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=10**8.5 * u.cm / u.s,
            description="Scale velocity of the shock near the stellar surface.",
            latex=r"v_*",
        ),
        "radius": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e13 * u.cm,
            description="Progenitor stellar radius.",
            latex=r"R",
        ),
        "opacity": Parameter(
            prior=ConstantPrior(value=1.0),
            scale=0.34 * u.cm**2 / u.g,
            description="Rosseland mean opacity. Fixed at the electron-scattering value.",
            latex=r"\kappa",
        ),
        "envelope_mass": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1.0 * u.Msun,
            description="Envelope mass.",
            latex=r"M_E",
        ),
        "core_mass": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1.0 * u.Msun,
            description="Core mass.",
            latex=r"M_C",
        ),
    }

    @classmethod
    def _components(
        cls,
        t: FloatArray,
        *,
        v_star: CGSParameterValue,
        radius: CGSParameterValue,
        opacity: CGSParameterValue,
        envelope_mass: CGSParameterValue,
        core_mass: CGSParameterValue,
        **_ignored: CGSParameterValue,
    ) -> _MoragComponents:
        """
        Compute this SED's :func:`_log_morag_components` from its own cgs parameter values.

        Parameters
        ----------
        t : numpy.ndarray
            Time since explosion, in seconds.
        v_star, radius, opacity, envelope_mass, core_mass : float or numpy.ndarray
            This model's parameter values, in cgs units; see the class docstring.
        **_ignored
            Any other model parameter values, ignored.

        Returns
        -------
        _MoragComponents
            The precomputed scalings/break points, from :func:`_log_morag_components`.
        """
        log_time = np.log(np.asarray(t, dtype=np.float64))
        return _log_morag_components(
            log_time=log_time,
            log_v_star=np.log(v_star),
            log_radius=np.log(radius),
            log_opacity=np.log(opacity),
            log_envelope_mass=np.log(envelope_mass),
            log_core_mass=np.log(core_mass),
        )

    @classmethod
    def temperature(cls, t: u.Quantity, **parameters: CGSParameterValue) -> u.Quantity:
        r"""
        :math:`T_\mathrm{col}(t)` in Kelvin -- the color temperature shared by both subclasses.

        For :class:`MoragShockCoolingSED`, this is the underlying blackbody temperature Eq. A7
        reshapes in frequency, not a temperature that alone characterizes its emergent spectrum.

        Parameters
        ----------
        t : ~astropy.units.Quantity
            Time since explosion.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`T_\mathrm{col}(t)`, in Kelvin.
        """
        cgs_parameters: dict[str, CGSParameterValue] = {name: to_cgs_value(value) for name, value in parameters.items()}
        log_time = np.log(t.cgs.value)
        components = cls._components(t.cgs.value, **cgs_parameters)
        return np.exp(_log_morag_temperature_log_K(components, log_time, mask_invalid=cls._MASK_INVALID)) * u.K


# ======================================== #
# Model: Full UV-Suppressed SED            #
# ======================================== #
class MoragShockCoolingSED(_MoragShockCoolingBase):
    r"""
    Full UV-suppressed shock-cooling SED (Eq. A7).

    Based on Morag+24\ :footcite:p:`2024MNRAS.528.7137M`.

    The frequency-dependent SED: line suppression above :math:`3.5\,T_\mathrm{col}`, plus a
    free-free correction below it. See :class:`MoragShockCoolingBlackbodySED` for the simpler
    pure-blackbody form (Eq. A8) instead.

    Notes
    -----
    Unlike :class:`MoragShockCoolingBlackbodySED`, this SED has no closed-form bolometric
    luminosity: Eq. A7's UV-suppression/free-free reshaping of the underlying blackbody is not
    constructed to conserve the Eq. A1 diffusion-envelope :math:`L_\mathrm{bol}(t)` upon frequency
    integration (by as much as ~15-20%, depending on epoch), so :meth:`_eval_bolometric` genuinely
    needs the frequency integral of :meth:`_eval` itself, not Eq. A1. A generic adaptive
    :func:`scipy.integrate.quad_vec` over :math:`(0, \infty)` converges too slowly through Eq. A7's
    line-suppression kink to be practical, so this is overridden with a fixed, log-spaced grid in
    :math:`h\nu` (0.1-10\ :sup:`4`\  eV, comfortably spanning the Wien tail on both sides of any
    physically reasonable :math:`T_\mathrm{col}`) and a trapezoidal quadrature in
    :math:`\ln\nu` -- accurate to <0.2% against an adaptive reference across the whole validity
    window, and orders of magnitude faster.

    References
    ----------
    .. footbibliography::
    """

    #: Fixed, log-spaced grid in :math:`h\nu` [eV] used by :meth:`_eval_bolometric`'s trapezoidal
    #: quadrature -- see the class docstring's Notes for why a fixed grid, and why these bounds.
    _LOG_HNU_EV_GRID: ClassVar[FloatArray] = np.log(np.geomspace(1e-1, 1e4, 300))

    @classmethod
    def _log_Lnu(
        cls,
        nu: FloatArray,
        t: FloatArray,
        *,
        v_star: CGSParameterValue,
        radius: CGSParameterValue,
        opacity: CGSParameterValue,
        envelope_mass: CGSParameterValue,
        core_mass: CGSParameterValue,
        **_ignored: CGSParameterValue,
    ) -> FloatArray:
        """
        Compute :func:`_log_morag_Lnu` from this SED's own cgs parameter values.

        Parameters
        ----------
        nu : numpy.ndarray
            Frequency, in Hz.
        t : numpy.ndarray
            Time since explosion, in seconds.
        v_star, radius, opacity, envelope_mass, core_mass : float or numpy.ndarray
            This model's parameter values, in cgs units; see the class docstring.
        **_ignored
            Any other model parameter values, ignored.

        Returns
        -------
        numpy.ndarray
            The natural log of the specific luminosity, in erg/s/Hz.
        """
        return _log_morag_Lnu(
            log_time=np.log(t),
            log_frequency=np.log(nu),
            log_v_star=np.log(v_star),
            log_radius=np.log(radius),
            log_opacity=np.log(opacity),
            log_envelope_mass=np.log(envelope_mass),
            log_core_mass=np.log(core_mass),
            mask_invalid=cls._MASK_INVALID,
        )

    @classmethod
    def _eval(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        :math:`\log L_\nu(\nu, t)`, delegated to :func:`_log_morag_Lnu`.

        Parameters
        ----------
        nu : numpy.ndarray
            Frequency, in Hz.
        t : numpy.ndarray
            Time since explosion, in seconds.
        **parameters
            This model's parameter values, in cgs units.

        Returns
        -------
        numpy.ndarray
            The natural log of :math:`L_\nu(\nu, t)`, in erg/s/Hz.
        """
        return cls._log_Lnu(nu, t, **parameters)

    @classmethod
    def _eval_bolometric(cls, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        :math:`\log L_\mathrm{bol}(t)`, via the fixed-grid quadrature described in the class docstring.

        Parameters
        ----------
        t : numpy.ndarray
            Time since explosion, in seconds.
        **parameters
            This model's parameter values, in cgs units.

        Returns
        -------
        numpy.ndarray
            The natural log of :math:`L_\mathrm{bol}(t)`, in erg/s.
        """
        t_grid, *param_arrays = np.broadcast_arrays(np.asarray(t, dtype=np.float64), *parameters.values())
        param_grids = dict(zip(parameters, param_arrays))

        # A new leading frequency axis, broadcasting against every existing `t`/parameter axis.
        expand = (slice(None),) + (np.newaxis,) * t_grid.ndim
        log_nu_grid = (cls._LOG_HNU_EV_GRID + LOG_EV_TO_ERG - LOG_H_CGS)[expand]
        nu_grid = np.exp(log_nu_grid)

        log_L_nu = cls._eval(
            nu_grid,
            t_grid[np.newaxis, ...],
            **{name: value[np.newaxis, ...] for name, value in param_grids.items()},
        )
        # d(ln nu) substitution: integrand is L_nu * nu, trapezoidal in ln(nu).
        integrand = np.where(np.isnan(log_L_nu), 0.0, np.exp(log_L_nu) * nu_grid)
        integral = np.trapezoid(integrand, x=cls._LOG_HNU_EV_GRID, axis=0)

        with np.errstate(divide="ignore"):
            log_L = np.log(integral)

        # nan wherever every frequency sample was masked invalid (integral == 0).
        return np.where(np.any(np.isfinite(log_L_nu), axis=0), log_L, np.nan)


# ======================================== #
# Model: Pure Blackbody SED                #
# ======================================== #
class MoragShockCoolingBlackbodySED(_MoragShockCoolingBase):
    r"""
    Pure-blackbody shock-cooling SED (Eq. A8).

    Based on Morag+24\ :footcite:p:`2024MNRAS.528.7137M`.

    A pure blackbody at the color temperature :math:`T_\mathrm{col}(t)`, which integrates exactly
    to the bolometric luminosity :math:`L(t)` (Eq. A1) -- unlike :class:`MoragShockCoolingSED`,
    whose UV line-suppression/free-free redistribution (Eq. A9) does not conserve bolometric energy
    relative to a blackbody, this variant needs no numerical frequency integration to recover
    :math:`L_\mathrm{bol}(t)`.

    References
    ----------
    .. footbibliography::
    """

    @classmethod
    def _eval_bolometric(cls, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        :math:`\log L_\mathrm{bol}(t)`, Eq. A1 -- exact, no integration needed.

        Parameters
        ----------
        t : numpy.ndarray
            Time since explosion, in seconds.
        **parameters
            This model's parameter values, in cgs units.

        Returns
        -------
        numpy.ndarray
            The natural log of :math:`L_\mathrm{bol}(t)`, in erg/s.
        """
        log_time = np.log(np.asarray(t, dtype=np.float64))
        components = cls._components(t, **parameters)
        return _log_morag_bolometric(components, log_time, mask_invalid=cls._MASK_INVALID)

    @classmethod
    def _eval_spectrum(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        :math:`\log S(\nu, T_\mathrm{col}(t))`, delegated to :func:`planck_shape_log_cgs`.

        Parameters
        ----------
        nu : numpy.ndarray
            Frequency, in Hz.
        t : numpy.ndarray
            Time since explosion, in seconds.
        **parameters
            This model's parameter values, in cgs units.

        Returns
        -------
        numpy.ndarray
            The natural log of the normalized spectral shape, in 1/Hz.
        """
        log_time = np.log(np.asarray(t, dtype=np.float64))
        components = cls._components(t, **parameters)
        log_T = _log_morag_temperature_log_K(components, log_time, mask_invalid=cls._MASK_INVALID)
        return planck_shape_log_cgs(nu, np.exp(log_T))

    @classmethod
    def _eval(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        :math:`\log L_\nu(\nu, t) = \log L_\mathrm{bol}(t) + \log S(\nu, T_\mathrm{col}(t))`.

        Parameters
        ----------
        nu : numpy.ndarray
            Frequency, in Hz.
        t : numpy.ndarray
            Time since explosion, in seconds.
        **parameters
            This model's parameter values, in cgs units.

        Returns
        -------
        numpy.ndarray
            The natural log of :math:`L_\nu(\nu, t)`, in erg/s/Hz.
        """
        return cls._eval_bolometric(t, **parameters) + cls._eval_spectrum(nu, t, **parameters)


# ======================================== #
# Model: Two-Component SED                 #
# ======================================== #
class TypeIIbSED(SpectralModel):
    r"""
    Phenomenological Type IIb supernova SED.

    Two superposed Bazin pulses times a cooling blackbody photosphere. Unlike
    :class:`MoragShockCoolingSED`/:class:`MoragShockCoolingBlackbodySED`, which model only
    the early shock-cooling phase from first principles, this is a purely empirical light-curve
    shape intended to span a full Type IIb light curve. It fits real double- and single-peaked
    Type IIb events better than a single modulated pulse:

    .. math::

        L_\mathrm{bol}(t) =
        A_0\,
        \frac{\exp[-(t-t_0)/\tau_{\mathrm{fall},0}]}{1 + \exp[-(t-t_0)/\tau_{\mathrm{rise},0}]}
        +
        A_1\,
        \frac{\exp[-(t-t_1)/\tau_{\mathrm{fall},1}]}{1 + \exp[-(t-t_1)/\tau_{\mathrm{rise},1}]},

    delegated directly to :class:`~uvex_transients.models.lightcurves.generic.TwoComponentBazinLightcurve`
    -- see that class's docstring for why an additive superposition of two independent pulses
    covers both single- and double-peaked light curves. The first component (centered on
    :math:`t_0`) stands in for the early shock-cooling peak, the second (centered on :math:`t_1`)
    for the radioactively powered main/nickel peak. The photospheric temperature follows the same
    single-power-law cooling law used elsewhere in this package
    (:func:`~uvex_transients.models._util_functions.cooling_temperature_cgs`):

    .. math::

        T(t) = T_\mathrm{floor} + (T_0 - T_\mathrm{floor})\left(1 + \frac{t}{\tau_T}\right)^{-\alpha_T}.

    .. rubric:: Parameters

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude_0``
         - :math:`A_0`
         - Luminosity normalization of the early, shock-cooling peak.
       * - ``t0``
         - :math:`t_0`
         - Characteristic transition time of the early peak.
       * - ``rise_0``
         - :math:`\tau_{\mathrm{rise},0}`
         - Logistic rise timescale of the early peak.
       * - ``fall_0``
         - :math:`\tau_{\mathrm{fall},0}`
         - Exponential decline timescale of the early peak.
       * - ``amplitude_1``
         - :math:`A_1`
         - Luminosity normalization of the main, radioactively powered peak.
       * - ``t1``
         - :math:`t_1`
         - Characteristic transition time of the main peak.
       * - ``rise_1``
         - :math:`\tau_{\mathrm{rise},1}`
         - Logistic rise timescale of the main peak.
       * - ``fall_1``
         - :math:`\tau_{\mathrm{fall},1}`
         - Exponential decline timescale of the main peak.
       * - ``T0``
         - :math:`T_0`
         - Photospheric temperature at t=0 (the T(t) -> T0 limit, not literally T at peak).
       * - ``T_floor``
         - :math:`T_\mathrm{floor}`
         - Asymptotic late-time photospheric temperature (T(t) -> T_floor as t -> infinity).
       * - ``tau_T``
         - :math:`\tau_T`
         - Photospheric cooling timescale.
       * - ``alpha_T``
         - :math:`\alpha_T`
         - Photospheric cooling power-law index.

    Notes
    -----
    ``amplitude_0`` -- the early peak's normalization -- is uniform in
    :math:`\log_{10}(A_0/\mathrm{erg\,s^{-1}})` between 39 and 43, spanning several decades from
    far fainter than the main peak up to brighter than it. This single prior covers both
    populations at once: draws where the early peak ends up negligible are effectively
    single-peaked Type IIb light curves, and draws where it is comparable to or exceeds the main
    peak are double-peaked -- roughly a third of draws from the default priors below are
    double-peaked. ``t0`` and ``T_floor`` are held fixed; every other parameter is drawn from a
    broad Uniform (or, for ``amplitude_1``/``T0``, Normal-in-log) prior over an
    order-of-magnitude-motivated range, not yet a fit to any specific real Type IIb event (compare
    :class:`~uvex_transients.models.supernovae.IIp.TypeIIPSED`, whose light-curve priors do come
    from fitting real SNe).
    """

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude_0": Parameter(
            prior=UniformPrior(lower=39, upper=43),
            scale=1 * u.erg / u.s,
            transform="log10",
            description="Luminosity normalization of the early, shock-cooling peak; uniform in "
            "log10(amplitude_0 / erg/s) between 39 and 43, spanning far fainter to brighter than "
            "the main peak.",
            latex=r"A_0",
        ),
        "t0": Parameter(
            prior=ConstantPrior(value=1.0),
            scale=2.0 * u.day,
            description="Characteristic transition time of the early peak.",
            latex=r"t_0",
        ),
        "rise_0": Parameter(
            prior=UniformPrior(lower=0.3, upper=1),
            scale=1 * u.day,
            description="Logistic rise timescale of the early peak.",
            latex=r"\tau_{\mathrm{rise},0}",
        ),
        "fall_0": Parameter(
            prior=UniformPrior(lower=3, upper=10),
            scale=1 * u.day,
            description="Exponential decline timescale of the early peak.",
            latex=r"\tau_{\mathrm{fall},0}",
        ),
        "amplitude_1": Parameter(
            prior=NormalPrior(mean=42.5, sigma=0.1),
            scale=1 * u.erg / u.s,
            transform="log10",
            description="Luminosity normalization of the main, radioactively powered peak.",
            latex=r"A_1",
        ),
        "t1": Parameter(
            prior=UniformPrior(lower=10, upper=20),
            scale=1 * u.day,
            description="Characteristic transition time of the main peak.",
            latex=r"t_1",
        ),
        "rise_1": Parameter(
            prior=UniformPrior(lower=2, upper=4),
            scale=1 * u.day,
            description="Logistic rise timescale of the main peak.",
            latex=r"\tau_{\mathrm{rise},1}",
        ),
        "fall_1": Parameter(
            prior=UniformPrior(lower=30, upper=55),
            scale=1 * u.day,
            description="Exponential decline timescale of the main peak.",
            latex=r"\tau_{\mathrm{fall},1}",
        ),
        "T0": Parameter(
            prior=NormalPrior(mean=4.1, sigma=0.05),
            scale=1 * u.K,
            transform="log10",
            description="Photospheric temperature at t=0 (the T(t) -> T0 limit, not literally T at peak).",
            latex=r"T_0",
        ),
        "T_floor": Parameter(
            prior=ConstantPrior(value=1.0),
            scale=4e3 * u.K,
            description="Asymptotic late-time photospheric temperature (T(t) -> T_floor as t -> infinity).",
            latex=r"T_\mathrm{floor}",
        ),
        "tau_T": Parameter(
            prior=UniformPrior(lower=5, upper=10),
            scale=1 * u.day,
            description="Photospheric cooling timescale.",
            latex=r"\tau_T",
        ),
        "alpha_T": Parameter(
            prior=UniformPrior(lower=0.7, upper=1.3),
            scale=1.0,
            description="Photospheric cooling power-law index.",
            latex=r"\alpha_T",
        ),
    }

    # -------------------------------------- #
    # Cooling Law: T(t)                       #
    # -------------------------------------- #
    @classmethod
    def _temperature_cgs(
        cls,
        t: FloatArray,
        *,
        T0: CGSParameterValue,
        T_floor: CGSParameterValue,
        tau_T: CGSParameterValue,
        alpha_T: CGSParameterValue,
        **_ignored: CGSParameterValue,
    ) -> FloatArray:
        r"""
        :math:`T(t) = T_\mathrm{floor} + (T_0 - T_\mathrm{floor})(1 + t/\tau_T)^{-\alpha_T}`.

        Parameters
        ----------
        t : numpy.ndarray
            Time since explosion, in seconds.
        T0, T_floor, tau_T, alpha_T : float or numpy.ndarray
            This model's parameter values, in cgs units; see the class docstring.
        **_ignored
            Any other model parameter values, ignored.

        Returns
        -------
        numpy.ndarray
            :math:`T(t)`, in Kelvin.
        """
        return cooling_temperature_cgs(t, T0=T0, T_floor=T_floor, timescale=tau_T, alpha=alpha_T)

    @classmethod
    def temperature(cls, t: u.Quantity, **parameters: CGSParameterValue) -> u.Quantity:
        r"""
        :math:`T(t)` in Kelvin.

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
        cgs_parameters: dict[str, CGSParameterValue] = {name: to_cgs_value(value) for name, value in parameters.items()}
        return cls._temperature_cgs(t.cgs.value, **cgs_parameters) * u.K

    # -------------------------------------- #
    # Bolometric Luminosity: L_bol(t)         #
    # -------------------------------------- #
    @classmethod
    def _eval_bolometric(cls, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        :math:`\log L_\mathrm{bol}(t)`, delegated directly to :class:`TwoComponentBazinLightcurve`.

        Exact -- no integration needed.

        Parameters
        ----------
        t : numpy.ndarray
            Time since explosion, in seconds.
        **parameters
            This model's parameter values, in cgs units.

        Returns
        -------
        numpy.ndarray
            The natural log of :math:`L_\mathrm{bol}(t)`, in erg/s.
        """
        lightcurve_parameters = {name: parameters[name] for name in TwoComponentBazinLightcurve._DEFAULT_PARAMETERS}
        return TwoComponentBazinLightcurve._eval(t, **lightcurve_parameters)

    # -------------------------------------- #
    # Normalized Spectral Shape: S(nu, t)    #
    # -------------------------------------- #
    @classmethod
    def _eval_spectrum(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        :math:`\log S(\nu, T(t))`, delegated to :class:`~uvex_transients.models.spectra.thermal.BlackbodySpectrum`.

        Evaluated at this ``t``'s own cooling-law temperature.

        Parameters
        ----------
        nu : numpy.ndarray
            Frequency, in Hz.
        t : numpy.ndarray
            Time since explosion, in seconds.
        **parameters
            This model's parameter values, in cgs units.

        Returns
        -------
        numpy.ndarray
            The natural log of the normalized spectral shape, in 1/Hz.
        """
        temperature = cls._temperature_cgs(t, **parameters)
        return BlackbodySpectrum._eval(nu, temperature=temperature)

    # -------------------------------------- #
    # Spectral Luminosity: L_nu(nu, t)        #
    # -------------------------------------- #
    @classmethod
    def _eval(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        :math:`\log L_\nu(\nu, t) = \log L_\mathrm{bol}(t) + \log S(\nu, T(t))`.

        Parameters
        ----------
        nu : numpy.ndarray
            Frequency, in Hz.
        t : numpy.ndarray
            Time since explosion, in seconds.
        **parameters
            This model's parameter values, in cgs units.

        Returns
        -------
        numpy.ndarray
            The natural log of :math:`L_\nu(\nu, t)`, in erg/s/Hz.
        """
        return cls._eval_bolometric(t, **parameters) + cls._eval_spectrum(nu, t, **parameters)
