r"""
Shock cooling emission with deviations from blackbody (Morag+24).

Implements the analytic model of :footcite:t:`2024MNRAS.528.7137M` for the early emission of a star with an
extended envelope (mass :math:`M_E`) around a core (mass :math:`M_C`), shocked to a surface velocity scale
:math:`v_*`.

As discussed in :footcite:t:`2024MNRAS.528.7137M`, this model is suitable only for a relatively short window
between shock breakout and the onset of recombination. A detailed read of the paper should be conducted before
using this class to ensure that it is being applied properly.

References
----------
.. footbibliography::
"""

from types import SimpleNamespace
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
from uvex_transients.models._util_functions import planck_Bnu_log_cgs, planck_shape_log_cgs
from uvex_transients.models._utils import to_cgs_value
from uvex_transients.models.arnett import ArnettDecaySED
from uvex_transients.models.core.base import SpectralModel
from uvex_transients.models.core.parameters import Parameter
from uvex_transients.models.core.priors import ConstantPrior, LogNormalPrior, TruncatedNormalPrior, UniformPrior

__all__ = ["MoragShockCoolingArnettSED", "MoragShockCoolingBlackbodySED", "MoragShockCoolingSED"]

# ================================================= #
# Constants                                         #
# ================================================= #
_LOG_A9_PREFACTOR = np.log((4.0 * np.pi) ** 2 / np.sqrt(3.0))
"""float: Natural log of the geometric prefactor :math:`(4\\pi)^2 / \\sqrt{3}` of Eq. A9."""

_INVALID_FILLS = {"zero": -np.inf, "nan": np.nan, None: None}
"""dict: Natural-log value written to a luminosity outside the regime of validity, per fill setting."""


# ================================================= #
# Physics (log, CGS)                                #
# ================================================= #
def _components_log_cgs_morag24(
    log_t: FloatArray,
    log_v_star: CGSParameterValue,
    log_radius: CGSParameterValue,
    log_opacity: CGSParameterValue,
    log_envelope_mass: CGSParameterValue,
    log_core_mass: CGSParameterValue,
) -> SimpleNamespace:
    """
    Compute the scalings, break points, and regime of validity shared by all of the Morag+24 relations.

    Parameters
    ----------
    log_t : ~numpy.ndarray or float
        Natural log of the time since explosion, in s.
    log_v_star : ~numpy.ndarray or float
        Natural log of the shock velocity scale near the surface, in cm/s (Eqs. 2 and 3 of Morag+24).
    log_radius : ~numpy.ndarray or float
        Natural log of the stellar radius, in cm.
    log_opacity : ~numpy.ndarray or float
        Natural log of the Rosseland mean opacity, in cm^2/g.
    log_envelope_mass : ~numpy.ndarray or float
        Natural log of the envelope mass, in g.
    log_core_mass : ~numpy.ndarray or float
        Natural log of the core mass, in g.

    Returns
    -------
    types.SimpleNamespace
        The scalings and break points shared by the temperature, luminosity, and spectral luminosity.
        ``log_kappa`` and ``log_R`` are normalized to the fiducial scalings of the paper, ``log_t_break`` (s),
        ``log_T_break`` (K), ``log_L_break`` (erg/s), and ``log_t_tr`` (s) are in CGS, and ``valid`` is the
        boolean mask of the epochs inside the regime of validity.
    """
    # The fits are normalized to v = 10^8.5 cm/s, R = 1e13 cm, kappa = 0.34 cm^2/g, and masses in solar masses.
    log_v = log_v_star - np.log(10**8.5)
    log_R = log_radius - np.log(1e13)
    log_kappa = log_opacity - np.log(0.34)
    log_M_E = log_envelope_mass - np.log(MSUN_G)
    log_M_C = log_core_mass - np.log(MSUN_G)

    # The polytropic coefficient f_rho times the total mass in solar masses.
    log_f_rho_M = 0.5 * (log_M_E - log_M_C) + np.logaddexp(log_M_C, log_M_E)

    # The break from the planar to the spherical phase, and the temperature and luminosity there.
    log_t_break = (
        np.log(0.86) + 1.26 * log_R - 1.13 * log_v - 0.13 * (log_f_rho_M + log_kappa) + np.log(SECONDS_PER_HOUR)
    )
    log_T_break = np.log(8.19) - 0.32 * log_R + 0.58 * log_v + 0.03 * log_f_rho_M - 0.22 * log_kappa + LOG_KELVIN_PER_EV
    log_L_break = np.log(3.69e42) + 0.78 * log_R + 2.11 * log_v + 0.11 * log_f_rho_M - 0.89 * log_kappa

    # The regime of validity (Eqs. 17 and 18 of Morag+24): after the light crossing (times 3) and breakout times, and
    # before t_0.7 and half the transparency time.
    log_t_tr = np.log(19.5) + 0.5 * (log_kappa + log_M_E - log_v) + np.log(SECONDS_PER_DAY)
    log_t_light = log_radius - np.log(C_CGS)
    log_t_breakout = np.log(30) + 2.16 * log_R - 1.58 * log_v - 0.58 * (log_f_rho_M + log_kappa)
    log_t_07 = (
        np.log(6.86) + 0.56 * log_R + 0.16 * log_v - 0.61 * log_kappa - 0.06 * log_f_rho_M + np.log(SECONDS_PER_DAY)
    )
    log_t_min = np.maximum(log_t_light + np.log(3), log_t_breakout)
    log_t_max = np.minimum(log_t_07, log_t_tr - np.log(2))

    return SimpleNamespace(
        log_kappa=log_kappa,
        log_R=log_R,
        log_t_break=log_t_break,
        log_T_break=log_T_break,
        log_L_break=log_L_break,
        log_t_tr=log_t_tr,
        valid=np.logical_and(log_t > log_t_min, log_t < log_t_max),
    )


def _temperature_log_cgs_morag24(
    log_t: FloatArray,
    log_v_star: CGSParameterValue,
    log_radius: CGSParameterValue,
    log_opacity: CGSParameterValue,
    log_envelope_mass: CGSParameterValue,
    log_core_mass: CGSParameterValue,
    log_fill: float | None = None,
) -> FloatArray:
    """
    Natural log of the color temperature in K.

    Parameters
    ----------
    log_t, log_v_star, log_radius, log_opacity, log_envelope_mass, log_core_mass : ~numpy.ndarray or float
        As in :func:`_components_log_cgs_morag24`.
    log_fill : float, optional
        If given, the natural-log value written outside the regime of validity. By default the formula is
        evaluated everywhere.
    """
    components = _components_log_cgs_morag24(
        log_t, log_v_star, log_radius, log_opacity, log_envelope_mass, log_core_mass
    )
    log_t_rel = log_t - components.log_t_break
    log_T = components.log_T_break + np.minimum(np.log(0.97) - (1 / 3) * log_t_rel, -0.45 * log_t_rel)
    return log_T if log_fill is None else np.where(components.valid, log_T, log_fill)


def _luminosity_log_cgs_morag24(
    log_t: FloatArray,
    log_v_star: CGSParameterValue,
    log_radius: CGSParameterValue,
    log_opacity: CGSParameterValue,
    log_envelope_mass: CGSParameterValue,
    log_core_mass: CGSParameterValue,
    log_fill: float | None = None,
) -> FloatArray:
    r"""
    Natural log of the bolometric luminosity in erg/s (Eq. A1 of Morag+24).

    .. math::

        L(t) = L_\mathrm{break} \left[ \tilde t^{-4/3} + 0.9\,e^{-\sqrt{2t/t_\mathrm{tr}}}\,\tilde t^{-0.17} \right],
        \quad \tilde t = t / t_\mathrm{break}.

    Parameters
    ----------
    log_t, log_v_star, log_radius, log_opacity, log_envelope_mass, log_core_mass : ~numpy.ndarray or float
        As in :func:`_components_log_cgs_morag24`.
    log_fill : float, optional
        If given, the natural-log value written outside the regime of validity (``-inf`` for zero luminosity,
        ``nan`` for undefined). By default the formula is evaluated everywhere, including where it is known to
        be wrong.
    """
    components = _components_log_cgs_morag24(
        log_t, log_v_star, log_radius, log_opacity, log_envelope_mass, log_core_mass
    )
    log_t_rel = log_t - components.log_t_break
    sqrt_2t_over_t_tr = np.exp(0.5 * (np.log(2.0) + log_t - components.log_t_tr))
    log_L = components.log_L_break + np.logaddexp(
        -(4.0 / 3.0) * log_t_rel,
        np.log(0.9) - sqrt_2t_over_t_tr - 0.17 * log_t_rel,
    )
    return log_L if log_fill is None else np.where(components.valid, log_L, log_fill)


def _spectral_luminosity_log_cgs_morag24(
    log_t: FloatArray,
    log_nu: FloatArray,
    log_v_star: CGSParameterValue,
    log_radius: CGSParameterValue,
    log_opacity: CGSParameterValue,
    log_envelope_mass: CGSParameterValue,
    log_core_mass: CGSParameterValue,
    log_fill: float | None = None,
) -> FloatArray:
    r"""
    Natural log of the full specific luminosity :math:`L_\nu` in erg/s/Hz (Eq. A7 of Morag+24).

    A blackbody at the color temperature with the UV suppressed above :math:`h\nu = 3.5\,T_\mathrm{col}`, plus a
    free-free correction below it (Eqs. A9 to A12). Unlike the pure blackbody (Eq. A8), this does not integrate back
    to the bolometric luminosity of Eq. A1.

    Parameters
    ----------
    log_t : ~numpy.ndarray or float
        Natural log of the time since explosion, in s.
    log_nu : ~numpy.ndarray or float
        Natural log of the rest-frame frequency, in Hz.
    log_v_star, log_radius, log_opacity, log_envelope_mass, log_core_mass : ~numpy.ndarray or float
        Natural log of the model parameters in CGS; see :func:`_components_log_cgs_morag24`.
    log_fill : float, optional
        As in :func:`_luminosity_log_cgs_morag24`.
    """
    nu = np.exp(log_nu)
    parameters = (log_t, log_v_star, log_radius, log_opacity, log_envelope_mass, log_core_mass)
    components = _components_log_cgs_morag24(*parameters)
    log_t_rel = log_t - components.log_t_break
    log_T = _temperature_log_cgs_morag24(*parameters)
    log_L = _luminosity_log_cgs_morag24(*parameters)

    # Eqs. A9 to A12 are fit with T and h*nu in eV, so only the blackbody evaluations below convert to Kelvin.
    log_T_eV = log_T - LOG_KELVIN_PER_EV
    log_Tb = components.log_T_break - LOG_KELVIN_PER_EV - np.log(5.0)
    log_Lb = components.log_L_break - 42.5 * np.log(10.0)
    log_hnu = LOG_H_CGS + log_nu - LOG_EV_TO_ERG
    log_k = components.log_kappa

    # The thermal depth in the low-absorption regime: radius [cm], temperature [eV], free-free opacity [cm^2/g].
    log_r_col = np.logaddexp(
        log_radius,
        np.log(2.18e13) + 0.48 * log_Lb - 1.97 * log_Tb - 0.07 * log_k + 0.80 * log_t_rel - 0.08 * log_hnu,
    )  # Eq. A10
    log_T_nu = np.log(5.47) + 0.05 * log_Lb + 0.92 * log_Tb + 0.22 * log_k - 0.42 * log_t_rel + 0.25 * log_hnu  # A11
    log_k_ff = np.log(0.03) - 0.37 * log_Lb + 0.56 * log_Tb - 0.47 * log_k - 0.19 * log_t_rel - 1.66 * log_hnu  # A12
    log_eps = log_k_ff - np.logaddexp(log_k_ff, log_opacity)
    sqrt_eps = np.exp(0.5 * log_eps)

    log_B_nu = planck_Bnu_log_cgs(nu, np.exp(log_T_nu + LOG_KELVIN_PER_EV))
    log_L_eps = _LOG_A9_PREFACTOR + 2.0 * log_r_col + 0.5 * log_eps - np.log1p(sqrt_eps) + log_B_nu  # Eq. A9

    # Eq. A7 for h*nu < 3.5 T_col: a smooth minimum (m = 5) of the blackbody and the free-free emission.
    m = 5.0
    log_L_blackbody = log_L + planck_shape_log_cgs(nu, np.exp(np.log(0.85) + log_T_eV + LOG_KELVIN_PER_EV))
    log_low = -(1.0 / m) * np.logaddexp(-m * log_L_blackbody, -m * log_L_eps)

    # Eq. A7 for h*nu > 3.5 T_col: a cooler blackbody with line suppression.
    log_t_days = log_t - np.log(SECONDS_PER_DAY)
    log_T_high = np.log(0.85) + 0.13 * components.log_R - 0.13 * log_t_days + log_T_eV + LOG_KELVIN_PER_EV
    log_high = np.log(1.2) + log_L + planck_shape_log_cgs(nu, np.exp(log_T_high))

    log_L_nu = np.where(log_hnu < np.log(3.5) + log_T_eV, log_low, log_high)
    return log_L_nu if log_fill is None else np.where(components.valid, log_L_nu, log_fill)


# ================================================= #
# Models                                            #
# ================================================= #
class _MoragShockCoolingBase(SpectralModel):
    r"""
    Parameters and regime of validity shared by the Morag+24 shock cooling SEDs.

    Not instantiable directly (``_eval`` is left abstract). :class:`MoragShockCoolingSED` and
    :class:`MoragShockCoolingBlackbodySED` share this parameter set and differ only in their spectral shape.
    Based on :footcite:t:`2024MNRAS.528.7137M`.

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
         - Hydrogen-rich envelope mass.
       * - ``core_mass``
         - :math:`M_C`
         - Core mass.

    .. rubric:: Regime of validity

    The relations hold only between :math:`t_\mathrm{min} = \max(3R/c, t_\mathrm{bo})` and
    :math:`t_\mathrm{max} = \min(t_{0.7}, t_\mathrm{tr}/2)` (Eqs. 17 and 18 of Morag+24). For Type IIb-like
    parameters that is only about a day or two. Outside it the luminosity is not just inaccurate: it diverges
    toward early times, reaching :math:`10^{45}\,\mathrm{erg\,s^{-1}}` in the unmasked tail of the priors. What is
    written there is set by ``_INVALID_FILL``.

    References
    ----------
    .. footbibliography::
    """

    _INVALID_FILL: ClassVar[str | None] = "zero"
    """
    str or None: What to write where the model is outside its regime of validity.

    ``"zero"`` gives zero luminosity (``-inf`` in log space), so an epoch the model cannot describe reads as
    a non-detection and cannot poison a minimum or maximum taken over epochs. ``"nan"`` marks it undefined, for
    plots and diagnostics. ``None`` evaluates the formulas everywhere, including where they are known to be
    wrong (the luminosity grows without bound toward early times, so this overstates the early emission by orders
    of magnitude). The color temperature is always ``nan`` outside the regime of validity unless this is ``None``.
    """

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "v_star": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.35),
            scale=2.0 * 10**8.5 * u.cm / u.s,
            description="Scale velocity of the shock near the stellar surface.",
            latex=r"v_*",
        ),
        "radius": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.85),
            scale=6e12 * u.cm,
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
            prior=LogNormalPrior(mean=0.0, sigma=1.0),
            scale=0.05 * u.Msun,
            description="Hydrogen-rich envelope mass.",
            latex=r"M_E",
        ),
        "core_mass": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.35),
            scale=2.8 * u.Msun,
            description="Core mass.",
            latex=r"M_C",
        ),
    }

    @classmethod
    def _log_fill(cls) -> float | None:
        """
        Resolve ``_INVALID_FILL`` to the natural-log value written to an invalid luminosity.

        Returns
        -------
        float or None
            ``-inf``, ``nan``, or ``None`` when masking is off.

        Raises
        ------
        ValueError
            If ``_INVALID_FILL`` is not ``"zero"``, ``"nan"`` or ``None``.
        """
        if cls._INVALID_FILL not in _INVALID_FILLS:
            raise ValueError(
                f"_INVALID_FILL must be one of {sorted(map(str, _INVALID_FILLS))}, got {cls._INVALID_FILL!r}."
            )
        return _INVALID_FILLS[cls._INVALID_FILL]

    @staticmethod
    def _log_arguments(
        t: FloatArray,
        *,
        v_star: CGSParameterValue,
        radius: CGSParameterValue,
        opacity: CGSParameterValue,
        envelope_mass: CGSParameterValue,
        core_mass: CGSParameterValue,
        **_ignored: CGSParameterValue,
    ) -> dict[str, FloatArray]:
        """Natural logs of the CGS time and parameters, keyed by the argument names of the Morag+24 functions."""
        return {
            "log_t": np.log(t),
            "log_v_star": np.log(v_star),
            "log_radius": np.log(radius),
            "log_opacity": np.log(opacity),
            "log_envelope_mass": np.log(envelope_mass),
            "log_core_mass": np.log(core_mass),
        }

    @classmethod
    def temperature(cls, t: u.Quantity, **parameters: CGSParameterValue) -> u.Quantity:
        r"""
        Color temperature :math:`T_\mathrm{col}(t)`, shared by both SEDs.

        For :class:`MoragShockCoolingSED` this is the underlying blackbody temperature that Eq. A7 of
        Morag+24 reshapes in frequency, not a temperature that alone characterizes its spectrum.

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
        cgs_parameters = {name: to_cgs_value(value) for name, value in parameters.items()}
        log_fill = None if cls._log_fill() is None else np.nan
        log_T = _temperature_log_cgs_morag24(log_fill=log_fill, **cls._log_arguments(t.cgs.value, **cgs_parameters))
        return np.exp(log_T) * u.K


class MoragShockCoolingSED(_MoragShockCoolingBase):
    r"""
    Full shock cooling SED with UV line suppression and a free-free correction (Eq. A7).

    Based on :footcite:t:`2024MNRAS.528.7137M`. See :class:`MoragShockCoolingBlackbodySED` for the simpler
    pure blackbody (Eq. A8).

    Notes
    -----
    This SED has no closed-form bolometric luminosity. The reshaping of Eq. A7 does not conserve the Eq. A1
    luminosity upon frequency integration (by as much as 15 to 20 percent, depending on epoch), so
    :meth:`_eval_bolometric` integrates :meth:`_eval` itself. A generic adaptive integral over
    :math:`(0, \infty)` converges too slowly through the line suppression kink to be practical, so it uses a fixed,
    log-spaced grid in :math:`h\nu` from 0.1 eV to :math:`10^4` eV (comfortably spanning the Wien tail on both
    sides of any physically reasonable :math:`T_\mathrm{col}`) with a trapezoidal rule in :math:`\ln\nu`. This
    agrees with an adaptive reference to better than 0.2 percent across the whole regime of validity, and is
    orders of magnitude faster.

    References
    ----------
    .. footbibliography::
    """

    _LOG_HNU_EV_GRID: ClassVar[FloatArray] = np.log(np.geomspace(1e-1, 1e4, 300))
    """numpy.ndarray: Natural log of the :math:`h\\nu` grid in eV used by :meth:`_eval_bolometric`."""

    @classmethod
    def _eval(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""Natural log of :math:`L_\nu(\nu, t)` in erg/s/Hz, from :func:`_spectral_luminosity_log_cgs_morag24`."""
        return _spectral_luminosity_log_cgs_morag24(
            log_nu=np.log(nu), log_fill=cls._log_fill(), **cls._log_arguments(t, **parameters)
        )

    @classmethod
    def _eval_bolometric(cls, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""Natural log of :math:`L_\mathrm{bol}(t)` in erg/s, by the fixed grid quadrature of the class notes."""
        t_grid, *param_arrays = np.broadcast_arrays(np.asarray(t, dtype=np.float64), *parameters.values())
        param_grids = dict(zip(parameters, param_arrays))

        # A new leading frequency axis, broadcasting against every existing `t`/parameter axis.
        expand = (slice(None),) + (np.newaxis,) * t_grid.ndim
        nu_grid = np.exp((cls._LOG_HNU_EV_GRID + LOG_EV_TO_ERG - LOG_H_CGS)[expand])

        log_L_nu = cls._eval(
            nu_grid,
            t_grid[np.newaxis, ...],
            **{name: value[np.newaxis, ...] for name, value in param_grids.items()},
        )
        # With d(ln nu) the integrand is L_nu * nu, integrated by the trapezoidal rule in ln(nu).
        integrand = np.where(np.isnan(log_L_nu), 0.0, np.exp(log_L_nu) * nu_grid)
        integral = np.trapezoid(integrand, x=cls._LOG_HNU_EV_GRID, axis=0)

        with np.errstate(divide="ignore"):
            log_L = np.log(integral)

        # With a zero fill an invalid epoch integrates to exactly zero (log_L == -inf). Otherwise it is nan
        # wherever every frequency sample was masked invalid.
        if cls._INVALID_FILL == "zero":
            return log_L
        return np.where(np.any(np.isfinite(log_L_nu), axis=0), log_L, np.nan)


class MoragShockCoolingBlackbodySED(_MoragShockCoolingBase):
    r"""
    Pure blackbody shock cooling SED (Eq. A8).

    Based on :footcite:t:`2024MNRAS.528.7137M`. A blackbody at the color temperature :math:`T_\mathrm{col}(t)`
    that integrates exactly to the bolometric luminosity :math:`L(t)` of Eq. A1, so it needs no numerical frequency
    integration. See :class:`MoragShockCoolingSED` for the full SED (Eq. A7), which does not conserve it.

    References
    ----------
    .. footbibliography::
    """

    @classmethod
    def _eval_bolometric(cls, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""Natural log of :math:`L_\mathrm{bol}(t)` in erg/s, Eq. A1 of Morag+24 (exact, no integration)."""
        return _luminosity_log_cgs_morag24(log_fill=cls._log_fill(), **cls._log_arguments(t, **parameters))

    @classmethod
    def _eval_spectrum(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""Natural log of the normalized blackbody shape :math:`S(\nu, T_\mathrm{col}(t))` in 1/Hz."""
        # Only a nan fill masks the shape. For a zero fill it stays finite, because the bolometric luminosity it is
        # added to in `_eval` is already -inf, and -inf + nan is nan.
        log_fill = np.nan if cls._INVALID_FILL == "nan" else None
        log_T = _temperature_log_cgs_morag24(log_fill=log_fill, **cls._log_arguments(t, **parameters))
        return planck_shape_log_cgs(nu, np.exp(log_T))

    @classmethod
    def _eval(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""Natural log of :math:`L_\nu(\nu, t) = L_\mathrm{bol}(t)\,S(\nu, T_\mathrm{col}(t))` in erg/s/Hz."""
        return cls._eval_bolometric(t, **parameters) + cls._eval_spectrum(nu, t, **parameters)


class _UncutMoragShockCoolingSED(MoragShockCoolingSED):
    """The full Morag+24 SED evaluated at every epoch, ignoring its regime of validity."""

    _INVALID_FILL = None


class MoragShockCoolingArnettSED(SpectralModel):
    r"""
    Type IIb SED: Morag+24 shock cooling plus Arnett nickel decay.

    The sum of two emission components, with the spectral luminosities added,

    .. math::

        L_\nu(\nu, t) = L_\nu^\mathrm{shock}(\nu, t) + L_\nu^\mathrm{Ni}(\nu, t).

    The first is the full shock cooling SED of :footcite:t:`2024MNRAS.528.7137M` (:class:`MoragShockCoolingSED`,
    Eq. A7), which dominates in the first days. The second is the diffusion-limited, nickel powered light curve of
    :footcite:t:`1982ApJ...253..785A` with a floored photospheric blackbody
    (:class:`~uvex_transients.models.arnett.ArnettDecaySED`), which produces the main peak at about 20 days.
    Together they give the characteristic Type IIb double structure: a luminous decline after breakout, a trough,
    then the rise to the nickel peak.

    The shock cooling component is evaluated at every epoch, without the cut at the edges of the regime of validity
    of Morag+24 that :class:`MoragShockCoolingSED` applies by default. The cut zeroes the shock cooling after about a
    day or two, which leaves the nickel component alone through the trough. The price is that the formulas are
    not valid before :math:`t_\mathrm{min}`, where the luminosity grows without bound toward :math:`t \to 0`.

    .. rubric:: Parameters

    The five parameters of :class:`MoragShockCoolingSED` (``v_star``, ``radius``, ``opacity``, ``envelope_mass``,
    ``core_mass``) and the six of :class:`~uvex_transients.models.arnett.ArnettDecaySED` (``M_Ni``, ``M_ej``,
    ``v_ej``, ``kappa``, ``kappa_gamma``, ``T_floor``).

    Notes
    -----
    The two components are independent, so nothing ties the envelope mass of the shock cooling to the ejecta mass
    of the nickel component. The nickel priors are broad, Type IIb-like placeholders, and have not been fit to
    data.

    References
    ----------
    .. footbibliography::
    """

    _ARNETT_SED: ClassVar[type[ArnettDecaySED]] = ArnettDecaySED
    """type of ArnettDecaySED: The nickel component. Subclasses swap in an SED with its own parameters and priors."""

    _MORAG_PARAMETERS: ClassVar[tuple[str, ...]] = tuple(_MoragShockCoolingBase._DEFAULT_PARAMETERS)
    _ARNETT_PARAMETERS: ClassVar[tuple[str, ...]] = ("M_Ni", "M_ej", "v_ej", "kappa", "kappa_gamma", "T_floor")

    # TODO: The nickel priors below (M_Ni through kappa_gamma) are placeholders, not fits to data.
    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        **_MoragShockCoolingBase._DEFAULT_PARAMETERS,
        "M_Ni": Parameter(
            prior=TruncatedNormalPrior(mean=np.log10(0.08), sigma=0.2, lower=np.log10(0.02), upper=np.log10(0.3)),
            scale=1 * u.Msun,
            transform="log10",
            description="Nickel-56 mass synthesized in the explosion.",
            latex=r"M_\mathrm{Ni}",
        ),
        "M_ej": Parameter(
            prior=TruncatedNormalPrior(mean=np.log10(2.5), sigma=0.15, lower=np.log10(0.8), upper=np.log10(8.0)),
            scale=1 * u.Msun,
            transform="log10",
            description="Ejecta mass.",
            latex=r"M_\mathrm{ej}",
        ),
        "v_ej": Parameter(
            prior=TruncatedNormalPrior(mean=0.9, sigma=0.15, lower=0.5, upper=1.5),
            scale=1e4 * u.km / u.s,
            description="Constant ejecta velocity, identified with the photospheric velocity.",
            latex=r"v_\mathrm{ej}",
        ),
        "kappa": Parameter(
            prior=UniformPrior(lower=0.06, upper=0.12),
            scale=1 * u.cm**2 / u.g,
            description="Grey optical opacity of the ejecta.",
            latex=r"\kappa",
        ),
        "kappa_gamma": Parameter(
            prior=UniformPrior(lower=-1.7, upper=-1.2),
            scale=1 * u.cm**2 / u.g,
            transform="log10",
            description="Opacity to high-energy photons.",
            latex=r"\kappa_\gamma",
        ),
        "T_floor": ArnettDecaySED._DEFAULT_PARAMETERS["T_floor"],
    }

    @classmethod
    def _split_parameters(
        cls, parameters: dict[str, CGSParameterValue]
    ) -> tuple[dict[str, CGSParameterValue], dict[str, CGSParameterValue]]:
        """Split the parameters into those of the shock cooling and of the nickel components."""
        return (
            {name: parameters[name] for name in cls._MORAG_PARAMETERS},
            {name: parameters[name] for name in cls._ARNETT_PARAMETERS},
        )

    @classmethod
    def temperature(cls, t: u.Quantity, **parameters: CGSParameterValue) -> u.Quantity:
        r"""
        Luminosity-weighted mean of the temperatures of the two components.

        This is a heuristic summary of a two-component spectrum. It follows the shock cooling color temperature
        early on, and the photospheric temperature of the nickel component once that dominates.

        Parameters
        ----------
        t : ~astropy.units.Quantity
            Time since explosion.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        ~astropy.units.Quantity
            The temperature, in Kelvin.
        """
        morag, arnett = cls._split_parameters(parameters)
        L_morag = _UncutMoragShockCoolingSED.eval_bolometric(t, **morag)
        L_arnett = cls._ARNETT_SED.eval_bolometric(t, **arnett)
        T_morag = _UncutMoragShockCoolingSED.temperature(t, **morag)
        T_arnett = cls._ARNETT_SED.temperature(t, **arnett)
        return (L_morag * T_morag + L_arnett * T_arnett) / (L_morag + L_arnett)

    @classmethod
    def _eval_bolometric(cls, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""Natural log of :math:`L_\mathrm{bol}(t)` in erg/s, the sum of the two components."""
        morag, arnett = cls._split_parameters(parameters)
        return np.logaddexp(
            _UncutMoragShockCoolingSED._eval_bolometric(t, **morag), cls._ARNETT_SED._eval_bolometric(t, **arnett)
        )

    @classmethod
    def _eval(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""Natural log of :math:`L_\nu(\nu, t)` in erg/s/Hz, the sum of the two components."""
        morag, arnett = cls._split_parameters(parameters)
        return np.logaddexp(_UncutMoragShockCoolingSED._eval(nu, t, **morag), cls._ARNETT_SED._eval(nu, t, **arnett))
