r"""SED model for Type IIP supernovae."""

from typing import ClassVar

import numpy as np
from astropy import units as u

from uvex_transients.models._constants import SECONDS_PER_DAY
from uvex_transients.models._typing import CGSParameterValue, FloatArray
from uvex_transients.models._util_functions import _log_sigmoid
from uvex_transients.models._utils import to_cgs_value
from uvex_transients.models.core.base import SpectralModel
from uvex_transients.models.core.parameters import Parameter
from uvex_transients.models.core.priors import ConstantPrior, NormalPrior, UniformPrior
from uvex_transients.models.spectra.thermal import BlackbodySpectrum

__all__ = ["TypeIIPExcessSED", "TypeIIPSED"]

#: :math:`^{56}\mathrm{Co}` decay e-folding time (half-life 77.236 d / ln 2), in days. The
#: radioactive tail's decay rate is fixed to this physical constant rather than sampled.
_TAU_CO_DAYS = 77.236 / np.log(2.0)

#: Numerical floor on L_bol, in erg/s. Well before t0 (where L_bol is already ~0 from the S_0(t)
#: switch) the bolometric light curve is unconstrained and can underflow toward 0 in float64,
#: which would otherwise propagate to -inf magnitudes downstream.
_L_BOL_FLOOR_CGS = 1e35

#: Numerical floor on the photospheric temperature, in K. Same rationale as `_L_BOL_FLOOR_CGS`:
#: well before t0, the doubly-broken power law's early t^alpha_r branch can underflow toward 0 K,
#: which drives the blackbody flux to exactly 0 and -inf magnitudes.
_T_FLOOR_K = 100.0


# ======================================== #
# Model                                    #
# ======================================== #
class TypeIIPSED(SpectralModel):
    r"""
    Type IIP supernova SED: two shared-onset exponentials and a switched radioactive tail.

    With a cooling blackbody photosphere.

    .. math::

        L_\mathrm{bol}(t) = S_0(t)\,\bigl[1-S_P(t)\bigr]
        \Bigl[L_\mathrm{pk}\,e^{-(t-t_0)/\tau_\mathrm{cool}} + L_p\Bigr]
        + S_P(t)\, L_\mathrm{Co}\, e^{-(t-t_P)/\tau_\mathrm{Co}},

    with two logistic switches

    .. math::

        S_0(t) = \frac{1}{1+e^{-(t-t_0)/\tau_\mathrm{rise}}}, \qquad
        S_P(t) = \frac{1}{1+e^{-(t-t_P)/\tau_\mathrm{drop}}}.

    :math:`S_0` turns the photospheric emission on at the explosion/rise epoch :math:`t_0`;
    :math:`S_P` turns it back off -- and the radioactive tail on -- at the plateau-end epoch
    :math:`t_P`. Well before :math:`t_0`, :math:`L_\mathrm{bol} \to 0`. Well after the rise but
    before the plateau ends, :math:`L_\mathrm{bol} \simeq L_\mathrm{pk}\,e^{-(t-t_0)/\tau_\mathrm{cool}}
    + L_p`: an early cooling-phase decline from :math:`L_\mathrm{pk}` onto a constant plateau
    :math:`L_p`. After :math:`t_P`, :math:`L_\mathrm{bol} \to L_\mathrm{Co}\, e^{-(t-t_P)/\tau_\mathrm{Co}}`,
    a pure radioactive-tail exponential.

    A deliberately minimum-complexity stand-in for the classic three-phase Type IIP morphology
    (shock-cooling decline, hydrogen-recombination plateau, radioactive tail): two logistic
    switches, two exponentials, three luminosity scales.

    The photospheric temperature reuses the light curve's own :math:`t_0` and :math:`t_P` and is a
    smooth, doubly-broken power law with three regimes -- an early regime, a cooling-phase regime
    around :math:`t_0`, and a recombination/plateau regime around :math:`t_P` --

    .. math::

        T(t) = T_0\, \left(\frac{t}{t_0}\right)^{\alpha_r}
        \left(\frac{1 + (t/t_0)^{s_0}}{2}\right)^{\frac{\alpha_c - \alpha_r}{s_0}}
        \left(\frac{1 + (t/\sqrt{t_0 t_P})^{s_1}}{2}\right)^{-\frac{\alpha_c - \alpha_p}{s_1}},

    (:math:`t` in days). Both bracketed terms are normalized to equal 1 exactly at their own break
    (:math:`t=t_0`, :math:`t=\sqrt{t_0 t_P}`), so :math:`T(t_0) \approx T_0` (up to a small
    correction from the second bracket, negligible when :math:`t_0 \ll t_P`). For :math:`t \ll t_0`,
    :math:`T(t) \to T_0\, (t/t_0)^{\alpha_r}`; between the two breaks
    (:math:`t_0 \ll t \ll \sqrt{t_0 t_P}`), :math:`T(t) \propto t^{\alpha_c}`; for
    :math:`t \gg \sqrt{t_0 t_P}` (approaching and past :math:`t_P`), :math:`T(t) \propto
    t^{\alpha_p}`. With :math:`\alpha_r > 0 > \alpha_c > \alpha_p`, the temperature rises through
    the early, pre-:math:`t_0` region (where :math:`L_\mathrm{bol} \approx 0` anyway, so this
    region is not otherwise constrained), peaks near :math:`t_0`, then declines -- fast through the
    cooling regime, and much more slowly (nearly flat) through the recombination/plateau regime.
    :math:`s_0`/:math:`s_1` set the sharpness of the two breaks (smooth, unlike the light curve's
    logistic switches, but the same role); the second break has no epoch of its own, so it is
    centered on the geometric mean of :math:`t_0` and :math:`t_P`.

    .. rubric:: Parameters

    .. list-table::
       :header-rows: 1
       :widths: 22 78

       * - Parameter
         - Symbol
       * - ``t0``
         - :math:`t_0`
       * - ``tau_rise``
         - :math:`\tau_\mathrm{rise}`
       * - ``L_pk``
         - :math:`L_\mathrm{pk}`
       * - ``tau_cool``
         - :math:`\tau_\mathrm{cool}`
       * - ``L_p``
         - :math:`L_p`
       * - ``t_P``
         - :math:`t_P`
       * - ``tau_drop``
         - :math:`\tau_\mathrm{drop}`
       * - ``L_Co``
         - :math:`L_\mathrm{Co}`
       * - ``tau_Co``
         - :math:`\tau_\mathrm{Co}`
       * - ``T_0``
         - :math:`T_0`
       * - ``alpha_r``
         - :math:`\alpha_r`
       * - ``alpha_c``
         - :math:`\alpha_c`
       * - ``alpha_p``
         - :math:`\alpha_p`
       * - ``s_0``
         - :math:`s_0`
       * - ``s_1``
         - :math:`s_1`

    Notes
    -----
    ``tau_drop``/``tau_Co`` are fixed shape constants. Light-curve priors (``t0`` through
    ``tau_Co``) come from a :func:`scipy.optimize.curve_fit` to SN 1999em and SN 2003hn;
    temperature-law priors (``T_0`` through ``s_1``) are hand-tuned against the aggregate Type IIP
    photospheric-temperature dataset in ``test_data/transients`` (SN 1999em, SN 2003hn, SN 2012aw,
    SN 2012A, SN 2008in), following :footcite:t:`dallora2014` and :footcite:t:`faran2018`.
    """

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "t0": Parameter(
            prior=UniformPrior(lower=4, upper=20),
            scale=1.0 * u.day,
            description="Explosion/rise reference epoch.",
            latex=r"t_0",
        ),
        "tau_rise": Parameter(
            prior=NormalPrior(mean=0, sigma=0.2),
            scale=1.0 * u.day,
            transform="log10",
            description="Rise timescale.",
            latex=r"\tau_\mathrm{rise}",
        ),
        "L_pk": Parameter(
            prior=NormalPrior(mean=np.log10(3e42), sigma=0.3),
            scale=1.0 * u.erg / u.s,
            transform="log10",
            description="Early cooling-phase peak luminosity scale.",
            latex=r"L_\mathrm{pk}",
        ),
        "tau_cool": Parameter(
            prior=UniformPrior(lower=4, upper=10),
            scale=1 * u.day,
            description="Early cooling-phase decay timescale.",
            latex=r"\tau_\mathrm{cool}",
        ),
        "L_p": Parameter(
            prior=NormalPrior(mean=42.1, sigma=0.2),
            scale=1.0 * u.erg / u.s,
            transform="log10",
            description="Plateau luminosity.",
            latex=r"L_p",
        ),
        "t_P": Parameter(
            prior=UniformPrior(lower=50, upper=150),
            scale=1 * u.day,
            description="Plateau-end / radioactive-tail-onset epoch.",
            latex=r"t_P",
        ),
        "tau_drop": Parameter(
            prior=ConstantPrior(value=10.5),
            scale=1 * u.day,
            description="Plateau-end transition width. Fixed.",
            latex=r"\tau_\mathrm{drop}",
        ),
        "L_Co": Parameter(
            prior=NormalPrior(mean=41, sigma=0.3),
            scale=1.0 * u.erg / u.s,
            transform="log10",
            description="Radioactive-tail luminosity at t_P.",
            latex=r"L_\mathrm{Co}",
        ),
        "tau_Co": Parameter(
            prior=ConstantPrior(value=1.0),
            scale=_TAU_CO_DAYS * u.day,
            description="Radioactive-tail decay timescale. Fixed to the 56Co decay e-folding time.",
            latex=r"\tau_\mathrm{Co}",
        ),
        "T_0": Parameter(
            prior=NormalPrior(mean=4.05, sigma=0.1),
            scale=1.0 * u.K,
            transform="log10",
            description="Doubly-broken power-law temperature normalization; T(t) ~ T_0 at t = t0 (~11200 K).",
            latex=r"T_0",
        ),
        "alpha_r": Parameter(
            prior=NormalPrior(mean=2, sigma=0.2),
            scale=1,
            description="Early-regime (t << t0) temperature power-law index; T rises toward t0.",
            latex=r"\alpha_r",
        ),
        "alpha_c": Parameter(
            prior=NormalPrior(mean=-0.45, sigma=0.1),
            scale=1,
            description="Cooling-regime (t0 << t << sqrt(t0 t_P)) temperature power-law index; T declines.",
            latex=r"\alpha_c",
        ),
        "alpha_p": Parameter(
            prior=NormalPrior(mean=-0.1, sigma=0.01),
            scale=1,
            description="Plateau-regime (t >> sqrt(t0 t_P)) temperature power-law index; T declines slowly.",
            latex=r"\alpha_p",
        ),
        "s_0": Parameter(
            prior=UniformPrior(lower=10, upper=20),
            scale=1,
            description="Sharpness of the break at t0 (early -> cooling regime).",
            latex=r"s_0",
        ),
        "s_1": Parameter(
            prior=UniformPrior(lower=10, upper=50),
            scale=1,
            description="Sharpness of the break at sqrt(t0 t_P) (cooling -> plateau regime).",
            latex=r"s_1",
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
        t0: CGSParameterValue,
        t_P: CGSParameterValue,
        T_0: CGSParameterValue,
        alpha_r: CGSParameterValue,
        alpha_c: CGSParameterValue,
        alpha_p: CGSParameterValue,
        s_0: CGSParameterValue,
        s_1: CGSParameterValue,
        **_ignored: CGSParameterValue,
    ) -> FloatArray:
        r"""Doubly-broken power-law photospheric temperature; see the class docstring.

        Reuses the light curve's own ``t0``/``t_P``. Evaluated in days -- ``t``, ``t0``, ``t_P``
        arrive in cgs (seconds), converted back here since (unlike the rest of this model) the
        formula is not scale-invariant to the day/second choice.

        Floored at ``_T_FLOOR_K`` -- well before ``t0`` (unconstrained; ``L_bol`` is already ~0
        there) the early branch can otherwise underflow toward 0 K.
        """
        t_day = t / SECONDS_PER_DAY
        t0_day = t0 / SECONDS_PER_DAY
        tP_day = t_P / SECONDS_PER_DAY
        t_break2 = np.sqrt(t0_day * tP_day)

        with np.errstate(divide="ignore", invalid="ignore"):
            early_to_cooling = ((1.0 + (t_day / t0_day) ** s_0) / 2) ** ((alpha_c - alpha_r) / s_0)
            cooling_to_plateau = ((1.0 + (t_day / t_break2) ** s_1) / 2) ** (-(alpha_c - alpha_p) / s_1)
            temperature = T_0 * (t_day / t0_day) ** alpha_r * early_to_cooling * cooling_to_plateau
        return np.maximum(temperature, _T_FLOOR_K)

    @classmethod
    def temperature(cls, t: u.Quantity, **parameters: CGSParameterValue) -> u.Quantity:
        r""":math:`T(t)` in Kelvin."""
        cgs_parameters: dict[str, CGSParameterValue] = {name: to_cgs_value(value) for name, value in parameters.items()}
        return cls._temperature_cgs(t.cgs.value, **cgs_parameters) * u.K

    # -------------------------------------- #
    # Bolometric Luminosity: L_bol(t)         #
    # -------------------------------------- #
    @classmethod
    def _eval_bolometric(cls, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r""":math:`\ln L_\mathrm{bol}(t)`; see the class docstring.

        Floored at ``_L_BOL_FLOOR_CGS`` -- well before ``t0``/well after ``t_P`` this is already
        ~0 and otherwise unconstrained, and can underflow toward 0 in float64.
        """
        t0 = parameters["t0"]
        t_P = parameters["t_P"]

        x0 = (t - t0) / parameters["tau_rise"]
        xP = (t - t_P) / parameters["tau_drop"]

        ln_S0 = _log_sigmoid(x0)
        ln_SP = _log_sigmoid(xP)
        ln_1mSP = _log_sigmoid(-xP)

        # L_pk * exp(-(t-t0)/tau_cool) + L_p, computed stably in log-space.
        ln_photospheric = np.logaddexp(
            np.log(parameters["L_pk"]) - (t - t0) / parameters["tau_cool"],
            np.log(parameters["L_p"]),
        )

        ln_term_photospheric = ln_S0 + ln_1mSP + ln_photospheric
        ln_term_radioactive = ln_SP + np.log(parameters["L_Co"]) - (t - t_P) / parameters["tau_Co"]

        ln_L_bol = np.logaddexp(ln_term_photospheric, ln_term_radioactive)
        return np.maximum(ln_L_bol, np.log(_L_BOL_FLOOR_CGS))

    # -------------------------------------- #
    # Normalized Spectral Shape: S(nu, t)    #
    # -------------------------------------- #
    @classmethod
    def _eval_spectrum(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r""":math:`\log S(\nu, T(t))`, delegated to :class:`BlackbodySpectrum`.

        Evaluated at this ``t``'s own cooling-law temperature.
        """
        temperature = cls._temperature_cgs(t, **parameters)
        return BlackbodySpectrum._eval(nu, temperature=temperature)

    # -------------------------------------- #
    # Spectral Luminosity: L_nu(nu, t)       #
    # -------------------------------------- #
    @classmethod
    def _eval(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r""":math:`\log L_\nu(\nu, t) = \log L_\mathrm{bol}(t) + \log S(\nu, T(t))`."""
        return cls._eval_bolometric(t, **parameters) + cls._eval_spectrum(nu, t, **parameters)


# ======================================== #
# Model: Early-Interacting Excess          #
# ======================================== #
class TypeIIPExcessSED(TypeIIPSED):
    r"""
    Early-interacting, IXF/GGI-like Type IIP supernova SED.

    Same light curve and temperature functional forms as :class:`TypeIIPSED` -- only the parameter
    priors differ, shifted toward the brighter, hotter, faster early cooling phase attributed to
    shock breakout through and/or collisional heating of close circumstellar material, rather than
    the smoother early decline of ordinary Type IIP SNe.

    Priors are hand-tuned against SN 2023ixf :footcite:p:`hsu2025` and SN 2024ggi
    :footcite:p:`chen2024`, the only two objects of this kind currently in
    ``test_data/transients``.
    """

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "t0": Parameter(
            prior=UniformPrior(lower=2, upper=5),
            scale=1.0 * u.day,
            description="Explosion/rise reference epoch.",
            latex=r"t_0",
        ),
        "tau_rise": Parameter(
            prior=NormalPrior(mean=-0.5, sigma=0.2),
            scale=1.0 * u.day,
            transform="log10",
            description="Rise timescale.",
            latex=r"\tau_\mathrm{rise}",
        ),
        "L_pk": Parameter(
            prior=NormalPrior(mean=np.log10(2e43), sigma=0.25),
            scale=1.0 * u.erg / u.s,
            transform="log10",
            description="Early cooling-phase peak luminosity scale.",
            latex=r"L_\mathrm{pk}",
        ),
        "tau_cool": Parameter(
            prior=UniformPrior(lower=4, upper=15),
            scale=1 * u.day,
            description="Early cooling-phase decay timescale.",
            latex=r"\tau_\mathrm{cool}",
        ),
        "L_p": Parameter(
            prior=NormalPrior(mean=42.2, sigma=0.2),
            scale=1.0 * u.erg / u.s,
            transform="log10",
            description="Plateau luminosity.",
            latex=r"L_p",
        ),
        "t_P": Parameter(
            prior=UniformPrior(lower=50, upper=150),
            scale=1 * u.day,
            description="Plateau-end / radioactive-tail-onset epoch.",
            latex=r"t_P",
        ),
        "tau_drop": Parameter(
            prior=ConstantPrior(value=10.5),
            scale=1 * u.day,
            description="Plateau-end transition width. Fixed.",
            latex=r"\tau_\mathrm{drop}",
        ),
        "L_Co": Parameter(
            prior=NormalPrior(mean=41.5, sigma=0.3),
            scale=1.0 * u.erg / u.s,
            transform="log10",
            description="Radioactive-tail luminosity at t_P.",
            latex=r"L_\mathrm{Co}",
        ),
        "tau_Co": Parameter(
            prior=ConstantPrior(value=1.0),
            scale=_TAU_CO_DAYS * u.day,
            description="Radioactive-tail decay timescale. Fixed to the 56Co decay e-folding time.",
            latex=r"\tau_\mathrm{Co}",
        ),
        "T_0": Parameter(
            prior=NormalPrior(mean=4.2, sigma=0.1),
            scale=1.0 * u.K,
            transform="log10",
            description="Doubly-broken power-law temperature normalization; T(t) ~ T_0 at t = t0 (~11200 K).",
            latex=r"T_0",
        ),
        "alpha_r": Parameter(
            prior=NormalPrior(mean=1, sigma=0.2),
            scale=1,
            description="Early-regime (t << t0) temperature power-law index; T rises toward t0.",
            latex=r"\alpha_r",
        ),
        "alpha_c": Parameter(
            prior=NormalPrior(mean=-0.45, sigma=0.1),
            scale=1,
            description="Cooling-regime (t0 << t << sqrt(t0 t_P)) temperature power-law index; T declines.",
            latex=r"\alpha_c",
        ),
        "alpha_p": Parameter(
            prior=NormalPrior(mean=-0.1, sigma=0.01),
            scale=1,
            description="Plateau-regime (t >> sqrt(t0 t_P)) temperature power-law index; T declines slowly.",
            latex=r"\alpha_p",
        ),
        "s_0": Parameter(
            prior=UniformPrior(lower=10, upper=20),
            scale=1,
            description="Sharpness of the break at t0 (early -> cooling regime).",
            latex=r"s_0",
        ),
        "s_1": Parameter(
            prior=UniformPrior(lower=10, upper=50),
            scale=1,
            description="Sharpness of the break at sqrt(t0 t_P) (cooling -> plateau regime).",
            latex=r"s_1",
        ),
    }
