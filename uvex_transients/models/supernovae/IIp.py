r"""SED model for Type IIP supernovae."""

from typing import ClassVar

import numpy as np
from astropy import units as u

from uvex_transients.models._typing import CGSParameterValue, FloatArray
from uvex_transients.models._util_functions import _log_sigmoid, cooling_temperature_cgs, temperature_bump_cgs
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

    The photospheric temperature follows the same single-power-law cooling law used by the Type IIb
    and Villar supernova models
    (:func:`~uvex_transients.models._util_functions.cooling_temperature_cgs`):

    .. math::

        T(t) = T_\mathrm{floor} + (T_0 - T_\mathrm{floor})\left(1 + \frac{t}{\tau_T}\right)^{-\alpha_T}.

    It starts hot at :math:`T_0` at :math:`t=0` (the shock-heated, pre-rise photosphere, :math:`\sim5\times10^4`
    K) and cools monotonically toward the hydrogen-recombination temperature :math:`T_\mathrm{floor}`
    (:math:`\sim5.6\times10^3` K). The data only constrain :math:`T` from a few days after explosion, where
    :math:`T - T_\mathrm{floor} \propto (T_0 - T_\mathrm{floor})\,\tau_T^{\alpha_T}\,t^{-\alpha_T}`, so
    :math:`T_0` and :math:`\tau_T` are degenerate and :math:`T_0` is set by the expected early-time temperature.

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
       * - ``T0``
         - :math:`T_0`
       * - ``T_floor``
         - :math:`T_\mathrm{floor}`
       * - ``tau_T``
         - :math:`\tau_T`
       * - ``alpha_T``
         - :math:`\alpha_T`

    Notes
    -----
    ``tau_drop``/``tau_Co`` are fixed shape constants. Light-curve priors (``t0`` through
    ``tau_Co``) come from a :func:`scipy.optimize.curve_fit` to SN 1999em and SN 2003hn;
    temperature-law priors (``T0`` through ``alpha_T``) are hand-tuned: ``T0`` to the expected
    :math:`\sim5\times10^4` K early-time temperature, the others against the aggregate Type IIP
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
        "T0": Parameter(
            prior=NormalPrior(mean=4.7, sigma=0.1),
            scale=1.0 * u.K,
            transform="log10",
            description="Photospheric temperature at t=0 (the T(t) -> T0 limit, not literally T at peak).",
            latex=r"T_0",
        ),
        "T_floor": Parameter(
            prior=NormalPrior(mean=3.73, sigma=0.08),
            scale=1.0 * u.K,
            transform="log10",
            description="Asymptotic late-time photospheric temperature (T(t) -> T_floor as t -> infinity).",
            latex=r"T_\mathrm{floor}",
        ),
        "tau_T": Parameter(
            prior=UniformPrior(lower=0.7, upper=2.0),
            scale=1 * u.day,
            description="Photospheric cooling timescale.",
            latex=r"\tau_T",
        ),
        "alpha_T": Parameter(
            prior=UniformPrior(lower=0.9, upper=1.2),
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
        :math:`\ln L_\mathrm{bol}(t)`; see the class docstring.

        Floored at ``_L_BOL_FLOOR_CGS`` -- well before ``t0``/well after ``t_P`` this is already
        ~0 and otherwise unconstrained, and can underflow toward 0 in float64.

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
        r"""
        :math:`\log S(\nu, T(t))`, delegated to :class:`BlackbodySpectrum`.

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
    # Spectral Luminosity: L_nu(nu, t)       #
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


# ======================================== #
# Model: Early-Interacting Excess          #
# ======================================== #
class TypeIIPExcessSED(TypeIIPSED):
    r"""
    Early-interacting, IXF/GGI-like Type IIP supernova SED.

    Same light curve as :class:`TypeIIPSED`, with different priors shifted toward the brighter,
    faster early cooling phase attributed to shock breakout through and/or collisional heating of
    close circumstellar material, rather than the smoother early decline of ordinary Type IIP SNe.

    The temperature law differs: the single-power-law cooling law of :class:`TypeIIPSED`, now describing a
    hot initial flash (:math:`T_0 \sim10^5` K) that has cooled by :math:`\sim1` d, plus a reheating bump
    (:func:`~uvex_transients.models._util_functions.temperature_bump_cgs`) that reproduces the rise of the
    photospheric temperature to :math:`\sim2\times10^4` K at :math:`\sim3` d seen in SN 2023ixf and SN 2024ggi:

    .. math::

        T(t) = T_\mathrm{floor} + (T_0 - T_\mathrm{floor})\left(1 + \frac{t}{\tau_T}\right)^{-\alpha_T}
        + \Delta T_\mathrm{bump}\,\frac{g(t/\tau_b)}{g(\beta_\mathrm{bump}/\alpha_\mathrm{bump})},
        \qquad g(x) = \frac{x^{\beta_\mathrm{bump}}}{(1 + x)^{\alpha_\mathrm{bump} + \beta_\mathrm{bump}}},

    with :math:`\tau_b = t_\mathrm{bump}\,\alpha_\mathrm{bump}/\beta_\mathrm{bump}`, so the bump peaks at
    exactly :math:`t_\mathrm{bump}` with height :math:`\Delta T_\mathrm{bump}`. The temperature is not
    monotonic. There is no temperature data before :math:`\sim0.8` d, so the flash parameters (``T0``,
    ``tau_T``, ``alpha_T``) are set by expectation rather than fit, and the bump parameters
    (``T_bump``, ``t_bump``, ``beta_bump``, ``alpha_bump``) and ``T_floor`` by fits to the data.

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
        "T0": Parameter(
            prior=NormalPrior(mean=5.0, sigma=0.1),
            scale=1.0 * u.K,
            transform="log10",
            description="Photospheric temperature at t=0, the hot shock-breakout flash (the T(t) -> T0 limit).",
            latex=r"T_0",
        ),
        "T_floor": Parameter(
            prior=NormalPrior(mean=3.69, sigma=0.06),
            scale=1.0 * u.K,
            transform="log10",
            description="Asymptotic late-time photospheric temperature (T(t) -> T_floor as t -> infinity).",
            latex=r"T_\mathrm{floor}",
        ),
        "tau_T": Parameter(
            prior=UniformPrior(lower=0.15, upper=0.3),
            scale=1 * u.day,
            description="Cooling timescale of the initial hot flash.",
            latex=r"\tau_T",
        ),
        "alpha_T": Parameter(
            prior=UniformPrior(lower=1.8, upper=2.4),
            scale=1.0,
            description="Cooling power-law index of the initial hot flash.",
            latex=r"\alpha_T",
        ),
        "T_bump": Parameter(
            prior=NormalPrior(mean=4.19, sigma=0.08),
            scale=1.0 * u.K,
            transform="log10",
            description="Peak temperature excess of the post-flash reheating bump, above the cooling law.",
            latex=r"\Delta T_\mathrm{bump}",
        ),
        "t_bump": Parameter(
            prior=UniformPrior(lower=2.4, upper=3.6),
            scale=1 * u.day,
            description="Epoch of the peak of the reheating bump.",
            latex=r"t_\mathrm{bump}",
        ),
        "beta_bump": Parameter(
            prior=UniformPrior(lower=5, upper=12),
            scale=1.0,
            description="Rise power-law index of the reheating bump.",
            latex=r"\beta_\mathrm{bump}",
        ),
        "alpha_bump": Parameter(
            prior=UniformPrior(lower=1.6, upper=2.4),
            scale=1.0,
            description="Decay power-law index of the reheating bump.",
            latex=r"\alpha_\mathrm{bump}",
        ),
    }

    # -------------------------------------- #
    # Cooling Law + Reheating Bump: T(t)      #
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
        T_bump: CGSParameterValue,
        t_bump: CGSParameterValue,
        beta_bump: CGSParameterValue,
        alpha_bump: CGSParameterValue,
        **_ignored: CGSParameterValue,
    ) -> FloatArray:
        r"""
        Cooling law plus a reheating bump; see the class docstring.

        Parameters
        ----------
        t : numpy.ndarray
            Time since explosion, in seconds.
        T0, T_floor, tau_T, alpha_T, T_bump, t_bump, beta_bump, alpha_bump : float or numpy.ndarray
            This model's parameter values, in cgs units; see the class docstring.
        **_ignored
            Any other model parameter values, ignored.

        Returns
        -------
        numpy.ndarray
            :math:`T(t)`, in Kelvin.
        """
        cooling = cooling_temperature_cgs(t, T0=T0, T_floor=T_floor, timescale=tau_T, alpha=alpha_T)
        bump = temperature_bump_cgs(t, amplitude=T_bump, t_peak=t_bump, beta=beta_bump, alpha=alpha_bump)
        return cooling + bump
