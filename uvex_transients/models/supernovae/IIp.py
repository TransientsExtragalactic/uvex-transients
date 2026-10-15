r"""SED model for Type IIP supernovae."""

from typing import ClassVar

import numpy as np
from astropy import units as u

from uvex_transients.models._constants import SECONDS_PER_DAY
from uvex_transients.models._typing import CGSParameterValue, FloatArray
from uvex_transients.models._util_functions import (
    _log_sigmoid,
    cooling_temperature_cgs,
    sigmoid_temperature_cgs,
    temperature_bump_cgs,
)
from uvex_transients.models._utils import to_cgs_value
from uvex_transients.models.core.base import SpectralModel
from uvex_transients.models.core.parameters import Parameter
from uvex_transients.models.core.priors import ConstantPrior, NormalPrior, TruncatedNormalPrior, UniformPrior
from uvex_transients.models.spectra.thermal import BlackbodySpectrum

__all__ = ["TypeIIPExcessSED", "TypeIIPSED"]

#: :math:`^{56}\mathrm{Co}` decay e-folding time (half-life 77.236 d / ln 2), in days. The
#: radioactive tail's decay rate is fixed to this physical constant rather than sampled.
_TAU_CO_DAYS = 77.236 / np.log(2.0)

#: Numerical floor on L_bol, in erg/s. Well before t0 (where L_bol is already ~0 from the S_0(t)
#: switch) the bolometric light curve is unconstrained and can underflow toward 0 in float64,
#: which would otherwise propagate to -inf magnitudes downstream.
_L_BOL_FLOOR_CGS = 1e35

#: Earliest epoch, in days, at which `TypeIIPSED`'s shock-cooling power law is evaluated. Earlier than this the
#: luminosity is held at its value here, since the power law diverges at :math:`t\to0` and the data it is calibrated
#: to do not reach back that far.
_T_SHOCK_MIN_DAYS = 0.1


# ======================================== #
# Model                                    #
# ======================================== #
class TypeIIPSED(SpectralModel):
    r"""
    Type IIP supernova SED: a cut-off power-law shock-cooling decline, a plateau and a radioactive tail.

    With a cooling blackbody photosphere.

    .. math::

        L_\mathrm{bol}(t) = L_1\left(\frac{t}{1\,\mathrm{d}}\right)^{-\alpha} e^{-t/t_c}
        + \bigl[1-S_P(t)\bigr]L_p
        + S_P(t)\, L_\mathrm{Co}\, e^{-(t-t_P)/\tau_\mathrm{Co}},

    with the plateau-end switch

    .. math::

        S_P(t) = \frac{1}{1+e^{-(t-t_P)/\tau_\mathrm{drop}}}.

    The first term is the shock-cooling decline of the extended envelope: a power law in time, cut off
    exponentially once the emission comes from deeper in the envelope and hydrogen starts to recombine.
    The cutoff has to sit inside this term, because adding a constant plateau to a bare power law can only make
    the curve flatter. The second term is the hydrogen-recombination plateau at :math:`L_p`, and :math:`S_P`
    turns it off, and the radioactive tail on, at the plateau-end epoch :math:`t_P`. The tail is a pure
    :math:`^{56}\mathrm{Co}` exponential from :math:`L_\mathrm{Co}` at :math:`t_P`.

    There is no rise: the bolometric luminosity is largest at the earliest times, as it is for most of the
    SNe II in the calibration sample. The power law is evaluated no earlier than
    :math:`0.1\,\mathrm{d}`, and :math:`L_\mathrm{bol}` is zero before explosion (:math:`t\le0`). The planar
    shock-breakout phase, :math:`L\propto t^{-4/3}` for the first few hours, is not modeled.

    The photospheric temperature falls from a hot start to a cool floor as a logistic in :math:`\ln t`
    (:func:`~uvex_transients.models._util_functions.sigmoid_temperature_cgs`):

    .. math::

        T(t) = T_\mathrm{floor} + (T_0 - T_\mathrm{floor})\,\frac{1}{1 + (t/t_T)^{k_T}}.

    It sits near :math:`T_0` until a fraction of :math:`t_T`, is halfway to the floor at :math:`t_T`, and
    falls as :math:`t^{-k_T}` afterwards. :math:`T_0` is set by expectation rather than by data: the shock-heated
    photosphere is expected to start at :math:`\sim5\times10^4` K or hotter, and the prior is centered on
    :math:`10^5` K. The transition epoch and steepness are tied to the Swift/UVOT UVW1 light curves, which put the
    photosphere at :math:`\sim10^4` K by :math:`\sim10` d.

    A blackbody does not describe the UV of a Type II SN well. Line blanketing makes the UV fainter than a
    blackbody at the optical temperature, by up to a factor of :math:`\sim3` at :math:`10`-:math:`30` d. The
    temperature is therefore a compromise between the UV and the optical data, and not a fit to the optical
    temperatures: the floor is :math:`\sim4.5\times10^3` K, about :math:`25\%` below the observed optical
    temperature at :math:`30`-:math:`100` d. The model is somewhat too bright in the UV from about 20 d (by
    :math:`\sim0.2` dex) and too faint at :math:`\sim10`-:math:`15` d (by :math:`\lesssim0.3` dex).

    .. rubric:: Parameters

    .. list-table::
       :header-rows: 1
       :widths: 22 78

       * - Parameter
         - Symbol
       * - ``L_1``
         - :math:`L_1`
       * - ``alpha``
         - :math:`\alpha`
       * - ``t_c``
         - :math:`t_c`
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
       * - ``t_T``
         - :math:`t_T`
       * - ``k_T``
         - :math:`k_T`

    Notes
    -----
    ``tau_drop``/``tau_Co`` are fixed shape constants. The light-curve priors come from fitting this form to
    each bolometric curve in the Type IIP calibration set (``test_data/transients``: the curves of
    :footcite:t:`dallora2014`, :footcite:t:`faran2018`, :footcite:t:`bersten2009` and the 35 comparison SNe II of
    :footcite:t:`jacobsongalan2024`), keeping the 28 with a resolved plateau drop and a good fit, and then
    adjusting the prior widths until the prior-predictive spread of :math:`L_\mathrm{bol}` matched the observed
    spread at epochs from 2 to 280 d. ``tau_drop`` is the median of the per-object fits. The fits prefer
    :math:`\alpha\simeq0` for most objects (a plain exponential decline with :math:`t_c\sim14` d), so ``alpha``
    is only bounded to :math:`[0, 0.35]`. ``L_1`` is a truncated normal, limited to the range of the fitted
    values, which keeps the bright tail of the population within what is observed. The priors are independent,
    although in the data :math:`L_1` and :math:`L_p` are correlated, so the population is somewhat broader than
    the observed one at fixed epoch pairs.

    ``T_floor``, ``t_T`` and ``k_T`` were fit against the Swift/UVOT UVW1 light curves of the Type IIP SNe in
    ``test_data/transients`` (epochs from 2 to 50 d), with ``T0`` held at its prior and the optical temperatures of
    SN 1999em, SN 2003hn, SN 2012aw, SN 2012A and SN 2008in weighted down. The UVOT light curves are mostly timed
    from discovery rather than explosion, which the calibration treats as a random 0 to 5 d lag.
    """

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "L_1": Parameter(
            prior=TruncatedNormalPrior(mean=42.45, sigma=1.1, lower=41.3, upper=43.5),
            scale=1.0 * u.erg / u.s,
            transform="log10",
            description="Shock-cooling luminosity scale: the power law's value at 1 d, before the cutoff.",
            latex=r"L_1",
        ),
        "alpha": Parameter(
            prior=UniformPrior(lower=0.0, upper=0.35),
            scale=1.0,
            description="Shock-cooling power-law decay index.",
            latex=r"\alpha",
        ),
        "t_c": Parameter(
            prior=UniformPrior(lower=9, upper=13),
            scale=1 * u.day,
            description="Exponential cutoff time of the shock-cooling decline.",
            latex=r"t_c",
        ),
        "L_p": Parameter(
            prior=NormalPrior(mean=41.76, sigma=0.35),
            scale=1.0 * u.erg / u.s,
            transform="log10",
            description="Plateau luminosity.",
            latex=r"L_p",
        ),
        "t_P": Parameter(
            prior=UniformPrior(lower=75, upper=135),
            scale=1 * u.day,
            description="Plateau-end / radioactive-tail-onset epoch.",
            latex=r"t_P",
        ),
        "tau_drop": Parameter(
            prior=ConstantPrior(value=7.0),
            scale=1 * u.day,
            description="Plateau-end transition width. Fixed.",
            latex=r"\tau_\mathrm{drop}",
        ),
        "L_Co": Parameter(
            prior=NormalPrior(mean=41.03, sigma=0.35),
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
            prior=NormalPrior(mean=5.0, sigma=0.25),
            scale=1.0 * u.K,
            transform="log10",
            description="Photospheric temperature at t=0 (the T(t) -> T0 limit), the hot shock-heated photosphere.",
            latex=r"T_0",
        ),
        "T_floor": Parameter(
            prior=NormalPrior(mean=3.65, sigma=0.04),
            scale=1.0 * u.K,
            transform="log10",
            description="Asymptotic late-time photospheric temperature (T(t) -> T_floor as t -> infinity).",
            latex=r"T_\mathrm{floor}",
        ),
        "t_T": Parameter(
            prior=NormalPrior(mean=0.30, sigma=0.23),
            scale=1 * u.day,
            transform="log10",
            description="Epoch at which the photosphere is halfway from T0 to T_floor.",
            latex=r"t_T",
        ),
        "k_T": Parameter(
            prior=UniformPrior(lower=1.4, upper=2.0),
            scale=1.0,
            description="Steepness of the photospheric cooling transition.",
            latex=r"k_T",
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
        t_T: CGSParameterValue,
        k_T: CGSParameterValue,
        **_ignored: CGSParameterValue,
    ) -> FloatArray:
        r"""
        :math:`T(t) = T_\mathrm{floor} + (T_0 - T_\mathrm{floor})[1 + (t/t_T)^{k_T}]^{-1}`.

        Parameters
        ----------
        t : numpy.ndarray
            Time since explosion, in seconds.
        T0, T_floor, t_T, k_T : float or numpy.ndarray
            This model's parameter values, in cgs units; see the class docstring.
        **_ignored
            Any other model parameter values, ignored.

        Returns
        -------
        numpy.ndarray
            :math:`T(t)`, in Kelvin.
        """
        return sigmoid_temperature_cgs(t, T0=T0, T_floor=T_floor, t_transition=t_T, steepness=k_T)

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

        Zero (the ``_L_BOL_FLOOR_CGS`` floor) before explosion, and floored there elsewhere, where the sum
        of the three terms can underflow in float64.

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
        t_P = parameters["t_P"]
        x_P = (t - t_P) / parameters["tau_drop"]

        # The shock-cooling power law, held at its 0.1 d value earlier than that. The exponential cutoff uses the
        # true t, so the term is smooth through the clip.
        t_days = np.maximum(t, _T_SHOCK_MIN_DAYS * SECONDS_PER_DAY) / SECONDS_PER_DAY
        ln_shock = np.log(parameters["L_1"]) - parameters["alpha"] * np.log(t_days) - t / parameters["t_c"]

        ln_plateau = _log_sigmoid(-x_P) + np.log(parameters["L_p"])
        ln_tail = _log_sigmoid(x_P) + np.log(parameters["L_Co"]) - (t - t_P) / parameters["tau_Co"]

        ln_L_bol = np.logaddexp(np.logaddexp(ln_shock, ln_plateau), ln_tail)
        return np.where(t > 0.0, np.maximum(ln_L_bol, np.log(_L_BOL_FLOOR_CGS)), np.log(_L_BOL_FLOOR_CGS))

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

    A subclass of :class:`TypeIIPSED` that keeps the earlier two-shared-onset-exponentials light curve, whose
    logistic rise and bright early cooling-phase peak describe the brighter, faster early phase attributed to
    shock breakout through and/or collisional heating of close circumstellar material:

    .. math::

        L_\mathrm{bol}(t) = S_0(t)\,\bigl[1-S_P(t)\bigr]
        \Bigl[L_\mathrm{pk}\,e^{-(t-t_0)/\tau_\mathrm{cool}} + L_p\Bigr]
        + S_P(t)\, L_\mathrm{Co}\, e^{-(t-t_P)/\tau_\mathrm{Co}},

    with logistic switches :math:`S_0(t) = [1+e^{-(t-t_0)/\tau_\mathrm{rise}}]^{-1}` and
    :math:`S_P(t) = [1+e^{-(t-t_P)/\tau_\mathrm{drop}}]^{-1}`. Its parameters are ``t0``, ``tau_rise``,
    ``L_pk``, ``tau_cool``, ``L_p``, ``t_P``, ``tau_drop``, ``L_Co`` and ``tau_Co``, rather than those of
    :class:`TypeIIPSED`.

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
