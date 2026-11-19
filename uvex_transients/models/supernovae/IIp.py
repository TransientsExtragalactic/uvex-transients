r"""SED model for Type IIP supernovae, with and without an early excess from confined circumstellar material."""

from functools import cache
from typing import ClassVar

import numpy as np
from astropy import units as u
from scipy.special import ndtr, ndtri

from uvex_transients.models._constants import C_CGS, SECONDS_PER_DAY, SIGMA_SB_CGS
from uvex_transients.models._typing import CGSParameterValue, FloatArray
from uvex_transients.models._util_functions import _log_sigmoid, planck_shape_log_cgs
from uvex_transients.models._utils import to_cgs_value
from uvex_transients.models.core.base import SpectralModel
from uvex_transients.models.core.parameters import Parameter
from uvex_transients.models.core.priors import (
    ConstantPrior,
    DiscretePrior,
    NormalPrior,
    TruncatedNormalPrior,
)

__all__ = ["TypeIIPSED"]

#: Specific heating rates of the :math:`^{56}\mathrm{Ni}\to{}^{56}\mathrm{Co}\to{}^{56}\mathrm{Fe}` chain, in erg/s/g,
#: and the e-folding times of the two decays, in seconds. The values are those of the model specification, not the
#: ones the Arnett models of :mod:`uvex_transients.models.arnett` use.
_EPS_NI_CGS = 3.9e10
_EPS_CO_CGS = 6.78e9
_TAU_NI_CGS = 8.8 * SECONDS_PER_DAY
_TAU_CO_CGS = 111.3 * SECONDS_PER_DAY

#: Numerical floor on :math:`L_\mathrm{bol}`, in erg/s. At and just after explosion the bolometric light curve is
#: zero (or underflows in float64), and would otherwise send a magnitude to infinity downstream.
_L_BOL_FLOOR_CGS = 1e35

#: Smallest time, in days, used inside the gamma-ray trapping term :math:`(T_0/t)^2`, which diverges at :math:`t=0`.
_T_TRAPPING_MIN_DAYS = 1e-6

_ANGSTROM_CM = 1e-8

#: The grid of the table of the energy-conserving normalization :math:`N(T)`: its temperatures (log10 K) and the
#: wavelengths it integrates over (log10 Angstrom). A photosphere outside the temperature range uses its end value.
_NORM_LOG10_TEMPERATURE = (3.3, 5.7, 400)
_NORM_LOG10_WAVELENGTH_ANGSTROM = (1.5, 6.5, 1500)

_LOG_4PI_SIGMA_SB = np.log(4.0 * np.pi * SIGMA_SB_CGS)
_LN10 = np.log(10.0)

#: Priors of the CSM branch of the events that are not those of the parameter's own prior (the plain events').
#: ``t_r``: :math:`\log_{10}(t_r/\mathrm{d})` is a normal of mean :math:`\log_{10}1.2` and width 0.40, truncated to
#: :math:`[-0.7, 0.9]`. ``R0``: :math:`1.4\times10^{14}\,\mathrm{cm}\times10^{N(0, 0.20)}`, capped at
#: :math:`3\times10^{14}` cm (a cap, not a truncation: the draws above it are set to it).
_CSM_T_R = {"log10_median": np.log10(1.2), "sigma": 0.40, "lower": -0.7, "upper": 0.9}
_CSM_R0 = {"median_cm": 1.4e14, "sigma": 0.20, "cap_cm": 3.0e14}


# ======================================== #
# Spectrum helpers                         #
# ======================================== #
def _log_suppression(
    wavelength: FloatArray,
    temperature: FloatArray,
    lam_c: FloatArray,
    beta_max: FloatArray,
    T_b: FloatArray,
    dT: FloatArray,
) -> FloatArray:
    r"""
    :math:`\ln S(\lambda, T)`, the UV line-blanketing suppression of the blackbody.

    :math:`S=(\lambda/\lambda_c)^{\beta(T)}` blueward of the cutoff wavelength :math:`\lambda_c` and 1 redward of it,
    with an index :math:`\beta(T)=\beta_\mathrm{max}/[1+e^{(T-T_b)/\Delta T}]` that is :math:`\beta_\mathrm{max}`
    for a cool photosphere and 0 (no suppression) for a hot one.

    Parameters
    ----------
    wavelength : numpy.ndarray
        Wavelength, in cm.
    temperature : numpy.ndarray
        Colour temperature, in K.
    lam_c : float or numpy.ndarray
        Cutoff wavelength, in cm.
    beta_max : float or numpy.ndarray
        Index of the suppression at full strength.
    T_b : float or numpy.ndarray
        Temperature at which the blanketing is half on, in K.
    dT : float or numpy.ndarray
        Width of the turn-on, in K.

    Returns
    -------
    numpy.ndarray
        The natural log of the suppression (dimensionless, :math:`\le 0`).
    """
    beta = beta_max * np.exp(_log_sigmoid(-(temperature - T_b) / dT))
    return beta * np.minimum(np.log(wavelength / lam_c), 0.0)


@cache
def _normalization_table(lam_c: float, beta_max: float, T_b: float, dT: float) -> tuple[FloatArray, FloatArray]:
    r"""
    Tabulate :math:`N(T)=\int\pi B_\nu(T)\,S\,d\nu/\sigma T^4`, the fraction of a blackbody left by the blanketing.

    The integral is taken over :data:`_NORM_LOG10_WAVELENGTH_ANGSTROM` by the trapezoid rule in frequency, for the
    temperatures of :data:`_NORM_LOG10_TEMPERATURE`. The table depends only on the four UV parameters, so it is built
    once for each set of them.

    Parameters
    ----------
    lam_c, beta_max, T_b, dT : float
        The UV parameters, as for :func:`_log_suppression` (``lam_c`` in cm).

    Returns
    -------
    ln_temperature : numpy.ndarray
        The natural log of the tabulated temperatures, in K.
    norm : numpy.ndarray
        :math:`N(T)` at those temperatures.
    """
    log_t_lo, log_t_hi, n_t = _NORM_LOG10_TEMPERATURE
    log_l_lo, log_l_hi, n_l = _NORM_LOG10_WAVELENGTH_ANGSTROM
    temperature = np.logspace(log_t_lo, log_t_hi, n_t)[:, None]
    wavelength = np.logspace(log_l_lo, log_l_hi, n_l)[None, :] * _ANGSTROM_CM
    nu = C_CGS / wavelength

    integrand = np.exp(
        planck_shape_log_cgs(nu, temperature) + _log_suppression(wavelength, temperature, lam_c, beta_max, T_b, dT)
    )
    # Frequency falls as wavelength rises, so reverse both for the trapezoid rule.
    norm = np.trapezoid(integrand[:, ::-1], nu[0, ::-1], axis=1)
    return np.log(temperature[:, 0]), norm


def _log_normalization(
    temperature: FloatArray, lam_c: FloatArray, beta_max: FloatArray, T_b: FloatArray, dT: FloatArray
) -> FloatArray:
    r"""
    :math:`\ln N(T)` by linear interpolation of the table in :math:`\ln T`.

    One table serves the usual case, where the four UV parameters are the same for every element. If they differ
    (a UV parameter given a prior that is not a constant), the elements are grouped by their set of values, and
    one table is built for each.

    Parameters
    ----------
    temperature : numpy.ndarray
        Colour temperature, in K.
    lam_c, beta_max, T_b, dT : float or numpy.ndarray
        The UV parameters, as for :func:`_log_suppression` (``lam_c`` in cm), broadcastable against ``temperature``.

    Returns
    -------
    numpy.ndarray
        :math:`\ln N(T)`, with the shape the arguments broadcast to.
    """
    uv = [np.asarray(x, dtype=np.float64) for x in (lam_c, beta_max, T_b, dT)]
    if all(x.size == 1 for x in uv):
        ln_t, norm = _normalization_table(*(x.item() for x in uv))
        return np.log(np.interp(np.log(temperature), ln_t, norm))

    shape = np.broadcast_shapes(np.shape(temperature), *(x.shape for x in uv))
    columns = np.stack([np.broadcast_to(x, shape).ravel() for x in uv])
    unique, inverse = np.unique(columns, axis=1, return_inverse=True)
    inverse = inverse.reshape(-1)
    ln_temperature = np.log(np.broadcast_to(temperature, shape).ravel())
    out = np.empty(ln_temperature.size)
    for j, key in enumerate(unique.T):
        ln_t, norm = _normalization_table(*(float(x) for x in key))
        selected = inverse == j
        out[selected] = np.interp(ln_temperature[selected], ln_t, norm)
    return np.log(out.reshape(shape))


# ======================================== #
# Sampling helpers                         #
# ======================================== #
def _truncated_normal(
    rng: np.random.Generator,
    mean: float | FloatArray,
    sigma: float,
    lower: float,
    upper: float,
    size: int,
) -> FloatArray:
    r"""
    Draw from a normal distribution truncated to :math:`[\mathrm{lower}, \mathrm{upper}]`, with a per-draw mean.

    By the inverse CDF, which uses one uniform draw per sample and so costs the same wherever the mean sits
    relative to the bounds (resampling out-of-range draws would not: a bright event's nickel mass sits at the
    edge of its range). An interval in the upper tail is reflected into the lower one first, where
    :func:`scipy.special.ndtr` keeps its precision.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random-number source.
    mean : float or numpy.ndarray
        Mean of the untruncated normal, a scalar or an array of shape ``(size,)``.
    sigma : float
        Width of the untruncated normal.
    lower, upper : float
        Bounds. Either may be infinite.
    size : int
        Number of draws.

    Returns
    -------
    numpy.ndarray
        The draws, with shape ``(size,)``.
    """
    mean = np.broadcast_to(np.asarray(mean, dtype=np.float64), (size,))
    a = (lower - mean) / sigma
    b = (upper - mean) / sigma
    flip = a > 0.0
    lo = np.where(flip, -b, a)
    hi = np.where(flip, -a, b)
    p_lo = ndtr(lo)
    z = ndtri(p_lo + rng.uniform(size=size) * (ndtr(hi) - p_lo))
    return np.clip(mean + sigma * np.where(flip, -z, z), lower, upper)


def _log10_physical(parameter: Parameter, latent: FloatArray) -> FloatArray:
    r""":math:`\log_{10}` of a luminosity parameter's physical value, in erg/s, from its latent values."""
    return np.log10(to_cgs_value(parameter.transform_latent_to_physical(latent)))


# ======================================== #
# Model                                    #
# ======================================== #
class TypeIIPSED(SpectralModel):
    r"""
    Type IIP supernova SED, with an optional early excess from confined circumstellar material (CSM).

    A semi-analytic model of the spectral luminosity :math:`L_\nu(t,\nu\mid\theta)` of Type IIP supernovae and of
    the IIP events whose early light curve is boosted by a confined shell of CSM, like SN 2023ixf and SN 2024ggi.
    Everything is closed form: there is no ODE integration and no grid beyond one small table of a normalization.
    Redshift, distance and dust are not part of it.

    The form follows the temperature and bolometric evolution of SNe II, with a recombination floor
    :footcite:p:`faran2017`, the colour temperature of shock cooling and the departures from a blackbody that
    suppress the UV :footcite:p:`morag2023,morag2024`, and the interaction power that emerges in the UV and shortens
    the optical rise of events with CSM :footcite:p:`dessart2022,dessart2026`. The prevalence and brightness of the
    CSM events are set from the ZTF samples :footcite:p:`bruch2023,hinds2025`, and their parameters from the fits to
    SN 2023ixf and SN 2024ggi (:footcite:t:`hiramatsu2023`, :footcite:t:`shrestha2024`), as the Notes describe.

    **Bolometric luminosity.** The sum of an envelope, a radioactive tail and a CSM pulse,

    .. math::

        L_\mathrm{bol}(t) = L_\mathrm{env}(t) + L_\mathrm{Ni}(t) + L_\mathrm{csm}(t),

    with the plateau cutoff :math:`F(t)=1/\{1+\exp[(t-t_p)/w]\}`. The envelope is the shock-cooling excess and the
    plateau,

    .. math::

        L_\mathrm{env}(t) = L_{50}\,10^{-0.4\,s\,(t-50)/100}\,\bigl[1+a_e e^{-t/\tau_e}\bigr]
        \bigl[1-e^{-(t/t_r)^2}\bigr]F(t),

    with :math:`t` in days, so that :math:`L_{50}` is the observed plateau luminosity at day 50. The tail is gated
    to emerge only after the plateau, which is what keeps that true,

    .. math::

        L_\mathrm{Ni}(t) = M_\mathrm{Ni}\bigl[(\epsilon_\mathrm{Ni}-\epsilon_\mathrm{Co})e^{-t/\tau_\mathrm{Ni}}
        + \epsilon_\mathrm{Co}e^{-t/\tau_\mathrm{Co}}\bigr]\bigl[1-e^{-(T_0/t)^2}\bigr]\bigl[1-F(t)\bigr],

    and the CSM pulse, with :math:`x=t/t_c`, is :math:`L_c\,x^k e^{k(1-x)}` up to its peak at :math:`t_c` and
    :math:`L_c\exp[1-\sqrt{1+((t-t_c)/\tau_d)^2}]` after it. It equals :math:`L_c` at :math:`t_c`, with zero slope on
    both sides, rises as :math:`t^k` and decays exponentially with the time :math:`\tau_d`.
    :math:`L_c=0` is a plain IIP.

    **Colour temperature.** An expanding photosphere with a recombination floor. Temperature is derived, never
    specified:

    .. math::

        R(t) = R_0 + v_\mathrm{ph}t,\quad
        T_\mathrm{exp} = \left[\frac{L_\mathrm{bol}}{4\pi\sigma R^2}\right]^{1/4},\quad
        T_\mathrm{col} = \bigl[(f_cT_\mathrm{exp})^n + T_\mathrm{rec}^n\bigr]^{1/n}.

    :math:`R_0` is the progenitor radius of a plain IIP and the breakout radius when there is CSM.

    **Spectrum.** A blackbody at :math:`T_\mathrm{col}`, suppressed in the UV by line blanketing and renormalized
    to :math:`L_\mathrm{bol}`,

    .. math::

        L_\nu = L_\mathrm{bol}\,\frac{\pi B_\nu(T_\mathrm{col})\,S(\lambda,T_\mathrm{col})}{\sigma T_\mathrm{col}^4\,
        N(T_\mathrm{col})},\qquad
        S=\begin{cases}(\lambda/\lambda_c)^{\beta(T)}&\lambda<\lambda_c\\1&\lambda\ge\lambda_c\end{cases},\qquad
        \beta(T)=\frac{\beta_\mathrm{max}}{1+e^{(T-T_b)/\Delta T}},

    where :math:`N(T)=\int\pi B_\nu S\,d\nu/\sigma T^4` is the fraction of the blackbody left after the suppression.
    The spectral shape therefore integrates to 1 at every time: **the flux removed from the UV is moved to longer
    wavelengths, not lost**, so that :math:`L_\mathrm{bol}` is the luminosity actually emitted. The blanketing
    is off above :math:`\sim10^4` K, where :math:`N=1`, so the early, hot photosphere is a plain blackbody and the
    renormalization only matters once the photosphere has cooled (:math:`N=0.94` at 6000 K). :math:`N(T)` is
    tabulated once for each set of the four UV parameters, on 400 temperatures from :math:`10^{3.3}` to
    :math:`10^{5.7}` K and 1500 wavelengths from :math:`10^{1.5}` to :math:`10^{6.5}` Angstrom, and interpolated
    linearly in :math:`\ln T`.

    .. rubric:: Population

    The parameters are drawn from a hierarchical population, as in the specification: an event is a CSM event with
    probability :math:`f_\mathrm{csm}=0.36` (the parameter ``csm``), then :math:`L_{50}` (and, for a CSM event,
    :math:`L_c`) is drawn, then every other parameter given them. This is not the independent draw of the other
    models, so :meth:`_sample_latent_parameters` is overridden, and the rules are these.

    * **Luminosity trends.** The priors of ``s``, ``M_Ni`` and ``v_ph`` are those of an event with
      :math:`\log_{10}L_{50}=42`, and of ``t_c`` of one with :math:`\log_{10}L_c=43.15`. The mean of each, in the
      variable the parameter is sampled in, moves by a slope times the distance of the event's luminosity from that
      reference (:attr:`LUMINOSITY_TRENDS`), with its truncation bounds unchanged. The plateau decline steepens
      and the nickel mass and the velocity rise with :math:`L_{50}`, and a brighter CSM pulse peaks later. A prior
      that is not a normal or a truncated normal is drawn as it is, without the trend.
    * **Event class.** ``t_r`` and ``R0`` have one prior for a plain event, the parameter's own, and another for a CSM
      event (:math:`\log_{10}(t_r/\mathrm{d})` normal about :math:`\log_{10}1.2` with width 0.40 and bounds
      :math:`[-0.7, 0.9]`; :math:`R_0=1.4\times10^{14}\,\mathrm{cm}\times10^{N(0,0.20)}` capped at
      :math:`3\times10^{14}` cm), which is not a parameter and is changed in a subclass only. An event without CSM has
      ``L_c = 0``, and the unused ``t_c``, ``k`` and ``tau_d`` are set to 3 d, 2 and 3 d.
    * **Fixing a parameter** gives that value to every event, and the trends of the others follow it. ``csm=1`` or
      ``csm=0`` selects the class, and fixing ``L_c`` makes every event a CSM event unless ``csm`` is fixed to 0
      (``L_c`` itself cannot be fixed to 0, because it is sampled in :math:`\log_{10}`).
    * **Replacing a parameter** with another :class:`~uvex_transients.models.core.parameters.Parameter` keeps the trend
      if its prior is a (truncated) normal in the same variable as the one it replaces.

    The result is a hierarchical population whose V-band peak luminosity function follows the ASAS-SN one, with
    the CSM share of a magnitude-limited sample near the observed :math:`\sim65\%` (see the Notes).

    .. rubric:: Parameters

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``csm``
         - :math:`x_\mathrm{csm}`
         - Whether the event has CSM, 1 or 0. A label of the population: the model uses ``L_c``.
       * - ``L50``
         - :math:`L_{50}`
         - Plateau luminosity at day 50.
       * - ``t_p``
         - :math:`t_p`
         - Plateau duration: the midpoint of the drop.
       * - ``s``
         - :math:`s`
         - Plateau decline rate, in mag per 100 d.
       * - ``a_e``
         - :math:`a_e`
         - Early excess amplitude, relative to the plateau.
       * - ``tau_e``
         - :math:`\tau_e`
         - Early excess decay time.
       * - ``t_r``
         - :math:`t_r`
         - Envelope rise time.
       * - ``M_Ni``
         - :math:`M_\mathrm{Ni}`
         - Nickel-56 mass.
       * - ``v_ph``
         - :math:`v_\mathrm{ph}`
         - Photospheric expansion velocity.
       * - ``R0``
         - :math:`R_0`
         - Initial photospheric radius.
       * - ``T_rec``
         - :math:`T_\mathrm{rec}`
         - Recombination floor temperature.
       * - ``L_c``
         - :math:`L_c`
         - CSM pulse peak luminosity (0 for a plain event).
       * - ``t_c``
         - :math:`t_c`
         - CSM pulse peak time.
       * - ``k``
         - :math:`k`
         - CSM pulse rise index.
       * - ``tau_d``
         - :math:`\tau_d`
         - CSM pulse decay time.
       * - ``w``
         - :math:`w`
         - Width of the plateau drop. Fixed.
       * - ``T0``
         - :math:`T_0`
         - Gamma-ray trapping timescale. Fixed.
       * - ``f_c``
         - :math:`f_c`
         - Colour to effective temperature ratio. Fixed.
       * - ``n``
         - :math:`n`
         - Smoothness of the transition to the temperature floor. Fixed.
       * - ``lam_c``
         - :math:`\lambda_c`
         - Blanketing cutoff wavelength. Fixed.
       * - ``beta_max``
         - :math:`\beta_\mathrm{max}`
         - Full-strength suppression index. Fixed.
       * - ``T_b``
         - :math:`T_b`
         - Temperature at which the blanketing is half on. Fixed.
       * - ``dT``
         - :math:`\Delta T`
         - Width of the blanketing turn-on. Fixed.

    Notes
    -----
    The priors carry these tags: **[LF]** tuned to the ASAS-SN luminosity function of SNe II, **[fit43]** from fits of
    the plain model to 43 IIP bolometric light curves, **[csm2]** from fits to SN 2023ixf and SN 2024ggi, **[lit]**
    set from the literature and **[assumed]** not constrained by the data.

    * ``csm``: 0.36 [lit], :footcite:t:`bruch2023` (more than 36%) and :footcite:t:`hinds2025` (:math:`36^{+5}_{-7}\%`).
    * ``L50``: :math:`\log_{10}L_{50}\sim N(41.26, 0.51)` truncated to :math:`[40.8, 43.3]` [LF]. The faint cutoff is
      the faintest object of the 43.
    * ``L_c``: :math:`\log_{10}L_c\sim N(43.00, 0.40)` truncated to :math:`[41.0, 44.8]` [lit], chosen so that the CSM
      share of a magnitude-limited sample approaches that of :footcite:t:`hinds2025`. The luminosity function does not
      constrain it.
    * ``t_p`` :math:`N(105, 20)` d truncated to :math:`[60, 150]`; ``s`` :math:`N(0.8+0.5(\log_{10}L_{50}-42), 0.5)`
      truncated to :math:`[0, 3]`; ``a_e`` :math:`4.0\times10^{N(0, 0.20)}`, bounded in :math:`\log_{10}` to
      :math:`[-1, 1]`; ``tau_e`` :math:`10\,\mathrm{d}\times10^{N(0, 0.18)}`, :math:`\log_{10}` to :math:`[0.5, 1.5]`;
      ``M_Ni`` :math:`\log_{10}\sim N(-1.45+(\log_{10}L_{50}-42), 0.17)` truncated to :math:`[-3.3, -0.5]`, all [fit43].
    * ``v_ph``: :math:`8\times10^8\,\mathrm{cm/s}\times10^{0.33(\log_{10}L_{50}-42)+N(0, 0.08)}` [assumed], from
      :math:`L\propto v^3`. ``T_rec``: :math:`N(6000, 500)` K truncated to :math:`[4500, 7000]` [csm2], with the
      observed temperatures.
    * Plain events: ``t_r`` :math:`1.0\,\mathrm{d}\times10^{N(0, 0.35)}`, :math:`\log_{10}` bounded to :math:`[-1, 0.7]`
      [fit43]; ``R0`` :math:`3.5\times10^{13}\,\mathrm{cm}\times10^{N(0, 0.15)}` [assumed], a red supergiant.
    * CSM events: ``t_c`` :math:`\log_{10}\sim N(\log_{10}3.5+0.5(\log_{10}L_c-43.15), 0.15)` truncated to
      :math:`[0, 1]`; ``tau_d`` :math:`1.5\,\mathrm{d}\times10^{N(0, 0.30)}`, :math:`\log_{10}` to :math:`[-0.5, 1.0]`,
      both [csm2]; ``k`` :math:`4.0\times10^{N(0, 0.30)}`, :math:`\log_{10}` to :math:`[0.18, 1.18]` [assumed], with
      SN 2023ixf and SN 2024ggi (:math:`k\ge12`) the steep tail; ``R0`` and ``t_r`` as above [csm2], [assumed] (5 to
      8 d for those two objects, the slow tail).

    The model was checked against:

    * the bolometric curves of 43 IIP (plain model: a median 0.02 dex each) and the bolometric curve and
      temperature of SN 2023ixf and SN 2024ggi (0.05 to 0.07 dex in :math:`L`, 0.03 to 0.05 dex in :math:`T`), with no
      UV data used (see below for those fits);
    * Swift/UVOT light curves (SOUSA, Brown et al. 2014): at the object level, SN 2012aw, SN 2008in, SN 2024ggi
      and SN 2023ixf (183 detections) scatter 0.20 dex about a +0.02 dex median offset, and at the population level
      (53 SNe II) the median offsets are below 0.2 dex in all three bands from day 0.6 to 15;
    * the ASAS-SN V-band peak luminosity function of SNe II, to 0.06 to 0.10 dex rms over the eight bins from
      :math:`M_V=-15.4` to :math:`-20.0` after fitting one overall normalization (about
      :math:`5.3\times10^{-5}` SNe/Mpc\ :sup:`3`/yr). The brightest bin (:math:`M_V=-21.2`) is not reproduced and was
      excluded;
    * a CSM share of a magnitude-limited sample (weights :math:`\propto10^{-0.6M_V}`) of about 60%, against
      :math:`67\pm6\%` from :footcite:t:`hinds2025`.

    The reference fits of the two CSM objects, which are curve-reproducing sets, not measurements (the early-excess and
    CSM terms trade off), are in the model's tests.

    **Limitations**

    * The late-time UV of the population is probably faint, by a factor of 2 to 4 after day 15. Selection effects in the
      Swift sample contribute (the data are 0.3 to 0.65 dex above the model median from day 15 to 110), but persistent
      interaction emission is not modeled. A one-parameter additive UV term was designed and deliberately left out.
    * The far UV (below 1900 Angstrom) is uncalibrated, since the blanketing was set on Swift bands only. There is
      no cut at the hydrogen Lyman limit.
    * The first day is an extrapolation: no plain IIP has a temperature before day 4 in the calibration set.
    * :math:`t=0` is the explosion, not first light. With CSM, the model can be faint for a short time before an
      observer would record first light.
    * About 17% of the population peaks fainter than the faintest bin of the luminosity function (:math:`M_V=-15.4`),
      and is unconstrained.
    * 37 of the 43 light curves come from a flash-spectroscopy sample, so part of the "plain" early excess is CSM.
    * After about day 80 the temperature of real events can fall below the floor (nebular phase), and the tail
      spectrum is a floor-temperature blackbody. A constant photospheric velocity makes the model cool slightly too
      fast over days 20 to 40 in some objects.
    * There is no host extinction, and the luminosity function was taken at face value.
    * ``v_ph``, ``R0`` of the plain events, ``k`` and the ``t_r`` of the CSM events are assumptions, not fits.

    References
    ----------
    .. footbibliography::
    """

    LUMINOSITY_TRENDS: ClassVar[dict[str, tuple[str, float, float]]] = {
        "s": ("L50", 42.0, 0.5),
        "M_Ni": ("L50", 42.0, 1.0),
        "v_ph": ("L50", 42.0, 0.33),
        "t_c": ("L_c", 43.15, 0.5),
    }
    r"""dict: For each parameter with a trend, ``(driver, reference, slope)``.

    The mean of the parameter's prior, in the variable it is sampled in, moves by ``slope`` times
    :math:`\log_{10}` of the ``driver`` parameter (in erg/s) minus ``reference``.
    """

    _PLAIN_CSM_PLACEHOLDERS: ClassVar[dict[str, u.Quantity | float]] = {
        "t_c": 3.0 * u.day,
        "k": 2.0,
        "tau_d": 3.0 * u.day,
    }
    """dict: The values of the CSM pulse's unused parameters for an event without CSM."""

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "csm": Parameter(
            prior=DiscretePrior(values=np.array([0.0, 1.0]), probabilities=np.array([0.64, 0.36])),
            scale=1.0,
            description="Whether the event has a confined CSM excess: 1 if so, 0 if not. The CSM fraction is 0.36.",
            latex=r"x_\mathrm{csm}",
        ),
        "L50": Parameter(
            prior=TruncatedNormalPrior(mean=41.26, sigma=0.51, lower=40.8, upper=43.3),
            scale=1.0 * u.erg / u.s,
            transform="log10",
            description="Plateau luminosity at day 50 (tuned to the ASAS-SN luminosity function).",
            latex=r"L_{50}",
        ),
        "t_p": Parameter(
            prior=TruncatedNormalPrior(mean=105.0, sigma=20.0, lower=60.0, upper=150.0),
            scale=1 * u.day,
            description="Plateau duration, the midpoint of the drop.",
            latex=r"t_p",
        ),
        "s": Parameter(
            prior=TruncatedNormalPrior(mean=0.8, sigma=0.5, lower=0.0, upper=3.0),
            scale=1.0,
            description="Plateau decline rate, in mag per 100 d. The prior is that at log L50 = 42; the mean "
            "rises by 0.5 per dex of L50.",
            latex=r"s",
        ),
        "a_e": Parameter(
            prior=TruncatedNormalPrior(mean=np.log10(4.0), sigma=0.20, lower=-1.0, upper=1.0),
            scale=1.0,
            transform="log10",
            description="Early excess amplitude, relative to the plateau.",
            latex=r"a_e",
        ),
        "tau_e": Parameter(
            prior=TruncatedNormalPrior(mean=1.0, sigma=0.18, lower=0.5, upper=1.5),
            scale=1 * u.day,
            transform="log10",
            description="Early excess decay time.",
            latex=r"\tau_e",
        ),
        "t_r": Parameter(
            prior=TruncatedNormalPrior(mean=0.0, sigma=0.35, lower=-1.0, upper=0.7),
            scale=1 * u.day,
            transform="log10",
            description="Envelope rise time. The prior is that of a plain event; a CSM event has its own.",
            latex=r"t_r",
        ),
        "M_Ni": Parameter(
            prior=TruncatedNormalPrior(mean=-1.45, sigma=0.17, lower=-3.3, upper=-0.5),
            scale=1 * u.Msun,
            transform="log10",
            description="Nickel-56 mass. The prior is that at log L50 = 42; the mean rises by 1 dex per dex of L50.",
            latex=r"M_\mathrm{Ni}",
        ),
        "v_ph": Parameter(
            prior=NormalPrior(mean=0.0, sigma=0.08),
            scale=8.0e8 * u.cm / u.s,
            transform="log10",
            description="Photospheric expansion velocity. The prior is that at log L50 = 42; the mean rises by "
            "0.33 dex per dex of L50 (L ~ v^3).",
            latex=r"v_\mathrm{ph}",
        ),
        "R0": Parameter(
            prior=NormalPrior(mean=0.0, sigma=0.15),
            scale=3.5e13 * u.cm,
            transform="log10",
            description="Initial photospheric radius: the progenitor radius, or the breakout radius for a CSM event. "
            "The prior is that of a plain event; a CSM event has its own.",
            latex=r"R_0",
        ),
        "T_rec": Parameter(
            prior=TruncatedNormalPrior(mean=6000.0, sigma=500.0, lower=4500.0, upper=7000.0),
            scale=1 * u.K,
            description="Recombination floor temperature.",
            latex=r"T_\mathrm{rec}",
        ),
        "L_c": Parameter(
            prior=TruncatedNormalPrior(mean=43.0, sigma=0.40, lower=41.0, upper=44.8),
            scale=1.0 * u.erg / u.s,
            transform="log10",
            description="CSM pulse peak luminosity, of the events that have CSM (0 for the others).",
            latex=r"L_c",
        ),
        "t_c": Parameter(
            prior=TruncatedNormalPrior(mean=np.log10(3.5), sigma=0.15, lower=0.0, upper=1.0),
            scale=1 * u.day,
            transform="log10",
            description="CSM pulse peak time. The prior is that at log L_c = 43.15; the mean rises by 0.5 dex per "
            "dex of L_c.",
            latex=r"t_c",
        ),
        "k": Parameter(
            prior=TruncatedNormalPrior(mean=np.log10(4.0), sigma=0.30, lower=0.18, upper=1.18),
            scale=1.0,
            transform="log10",
            description="CSM pulse rise index.",
            latex=r"k",
        ),
        "tau_d": Parameter(
            prior=TruncatedNormalPrior(mean=np.log10(1.5), sigma=0.30, lower=-0.5, upper=1.0),
            scale=1 * u.day,
            transform="log10",
            description="CSM pulse decay time.",
            latex=r"\tau_d",
        ),
        "w": Parameter(
            prior=ConstantPrior(value=5.0),
            scale=1 * u.day,
            description="Width of the plateau drop. Fixed.",
            latex=r"w",
        ),
        "T0": Parameter(
            prior=ConstantPrior(value=400.0),
            scale=1 * u.day,
            description="Gamma-ray trapping timescale. Fixed.",
            latex=r"T_0",
        ),
        "f_c": Parameter(
            prior=ConstantPrior(value=1.25),
            scale=1.0,
            description="Ratio of the colour to the effective temperature. Fixed.",
            latex=r"f_c",
        ),
        "n": Parameter(
            prior=ConstantPrior(value=6.0),
            scale=1.0,
            description="Smoothness of the transition of the temperature to its floor. Fixed.",
            latex=r"n",
        ),
        "lam_c": Parameter(
            prior=ConstantPrior(value=3975.0),
            scale=1 * u.AA,
            description="UV line-blanketing cutoff wavelength. Fixed (calibrated against Swift/UVOT).",
            latex=r"\lambda_c",
        ),
        "beta_max": Parameter(
            prior=ConstantPrior(value=3.6),
            scale=1.0,
            description="Index of the UV suppression at full strength. Fixed (calibrated against Swift/UVOT).",
            latex=r"\beta_\mathrm{max}",
        ),
        "T_b": Parameter(
            prior=ConstantPrior(value=8000.0),
            scale=1 * u.K,
            description="Temperature at which the UV blanketing is half on. Fixed (calibrated against Swift/UVOT).",
            latex=r"T_b",
        ),
        "dT": Parameter(
            prior=ConstantPrior(value=660.0),
            scale=1 * u.K,
            description="Width of the turn-on of the UV blanketing. Fixed (calibrated against Swift/UVOT).",
            latex=r"\Delta T",
        ),
    }

    # -------------------------------------- #
    # Sampling                               #
    # -------------------------------------- #
    def _draw(self, name: str, size: int, rng: np.random.Generator, shift: float | FloatArray = 0.0) -> FloatArray:
        """
        Draw ``size`` latent values of one parameter, with the mean of its prior moved by ``shift``.

        A fixed parameter returns its fixed value. A normal or truncated normal prior is drawn with its mean moved,
        and its bounds as they are. Any other prior is drawn as it is.

        Parameters
        ----------
        name : str
            The parameter.
        size : int
            Number of draws.
        rng : numpy.random.Generator
            Random-number source.
        shift : float or numpy.ndarray, optional
            The movement of the mean, in the parameter's latent variable. The default is 0.

        Returns
        -------
        numpy.ndarray
            The latent values, with shape ``(size,)``.
        """
        parameter = self._parameters[name]
        if parameter.is_fixed:
            return parameter.sample_latent_variable(size=size, rng=rng)
        prior = parameter.prior
        if isinstance(prior, TruncatedNormalPrior):
            return _truncated_normal(rng, prior.mean + shift, prior.sigma, prior.lower, prior.upper, size)
        if isinstance(prior, NormalPrior):
            return _truncated_normal(rng, prior.mean + shift, prior.sigma, -np.inf, np.inf, size)
        return prior.sample(size=size, rng=rng)

    def _draw_trend(
        self, name: str, size: int, rng: np.random.Generator, log10_driver: float | FloatArray
    ) -> FloatArray:
        """Draw a parameter with a luminosity trend (:attr:`LUMINOSITY_TRENDS`), given ``log10`` of its driver."""
        _, reference, slope = self.LUMINOSITY_TRENDS[name]
        return self._draw(name, size, rng, shift=slope * (log10_driver - reference))

    def _draw_csm_branch(self, name: str, size: int, rng: np.random.Generator) -> FloatArray:
        """Draw ``t_r`` or ``R0`` of ``size`` CSM events: a fixed parameter keeps its value, else the CSM prior."""
        parameter = self._parameters[name]
        if parameter.is_fixed:
            return parameter.sample_latent_variable(size=size, rng=rng)
        if name == "t_r":
            log10_t_r = _truncated_normal(
                rng, _CSM_T_R["log10_median"], _CSM_T_R["sigma"], _CSM_T_R["lower"], _CSM_T_R["upper"], size
            )
            physical = 10.0**log10_t_r * u.day
        else:
            radius = _CSM_R0["median_cm"] * 10.0 ** rng.normal(0.0, _CSM_R0["sigma"], size)
            physical = np.minimum(radius, _CSM_R0["cap_cm"]) * u.cm
        return parameter.transform_physical_to_latent(physical)

    def _sample_latent_parameters(self, size: int, rng: np.random.Generator) -> dict[str, FloatArray]:
        """
        Draw every parameter's latent variable from the hierarchical population; see the class docstring.

        The event class first, then :math:`L_{50}` and, for a CSM event, :math:`L_c`, then the other parameters given
        them. Everything is drawn from ``rng``, so that a seed fixes the draw.

        Parameters
        ----------
        size : int
            Number of events.
        rng : numpy.random.Generator
            The random-number source.

        Returns
        -------
        dict
            ``{name: latent samples}`` for each parameter, each with shape ``(size,)``.
        """
        parameters = self._parameters
        csm = self._draw("csm", size, rng) > 0.5
        if parameters["L_c"].is_fixed and not parameters["csm"].is_fixed:
            csm = np.ones(size, dtype=bool)
        n_csm = int(csm.sum())
        plain = ~csm

        latent: dict[str, FloatArray] = {"csm": csm.astype(np.float64)}
        latent["L50"] = self._draw("L50", size, rng)
        log10_l50 = _log10_physical(parameters["L50"], latent["L50"])

        for name in ("t_p", "a_e", "tau_e", "T_rec"):
            latent[name] = self._draw(name, size, rng)
        for name in ("s", "M_Ni", "v_ph"):
            latent[name] = self._draw_trend(name, size, rng, log10_l50)

        # The CSM pulse: drawn for the events that have one, and a placeholder where there is none.
        for name in ("L_c", "t_c", "k", "tau_d"):
            latent[name] = np.empty(size)
        latent["L_c"][plain] = -np.inf  # L_c = 0
        for name, value in self._PLAIN_CSM_PLACEHOLDERS.items():
            latent[name][plain] = parameters[name].transform_physical_to_latent(value)
        if n_csm:
            latent["L_c"][csm] = self._draw("L_c", n_csm, rng)
            log10_l_c = _log10_physical(parameters["L_c"], latent["L_c"][csm])
            latent["t_c"][csm] = self._draw_trend("t_c", n_csm, rng, log10_l_c)
            latent["k"][csm] = self._draw("k", n_csm, rng)
            latent["tau_d"][csm] = self._draw("tau_d", n_csm, rng)

        # The two parameters with a prior for each class of event.
        for name in ("t_r", "R0"):
            latent[name] = np.empty(size)
            if n_csm:
                latent[name][csm] = self._draw_csm_branch(name, n_csm, rng)
            if n_csm < size:
                latent[name][plain] = self._draw(name, size - n_csm, rng)

        # The constants, last and in declaration order.
        for name, parameter in parameters.items():
            if name not in latent:
                latent[name] = parameter.sample_latent_variable(size=size, rng=rng)
        return latent

    # -------------------------------------- #
    # Bolometric Luminosity: L_bol(t)         #
    # -------------------------------------- #
    @classmethod
    def _log_components(
        cls, t: FloatArray, **parameters: CGSParameterValue
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        r"""
        :math:`\ln L_\mathrm{env}`, :math:`\ln L_\mathrm{Ni}` and :math:`\ln L_\mathrm{csm}`, in erg/s.

        Parameters
        ----------
        t : numpy.ndarray
            Time since explosion, in seconds.
        **parameters
            This model's parameter values, in cgs units.

        Returns
        -------
        tuple of numpy.ndarray
            The natural logs of the envelope, the radioactive tail and the CSM pulse. Each is :math:`-\infty` where
            the component is zero.
        """
        p = parameters
        t_days = t / SECONDS_PER_DAY
        x_p = (t - p["t_p"]) / p["w"]
        log_f = _log_sigmoid(-x_p)  # ln F(t)
        log_1mf = _log_sigmoid(x_p)  # ln [1 - F(t)]

        with np.errstate(divide="ignore"):
            log_rise = np.log(-np.expm1(-((t / p["t_r"]) ** 2)))
            ln_env = (
                np.log(p["L50"])
                - 0.4 * _LN10 * p["s"] * (t_days - 50.0) / 100.0
                + np.log1p(p["a_e"] * np.exp(-t / p["tau_e"]))
                + log_rise
                + log_f
            )

            t_trap = np.maximum(t_days, _T_TRAPPING_MIN_DAYS) * SECONDS_PER_DAY
            ln_decay = np.logaddexp(
                np.log(_EPS_NI_CGS - _EPS_CO_CGS) - t / _TAU_NI_CGS, np.log(_EPS_CO_CGS) - t / _TAU_CO_CGS
            )
            ln_ni = np.log(p["M_Ni"]) + ln_decay + np.log(-np.expm1(-((p["T0"] / t_trap) ** 2))) + log_1mf

            x = t / p["t_c"]
            before_peak = p["k"] * (np.log(x) + 1.0 - np.minimum(x, 1.0))
            after_peak = 1.0 - np.sqrt(1.0 + (np.maximum(t - p["t_c"], 0.0) / p["tau_d"]) ** 2)
            ln_csm = np.log(p["L_c"]) + np.where(x <= 1.0, before_peak, after_peak)
        return ln_env, ln_ni, ln_csm

    @classmethod
    def _eval_bolometric(cls, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        :math:`\ln L_\mathrm{bol}(t)`, the sum of the envelope, the radioactive tail and the CSM pulse.

        Floored at ``_L_BOL_FLOOR_CGS``, which only matters at and just after explosion.

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
        ln_env, ln_ni, ln_csm = cls._log_components(t, **parameters)
        return np.maximum(np.logaddexp(np.logaddexp(ln_env, ln_ni), ln_csm), np.log(_L_BOL_FLOOR_CGS))

    # -------------------------------------- #
    # Colour Temperature: T(t)                #
    # -------------------------------------- #
    @classmethod
    def _temperature_from_log_lbol(
        cls, log_l_bol: FloatArray, t: FloatArray, **parameters: CGSParameterValue
    ) -> FloatArray:
        r"""
        :math:`T_\mathrm{col}` of the photosphere radiating :math:`\ln L_\mathrm{bol}` at time ``t``, in K.

        Parameters
        ----------
        log_l_bol : numpy.ndarray
            The natural log of :math:`L_\mathrm{bol}(t)`, in erg/s.
        t : numpy.ndarray
            Time since explosion, in seconds.
        **parameters
            This model's parameter values, in cgs units.

        Returns
        -------
        numpy.ndarray
            The colour temperature, in K.
        """
        p = parameters
        radius = p["R0"] + p["v_ph"] * t
        log_t_exp = 0.25 * (log_l_bol - _LOG_4PI_SIGMA_SB - 2.0 * np.log(radius))
        n = p["n"]
        # [(f_c T_exp)^n + T_rec^n]^(1/n), in log space so that it cannot overflow.
        return np.exp(np.logaddexp(n * (np.log(p["f_c"]) + log_t_exp), n * np.log(p["T_rec"])) / n)

    @classmethod
    def _temperature_cgs(cls, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        :math:`T_\mathrm{col}(t)`, the expanding photosphere with a recombination floor, in K.

        Parameters
        ----------
        t : numpy.ndarray
            Time since explosion, in seconds.
        **parameters
            This model's parameter values, in cgs units.

        Returns
        -------
        numpy.ndarray
            The colour temperature, in K.
        """
        return cls._temperature_from_log_lbol(cls._eval_bolometric(t, **parameters), t, **parameters)

    @classmethod
    def temperature(cls, t: u.Quantity, **parameters: CGSParameterValue) -> u.Quantity:
        r"""
        :math:`T_\mathrm{col}(t)` in Kelvin.

        Parameters
        ----------
        t : ~astropy.units.Quantity
            Time since explosion.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        ~astropy.units.Quantity
            The colour temperature, in Kelvin.
        """
        cgs_parameters: dict[str, CGSParameterValue] = {name: to_cgs_value(value) for name, value in parameters.items()}
        return cls._temperature_cgs(t.cgs.value, **cgs_parameters) * u.K

    # -------------------------------------- #
    # Spectral Shape: S(nu, t)               #
    # -------------------------------------- #
    @classmethod
    def _log_spectral_shape(
        cls, nu: FloatArray, temperature: FloatArray, **parameters: CGSParameterValue
    ) -> FloatArray:
        r"""
        :math:`\ln[\pi B_\nu(T)\,S(\lambda,T)/\sigma T^4N(T)]`, the blanketed blackbody with unit integral, in 1/Hz.

        Parameters
        ----------
        nu : numpy.ndarray
            Frequency, in Hz.
        temperature : numpy.ndarray
            Colour temperature, in K.
        **parameters
            This model's parameter values, in cgs units.

        Returns
        -------
        numpy.ndarray
            The natural log of the spectral shape, in 1/Hz.
        """
        uv = (parameters["lam_c"], parameters["beta_max"], parameters["T_b"], parameters["dT"])
        return (
            planck_shape_log_cgs(nu, temperature)
            + _log_suppression(C_CGS / nu, temperature, *uv)
            - _log_normalization(temperature, *uv)
        )

    @classmethod
    def _eval_spectrum(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        :math:`\ln S(\nu, t)`, the blanketed blackbody at the colour temperature, integrating to 1.

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
            The natural log of the spectral shape, in 1/Hz.
        """
        return cls._log_spectral_shape(nu, cls._temperature_cgs(t, **parameters), **parameters)

    # -------------------------------------- #
    # Spectral Luminosity: L_nu(nu, t)       #
    # -------------------------------------- #
    @classmethod
    def _eval(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        :math:`\ln L_\nu(\nu, t) = \ln L_\mathrm{bol}(t) + \ln S(\nu, T_\mathrm{col}(t))`.

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
        log_l_bol = cls._eval_bolometric(t, **parameters)
        temperature = cls._temperature_from_log_lbol(log_l_bol, t, **parameters)
        return log_l_bol + cls._log_spectral_shape(nu, temperature, **parameters)
