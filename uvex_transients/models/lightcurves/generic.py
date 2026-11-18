"""Generic, reusable `~uvex_transients.models.core.Lightcurve` shapes."""

from typing import ClassVar

import numpy as np
from astropy import units as u
from numpy.typing import NDArray

from uvex_transients.models.core import Lightcurve, LogNormalPrior, NormalPrior, Parameter, UniformPrior

from .._typing import CGSParameterValue
from .._utils import _BOL_LUM_UNIT

__all__ = [
    "BazinLightcurve",
    "BrokenPowerLawLightcurve",
    "DelayedExponentialLightcurve",
    "FREDLightcurve",
    "GREDLightcurve",
    "GaussianPulseLightcurve",
    "GaussianRiseBrokenPowerLawLightcurve",
    "GaussianRisePowerLawLightcurve",
    "InverseGammaBrokenPowerLawLightcurve",
    "KhatamiSBOLightcurve",
    "LogNormalPulseLightcurve",
    "PlateauPowerLawLightcurve",
    "PowerLawLightcurve",
    "SBOPhenomenologicalLightcurve",
    "SmoothBrokenPowerLawLightcurve",
    "TopHatLightcurve",
    "TwoComponentBazinLightcurve",
    "VillarLightcurve",
]


class TopHatLightcurve(Lightcurve):
    r"""
    A constant luminosity for a fixed duration, zero before and after.

    .. math::

        L(t) = \begin{cases} A & 0 \le t \le \Delta t \\ 0 & t > \Delta t \end{cases}

    The simplest possible pulse shape: a plateau at the peak amplitude :math:`A` for a
    duration :math:`\Delta t`, with an instantaneous rise and cutoff.

    .. rubric:: Parameters

    The light curve parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude``
         - :math:`A`
         - Plateau bolometric luminosity.
       * - ``duration``
         - :math:`\Delta t`
         - Duration of the plateau.
    """

    _LIGHTCURVE_TYPE = "bolometric"

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e43 * _BOL_LUM_UNIT,
            description="Plateau bolometric luminosity.",
            latex=r"A",
        ),
        "duration": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1.0 * u.day,
            description="Duration of the plateau.",
            latex=r"\Delta t",
        ),
    }

    @classmethod
    def _eval(cls, t: NDArray[np.float64], **parameters: CGSParameterValue) -> NDArray[np.float64]:
        r"""
        Evaluate the natural log of :math:`L_\mathrm{bol}(t)`; see the class docstring for the functional form.

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
        amplitude, duration = parameters["amplitude"], parameters["duration"]

        with np.errstate(divide="ignore"):
            return np.where(t <= duration, np.log(amplitude), -np.inf)


class GaussianPulseLightcurve(Lightcurve):
    r"""
    A Gaussian pulse in luminosity, peaked at ``t_peak``.

    .. math::

        L(t) = A \exp\left(-\frac{(t - t_\mathrm{peak})^2}{2\sigma^2}\right)

    :math:`L(t_\mathrm{peak}) = A` exactly, by construction.

    .. rubric:: Parameters

    The light curve parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude``
         - :math:`A`
         - Peak bolometric luminosity.
       * - ``t_peak``
         - :math:`t_\mathrm{peak}`
         - Time of peak luminosity since explosion.
       * - ``sigma``
         - :math:`\sigma`
         - Gaussian width of the pulse.
    """

    _LIGHTCURVE_TYPE = "bolometric"

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e43 * _BOL_LUM_UNIT,
            description="Peak bolometric luminosity.",
            latex=r"A",
        ),
        "t_peak": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1.0 * u.day,
            description="Time of peak luminosity since explosion.",
            latex=r"t_\mathrm{peak}",
        ),
        "sigma": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1.0 * u.day,
            description="Gaussian width of the pulse.",
            latex=r"\sigma",
        ),
    }

    @classmethod
    def _eval(cls, t: NDArray[np.float64], **parameters: CGSParameterValue) -> NDArray[np.float64]:
        r"""
        Evaluate the natural log of :math:`L_\mathrm{bol}(t)`; see the class docstring for the functional form.

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
        amplitude, t_peak, sigma = (
            parameters["amplitude"],
            parameters["t_peak"],
            parameters["sigma"],
        )

        return np.log(amplitude) - 0.5 * ((t - t_peak) / sigma) ** 2


class FREDLightcurve(Lightcurve):
    r"""
    A fast-rise, exponential-decay pulse (the Norris et al. 1996 GRB pulse shape).

    .. math::

        L(t) = A \exp\left(2\sqrt{\tau_1/\tau_2} - \frac{\tau_1}{t} - \frac{t}{\tau_2}\right), \quad t > 0

    with :math:`L(0) = 0`. Despite the apparent singularity at :math:`\tau_1/t`, this is
    well-behaved: as :math:`t \to 0^+`, the :math:`-\tau_1/t` term dominates and drives
    :math:`L \to 0`. The pulse peaks exactly at :math:`t_\mathrm{peak} = \sqrt{\tau_1 \tau_2}`,
    where :math:`L(t_\mathrm{peak}) = A`; :math:`\tau_1 \ll \tau_2` gives the characteristic
    fast rise / slow, exponential-looking decay.

    .. rubric:: Parameters

    The light curve parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude``
         - :math:`A`
         - Peak bolometric luminosity.
       * - ``rise``
         - :math:`\tau_1`
         - Rise timescale tau_1.
       * - ``decay``
         - :math:`\tau_2`
         - Decay timescale tau_2.
    """

    _LIGHTCURVE_TYPE = "bolometric"

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e43 * _BOL_LUM_UNIT,
            description="Peak bolometric luminosity.",
            latex=r"A",
        ),
        "rise": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=0.5 * u.day,
            description="Rise timescale tau_1.",
            latex=r"\tau_1",
        ),
        "decay": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=3.0 * u.day,
            description="Decay timescale tau_2.",
            latex=r"\tau_2",
        ),
    }

    @classmethod
    def _eval(cls, t: NDArray[np.float64], **parameters: CGSParameterValue) -> NDArray[np.float64]:
        r"""
        Evaluate the natural log of :math:`L_\mathrm{bol}(t)`; see the class docstring for the functional form.

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
        amplitude, rise, decay = (
            parameters["amplitude"],
            parameters["rise"],
            parameters["decay"],
        )

        # `t` includes `t = 0` by convention (see `Lightcurve`'s class docstring), which
        # divides by zero here. That's the correct limit, not an error -- `rise / t -> inf`
        # drives the exponent to `-inf` and `L -> 0` -- so the warning is suppressed rather
        # than worked around.
        with np.errstate(divide="ignore"):
            return np.log(amplitude) + 2 * np.sqrt(rise / decay) - rise / t - t / decay


class GREDLightcurve(Lightcurve):
    r"""
    A Gaussian rise followed by an exponential decline, peaked at ``t_peak``.

    .. math::

        L(t) = A \times \begin{cases}
            \exp\left(-\dfrac{(t - t_\mathrm{peak})^2}{2\sigma_\mathrm{rise}^2}\right)
                & t \le t_\mathrm{peak} \\[4pt]
            \exp\left(-\dfrac{t - t_\mathrm{peak}}{\tau_\mathrm{decline}}\right)
                & t > t_\mathrm{peak}
        \end{cases}

    The two pieces agree at :math:`t = t_\mathrm{peak}`, where :math:`L(t_\mathrm{peak}) = A`
    exactly, by construction. ``t_peak`` is not itself a free parameter --
    it is fixed at :math:`t_\mathrm{peak} = 5\sigma_\mathrm{rise}`, five
    Gaussian widths after explosion.

    .. rubric:: Parameters

    The light curve parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude``
         - :math:`A`
         - Peak bolometric luminosity.
       * - ``sigma_rise``
         - :math:`\sigma_\mathrm{rise}`
         - Gaussian width of the rise.
       * - ``tau_decline``
         - :math:`\tau_\mathrm{decline}`
         - Exponential decline timescale.
    """

    _LIGHTCURVE_TYPE = "bolometric"

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e43 * _BOL_LUM_UNIT,
            description="Peak bolometric luminosity.",
            latex=r"A",
        ),
        "sigma_rise": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=0.5 * u.day,
            description="Gaussian width of the rise.",
            latex=r"\sigma_\mathrm{rise}",
        ),
        "tau_decline": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=3.0 * u.day,
            description="Exponential decline timescale.",
            latex=r"\tau_\mathrm{decline}",
        ),
    }

    @classmethod
    def _eval(cls, t: NDArray[np.float64], **parameters: CGSParameterValue) -> NDArray[np.float64]:
        r"""
        Evaluate the natural log of :math:`L_\mathrm{bol}(t)`; see the class docstring for the functional form.

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
        amplitude = parameters["amplitude"]
        sigma_rise = parameters["sigma_rise"]
        t_peak = 5 * sigma_rise
        tau_decline = parameters["tau_decline"]

        is_rising = t <= t_peak
        log_rise = -0.5 * ((t - t_peak) / sigma_rise) ** 2
        log_decline = -(t - t_peak) / tau_decline

        return np.log(amplitude) + np.where(is_rising, log_rise, log_decline)


class GaussianRisePowerLawLightcurve(Lightcurve):
    r"""
    A Gaussian rise followed by a power-law decline, peaked at ``t_peak``.

    .. math::

        L(t) = A \times \begin{cases}
            \exp\left(-\dfrac{(t - t_\mathrm{peak})^2}{2\sigma_\mathrm{rise}^2}\right)
                & t \le t_\mathrm{peak} \\[4pt]
            \left(\dfrac{t}{t_\mathrm{peak}}\right)^{-\alpha_\mathrm{decline}}
                & t > t_\mathrm{peak}
        \end{cases}

    The two branches agree exactly at :math:`t = t_\mathrm{peak}`, where
    :math:`L(t_\mathrm{peak}) = A` -- the power-law branch is referenced to
    :math:`t_\mathrm{peak}` rather than :math:`t = 0` for exactly this
    reason. Unlike `GREDLightcurve`'s exponential cutoff, the power-law tail
    here (:math:`L \propto t^{-\alpha_\mathrm{decline}}` for :math:`t \gg
    t_\mathrm{peak}`) fades slowly at late times -- the qualitative
    late-time behavior of transients with an extended power source (e.g.
    luminous fast blue optical transients) rather than a sharply cut-off
    pulse.

    ``sigma_rise`` is not itself a free parameter -- it is fixed at
    :math:`\sigma_\mathrm{rise} = t_\mathrm{peak}/5`, one fifth of the time
    to peak, the same ratio `GREDLightcurve` fixes between the two
    quantities.

    .. rubric:: Parameters

    The light curve parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude``
         - :math:`A`
         - Peak bolometric luminosity.
       * - ``t_peak``
         - :math:`t_\mathrm{peak}`
         - Time of peak luminosity since explosion.
       * - ``decline_index``
         - :math:`\alpha_\mathrm{decline}`
         - Positive post-peak power-law decline index.
    """

    _LIGHTCURVE_TYPE = "bolometric"

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e43 * _BOL_LUM_UNIT,
            description="Peak bolometric luminosity.",
            latex=r"A",
        ),
        "t_peak": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=5.0 * u.day,
            description="Time of peak luminosity since explosion.",
            latex=r"t_\mathrm{peak}",
        ),
        "decline_index": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1.5 * u.dimensionless_unscaled,
            description="Positive post-peak power-law decline index.",
            latex=r"\alpha_\mathrm{decline}",
        ),
    }

    # Narrowing `**parameters` to this model's own named parameters (rather
    # than matching the base class's generic `**parameters` exactly) is the
    # intended pattern for every `_eval` override -- see `Lightcurve._eval`.
    @classmethod
    def _eval(  # type: ignore[override]
        cls,
        t: NDArray[np.float64],
        *,
        amplitude: CGSParameterValue,
        t_peak: CGSParameterValue,
        decline_index: CGSParameterValue,
    ) -> NDArray[np.float64]:
        r"""
        Evaluate the natural log of :math:`L_\mathrm{bol}(t)`; see the class docstring for the functional form.

        Parameters
        ----------
        t : numpy.ndarray
            Time since explosion, in seconds.
        amplitude, t_peak, decline_index : float or numpy.ndarray
            This model's parameter values, in cgs units; see the class docstring.

        Returns
        -------
        numpy.ndarray
            The natural log of :math:`L_\mathrm{bol}(t)`, in erg/s.
        """
        sigma_rise = t_peak / 5
        log_rise = -0.5 * ((t - t_peak) / sigma_rise) ** 2

        # `t_peak` is strictly positive (its prior's support excludes 0), so
        # `t = 0` always falls on the rising branch -- the `log(0) = -inf`
        # this produces on the discarded decline branch is a harmless
        # masked-out value, not an error, exactly as in `FREDLightcurve._eval`.
        with np.errstate(divide="ignore"):
            log_decline = -decline_index * np.log(t / t_peak)

        return np.log(amplitude) + np.where(t <= t_peak, log_rise, log_decline)


class GaussianRiseBrokenPowerLawLightcurve(Lightcurve):
    r"""
    A Gaussian rise followed by a broken power-law decline, peaked at ``t_peak``.

    .. math::

        L(t) = A \times \begin{cases}
            \exp\left(-\dfrac{(t - t_\mathrm{peak})^2}{2\sigma_\mathrm{rise}^2}\right)
                & t \le t_\mathrm{peak} \\[4pt]
            \left(\dfrac{t}{t_\mathrm{peak}}\right)^{-\alpha_1}
                & t_\mathrm{peak} < t \le t_\mathrm{break} \\[4pt]
            \left(\dfrac{t_\mathrm{break}}{t_\mathrm{peak}}\right)^{-\alpha_1}
            \left(\dfrac{t}{t_\mathrm{break}}\right)^{-\alpha_2}
                & t > t_\mathrm{break}
        \end{cases}

    Extends `GaussianRisePowerLawLightcurve` with a second, steeper power-law
    segment past a break time ``t_break``. Every branch agrees exactly at its
    boundary, by construction: :math:`L(t_\mathrm{peak}) = A`, and the two
    decline branches meet at :math:`t_\mathrm{break}` without a jump. Useful
    for transients whose late-time decline steepens relative to the
    early-time behavior -- e.g. a kilonova's blue/early ejecta component
    fading faster once it becomes optically thin.

    ``t_break`` is assumed to exceed ``t_peak``, although this ordering is
    not enforced by the model itself.

    ``sigma_rise`` is not itself a free parameter -- it is fixed at
    :math:`\sigma_\mathrm{rise} = t_\mathrm{peak}/5`, one fifth of the time
    to peak, the same ratio `GREDLightcurve` fixes between the two
    quantities.

    .. rubric:: Parameters

    The light curve parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude``
         - :math:`A`
         - Peak bolometric luminosity.
       * - ``t_peak``
         - :math:`t_\mathrm{peak}`
         - Time of peak luminosity since explosion.
       * - ``decline_index_1``
         - :math:`\alpha_1`
         - Positive power-law decline index between ``t_peak`` and ``t_break``.
       * - ``decline_index_2``
         - :math:`\alpha_2`
         - Positive power-law decline index past ``t_break``.
       * - ``t_break``
         - :math:`t_\mathrm{break}`
         - Time at which the decline steepens from :math:`\alpha_1` to :math:`\alpha_2`.
    """

    _LIGHTCURVE_TYPE = "bolometric"

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e43 * _BOL_LUM_UNIT,
            description="Peak bolometric luminosity.",
            latex=r"A",
        ),
        "t_peak": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=5.0 * u.day,
            description="Time of peak luminosity since explosion.",
            latex=r"t_\mathrm{peak}",
        ),
        "decline_index_1": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1.0 * u.dimensionless_unscaled,
            description="Positive power-law decline index between t_peak and t_break.",
            latex=r"\alpha_1",
        ),
        "decline_index_2": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=3.0 * u.dimensionless_unscaled,
            description="Positive power-law decline index past t_break.",
            latex=r"\alpha_2",
        ),
        "t_break": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=10.0 * u.day,
            description="Time at which the decline steepens from decline_index_1 to decline_index_2.",
            latex=r"t_\mathrm{break}",
        ),
    }

    # Narrowing `**parameters` to this model's own named parameters (rather
    # than matching the base class's generic `**parameters` exactly) is the
    # intended pattern for every `_eval` override -- see `Lightcurve._eval`.
    @classmethod
    def _eval(  # type: ignore[override]
        cls,
        t: NDArray[np.float64],
        *,
        amplitude: CGSParameterValue,
        t_peak: CGSParameterValue,
        decline_index_1: CGSParameterValue,
        decline_index_2: CGSParameterValue,
        t_break: CGSParameterValue,
    ) -> NDArray[np.float64]:
        r"""
        Evaluate the natural log of :math:`L_\mathrm{bol}(t)`; see the class docstring for the functional form.

        Parameters
        ----------
        t : numpy.ndarray
            Time since explosion, in seconds.
        amplitude, t_peak, decline_index_1, decline_index_2, t_break : float or numpy.ndarray
            This model's parameter values, in cgs units; see the class docstring.

        Returns
        -------
        numpy.ndarray
            The natural log of :math:`L_\mathrm{bol}(t)`, in erg/s.
        """
        sigma_rise = t_peak / 5
        log_rise = -0.5 * ((t - t_peak) / sigma_rise) ** 2

        # `t_peak`/`t_break` are strictly positive (their priors' support
        # excludes 0), so `t = 0` always falls on the rising branch -- the
        # `log(0) = -inf` this produces on the discarded decline branches is
        # a harmless masked-out value, not an error, exactly as in
        # `GaussianRisePowerLawLightcurve._eval`.
        with np.errstate(divide="ignore"):
            log_decline_early = -decline_index_1 * np.log(t / t_peak)
            log_break_offset = -decline_index_1 * np.log(t_break / t_peak)
            log_decline_late = log_break_offset - decline_index_2 * np.log(t / t_break)

        log_shape = np.where(
            t <= t_peak,
            log_rise,
            np.where(t <= t_break, log_decline_early, log_decline_late),
        )

        return np.log(amplitude) + log_shape


class BazinLightcurve(Lightcurve):
    r"""
    A smooth asymmetric transient following the Bazin functional form.

    .. math::

        L(t) =
        A\,
        \frac{
            \exp[-(t-t_0)/\tau_\mathrm{fall}]
        }{
            1 + \exp[-(t-t_0)/\tau_\mathrm{rise}]
        }.

    The model combines a logistic-like rise with an exponential decline and is
    commonly useful as a generic phenomenological approximation to supernova-like
    transients.

    A finite maximum exists when

    .. math::

        \tau_\mathrm{fall} > \tau_\mathrm{rise}.

    Unlike most of the other pulse models in this module, ``amplitude`` is the
    normalization of the Bazin function and is not, in general, exactly the peak
    luminosity.

    .. rubric:: Parameters

    The light curve parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude``
         - :math:`A`
         - Luminosity normalization.
       * - ``t0``
         - :math:`t_0`
         - Characteristic transition time.
       * - ``rise``
         - :math:`\tau_\mathrm{rise}`
         - Logistic rise timescale.
       * - ``fall``
         - :math:`\tau_\mathrm{fall}`
         - Exponential decline timescale.
    """

    _LIGHTCURVE_TYPE = "bolometric"

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e43 * _BOL_LUM_UNIT,
            description="Luminosity normalization.",
            latex=r"A",
        ),
        "t0": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=10.0 * u.day,
            description="Characteristic transition time.",
            latex=r"t_0",
        ),
        "rise": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=5.0 * u.day,
            description="Logistic rise timescale.",
            latex=r"\tau_\mathrm{rise}",
        ),
        "fall": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.3),
            scale=30.0 * u.day,
            description="Exponential decline timescale.",
            latex=r"\tau_\mathrm{fall}",
        ),
    }

    @classmethod
    def _eval(
        cls,
        t: NDArray[np.float64],
        **parameters: CGSParameterValue,
    ) -> NDArray[np.float64]:
        r"""
        Evaluate the natural log of :math:`L_\mathrm{bol}(t)`; see the class docstring for the functional form.

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
        amplitude = parameters["amplitude"]
        t0 = parameters["t0"]
        rise = parameters["rise"]
        fall = parameters["fall"]

        x = t - t0

        return np.log(amplitude) - x / fall - np.logaddexp(0.0, -x / rise)


class PowerLawLightcurve(Lightcurve):
    r"""
    A power-law decline beginning at a reference time.

    .. math::

        L(t) =
        \begin{cases}
            0, & t < t_\mathrm{ref}, \\[4pt]
            A \left(\dfrac{t}{t_\mathrm{ref}}\right)^{-\alpha},
            & t \ge t_\mathrm{ref}.
        \end{cases}

    The luminosity is therefore exactly :math:`A` at ``t_ref``. This form is
    useful for generic afterglows and other transients whose post-onset emission
    is approximately scale free.

    Restricting the model to ``t >= t_ref`` avoids the formal divergence of a
    declining power law as :math:`t \rightarrow 0`.

    .. rubric:: Parameters

    The light curve parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude``
         - :math:`A`
         - Luminosity at the reference time.
       * - ``t_ref``
         - :math:`t_\mathrm{ref}`
         - Reference time at which the power law begins.
       * - ``index``
         - :math:`\alpha`
         - Positive power-law decline index.
    """

    _LIGHTCURVE_TYPE = "bolometric"

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e43 * _BOL_LUM_UNIT,
            description="Luminosity at the reference time.",
            latex=r"A",
        ),
        "t_ref": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1.0 * u.day,
            description="Reference time at which the power law begins.",
            latex=r"t_\mathrm{ref}",
        ),
        "index": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1.0 * u.dimensionless_unscaled,
            description="Positive power-law decline index.",
            latex=r"\alpha",
        ),
    }

    @classmethod
    def _eval(
        cls,
        t: NDArray[np.float64],
        **parameters: CGSParameterValue,
    ) -> NDArray[np.float64]:
        r"""
        Evaluate the natural log of :math:`L_\mathrm{bol}(t)`; see the class docstring for the functional form.

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
        amplitude = parameters["amplitude"]
        t_ref = parameters["t_ref"]
        index = parameters["index"]

        with np.errstate(divide="ignore", invalid="ignore"):
            log_luminosity = np.log(amplitude) - index * np.log(t / t_ref)

        return np.where(t >= t_ref, log_luminosity, -np.inf)


class BrokenPowerLawLightcurve(Lightcurve):
    r"""
    A sharply broken power-law transient with a rise and decline.

    .. math::

        L(t) =
        A
        \begin{cases}
            \left(\dfrac{t}{t_\mathrm{peak}}\right)^{\alpha_\mathrm{rise}},
                & 0 < t \le t_\mathrm{peak}, \\[6pt]
            \left(\dfrac{t}{t_\mathrm{peak}}\right)^{-\alpha_\mathrm{decline}},
                & t > t_\mathrm{peak}.
        \end{cases}

    Both indices are defined to be positive. The two branches meet exactly at
    ``t_peak``, where :math:`L(t_\mathrm{peak}) = A`. For positive rise index,
    the luminosity tends continuously to zero as :math:`t \rightarrow 0`.

    .. rubric:: Parameters

    The light curve parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude``
         - :math:`A`
         - Peak bolometric luminosity.
       * - ``t_peak``
         - :math:`t_\mathrm{peak}`
         - Time of the power-law break and peak.
       * - ``rise_index``
         - :math:`\alpha_\mathrm{rise}`
         - Positive pre-peak power-law index.
       * - ``decline_index``
         - :math:`\alpha_\mathrm{decline}`
         - Positive post-peak power-law decline index.
    """

    _LIGHTCURVE_TYPE = "bolometric"

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e43 * _BOL_LUM_UNIT,
            description="Peak bolometric luminosity.",
            latex=r"A",
        ),
        "t_peak": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=10.0 * u.day,
            description="Time of the power-law break and peak.",
            latex=r"t_\mathrm{peak}",
        ),
        "rise_index": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=2.0 * u.dimensionless_unscaled,
            description="Positive pre-peak power-law index.",
            latex=r"\alpha_\mathrm{rise}",
        ),
        "decline_index": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1.5 * u.dimensionless_unscaled,
            description="Positive post-peak power-law decline index.",
            latex=r"\alpha_\mathrm{decline}",
        ),
    }

    @classmethod
    def _eval(
        cls,
        t: NDArray[np.float64],
        **parameters: CGSParameterValue,
    ) -> NDArray[np.float64]:
        r"""
        Evaluate the natural log of :math:`L_\mathrm{bol}(t)`; see the class docstring for the functional form.

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
        amplitude = parameters["amplitude"]
        t_peak = parameters["t_peak"]
        rise_index = parameters["rise_index"]
        decline_index = parameters["decline_index"]

        with np.errstate(divide="ignore"):
            log_time = np.log(t / t_peak)

        log_shape = np.where(
            t <= t_peak,
            rise_index * log_time,
            -decline_index * log_time,
        )

        return np.log(amplitude) + log_shape


class SmoothBrokenPowerLawLightcurve(Lightcurve):
    r"""
    A smoothly broken power-law transient.

    The asymptotic behavior is

    .. math::

        L(t) \propto
        \begin{cases}
            t^{\alpha_\mathrm{rise}}, & t \ll t_\mathrm{peak}, \\
            t^{-\alpha_\mathrm{decline}}, & t \gg t_\mathrm{peak},
        \end{cases}

    with a smooth transition between the two branches. The implementation uses

    .. math::

        f(x) =
        \left[
            x^{-s\alpha_\mathrm{rise}}
            +
            x^{s\alpha_\mathrm{decline}}
        \right]^{-1/s},

    but rescales the argument and normalization so that the maximum occurs
    exactly at ``t_peak`` and

    .. math::

        L(t_\mathrm{peak}) = A.

    The positive ``smoothness`` parameter :math:`s` controls the sharpness of
    the transition: larger values approach a sharply broken power law.

    .. rubric:: Parameters

    The light curve parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude``
         - :math:`A`
         - Peak bolometric luminosity.
       * - ``t_peak``
         - :math:`t_\mathrm{peak}`
         - Time of peak luminosity.
       * - ``rise_index``
         - :math:`\alpha_\mathrm{rise}`
         - Positive asymptotic rise index.
       * - ``decline_index``
         - :math:`\alpha_\mathrm{decline}`
         - Positive asymptotic decline index.
       * - ``smoothness``
         - :math:`s`
         - Sharpness of the transition between power-law branches.
    """

    _LIGHTCURVE_TYPE = "bolometric"

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e43 * _BOL_LUM_UNIT,
            description="Peak bolometric luminosity.",
            latex=r"A",
        ),
        "t_peak": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=10.0 * u.day,
            description="Time of peak luminosity.",
            latex=r"t_\mathrm{peak}",
        ),
        "rise_index": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=2.0 * u.dimensionless_unscaled,
            description="Positive asymptotic rise index.",
            latex=r"\alpha_\mathrm{rise}",
        ),
        "decline_index": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1.5 * u.dimensionless_unscaled,
            description="Positive asymptotic decline index.",
            latex=r"\alpha_\mathrm{decline}",
        ),
        "smoothness": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.4),
            scale=2.0 * u.dimensionless_unscaled,
            description="Sharpness of the transition between power-law branches.",
            latex=r"s",
        ),
    }

    @classmethod
    def _eval(
        cls,
        t: NDArray[np.float64],
        **parameters: CGSParameterValue,
    ) -> NDArray[np.float64]:
        r"""
        Evaluate the natural log of :math:`L_\mathrm{bol}(t)`; see the class docstring for the functional form.

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
        amplitude = parameters["amplitude"]
        t_peak = parameters["t_peak"]
        rise_index = parameters["rise_index"]
        decline_index = parameters["decline_index"]
        smoothness = parameters["smoothness"]

        # For
        #
        #   f(y) = [y^(-s a) + y^(s b)]^(-1/s),
        #
        # the maximum occurs at
        #
        #   y_peak = (a / b)^[1 / (s(a+b))].
        #
        # Scale y so that t=t_peak corresponds exactly to y=y_peak.
        log_y_peak = np.log(rise_index / decline_index) / (smoothness * (rise_index + decline_index))

        with np.errstate(divide="ignore"):
            log_y = np.log(t / t_peak) + log_y_peak

        log_shape = -(1.0 / smoothness) * np.logaddexp(
            -smoothness * rise_index * log_y,
            smoothness * decline_index * log_y,
        )

        log_shape_peak = -(1.0 / smoothness) * np.logaddexp(
            -smoothness * rise_index * log_y_peak,
            smoothness * decline_index * log_y_peak,
        )

        return np.log(amplitude) + log_shape - log_shape_peak


class DelayedExponentialLightcurve(Lightcurve):
    r"""
    A polynomial rise followed by an exponential decline.

    .. math::

        L(t) =
        A
        \left(\frac{t}{t_\mathrm{peak}}\right)^\alpha
        \exp\left[
            \alpha\left(1-\frac{t}{t_\mathrm{peak}}\right)
        \right],
        \qquad t > 0.

    This is a gamma-like transient pulse. The chosen parameterization makes

    .. math::

        L(t_\mathrm{peak}) = A

    exactly, while :math:`L \rightarrow 0` as :math:`t \rightarrow 0^+` and
    the late-time emission declines exponentially.

    The positive shape parameter :math:`\alpha` controls both the steepness of
    the rise and the relation between the peak time and exponential timescale.

    .. rubric:: Parameters

    The light curve parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude``
         - :math:`A`
         - Peak bolometric luminosity.
       * - ``t_peak``
         - :math:`t_\mathrm{peak}`
         - Time of peak luminosity.
       * - ``shape``
         - :math:`\alpha`
         - Positive dimensionless pulse-shape parameter.
    """

    _LIGHTCURVE_TYPE = "bolometric"

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e43 * _BOL_LUM_UNIT,
            description="Peak bolometric luminosity.",
            latex=r"A",
        ),
        "t_peak": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=5.0 * u.day,
            description="Time of peak luminosity.",
            latex=r"t_\mathrm{peak}",
        ),
        "shape": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=2.0 * u.dimensionless_unscaled,
            description="Positive dimensionless pulse-shape parameter.",
            latex=r"\alpha",
        ),
    }

    @classmethod
    def _eval(
        cls,
        t: NDArray[np.float64],
        **parameters: CGSParameterValue,
    ) -> NDArray[np.float64]:
        r"""
        Evaluate the natural log of :math:`L_\mathrm{bol}(t)`; see the class docstring for the functional form.

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
        amplitude = parameters["amplitude"]
        t_peak = parameters["t_peak"]
        shape = parameters["shape"]

        x = t / t_peak

        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(amplitude) + shape * np.log(x) + shape * (1.0 - x)


class LogNormalPulseLightcurve(Lightcurve):
    r"""
    A log-normal pulse in time.

    .. math::

        L(t) =
        A
        \exp\left[
            -\frac{1}{2}
            \left(
                \frac{\ln(t/t_\mathrm{peak})}{\sigma}
            \right)^2
        \right],
        \qquad t > 0.

    The pulse peaks exactly at ``t_peak`` with

    .. math::

        L(t_\mathrm{peak}) = A.

    Unlike a Gaussian pulse in linear time, a log-normal pulse is intrinsically
    asymmetric and has support only at positive times. The dimensionless width
    ``sigma`` controls the width in logarithmic time.

    .. rubric:: Parameters

    The light curve parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude``
         - :math:`A`
         - Peak bolometric luminosity.
       * - ``t_peak``
         - :math:`t_\mathrm{peak}`
         - Time of peak luminosity.
       * - ``sigma``
         - :math:`\sigma`
         - Width of the pulse in logarithmic time.
    """

    _LIGHTCURVE_TYPE = "bolometric"

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e43 * _BOL_LUM_UNIT,
            description="Peak bolometric luminosity.",
            latex=r"A",
        ),
        "t_peak": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=5.0 * u.day,
            description="Time of peak luminosity.",
            latex=r"t_\mathrm{peak}",
        ),
        "sigma": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.4),
            scale=0.5 * u.dimensionless_unscaled,
            description="Width of the pulse in logarithmic time.",
            latex=r"\sigma",
        ),
    }

    @classmethod
    def _eval(
        cls,
        t: NDArray[np.float64],
        **parameters: CGSParameterValue,
    ) -> NDArray[np.float64]:
        r"""
        Evaluate the natural log of :math:`L_\mathrm{bol}(t)`; see the class docstring for the functional form.

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
        amplitude = parameters["amplitude"]
        t_peak = parameters["t_peak"]
        sigma = parameters["sigma"]

        with np.errstate(divide="ignore"):
            log_time = np.log(t / t_peak)

        return np.log(amplitude) - 0.5 * (log_time / sigma) ** 2


class PlateauPowerLawLightcurve(Lightcurve):
    r"""
    A constant plateau followed by a power-law decline.

    .. math::

        L(t) =
        A
        \begin{cases}
            1, & 0 \le t \le t_\mathrm{break}, \\[4pt]
            \left(\dfrac{t}{t_\mathrm{break}}\right)^{-\alpha},
                & t > t_\mathrm{break}.
        \end{cases}

    This provides a minimal phenomenological description of a transient with a
    sustained plateau followed by scale-free fading, such as some GRB afterglow
    and engine-powered transient light curves.

    .. rubric:: Parameters

    The light curve parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude``
         - :math:`A`
         - Plateau bolometric luminosity.
       * - ``t_break``
         - :math:`t_\mathrm{break}`
         - Time at which the plateau transitions to a power-law decline.
       * - ``index``
         - :math:`\alpha`
         - Positive post-plateau power-law decline index.
    """

    _LIGHTCURVE_TYPE = "bolometric"

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e43 * _BOL_LUM_UNIT,
            description="Plateau bolometric luminosity.",
            latex=r"A",
        ),
        "t_break": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1.0 * u.day,
            description="Time at which the plateau transitions to a power-law decline.",
            latex=r"t_\mathrm{break}",
        ),
        "index": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1.5 * u.dimensionless_unscaled,
            description="Positive post-plateau power-law decline index.",
            latex=r"\alpha",
        ),
    }

    @classmethod
    def _eval(
        cls,
        t: NDArray[np.float64],
        **parameters: CGSParameterValue,
    ) -> NDArray[np.float64]:
        r"""
        Evaluate the natural log of :math:`L_\mathrm{bol}(t)`; see the class docstring for the functional form.

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
        amplitude = parameters["amplitude"]
        t_break = parameters["t_break"]
        index = parameters["index"]

        with np.errstate(divide="ignore"):
            log_decline = -index * np.log(t / t_break)

        return np.log(amplitude) + np.where(
            t <= t_break,
            0.0,
            log_decline,
        )


class VillarLightcurve(Lightcurve):
    r"""
    A parametric supernova-like light curve.

    .. math::

        L(t) =
        A \times
        \begin{cases}
            \dfrac{1 + \beta(t - t_0)}{1 + \exp[-(t-t_0)/\tau_\mathrm{rise}]},
                & t < t_1, \\[8pt]
            \dfrac{(1 + \beta\gamma)\,
                \exp[-(t-t_1)/\tau_\mathrm{fall}]}{1 + \exp[-(t-t_0)/\tau_\mathrm{rise}]},
                & t \ge t_1,
        \end{cases}

    with :math:`t_1 = t_0 + \gamma`. A logistic rise (timescale
    :math:`\tau_\mathrm{rise}`) turns on around :math:`t_0`; the light curve
    then declines linearly with slope :math:`\beta` (a plateau, for small
    :math:`|\beta|`) until :math:`t_1`, after which it switches to an
    exponential decline with timescale :math:`\tau_\mathrm{fall}`. The two
    branches agree exactly at :math:`t_1`, by construction.

    This is the parametric form used by :footcite:t:`2019ApJ...884...83V` to
    fit multi-band supernova photometry, adapted here to a bolometric
    luminosity rather than a per-band flux.

    .. rubric:: Parameters

    The light curve parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude``
         - :math:`A`
         - Overall luminosity normalization.
       * - ``t0``
         - :math:`t_0`
         - Reference time at which the logistic rise is centered.
       * - ``gamma``
         - :math:`\gamma`
         - Duration of the plateau, measured from t0.
       * - ``beta``
         - :math:`\beta`
         - Linear slope of the plateau.
       * - ``tau_rise``
         - :math:`\tau_\mathrm{rise}`
         - Logistic rise timescale.
       * - ``tau_fall``
         - :math:`\tau_\mathrm{fall}`
         - Exponential decline timescale, after the plateau.

    See Also
    --------
    uvex_transients.models.supernovae.VillarCoolingBlackbodySED :
        Pairs this bolometric light curve with a cooling-blackbody spectrum.

    References
    ----------
    .. footbibliography::
    """

    _LIGHTCURVE_TYPE = "bolometric"

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e43 * _BOL_LUM_UNIT,
            description="Overall luminosity normalization.",
            latex=r"A",
        ),
        "t0": Parameter(
            prior=NormalPrior(mean=0.0, sigma=1.0),
            scale=1.0 * u.day,
            description="Reference time at which the logistic rise is centered.",
            latex=r"t_0",
        ),
        "gamma": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=10.0 * u.day,
            description="Duration of the plateau, measured from t0.",
            latex=r"\gamma",
        ),
        "beta": Parameter(
            prior=NormalPrior(mean=0.0, sigma=0.5),
            scale=1e-2 / u.day,
            description="Linear slope of the plateau.",
            latex=r"\beta",
        ),
        "tau_rise": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=5.0 * u.day,
            description="Logistic rise timescale.",
            latex=r"\tau_\mathrm{rise}",
        ),
        "tau_fall": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=30.0 * u.day,
            description="Exponential decline timescale, after the plateau.",
            latex=r"\tau_\mathrm{fall}",
        ),
    }

    # Narrowing `**parameters` to this model's own named parameters (rather
    # than matching the base class's generic `**parameters` exactly) is the
    # intended pattern for every `_eval` override -- see `Lightcurve._eval`.
    @classmethod
    def _eval(  # type: ignore[override]
        cls,
        t: NDArray[np.float64],
        *,
        amplitude: CGSParameterValue,
        t0: CGSParameterValue,
        gamma: CGSParameterValue,
        beta: CGSParameterValue,
        tau_rise: CGSParameterValue,
        tau_fall: CGSParameterValue,
    ) -> NDArray[np.float64]:
        r"""
        Evaluate the natural log of :math:`L_\mathrm{bol}(t)`; see the class docstring for the functional form.

        Parameters
        ----------
        t : numpy.ndarray
            Time since explosion, in seconds.
        amplitude, t0, gamma, beta, tau_rise, tau_fall : float or numpy.ndarray
            This model's parameter values, in cgs units; see the class docstring.

        Returns
        -------
        numpy.ndarray
            The natural log of :math:`L_\mathrm{bol}(t)`, in erg/s.
        """
        x = t - t0
        t1 = t0 + gamma
        is_rise = t < t1

        with np.errstate(invalid="ignore"):
            linear = np.where(
                is_rise,
                1.0 + beta * x,
                (1.0 + beta * gamma) * np.exp(-(t - t1) / tau_fall),
            )
            log_shape = np.log(linear) - np.logaddexp(0.0, -x / tau_rise)

        return np.log(amplitude) + log_shape


class TwoComponentBazinLightcurve(Lightcurve):
    r"""
    A superposition of two independent Bazin pulses.

    .. math::

        L(t) =
        A_0\,
        \frac{\exp[-(t-t_0)/\tau_{\mathrm{fall},0}]}{1 + \exp[-(t-t_0)/\tau_{\mathrm{rise},0}]}
        +
        A_1\,
        \frac{\exp[-(t-t_1)/\tau_{\mathrm{fall},1}]}{1 + \exp[-(t-t_1)/\tau_{\mathrm{rise},1}]}.

    Two ordinary :class:`BazinLightcurve` pulses, added rather than multiplied, each with its own
    amplitude, transition time, and rise/fall timescales. Because the two components are
    independent and additive, this covers both single- and double-peaked light curves in the same
    functional form: with :math:`A_0 \ll A_1` (or a comparable but much earlier/narrower first
    pulse well clear of the second), the second component alone sets the observed shape -- a
    single peak; with :math:`A_0` comparable to :math:`A_1`, both pulses are visible, with a dip
    between them where each has decayed enough to let the other dominate. Since the two peaks are
    genuinely independent components rather than one shape reweighted by another, their relative
    amplitude, timing, and widths can be tuned separately to fit real double- and single-peaked
    events alike.

    .. rubric:: Parameters

    The light curve parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude_0``
         - :math:`A_0`
         - Luminosity normalization of the first (typically earlier) pulse.
       * - ``t0``
         - :math:`t_0`
         - Characteristic transition time of the first pulse.
       * - ``rise_0``
         - :math:`\tau_{\mathrm{rise},0}`
         - Logistic rise timescale of the first pulse.
       * - ``fall_0``
         - :math:`\tau_{\mathrm{fall},0}`
         - Exponential decline timescale of the first pulse.
       * - ``amplitude_1``
         - :math:`A_1`
         - Luminosity normalization of the second (typically later) pulse.
       * - ``t1``
         - :math:`t_1`
         - Characteristic transition time of the second pulse.
       * - ``rise_1``
         - :math:`\tau_{\mathrm{rise},1}`
         - Logistic rise timescale of the second pulse.
       * - ``fall_1``
         - :math:`\tau_{\mathrm{fall},1}`
         - Exponential decline timescale of the second pulse.

    See Also
    --------
    BazinLightcurve :
        The single-pulse form this combines two of.
    """

    _LIGHTCURVE_TYPE = "bolometric"

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude_0": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e42 * _BOL_LUM_UNIT,
            description="Luminosity normalization of the first (typically earlier) pulse.",
            latex=r"A_0",
        ),
        "t0": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=2.0 * u.day,
            description="Characteristic transition time of the first pulse.",
            latex=r"t_0",
        ),
        "rise_0": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=0.3 * u.day,
            description="Logistic rise timescale of the first pulse.",
            latex=r"\tau_{\mathrm{rise},0}",
        ),
        "fall_0": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.3),
            scale=5.0 * u.day,
            description="Exponential decline timescale of the first pulse.",
            latex=r"\tau_{\mathrm{fall},0}",
        ),
        "amplitude_1": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e43 * _BOL_LUM_UNIT,
            description="Luminosity normalization of the second (typically later) pulse.",
            latex=r"A_1",
        ),
        "t1": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=18.0 * u.day,
            description="Characteristic transition time of the second pulse.",
            latex=r"t_1",
        ),
        "rise_1": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=5.0 * u.day,
            description="Logistic rise timescale of the second pulse.",
            latex=r"\tau_{\mathrm{rise},1}",
        ),
        "fall_1": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.3),
            scale=30.0 * u.day,
            description="Exponential decline timescale of the second pulse.",
            latex=r"\tau_{\mathrm{fall},1}",
        ),
    }

    @classmethod
    def _eval(
        cls,
        t: NDArray[np.float64],
        **parameters: CGSParameterValue,
    ) -> NDArray[np.float64]:
        r"""
        Evaluate the natural log of :math:`L_\mathrm{bol}(t)`; see the class docstring for the functional form.

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
        x0 = t - parameters["t0"]
        log_bazin_0 = -x0 / parameters["fall_0"] - np.logaddexp(0.0, -x0 / parameters["rise_0"])

        x1 = t - parameters["t1"]
        log_bazin_1 = -x1 / parameters["fall_1"] - np.logaddexp(0.0, -x1 / parameters["rise_1"])

        return np.logaddexp(
            np.log(parameters["amplitude_0"]) + log_bazin_0,
            np.log(parameters["amplitude_1"]) + log_bazin_1,
        )


class InverseGammaBrokenPowerLawLightcurve(Lightcurve):
    r"""
    An inverse-gamma rise with a smoothly broken power-law decline, normalized to a given peak luminosity.

    .. math::

        L(t) = \frac{L_0}{\mathcal{N}}\; x^{-p}\, \exp\left[p\left(1-\frac{1}{x}\right)\right]
        \left[1+\left(\frac{x}{x_s}\right)^{k(p-q)}\right]^{1/k},
        \qquad x = \frac{t}{t_0},\quad t > 0.

    The factor :math:`x^{-p} \exp[p(1-1/x)]` switches the emission on as :math:`t \rightarrow 0`
    (it vanishes faster than any power of :math:`x`), peaks at :math:`x = 1`, and then declines as
    :math:`x^{-p}`. The bracket is a smooth break in that power law, centred on :math:`x_s` with sharpness
    :math:`k` (larger is sharper). For :math:`p > q` the light curve follows :math:`x^{-p}` before the break and
    flattens onto

    .. math::

        L \propto x^{-q}, \qquad x \gg x_s,

    after it. For :math:`p < q` the roles of the two indices are exchanged. A peak exists for :math:`q > 0`.

    The normalization :math:`\mathcal{N}` is the maximum of the unnormalized expression, so that

    .. math::

        \max_t L(t) = L_0

    exactly. The maximum has no closed form once the break is present. It is bracketed analytically, since
    it lies between :math:`x = 1` and :math:`x = p/q`, and then located numerically, independently for each
    set of parameters.

    .. rubric:: Parameters

    The light curve parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude``
         - :math:`L_0`
         - Peak bolometric luminosity.
       * - ``t_scale``
         - :math:`t_0`
         - Time scale; the dimensionless time is :math:`x = t / t_0`.
       * - ``p``
         - :math:`p`
         - Index of the rise-and-decline factor, and of the decline before the break.
       * - ``q``
         - :math:`q`
         - Index of the late-time decline after the break.
       * - ``x_break``
         - :math:`x_s`
         - Location of the break, in units of :math:`t_0`.
       * - ``smoothness``
         - :math:`k`
         - Sharpness of the break.

    See Also
    --------
    SmoothBrokenPowerLawLightcurve :
        A smoothly broken power law with a power-law rise, normalized to peak at a given time.
    DelayedExponentialLightcurve :
        A gamma-like pulse with an exponential decline.
    """

    _LIGHTCURVE_TYPE = "bolometric"

    # Number of grid points, and of bisection steps, used to find the peak of the shape.
    _N_PEAK_GRID: ClassVar[int] = 65
    _N_PEAK_BISECT: ClassVar[int] = 50

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e43 * _BOL_LUM_UNIT,
            description="Peak bolometric luminosity.",
            latex=r"L_0",
        ),
        "t_scale": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1.0 * u.day,
            description="Time scale; the dimensionless time is t / t_scale.",
            latex=r"t_0",
        ),
        "p": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.3),
            scale=2.0 * u.dimensionless_unscaled,
            description="Index of the rise-and-decline factor, and of the decline before the break.",
            latex=r"p",
        ),
        "q": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.3),
            scale=1.0 * u.dimensionless_unscaled,
            description="Index of the late-time decline after the break.",
            latex=r"q",
        ),
        "x_break": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=5.0 * u.dimensionless_unscaled,
            description="Location of the break, in units of t_scale.",
            latex=r"x_s",
        ),
        "smoothness": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.4),
            scale=2.0 * u.dimensionless_unscaled,
            description="Sharpness of the break.",
            latex=r"k",
        ),
    }

    @staticmethod
    def _log_shape(
        log_x: NDArray[np.float64],
        p: NDArray[np.float64],
        q: NDArray[np.float64],
        x_break: NDArray[np.float64],
        smoothness: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Natural log of the unnormalized shape as a function of the log of the dimensionless time.

        Parameters
        ----------
        log_x : numpy.ndarray
            Natural log of the dimensionless time :math:`x = t/t_0`. Must be finite.
        p, q, x_break, smoothness : numpy.ndarray
            The shape parameters, broadcastable against `log_x`.

        Returns
        -------
        numpy.ndarray
            The natural log of the unnormalized shape.
        """
        return (
            -p * log_x
            + p * (1.0 - np.exp(-log_x))
            + np.logaddexp(0.0, smoothness * (p - q) * (log_x - np.log(x_break))) / smoothness
        )

    @classmethod
    def _peak(
        cls,
        p: NDArray[np.float64],
        q: NDArray[np.float64],
        x_break: NDArray[np.float64],
        smoothness: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Location and height of the maximum of the unnormalized shape, for each set of parameters.

        The maximum lies between :math:`x = 1` and :math:`x = p/q`. It is found on a grid in :math:`\ln x`
        over that interval, and then refined by bisection on the sign of the logarithmic derivative.

        Parameters
        ----------
        p, q, x_break, smoothness : numpy.ndarray
            The shape parameters, all broadcastable against one another.

        Returns
        -------
        log_x_peak : numpy.ndarray
            The natural log of the dimensionless time of the maximum, :math:`x_\mathrm{pk}`.
        log_shape_peak : numpy.ndarray
            The natural log of the maximum.
        """
        p, q, x_break, smoothness = np.broadcast_arrays(p, q, x_break, smoothness)

        # The interval bracketing the peak, as ln x. It collapses to a point at p = q.
        log_a = np.zeros_like(p)
        log_b = np.log(p / q)
        lo, hi = np.minimum(log_a, log_b), np.maximum(log_a, log_b)

        params = tuple(arr[..., None] for arr in (p, q, x_break, smoothness))
        grid = lo[..., None] + (hi - lo)[..., None] * np.linspace(0.0, 1.0, cls._N_PEAK_GRID)
        i_max = np.argmax(cls._log_shape(grid, *params), axis=-1)[..., None]

        # Bisect within the two grid cells around the grid maximum.
        left = np.take_along_axis(grid, np.maximum(i_max - 1, 0), axis=-1)[..., 0]
        right = np.take_along_axis(grid, np.minimum(i_max + 1, cls._N_PEAK_GRID - 1), axis=-1)[..., 0]

        def slope(log_x):
            """
            Return the logarithmic derivative of the shape with respect to x, times x.

            Parameters
            ----------
            log_x : numpy.ndarray
                The natural log of the dimensionless time.

            Returns
            -------
            numpy.ndarray
                The derivative, which is positive left of the peak.
            """
            weight = 0.5 * (1.0 + np.tanh(0.5 * smoothness * (p - q) * (log_x - np.log(x_break))))
            return p * (np.exp(-log_x) - 1.0) + (p - q) * weight

        for _ in range(cls._N_PEAK_BISECT):
            mid = 0.5 * (left + right)
            rising = slope(mid) > 0
            left = np.where(rising, mid, left)
            right = np.where(rising, right, mid)

        log_x_peak = 0.5 * (left + right)
        return log_x_peak, cls._log_shape(log_x_peak, p, q, x_break, smoothness)

    @classmethod
    def _peak_time(cls, **parameters: CGSParameterValue) -> NDArray[np.float64]:
        """
        Time of the maximum of the light curve.

        Parameters
        ----------
        **parameters
            This model's parameter values, in cgs units.

        Returns
        -------
        numpy.ndarray
            The time of peak luminosity since explosion, in seconds. It lies between ``t_scale`` and
            ``t_scale * p / q``.
        """
        log_x_peak, _ = cls._peak(parameters["p"], parameters["q"], parameters["x_break"], parameters["smoothness"])
        return parameters["t_scale"] * np.exp(log_x_peak)

    @classmethod
    def _eval(
        cls,
        t: NDArray[np.float64],
        **parameters: CGSParameterValue,
    ) -> NDArray[np.float64]:
        r"""
        Evaluate the natural log of :math:`L_\mathrm{bol}(t)`; see the class docstring for the functional form.

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
        amplitude = parameters["amplitude"]
        t_scale = parameters["t_scale"]
        p = parameters["p"]
        q = parameters["q"]
        x_break = parameters["x_break"]
        smoothness = parameters["smoothness"]

        # The shape is evaluated only at t > 0; elsewhere the luminosity is exactly zero.
        positive = t > 0
        log_x = np.log(np.where(positive, t, t_scale) / t_scale)

        log_shape = cls._log_shape(log_x, p, q, x_break, smoothness)
        _, log_shape_peak = cls._peak(p, q, x_break, smoothness)
        log_luminosity = np.log(amplitude) + log_shape - log_shape_peak

        return np.where(positive, log_luminosity, -np.inf)


class SBOPhenomenologicalLightcurve(Lightcurve):
    r"""
    A phenomenological shock-breakout light curve: a broken power law plus a log-time bump, with an early cutoff.

    .. math::

        L(t) = \exp\left[-\left(\frac{t_r}{t}\right)^{n}\right]
        \left\{
            L_p \left[1 + \left(\frac{t}{t_p}\right)^{s}\right]^{-\beta/s}
            + L_b \exp\left[-\frac{\ln^2(t/t_b)}{2\sigma^2}\right]
        \right\}.

    The light curve is the sum of two components, multiplied by an overall early-time cutoff:

    - **Smoothly broken power law.** :math:`L_p [1 + (t/t_p)^s]^{-\beta/s}` is a plateau at
      :math:`L_p` for :math:`t \ll t_p` that turns over into a power-law decline
      :math:`\propto t^{-\beta}` for :math:`t \gg t_p`. The smoothness :math:`s` sets the sharpness
      of the turnover (larger is sharper). Note that :math:`L_p` is the plateau level, and the peak
      of this component is only equal to it when the cutoff has already switched off.
    - **Log-normal bump.** :math:`L_b \exp[-\ln^2(t/t_b)/2\sigma^2]` is a Gaussian in :math:`\ln t`
      centred on :math:`t_b`, with peak :math:`L_b` and logarithmic width :math:`\sigma`. It models
      a secondary component (for example a delayed cooling or interaction bump) on top of the
      declining component.
    - **Early-time cutoff.** :math:`\exp[-(t_r/t)^n]` suppresses the emission at
      :math:`t \lesssim t_r`, going to zero at :math:`t = 0`, with the sharpness of the switch-on
      set by :math:`n`. It is applied to both components.

    The luminosity is zero for :math:`t \le 0`.

    .. rubric:: Parameters

    The light curve parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``cutoff_time``
         - :math:`t_r`
         - Time scale of the early-time cutoff.
       * - ``cutoff_index``
         - :math:`n`
         - Sharpness of the early-time cutoff.
       * - ``amplitude``
         - :math:`L_p`
         - Plateau luminosity of the broken power law.
       * - ``t_break``
         - :math:`t_p`
         - Time at which the broken power law turns over.
       * - ``smoothness``
         - :math:`s`
         - Sharpness of the turnover of the broken power law.
       * - ``decline_index``
         - :math:`\beta`
         - Late-time power-law decline index.
       * - ``bump_amplitude``
         - :math:`L_b`
         - Peak luminosity of the log-normal bump.
       * - ``bump_time``
         - :math:`t_b`
         - Time of the peak of the log-normal bump.
       * - ``bump_width``
         - :math:`\sigma`
         - Width of the bump in logarithmic time.

    See Also
    --------
    SmoothBrokenPowerLawLightcurve :
        A smoothly broken power law with a rise branch, normalized to peak at a given time.
    LogNormalPulseLightcurve :
        The log-normal pulse shape that the bump component is built from.
    """

    _LIGHTCURVE_TYPE = "bolometric"

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "cutoff_time": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=0.1 * u.day,
            description="Time scale of the early-time cutoff.",
            latex=r"t_r",
        ),
        "cutoff_index": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.3),
            scale=2.0 * u.dimensionless_unscaled,
            description="Sharpness of the early-time cutoff.",
            latex=r"n",
        ),
        "amplitude": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e43 * _BOL_LUM_UNIT,
            description="Plateau luminosity of the broken power law.",
            latex=r"L_p",
        ),
        "t_break": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1.0 * u.day,
            description="Time at which the broken power law turns over.",
            latex=r"t_p",
        ),
        "smoothness": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.4),
            scale=2.0 * u.dimensionless_unscaled,
            description="Sharpness of the turnover of the broken power law.",
            latex=r"s",
        ),
        "decline_index": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.3),
            scale=2.0 * u.dimensionless_unscaled,
            description="Late-time power-law decline index.",
            latex=r"\beta",
        ),
        "bump_amplitude": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=3e42 * _BOL_LUM_UNIT,
            description="Peak luminosity of the log-normal bump.",
            latex=r"L_b",
        ),
        "bump_time": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=5.0 * u.day,
            description="Time of the peak of the log-normal bump.",
            latex=r"t_b",
        ),
        "bump_width": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.4),
            scale=0.5 * u.dimensionless_unscaled,
            description="Width of the bump in logarithmic time.",
            latex=r"\sigma",
        ),
    }

    @classmethod
    def _eval(
        cls,
        t: NDArray[np.float64],
        **parameters: CGSParameterValue,
    ) -> NDArray[np.float64]:
        r"""
        Evaluate the natural log of :math:`L_\mathrm{bol}(t)`; see the class docstring for the functional form.

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
        cutoff_time = parameters["cutoff_time"]
        cutoff_index = parameters["cutoff_index"]
        amplitude = parameters["amplitude"]
        t_break = parameters["t_break"]
        smoothness = parameters["smoothness"]
        decline_index = parameters["decline_index"]
        bump_amplitude = parameters["bump_amplitude"]
        bump_time = parameters["bump_time"]
        bump_width = parameters["bump_width"]

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            log_t = np.log(t)

            # ln of the broken power law: ln Lp - (beta/s) ln[1 + (t/tp)^s], via logaddexp to avoid overflow.
            log_broken = np.log(amplitude) - (decline_index / smoothness) * np.logaddexp(
                0.0, smoothness * (log_t - np.log(t_break))
            )

            # ln of the log-normal bump.
            log_bump = np.log(bump_amplitude) - 0.5 * ((log_t - np.log(bump_time)) / bump_width) ** 2

            # ln of the early-time cutoff; the exponent overflows to +inf as t -> 0, giving exp(-inf) = 0.
            log_cutoff = -np.exp(cutoff_index * (np.log(cutoff_time) - log_t))

            log_luminosity = log_cutoff + np.logaddexp(log_broken, log_bump)

        return np.where(t > 0, log_luminosity, -np.inf)


# --------------------------------------------------------------------------- #
# Calibration of the one-parameter shape family behind `KhatamiSBOLightcurve`. #
# --------------------------------------------------------------------------- #
_DAY = u.day.to(u.s)

# Shape parameters of `SBOPhenomenologicalLightcurve` that are the same for every member of the family.
_KHATAMI_FIXED = {"cutoff_index": 1.5, "t_break": 0.78, "smoothness": 4.0, "bump_width": 0.15}  # t_break in days

# The remaining shape parameters, (log10 L_b / L_p, t_b [d], t_r [d], beta), at the two ends of the family
# (q = 0 and q = 1). They are interpolated linearly in the shape variable q in between.
_KHATAMI_SHAPE_Q0 = np.array([-1.81, 1.75, 0.52, 2.05])
_KHATAMI_SHAPE_Q1 = np.array([0.56, 1.99, 0.79, 0.33])

# Time grid [d] over which the luminosity-weighted duration is taken, and the number of shape-variable grid points.
_KHATAMI_T_GRID = np.linspace(0.02, 4.4, 1500)
_KHATAMI_N_Q = 201


def _khatami_log_shape(t_day: NDArray[np.float64], q: NDArray[np.float64] | float) -> NDArray[np.float64]:
    """
    Natural log of the unnormalized light-curve shape of the family at shape variable `q`.

    Parameters
    ----------
    t_day : numpy.ndarray
        Time since breakout, in days.
    q : numpy.ndarray or float
        Shape variable in [0, 1], broadcastable against `t_day`.

    Returns
    -------
    numpy.ndarray
        The natural log of the shape, which has unit plateau amplitude (not unit peak).
    """
    log_bump_ratio, t_bump, t_cutoff, beta = (
        p0 + (p1 - p0) * q for p0, p1 in zip(_KHATAMI_SHAPE_Q0, _KHATAMI_SHAPE_Q1)
    )

    return SBOPhenomenologicalLightcurve._eval(
        t_day * _DAY,
        cutoff_time=t_cutoff * _DAY,
        cutoff_index=_KHATAMI_FIXED["cutoff_index"],
        amplitude=1.0,
        t_break=_KHATAMI_FIXED["t_break"] * _DAY,
        smoothness=_KHATAMI_FIXED["smoothness"],
        decline_index=beta,
        bump_amplitude=10.0**log_bump_ratio,
        bump_time=t_bump * _DAY,
        bump_width=_KHATAMI_FIXED["bump_width"],
    )


def _khatami_tables() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Tabulate the family on a grid of the shape variable.

    Returns
    -------
    q : numpy.ndarray
        Shape variable grid, increasing from 0 to 1.
    t_bar : numpy.ndarray
        The luminosity-weighted duration (days) at each `q`. Monotonically increasing in `q`.
    log_peak : numpy.ndarray
        The natural log of the peak of the unnormalized shape at each `q`.
    """
    q = np.linspace(0.0, 1.0, _KHATAMI_N_Q)
    shape = np.exp(_khatami_log_shape(_KHATAMI_T_GRID, q[:, None]))  # (n_q, n_t)

    t_bar = np.trapezoid(_KHATAMI_T_GRID * shape, _KHATAMI_T_GRID) / np.trapezoid(shape, _KHATAMI_T_GRID)
    return q, t_bar, np.log(shape.max(axis=-1))


class KhatamiSBOLightcurve(Lightcurve):
    r"""
    A shock-breakout light curve family set by its peak luminosity and luminosity-weighted duration.

    The light curve has the functional form of :class:`SBOPhenomenologicalLightcurve`,

    .. math::

        L(t) = \exp\left[-\left(\frac{t_r}{t}\right)^{n}\right]
        \left\{
            \left[1 + \left(\frac{t}{t_p}\right)^{s}\right]^{-\beta/s}
            + \frac{L_b}{L_p} \exp\left[-\frac{\ln^2(t/t_b)}{2\sigma^2}\right]
        \right\},

    but rather than being free, its shape is a one-parameter family. The exponent :math:`n`, break time
    :math:`t_p`, smoothness :math:`s` and bump width :math:`\sigma` are fixed for every member, and
    the other four shape parameters :math:`(\log_{10} L_b/L_p,\ t_b,\ t_r,\ \beta)` are linearly interpolated
    between two calibrated endpoints by a shape variable :math:`q \in [0, 1]`. The light curve is then
    normalized so that its maximum is :math:`L_\mathrm{pk}` (to about one part in :math:`10^{5}`).

    The shape variable is not a parameter of the model. It is set by the luminosity-weighted duration

    .. math::

        \bar{t} = \frac{\int t\, L\, \mathrm{d}t}{\int L\, \mathrm{d}t},

    taken over 0.02 to 4.4 days, which increases monotonically with :math:`q`, so that
    :math:`\bar{t}` and :math:`q` are in one-to-one correspondence. This correspondence is tabulated once, at
    import, in the class attributes ``_Q_GRID``, ``_T_BAR_GRID`` and ``_LOG_PEAK_GRID``, and then interpolated.
    :math:`\bar{t}` ranges from about 1.5 days (:math:`q = 0`, dominated by the main peak) to about
    2.3 days (:math:`q = 1`, dominated by the bump); ``t_bar`` outside of this range is clamped to the
    nearest end of it.

    The endpoints were calibrated by eye-digitizing curves, so the family should be recalibrated
    against real simulation output before it is relied on quantitatively.

    .. rubric:: Parameters

    The light curve parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``peak_luminosity``
         - :math:`L_\mathrm{pk}`
         - Peak bolometric luminosity.
       * - ``t_bar``
         - :math:`\bar{t}`
         - Luminosity-weighted duration, which sets the shape of the light curve.

    See Also
    --------
    SBOPhenomenologicalLightcurve :
        The unconstrained nine-parameter form this family is a slice of.
    """

    _LIGHTCURVE_TYPE = "bolometric"

    _Q_GRID: ClassVar[NDArray[np.float64]]
    """numpy.ndarray: Grid of the shape variable :math:`q`, from 0 to 1."""
    _T_BAR_GRID: ClassVar[NDArray[np.float64]]
    """numpy.ndarray: The luminosity-weighted duration :math:`\\bar{t}` (days) on :attr:`_Q_GRID`."""
    _LOG_PEAK_GRID: ClassVar[NDArray[np.float64]]
    """numpy.ndarray: The natural log of the maximum of the unnormalized shape on :attr:`_Q_GRID`."""
    _Q_GRID, _T_BAR_GRID, _LOG_PEAK_GRID = _khatami_tables()

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "peak_luminosity": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e43 * _BOL_LUM_UNIT,
            description="Peak bolometric luminosity.",
            latex=r"L_\mathrm{pk}",
        ),
        "t_bar": Parameter(
            prior=UniformPrior(lower=float(_T_BAR_GRID[0]), upper=float(_T_BAR_GRID[-1])),
            scale=1.0 * u.day,
            description="Luminosity-weighted duration, which sets the shape of the light curve.",
            latex=r"\bar{t}",
        ),
    }

    @classmethod
    def _eval(
        cls,
        t: NDArray[np.float64],
        **parameters: CGSParameterValue,
    ) -> NDArray[np.float64]:
        r"""
        Evaluate the natural log of :math:`L_\mathrm{bol}(t)`; see the class docstring for the functional form.

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
        q = np.interp(parameters["t_bar"] / _DAY, cls._T_BAR_GRID, cls._Q_GRID)
        log_peak = np.interp(q, cls._Q_GRID, cls._LOG_PEAK_GRID)

        return np.log(parameters["peak_luminosity"]) + _khatami_log_shape(t / _DAY, q) - log_peak
