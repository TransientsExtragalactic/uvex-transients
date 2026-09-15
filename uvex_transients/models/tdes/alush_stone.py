r"""Tidal disruption event SED with an early decline and a late-time disk plateau.

Following :footcite:t:`2025arXiv250303811A`.
"""

from typing import ClassVar

import numpy as np
from astropy import units as u

from uvex_transients.models._typing import CGSParameterValue, FloatArray
from uvex_transients.models.core.base import SpectralModel
from uvex_transients.models.core.parameters import Parameter
from uvex_transients.models.core.priors import NormalPrior, UniformPrior
from uvex_transients.models.lightcurves.generic import GREDLightcurve
from uvex_transients.models.spectra.thermal import BlackbodySpectrum

__all__ = ["AlushStoneTDESED"]


class AlushStoneTDESED(SpectralModel):
    r"""
    A Gaussian-rise/exponential-decline photosphere followed by a magnetized-disk plateau.

    .. math::

        L_\nu(\nu, t) = L_\mathrm{bol}^\mathrm{early}(t) \cdot S(\nu, T)
            + L_\mathrm{bol}^\mathrm{plat}(t) \cdot S(\nu, T_\mathrm{p}),

    where :math:`S(\nu, T)` is :class:`~uvex_transients.models.spectra.thermal.BlackbodySpectrum`'s
    normalized shape, :math:`L_\mathrm{bol}^\mathrm{early}(t)` is
    :class:`~uvex_transients.models.lightcurves.generic.GREDLightcurve`
    (peaking at :math:`t_\mathrm{peak} = 5\sigma`, that class's own convention), and

    .. math::

        L_\mathrm{bol}^\mathrm{plat}(t) = L_\mathrm{p}
            \left(1 + \frac{t - t_\mathrm{peak}}{\tau_\mathrm{p}}\right)^{-\alpha_\mathrm{p}}.

    The plateau branch is referenced to the same :math:`t_\mathrm{peak}` as the early
    branch (the same convention `GaussianRisePowerLawLightcurve` uses for its power-law
    tail): :math:`L_\mathrm{bol}^\mathrm{plat}(t_\mathrm{peak}) = L_\mathrm{p}` exactly,
    and it smoothly softens from a flat plateau (for :math:`t - t_\mathrm{peak} \ll
    \tau_\mathrm{p}`) to a :math:`t^{-\alpha_\mathrm{p}}` power-law decline (for
    :math:`t - t_\mathrm{peak} \gg \tau_\mathrm{p}`) -- exactly the family of
    theory-agnostic plateau shapes fit to late-time TDE disks by
    :footcite:t:`2025arXiv251024696A`. The two components are summed in linear space
    (:func:`numpy.logaddexp` on their logs) rather than switched between, since a real
    disk plateau does not sharply replace the fading early-time emission.

    Physically, the early branch is the same reprocessed/photospheric emission
    :class:`~uvex_transients.models.tdes.van_velzen.VanVelzenTDESED` models, while the
    plateau is the late-time, UV-bright emission from a magnetized accretion disk that
    :footcite:t:`2025arXiv250303811A` predict settles onto an asymptotic
    :math:`L \propto t^{-5/6}` decline persisting for decades to centuries --
    ``plateau_decline``'s default prior is centered on that theoretical value, though it
    is left free since :footcite:t:`2025arXiv251024696A` find real plateaus vary in how
    flat/evolving they are.

    ``temperature``/``sigma_rise``/``tau_decline`` reuse the same ZTF-sample-informed
    defaults as :class:`~uvex_transients.models.tdes.van_velzen.VanVelzenTDESED` (see that
    class's docstring); ``plateau_temperature``, ``plateau_amplitude``, and
    ``plateau_timescale`` do not yet have literature-calibrated defaults and use
    order-of-magnitude fiducial scales instead -- narrow these once fit to real
    late-time photometry.

    .. rubric:: Parameters

    The model parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude``
         - :math:`L_0`
         - Peak bolometric luminosity of the early-time component,
           :math:`L_0 = L_\mathrm{bol}^\mathrm{early}(t_\mathrm{peak})`.
           log10(L_0/[erg/s]) ~ N(43.8, 0.2^2).
       * - ``temperature``
         - :math:`T`
         - Early-time photospheric blackbody temperature. log10(T/K) ~ N(4.3, 0.1^2).
       * - ``sigma_rise``
         - :math:`\sigma`
         - Gaussian width of the pre-peak rise. log10(sigma/day) ~ N(1.3, 0.3^2).
       * - ``tau_decline``
         - :math:`\tau`
         - Exponential decline timescale of the early-time component, after peak.
           log10(tau/day) ~ N(2, 0.1^2).
       * - ``plateau_temperature``
         - :math:`T_\mathrm{p}`
         - Blackbody temperature of the late-time disk plateau. log10(T_p/K) ~ N(4.0, 0.3^2).
       * - ``plateau_amplitude``
         - :math:`L_\mathrm{p}`
         - Plateau bolometric luminosity at :math:`t_\mathrm{peak}`,
           :math:`L_\mathrm{p} = L_\mathrm{bol}^\mathrm{plat}(t_\mathrm{peak})`.
           log10(L_p/[erg/s]) ~ N(41.5, 0.3^2).
       * - ``plateau_timescale``
         - :math:`\tau_\mathrm{p}`
         - Timescale over which the plateau softens into its power-law decline.
           log10(tau_p/day) ~ N(2.3, 0.3^2).
       * - ``plateau_decline``
         - :math:`\alpha_\mathrm{p}`
         - Late-time power-law decline index, :math:`L \propto t^{-\alpha_\mathrm{p}}`
           for :math:`t - t_\mathrm{peak} \gg \tau_\mathrm{p}`.
           :math:`\alpha_\mathrm{p} \sim \mathcal{N}(5/6,\, 0.2^2)`, centered on the
           magnetized-disk prediction of :footcite:t:`2025arXiv250303811A`.

    See Also
    --------
    uvex_transients.models.tdes.VanVelzenTDESED

    References
    ----------
    .. footbibliography::
    """

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=NormalPrior(mean=43.8, sigma=0.3),
            scale=1.0 * u.erg / u.s,
            transform="log10",
            description="Peak bolometric luminosity, L_0 = L_bol(t_peak). log10(L_0/[erg/s]) ~ N(43.8, 0.2^2).",
            latex=r"L_0",
        ),
        "temperature": Parameter(
            prior=NormalPrior(mean=4.3, sigma=0.1),
            scale=1.0 * u.K,
            transform="log10",
            description="Photospheric blackbody temperature. log10(T/K) ~ N(4.3, 0.1^2).",
            latex=r"T",
        ),
        "sigma_rise": Parameter(
            prior=NormalPrior(mean=0.91, sigma=0.25),
            scale=1.0 * u.day,
            transform="log10",
            description="Gaussian width of the pre-peak rise. log10(sigma/day) ~ N(1.3, 0.3^2).",
            latex=r"\sigma",
        ),
        "tau_decline": Parameter(
            prior=NormalPrior(mean=1.8, sigma=0.2),
            scale=1.0 * u.day,
            transform="log10",
            description="Exponential decline timescale after peak. log10(tau/day) ~ N(2, 0.1^2).",
            latex=r"\tau",
        ),
        "plateau_temperature": Parameter(
            prior=NormalPrior(mean=4.0, sigma=0.3),
            scale=1.0 * u.K,
            transform="log10",
            description="Blackbody temperature of the late-time disk plateau. log10(T_p/K) ~ N(4.0, 0.3^2).",
            latex=r"T_\mathrm{p}",
        ),
        "plateau_amplitude": Parameter(
            prior=NormalPrior(mean=41.5, sigma=0.2),
            scale=1.0 * u.erg / u.s,
            transform="log10",
            description="Plateau bolometric luminosity, L_p = L_bol_plat(t_peak). log10(L_p/[erg/s]) ~ N(41.5, 0.3^2).",
            latex=r"L_\mathrm{p}",
        ),
        "plateau_timescale": Parameter(
            prior=NormalPrior(mean=2.3, sigma=0.3),
            scale=1.0 * u.day,
            transform="log10",
            description=(
                "Timescale over which the plateau softens into a power-law decline. log10(tau_p/day) ~ N(2.3, 0.3^2)."
            ),
            latex=r"\tau_\mathrm{p}",
        ),
        "plateau_decline": Parameter(
            prior=UniformPrior(lower=0, upper=2),
            scale=1.0 * u.dimensionless_unscaled,
            description=(
                "Late-time power-law decline index, L ~ t^-alpha_p. Centered on the "
                "magnetized-disk prediction alpha_p = 5/6."
            ),
            latex=r"\alpha_\mathrm{p}",
        ),
    }

    # ============================================== #
    # SED Methods                                    #
    # ============================================== #
    @classmethod
    def _eval(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        Evaluate the natural log of :math:`L_\nu(\nu, t)`, in cgs units.

        This is the one method every model must implement. ``nu``, ``t``,
        and every parameter value are combined using plain NumPy
        broadcasting -- no axes are inserted automatically. See the module
        docstring for what that means in practice.

        Parameters
        ----------
        nu
            Frequency, in Hz.
        t
            Time since explosion, in seconds. Always non-negative.
        **parameters
            This model's parameter values, in cgs units, broadcastable
            against ``nu`` and ``t``.

        Returns
        -------
        numpy.ndarray
            The natural log of :math:`L_\nu(\nu, t)`, in erg/s/Hz.
        """
        # Evaluate the early- and late-time spectral shapes using the
        # blackbody function provided in the thermal spectrum module.
        early_spec = BlackbodySpectrum._eval(nu, temperature=parameters["temperature"])
        late_spec = BlackbodySpectrum._eval(nu, temperature=parameters["plateau_temperature"])

        # Evaluate the two components' bolometric envelopes, both referenced
        # to the same t_peak (GREDLightcurve's own convention -- see its
        # docstring). The early envelope is a Gaussian rise / exponential
        # decline in time, not frequency, so `t` (not `nu`) is what it's
        # evaluated at.
        t_peak = 5 * parameters["sigma_rise"]
        early_amp = GREDLightcurve._eval(
            t,
            amplitude=parameters["amplitude"],
            sigma_rise=parameters["sigma_rise"],
            tau_decline=parameters["tau_decline"],
        )

        # The plateau softens from flat (near t_peak) to a t^-plateau_decline
        # power law once t - t_peak >> plateau_timescale -- see the class
        # docstring, which only defines this behavior for t >= t_peak (x >=
        # 0). `(1+x)^-plateau_decline` is not just undefined but singular as
        # x -> -1 from above (since 1+x -> 0+ with a negative exponent), so
        # letting the branch run for -1 < x < 0 doesn't just risk a NaN at
        # the boundary -- it produces a spurious blow-up in luminosity
        # immediately after it, whenever `plateau_timescale` is smaller than
        # `t_peak`. Restricting to x >= 0 keeps evaluation inside the domain
        # the formula (and docstring) actually describe, where it's smooth
        # and bounded by L_p; a plateau that hasn't reached t_peak yet
        # contributes zero luminosity, i.e. -inf in log space.
        x = (t - t_peak) / parameters["plateau_timescale"]
        plateau_active = x >= 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            plateau_shape = -parameters["plateau_decline"] * np.log1p(np.where(plateau_active, x, 0.0))
        late_amp = np.where(
            plateau_active,
            np.log(parameters["plateau_amplitude"]) + plateau_shape,
            -np.inf,
        )

        # Combine each component's shape and envelope in log space (shape +
        # envelope, since `early_spec`/`late_spec` are already normalized
        # log-shapes -- see `BlackbodySpectrum`), then sum the two
        # components themselves via logaddexp rather than a linear-space
        # sum, since both are already logarithmic.
        early_sed = early_spec + early_amp
        late_sed = late_spec + late_amp

        return np.logaddexp(early_sed, late_sed)
