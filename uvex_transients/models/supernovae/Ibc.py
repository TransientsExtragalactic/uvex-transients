"""Models of SEDs from type Ib and Ic (stripped-envelope) supernovae."""

from typing import ClassVar

import numpy as np
from astropy import units as u

from uvex_transients.models.arnett import LineBlanketedArnettDecaySED
from uvex_transients.models.core.parameters import Parameter
from uvex_transients.models.core.priors import (
    ConstantPrior,
    ExponentialPrior,
    TruncatedNormalPrior,
)
from uvex_transients.models.host_dust import HostDustMixin

__all__ = ["TypeIbSED", "TypeIcSED"]


def _nickel_mass_parameter() -> Parameter:
    """Return the ``M_Ni`` prior of Type Ib and Ic: log-normal, median 0.06 Msun, sigma_ln 0.7, 0.005 to 1.5 Msun."""
    return Parameter(
        # A normal in ln(M_Ni / Msun), truncated. Quantiles: 0.03 [16%], 0.06 [50%], 0.12 [84%], 0.24 [97.5%] Msun.
        prior=TruncatedNormalPrior(mean=np.log(0.06), sigma=0.7, lower=np.log(0.005), upper=np.log(1.5)),
        scale=1 * u.Msun,
        transform="log",
        description="Nickel-56 mass synthesized in the explosion (log-normal, set against the ASAS-SN LF).",
        latex=r"M_\mathrm{Ni}",
    )


def _host_attenuation_parameter() -> Parameter:
    """Return the ``A_V`` prior of Type Ib and Ic: exponential with mean 0.6 mag."""
    return Parameter(
        prior=ExponentialPrior(scale=0.6),
        scale=1.0,
        description="V-band attenuation by dust in the host galaxy, in magnitudes.",
        latex=r"A_V",
    )


class _TypeIbIntrinsicSED(LineBlanketedArnettDecaySED):
    """
    The Type Ib SED before host dust: `TypeIbSED` without ``A_V`` and ``R_V``.

    It holds the priors of `TypeIbSED`, and is the nickel component of
    :class:`~uvex_transients.models.supernovae.IIb.TypeIIbSED`, which has no host dust of its own and so must not
    inherit the dust of a Type Ib.
    """

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "M_Ni": _nickel_mass_parameter(),
        "M_ej": Parameter(
            # Truncated quantiles: 2.75 [1.10, 4.76] Msun (target 2.65 [1.17, 5.22]).
            prior=TruncatedNormalPrior(mean=2.3, sigma=2.2, lower=0.1, upper=20.0),
            scale=1 * u.Msun,
            description="Ejecta mass.",
            latex=r"M_\mathrm{ej}",
        ),
        "v_ej": Parameter(
            # Truncated quantiles: 9.8 [6.7, 13.2] (target 9.84 [6.6, 13.0]).
            prior=TruncatedNormalPrior(mean=9.6, sigma=3.5, lower=4.0, upper=25.0),
            scale=1e3 * u.km / u.s,
            description="Constant ejecta velocity, identified with the photospheric velocity.",
            latex=r"v_\mathrm{ej}",
        ),
        "kappa": Parameter(
            prior=ConstantPrior(value=0.07),
            scale=1 * u.cm**2 / u.g,
            description="Grey optical opacity.",
            latex=r"\kappa",
        ),
        "kappa_gamma": Parameter(
            prior=ConstantPrior(value=0.04),
            scale=1 * u.cm**2 / u.g,
            description="Opacity to high-energy photons. Fixed (comparable to TypeIaSED's Scalzo+14 value).",
            latex=r"\kappa_\gamma",
        ),
        "T_floor": Parameter(
            prior=TruncatedNormalPrior(mean=5000.0, sigma=1000.0, lower=3000.0, upper=10000.0),
            scale=1 * u.K,
            description="Minimum photospheric temperature.",
            latex=r"T_\mathrm{floor}",
        ),
        "break_frequency": Parameter(
            prior=ConstantPrior(value=1.4),
            scale=1e15 * u.Hz,
            description="Frequency above which the line-blanketing suppression sets in (set against Swift/UVOT).",
            latex=r"\nu_\mathrm{break}",
        ),
        "beta": Parameter(
            prior=ConstantPrior(value=3.0),
            scale=1.0,
            description="Power-law index of the suppression above the break (not constrained by Swift/UVOT).",
            latex=r"\beta",
        ),
    }


class TypeIbSED(HostDustMixin, _TypeIbIntrinsicSED):
    r"""
    Type Ib supernova SED, attenuated by host-galaxy dust, with priors from the ZTF stripped-envelope SN sample.

    Reuses :class:`~uvex_transients.models.arnett.LineBlanketedArnettDecaySED` entirely, exactly as
    :class:`~uvex_transients.models.supernovae.Ia.TypeIaSED` does (radioactive-decay Arnett diffusion,
    floored photospheric temperature and a line-blanketed blackbody); only :attr:`_DEFAULT_PARAMETERS`
    differs. ``break_frequency`` and ``beta`` are set against Swift/UVOT (see below), not taken from Type Ia.

    The priors on ``M_ej`` and ``v_ej`` follow the modified-Arnett fits to the 88 SNe Ib of the
    volume-limited (< 150 Mpc) ZTF CLU + CATS150 sample [citation pending]. Each truncated normal
    is tuned so that its 16th, 50th and 84th percentiles (after truncation) reproduce the CLU subtype
    distribution; the ``mean`` and ``sigma`` arguments are therefore fit parameters, not the sample moments.
    ``M_Ni`` is not taken from the sample (first bullet).

    * ``M_Ni``: a log-normal truncated to 0.005 to 1.5 :math:`M_\odot`, with median :math:`0.06\,M_\odot` and
      :math:`\sigma_{\ln}=0.7` (16th to 84th percentile 0.03 to 0.12, 97.5th percentile about 0.24
      :math:`M_\odot`). It replaces a truncated normal tuned to the survey-efficiency-corrected CLU median of
      0.093 :math:`M_\odot`, which put the model's peak absolute magnitudes in a narrow band near
      :math:`M_V\approx-17.5` where the ASAS-SN luminosity function of SNe Ib rises toward the faint end. A
      skewed distribution, many low-mass events and few high-mass ones, is what that function needs. It is
      chosen by comparison with that function, not fit to the CLU nickel masses, and is constrained only in
      combination with the :math:`A_V` prior below.
    * ``M_ej``: the observed Ib median and 16th to 84th percentiles (2.95 [1.3, 5.8] :math:`M_\odot`),
      scaled by 0.90, the sample-wide shift in median ejecta mass under :math:`1/\epsilon` weighting
      (2.90 to 2.61 :math:`M_\odot`); the sample analysis does not give a per-subtype corrected value.
    * ``v_ej``: the CLU Ib median velocity proxy (9.84e3 km/s; the He I 5876 absorption velocity
      converted to the characteristic velocity :math:`v_m = (2E_K/M_\mathrm{ej})^{1/2}`), with a width of
      about 3.2e3 km/s estimated from the measured velocities of the SNe Ib in the printed portion of the
      sample's velocity table.

    ``kappa`` is fixed at :math:`0.07\,\mathrm{cm^2\,g^{-1}}`, the grey opacity the sample analysis uses to
    turn diffusion times into ejecta masses; the ``M_ej`` prior is only consistent with that value.
    ``kappa_gamma`` is fixed at :math:`0.04\,\mathrm{cm^2\,g^{-1}}`, comparable to :class:`TypeIaSED`'s
    Scalzo+14-derived value (:math:`0.03\,\mathrm{cm^2\,g^{-1}}`); the sample analysis fits a free gamma-ray
    trapping time per event rather than an opacity, so it is not taken from the sample. ``T_floor`` is not
    constrained by the g- and r-band CLU photometry and is left as it was.

    .. rubric:: UV line blanketing

    ``break_frequency`` (:math:`1.4\times10^{15}` Hz, about 2140 Angstrom) and ``beta`` (3) were set against the
    peak :math:`\nu L_\nu` in UVW1 and UVW2 and the UVW1 to UVW2 colour of the Swift/UVOT supernovae of Type Ib and
    Ic together (11 events in UVW1, 5 in UVW2, 5 with both), by a coarse grid search with the other priors fixed. The
    original Type Ia values (:math:`0.75\times10^{15}` Hz, 5) left both types 0.5 to 1 dex too faint in the UV. The
    data are not enough to determine both parameters: any break above about :math:`1.4\times10^{15}` Hz fits about
    equally well, up to no blanketing in the UVOT bands, and ``beta`` is not constrained at all. The values adopted
    keep some suppression at the blue end, which is an extrapolation, as the UVOT bands do not reach the UVEX FUV
    band. The Swift supernovae are a brighter sample than this model's population, so the level carries that
    caveat. :class:`TypeIcSED` has the same values.

    .. rubric:: Host dust

    The SED is reddened by dust in the host galaxy, with the G23 law
    (:attr:`~uvex_transients.models.host_dust.HostDustMixin.DUST_LAW`) and a V-band attenuation :math:`A_V`
    drawn from an exponential distribution with mean 0.6 mag (a plausible range for the mean is 0.3 to 1.0).
    :math:`R_V` is the default of :class:`~uvex_transients.models.host_dust.HostDustMixin`, normal with
    :math:`2.70 \pm 0.25` truncated to G23's range of 2.3 to 5.6, which was set for Type Ia and is not measured
    for stripped-envelope events. The two are independent. As for :class:`TypeIaSED`, the dust is part of the
    model rather than a correction applied on top, so ``eval_bolometric`` is the integral of the attenuated
    :math:`L_\nu` and ``A_V=0`` recovers the dust-free SED. Milky Way foreground dust is separate (the ``ebv``
    argument of the photometry methods).

    .. rubric:: Parameters

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``M_Ni``
         - :math:`M_\mathrm{Ni}`
         - Nickel-56 mass synthesized in the explosion.
       * - ``M_ej``
         - :math:`M_\mathrm{ej}`
         - Ejecta mass.
       * - ``v_ej``
         - :math:`v_\mathrm{ej}`
         - Ejecta velocity, taken to be constant and equal to the photospheric velocity.
       * - ``kappa``
         - :math:`\kappa`
         - Grey optical opacity.
       * - ``kappa_gamma``
         - :math:`\kappa_\gamma`
         - Opacity to high-energy photons. Fixed.
       * - ``T_floor``
         - :math:`T_\mathrm{floor}`
         - Minimum photospheric temperature.
       * - ``break_frequency``
         - :math:`\nu_\mathrm{break}`
         - Frequency blueward of which line blanketing suppresses the flux.
       * - ``beta``
         - :math:`\beta`
         - Power-law index of the suppression above the break.
       * - ``A_V``
         - :math:`A_V`
         - V-band attenuation by dust in the host galaxy.
       * - ``R_V``
         - :math:`R_V`
         - Total-to-selective extinction ratio of the host dust.
    """

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        **_TypeIbIntrinsicSED._DEFAULT_PARAMETERS,
        **HostDustMixin.HOST_DUST_PARAMETERS,
        "A_V": _host_attenuation_parameter(),
    }


class TypeIcSED(HostDustMixin, LineBlanketedArnettDecaySED):
    r"""
    Type Ic supernova SED, attenuated by host-galaxy dust, with priors from the ZTF stripped-envelope SN sample.

    Same construction as :class:`TypeIbSED` (see its docstring for how the priors are built and for the
    reasoning behind ``kappa``, ``kappa_gamma``, ``T_floor``, the UV line blanketing and the host dust), but
    with ``M_ej`` and ``v_ej`` tuned to the 113 normal SNe Ic (Ic-BL excluded) of the same sample
    [citation pending]:

    * ``M_Ni``: the same log-normal as :class:`TypeIbSED` (median :math:`0.06\,M_\odot`,
      :math:`\sigma_{\ln}=0.7`, truncated to 0.005 to 1.5 :math:`M_\odot`), set against the ASAS-SN luminosity
      function rather than to the CLU Ic nickel masses (efficiency-corrected median 0.116 :math:`M_\odot`).
    * ``M_ej``: the observed Ic distribution (3.62 [1.4, 7.3] :math:`M_\odot`) scaled by 0.90.
    * ``v_ej``: the CLU Ic median velocity proxy (1.42e4 km/s, from O I 7774 absorption converted to
      :math:`v_m`), with a width of about 3.6e3 km/s from the measured SNe Ic in the printed portion of the
      sample's velocity table. This is substantially faster than Fe II photospheric velocities (the raw O I
      absorption median is 8.9e3 km/s).

    The host dust is that of :class:`TypeIbSED`: :math:`A_V` exponential with mean 0.6 mag.

    .. rubric:: Parameters

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``M_Ni``
         - :math:`M_\mathrm{Ni}`
         - Nickel-56 mass synthesized in the explosion.
       * - ``M_ej``
         - :math:`M_\mathrm{ej}`
         - Ejecta mass.
       * - ``v_ej``
         - :math:`v_\mathrm{ej}`
         - Ejecta velocity, taken to be constant and equal to the photospheric velocity.
       * - ``kappa``
         - :math:`\kappa`
         - Grey optical opacity.
       * - ``kappa_gamma``
         - :math:`\kappa_\gamma`
         - Opacity to high-energy photons. Fixed.
       * - ``T_floor``
         - :math:`T_\mathrm{floor}`
         - Minimum photospheric temperature.
       * - ``break_frequency``
         - :math:`\nu_\mathrm{break}`
         - Frequency blueward of which line blanketing suppresses the flux.
       * - ``beta``
         - :math:`\beta`
         - Power-law index of the suppression above the break.
       * - ``A_V``
         - :math:`A_V`
         - V-band attenuation by dust in the host galaxy.
       * - ``R_V``
         - :math:`R_V`
         - Total-to-selective extinction ratio of the host dust.
    """

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "M_Ni": _nickel_mass_parameter(),
        "M_ej": Parameter(
            # Truncated quantiles: 3.34 [1.21, 6.19] Msun (target 3.26 [1.26, 6.57]).
            prior=TruncatedNormalPrior(mean=2.2, sigma=3.35, lower=0.1, upper=20.0),
            scale=1 * u.Msun,
            description="Ejecta mass.",
            latex=r"M_\mathrm{ej}",
        ),
        "v_ej": Parameter(
            # Truncated quantiles: 14.2 [10.6, 17.8] (target 14.2 [10.6, 17.8]).
            prior=TruncatedNormalPrior(mean=14.2, sigma=3.6, lower=4.0, upper=25.0),
            scale=1e3 * u.km / u.s,
            description="Constant ejecta velocity, identified with the photospheric velocity.",
            latex=r"v_\mathrm{ej}",
        ),
        "kappa": Parameter(
            prior=ConstantPrior(value=0.07),
            scale=1 * u.cm**2 / u.g,
            description="Grey optical opacity.",
            latex=r"\kappa",
        ),
        "kappa_gamma": Parameter(
            prior=ConstantPrior(value=0.04),
            scale=1 * u.cm**2 / u.g,
            description="Opacity to high-energy photons. Fixed (comparable to TypeIaSED's Scalzo+14 value).",
            latex=r"\kappa_\gamma",
        ),
        "T_floor": Parameter(
            prior=TruncatedNormalPrior(mean=6000.0, sigma=1000.0, lower=3000.0, upper=10000.0),
            scale=1 * u.K,
            description="Minimum photospheric temperature.",
            latex=r"T_\mathrm{floor}",
        ),
        "break_frequency": Parameter(
            prior=ConstantPrior(value=1.4),
            scale=1e15 * u.Hz,
            description="Frequency above which the line-blanketing suppression sets in (set against Swift/UVOT).",
            latex=r"\nu_\mathrm{break}",
        ),
        "beta": Parameter(
            prior=ConstantPrior(value=3.0),
            scale=1.0,
            description="Power-law index of the suppression above the break (not constrained by Swift/UVOT).",
            latex=r"\beta",
        ),
        **HostDustMixin.HOST_DUST_PARAMETERS,
        "A_V": _host_attenuation_parameter(),
    }
