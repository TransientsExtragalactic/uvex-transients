"""Model of the SED from type Ia (thermonuclear) supernovae."""

from typing import ClassVar

from astropy import units as u

from uvex_transients.models.arnett import LineBlanketedArnettDecaySED
from uvex_transients.models.core.parameters import Parameter
from uvex_transients.models.core.priors import (
    ConstantPrior,
    NormalPrior,
    TruncatedNormalPrior,
    UniformPrior,
)
from uvex_transients.models.host_dust import HostDustMixin

__all__ = ["TypeIaSED"]


class TypeIaSED(HostDustMixin, LineBlanketedArnettDecaySED):
    r"""
    Type Ia supernova SED, attenuated by host-galaxy dust, with priors tuned to the ASAS-SN luminosity function.

    Reuses :class:`~uvex_transients.models.arnett.LineBlanketedArnettDecaySED` (radioactive-decay Arnett
    diffusion, floored photospheric temperature and a line-blanketed blackbody), with its parameters' priors
    replaced and the host dust of :class:`~uvex_transients.models.host_dust.HostDustMixin` added. The priors
    follow :footcite:t:`sarin2026` (Sarin et al. 2026, a sample of 2205 SNe Ia from ZTF), except for ``M_Ni``:
    ``M_ej`` is the population-level Gaussian from that paper's hierarchical Arnett-model fit
    (:math:`\mu_\mathrm{ej}=1.26\,M_\odot`, :math:`\sigma_\mathrm{ej}=0.33\,M_\odot`), ``kappa_gamma`` is
    fixed at the value adopted by :footcite:t:`scalzo2014` (Scalzo et al. 2014), and ``kappa`` is uniform
    over that paper's marginalization range.

    .. rubric:: Tuning to the ASAS-SN luminosity function

    The paper's ``M_Ni`` Gaussian (:math:`\mu_\mathrm{Ni}=0.64\,M_\odot`,
    :math:`\sigma_\mathrm{Ni}=0.42\,M_\odot`) gives a distribution of peak V-band absolute magnitude that is about
    0.3 mag too bright, and too broad, against the ASAS-SN luminosity function of SNe Ia, even with host dust. The
    ``M_Ni`` prior is therefore a truncated Gaussian with :math:`\mu_\mathrm{Ni}=0.48\,M_\odot` and
    :math:`\sigma_\mathrm{Ni}=0.2\,M_\odot`, chosen together with the mean of the :math:`A_V` prior below by trial
    and error (compared by the total variation distance of the binned distributions), not by a fit. The peak
    V-band magnitude constrains only the combination of the two: other pairs fit equally well, so
    neither value is measured here. The distribution is still too thin at the faintest end (:math:`M_V\gtrsim-17.5`),
    where subluminous events (91bg-like) are missing from a single Gaussian.

    .. rubric:: UV line blanketing

    ``break_frequency`` (:math:`0.9\times10^{15}` Hz, about 3330 Angstrom) and ``beta`` (5.5) were set against the
    peak :math:`\nu L_\nu` of the Swift/UVOT SNe Ia (UVW1, UVM2 and UVW2), with the priors above. The first
    calibration (:math:`0.75\times10^{15}` Hz, 5) predates the host dust and the tuned ``M_Ni``, and left the model
    about 0.35 dex too faint in UVW1 and UVW2. The median peaks in both bands and the UVW1 to UVW2 colour now agree
    with the data (to 0.02 dex), but the UVW1 to UVM2 colour does not: the model is too bright in UVM2 by about 0.35
    dex, and no choice of the two parameters fixes that. The Swift SNe Ia are a brighter and narrower sample than the
    ASAS-SN-like population of the model, so the agreement in level is only as good as that comparison, the
    colours are the more robust of the two, and the observed UVOT bands do not reach the UVEX FUV band, which is
    an extrapolation. Because the blanketing returns the flux it removes to longer wavelengths, changing it moves
    the V band too, so ``M_Ni`` was tuned again after it.

    .. rubric:: Host dust

    The SED is reddened by dust in the host galaxy, with the G23 law
    (:attr:`~uvex_transients.models.host_dust.HostDustMixin.DUST_LAW`) and a V-band attenuation :math:`A_V`
    drawn from an exponential distribution with mean 0.4 mag. :math:`R_V` is normal with
    :math:`2.70 \pm 0.25`, truncated to G23's range of 2.3 to 5.6; the width is that of the uncertainty on the
    mean, which is within the 95% upper limit of 0.61 on the scatter of the population. The two are independent.
    The dust is part of the model, not a correction applied on top: nothing is known of the dust that shaped any
    given event, so the SED describes the luminosity we observe and infer, and ``eval_bolometric`` is the
    integral of the attenuated :math:`L_\nu`. ``A_V=0`` recovers the dust-free SED. Milky Way foreground dust
    is separate (the ``ebv`` argument of the photometry methods).

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

    References
    ----------
    .. footbibliography::
    """

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "M_Ni": Parameter(
            prior=TruncatedNormalPrior(mean=0.48, sigma=0.2, lower=0.05, upper=3.0),
            scale=1 * u.Msun,
            description="Nickel-56 mass synthesized in the explosion (tuned to the ASAS-SN luminosity function).",
            latex=r"M_\mathrm{Ni}",
        ),
        "M_ej": Parameter(
            prior=TruncatedNormalPrior(mean=1.26, sigma=0.33, lower=0.05, upper=3.0),
            scale=1 * u.Msun,
            description="Ejecta mass.",
            latex=r"M_\mathrm{ej}",
        ),
        "v_ej": Parameter(
            prior=NormalPrior(mean=11e3, sigma=1e3),
            scale=1.0 * u.km / u.s,
            description="Constant ejecta velocity, identified with the photospheric velocity.",
            latex=r"v_\mathrm{ej}",
        ),
        "kappa": Parameter(
            prior=UniformPrior(lower=0.05, upper=0.15),
            scale=1 * u.cm**2 / u.g,
            description="Grey optical opacity.",
            latex=r"\kappa",
        ),
        "kappa_gamma": Parameter(
            prior=ConstantPrior(value=0.03),
            scale=1 * u.cm**2 / u.g,
            description="Opacity to high-energy photons. Fixed (Scalzo+14).",
            latex=r"\kappa_\gamma",
        ),
        "T_floor": Parameter(
            prior=TruncatedNormalPrior(mean=6000.0, sigma=1000.0, lower=3000.0, upper=10000.0),
            scale=1 * u.K,
            description="Minimum photospheric temperature.",
            latex=r"T_\mathrm{floor}",
        ),
        "break_frequency": Parameter(
            prior=ConstantPrior(value=0.9),
            scale=1e15 * u.Hz,
            description="Frequency above which the line-blanketing suppression sets in (set against Swift/UVOT).",
            latex=r"\nu_\mathrm{break}",
        ),
        "beta": Parameter(
            prior=ConstantPrior(value=5.5),
            scale=1.0,
            description="Power-law index of the suppression above the break.",
            latex=r"\beta",
        ),
        **HostDustMixin.HOST_DUST_PARAMETERS,
    }
