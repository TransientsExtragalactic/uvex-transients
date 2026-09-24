"""Model of the SED from type Ia (thermonuclear) supernovae."""

from typing import ClassVar

from astropy import units as u

from uvex_transients.models.arnett import ArnettDecaySED
from uvex_transients.models.core.parameters import Parameter
from uvex_transients.models.core.priors import (
    ConstantPrior,
    NormalPrior,
    TruncatedNormalPrior,
    UniformPrior,
)

__all__ = ["TypeIaSED"]


class TypeIaSED(ArnettDecaySED):
    r"""
    Type Ia supernova SED, with priors fit to a ZTF sample of SNe Ia.

    Reuses :class:`~uvex_transients.models.arnett.ArnettDecaySED`'s radioactive-decay Arnett diffusion and
    floored-photosphere blackbody entirely; only :attr:`_DEFAULT_PARAMETERS` differs. The priors follow
    :footcite:t:`sarin2026` (Sarin et al. 2026, a sample of 2205 SNe Ia from ZTF): ``M_Ni`` and ``M_ej``
    are the population-level Gaussians from that paper's hierarchical Arnett-model fit
    (:math:`\mu_\mathrm{Ni}=0.64\,M_\odot`, :math:`\sigma_\mathrm{Ni}=0.42\,M_\odot`;
    :math:`\mu_\mathrm{ej}=1.26\,M_\odot`, :math:`\sigma_\mathrm{ej}=0.33\,M_\odot`), ``kappa_gamma`` is
    fixed at the value adopted by :footcite:t:`scalzo2014` (Scalzo et al. 2014), and ``kappa`` is uniform
    over that paper's marginalization range.

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

    References
    ----------
    .. footbibliography::
    """

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "M_Ni": Parameter(
            prior=TruncatedNormalPrior(mean=0.64, sigma=0.42, lower=0.05, upper=3.0),
            scale=1 * u.Msun,
            description="Nickel-56 mass synthesized in the explosion.",
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
    }
