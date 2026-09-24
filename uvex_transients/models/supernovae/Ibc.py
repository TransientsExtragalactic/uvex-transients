"""Models of SEDs from type Ib and Ic (stripped-envelope) supernovae."""

from typing import ClassVar

from astropy import units as u

from uvex_transients.models.arnett import ArnettDecaySED
from uvex_transients.models.core.parameters import Parameter
from uvex_transients.models.core.priors import (
    ConstantPrior,
    TruncatedNormalPrior,
)

__all__ = ["TypeIbSED", "TypeIcSED"]


class TypeIbSED(ArnettDecaySED):
    r"""
    Type Ib supernova SED, with priors fit to a literature sample of stripped-envelope SNe.

    Reuses :class:`~uvex_transients.models.arnett.ArnettDecaySED`'s radioactive-decay Arnett
    diffusion and floored-photosphere blackbody entirely, exactly as
    :class:`~uvex_transients.models.supernovae.Ia.TypeIaSED` and
    :class:`~uvex_transients.models.supernovae.IcBL.TypeIcBLSED` do; only :attr:`_DEFAULT_PARAMETERS`
    differs. The priors on ``M_Ni``, ``M_ej`` and ``v_ej`` are the SN Ib subsample statistics (mean,
    sample standard deviation of the analytical-model fits in Table 6 of :footcite:t:`lyman2016`, 13
    events). ``kappa`` is fixed at :math:`0.06\,\mathrm{cm^2\,g^{-1}}`, the single grey optical opacity
    value Lyman et al. 2016 assume (rather than fit) for every event in their sample. ``kappa_gamma``
    is fixed at :math:`0.04\,\mathrm{cm^2\,g^{-1}}`, comparable to :class:`TypeIaSED`'s Scalzo+14-derived
    value (:math:`0.03\,\mathrm{cm^2\,g^{-1}}`): Lyman et al. 2016's own analytical model has no
    gamma-ray leakage term at all (it is the original Arnett 1982 diffusion formalism), so their table
    gives no direct constraint on it, and this package's :class:`TypeIaSED` value is used as the closest
    available analogue. ``T_floor`` is left close to :class:`TypeIaSED`'s value.

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
            prior=TruncatedNormalPrior(mean=0.17, sigma=0.16, lower=0.01, upper=3),
            scale=1 * u.Msun,
            description="Nickel-56 mass synthesized in the explosion.",
            latex=r"M_\mathrm{Ni}",
        ),
        "M_ej": Parameter(
            prior=TruncatedNormalPrior(mean=2.6, sigma=1.1, lower=0.1, upper=8.0),
            scale=1 * u.Msun,
            description="Ejecta mass.",
            latex=r"M_\mathrm{ej}",
        ),
        "v_ej": Parameter(
            prior=TruncatedNormalPrior(mean=9.9, sigma=1.4, lower=4.0, upper=16.0),
            scale=1e3 * u.km / u.s,
            description="Constant ejecta velocity, identified with the photospheric velocity.",
            latex=r"v_\mathrm{ej}",
        ),
        "kappa": Parameter(
            prior=ConstantPrior(value=0.06),
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
    }


class TypeIcSED(ArnettDecaySED):
    r"""
    Type Ic supernova SED, with priors fit to a literature sample of stripped-envelope SNe.

    Same construction as :class:`TypeIbSED` (see its docstring for the shared reasoning behind
    ``kappa``, ``kappa_gamma`` and ``T_floor``), but with ``M_Ni``, ``M_ej`` and ``v_ej`` drawn from
    the SN Ic subsample statistics of Table 6 of :footcite:t:`lyman2016` (8 events) instead of the
    Type Ib ones.

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
            prior=TruncatedNormalPrior(mean=0.22, sigma=0.16, lower=0.01, upper=3),
            scale=1 * u.Msun,
            description="Nickel-56 mass synthesized in the explosion.",
            latex=r"M_\mathrm{Ni}",
        ),
        "M_ej": Parameter(
            prior=TruncatedNormalPrior(mean=3, sigma=2.8, lower=0.1, upper=6),
            scale=1 * u.Msun,
            description="Ejecta mass.",
            latex=r"M_\mathrm{ej}",
        ),
        "v_ej": Parameter(
            prior=TruncatedNormalPrior(mean=10.4, sigma=1.2, lower=4.0, upper=16.0),
            scale=1e3 * u.km / u.s,
            description="Constant ejecta velocity, identified with the photospheric velocity.",
            latex=r"v_\mathrm{ej}",
        ),
        "kappa": Parameter(
            prior=ConstantPrior(value=0.06),
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
    }
