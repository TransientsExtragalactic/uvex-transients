r"""SED model for low-luminosity GRBs (llGRBs) and their broad-lined supernovae."""

from typing import ClassVar

import numpy as np
from astropy import units as u

from uvex_transients.models._constants import SECONDS_PER_DAY
from uvex_transients.models._typing import CGSParameterValue, FloatArray
from uvex_transients.models._utils import to_cgs_value
from uvex_transients.models.arnett import ArnettDecaySED
from uvex_transients.models.core.base import SpectralModel
from uvex_transients.models.core.parameters import Parameter
from uvex_transients.models.core.priors import NormalPrior
from uvex_transients.models.lightcurves.generic import BazinLightcurve
from uvex_transients.models.spectra.thermal import BlackbodySpectrum
from uvex_transients.models.supernovae.IcBL import TypeIcBLSED

__all__ = ["LLGRBSED"]

#: Earliest epoch, in days, at which the early excess's power-law temperature is evaluated. The power law
#: diverges as :math:`t\to0`. Holding it at its value here, about 15 minutes after the burst and well before the
#: first UVOT epoch at 0.065 d, keeps the blackbody shape finite at :math:`t=0`.
_T_BUMP_MIN_DAYS = 0.01


# ======================================== #
# Model                                    #
# ======================================== #
class LLGRBSED(SpectralModel):
    r"""
    Low-luminosity GRB SED: an early thermal excess plus a radioactively powered Ic-BL supernova.

    The emission is the sum of two blackbody components, each a unit-integral blackbody shape
    (:class:`~uvex_transients.models.spectra.thermal.BlackbodySpectrum`) scaled by its own bolometric light curve,

    .. math::

        L_\nu(\nu, t) = L_\mathrm{SN}(t)\,S_\mathrm{BB}\!\left[\nu, T_\mathrm{SN}(t)\right]
        + L_\mathrm{bump}(t)\,S_\mathrm{BB}\!\left[\nu, T_\mathrm{bump}(t)\right],

    so that :math:`L_\mathrm{bol}(t) = L_\mathrm{SN}(t) + L_\mathrm{bump}(t)` exactly.

    .. rubric:: Supernova

    :math:`L_\mathrm{SN}` is the Arnett diffusion light curve of
    :class:`~uvex_transients.models.arnett.ArnettDecaySED`, powered by :math:`^{56}\mathrm{Ni}` decay, with its floored
    photospheric temperature :math:`T_\mathrm{SN}(t)` and a plain blackbody (no line blanketing). It produces the late
    bump, which peaks at roughly :math:`10` d.

    .. rubric:: Early excess

    :math:`L_\mathrm{bump}` is a :class:`~uvex_transients.models.lightcurves.generic.BazinLightcurve`,

    .. math::

        L_\mathrm{bump}(t) = A\,\frac{e^{-(t-t_0)/\tau_\mathrm{fall}}}{1 + e^{-(t-t_0)/\tau_\mathrm{rise}}},

    with a power-law photospheric temperature,

    .. math::

        T_\mathrm{bump}(t) = T_1\left(\frac{t}{1\,\mathrm{d}}\right)^{-\alpha_T},

    which is held at its :math:`0.01` d value at earlier times and has no floor. The excess is attributed to shock
    breakout and cooling of material around the progenitor.

    The Bazin maximum is at :math:`t_0 - \tau_\mathrm{rise}\ln[\tau_\mathrm{rise}/(\tau_\mathrm{fall} -
    \tau_\mathrm{rise})]`, which is before the burst for the default priors. The excess is then already declining at
    :math:`t=0`, where :math:`L_\mathrm{bump}\sim8\times10^{43}` erg/s, and the rise of the UV light curves over the
    first :math:`\sim0.5` d comes from the photosphere cooling, not from the bolometric luminosity.

    .. rubric:: Parameters

    The first six parameters are those of :class:`~uvex_transients.models.arnett.ArnettDecaySED`, with the priors of
    :class:`~uvex_transients.models.supernovae.IcBL.TypeIcBLSED`. The last six describe the early excess.

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
         - Opacity to high-energy photons.
       * - ``T_floor``
         - :math:`T_\mathrm{floor}`
         - Minimum photospheric temperature of the supernova.
       * - ``A_bump``
         - :math:`A`
         - Bazin normalization of the early excess.
       * - ``t0_bump``
         - :math:`t_0`
         - Bazin transition time of the early excess.
       * - ``tau_rise``
         - :math:`\tau_\mathrm{rise}`
         - Bazin rise timescale of the early excess.
       * - ``tau_fall``
         - :math:`\tau_\mathrm{fall}`
         - Bazin decline timescale of the early excess.
       * - ``T_bump``
         - :math:`T_1`
         - Photospheric temperature of the early excess at 1 d.
       * - ``alpha_T``
         - :math:`\alpha_T`
         - Cooling power-law index of the early excess.

    See Also
    --------
    uvex_transients.models.arnett.ArnettDecaySED : The supernova component on its own.
    uvex_transients.models.supernovae.IcBL.TypeIcBLSED : The Ic-BL population this model's supernova priors come from.
    uvex_transients.models.lightcurves.generic.BazinLightcurve : The early-excess light curve.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy import units as u
    >>> from uvex_transients.models import LLGRBSED
    >>> sed = LLGRBSED()
    >>> params = sed.sample_parameters(
    ...     size=1, rng=np.random.default_rng(0)
    ... )
    >>> t = np.array([0.1, 1.0, 10.0]) * u.day
    >>> sed.eval_bolometric(t, **params)  # doctest: +SKIP
    """

    _ARNETT_PARAMETERS: ClassVar[tuple[str, ...]] = tuple(ArnettDecaySED._DEFAULT_PARAMETERS)
    """tuple of str: The parameters handed to the Arnett light curve."""

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        **{name: TypeIcBLSED._DEFAULT_PARAMETERS[name] for name in ArnettDecaySED._DEFAULT_PARAMETERS},
        "A_bump": Parameter(
            prior=NormalPrior(mean=44.19, sigma=0.3),
            scale=1.0 * u.erg / u.s,
            transform="log10",
            description="Bazin normalization of the early excess.",
            latex=r"A",
        ),
        "t0_bump": Parameter(
            prior=NormalPrior(mean=-1.0, sigma=0.25),
            scale=1.0 * u.day,
            transform="log10",
            description="Bazin transition time of the early excess.",
            latex=r"t_0",
        ),
        "tau_rise": Parameter(
            prior=NormalPrior(mean=-0.34, sigma=0.1),
            scale=1.0 * u.day,
            transform="log10",
            description="Bazin rise timescale of the early excess.",
            latex=r"\tau_\mathrm{rise}",
        ),
        "tau_fall": Parameter(
            prior=NormalPrior(mean=-0.16, sigma=0.1),
            scale=1.0 * u.day,
            transform="log10",
            description="Bazin decline timescale of the early excess.",
            latex=r"\tau_\mathrm{fall}",
        ),
        "T_bump": Parameter(
            prior=NormalPrior(mean=4.21, sigma=0.1),
            scale=1.0 * u.K,
            transform="log10",
            description="Photospheric temperature of the early excess at 1 d.",
            latex=r"T_1",
        ),
        "alpha_T": Parameter(
            prior=NormalPrior(mean=0.48, sigma=0.1),
            scale=1.0,
            description="Cooling power-law index of the early excess.",
            latex=r"\alpha_T",
        ),
    }

    # -------------------------------------- #
    # Components: L(t) and T(t)               #
    # -------------------------------------- #
    @classmethod
    def _bump_temperature_cgs(
        cls,
        t: FloatArray,
        *,
        T_bump: CGSParameterValue,
        alpha_T: CGSParameterValue,
        **_ignored: CGSParameterValue,
    ) -> FloatArray:
        r"""
        :math:`T_\mathrm{bump}(t) = T_1\,(t / 1\,\mathrm{d})^{-\alpha_T}`, held at its :math:`0.01` d value earlier.

        Parameters
        ----------
        t : numpy.ndarray
            Time since explosion, in seconds.
        T_bump, alpha_T : float or numpy.ndarray
            This model's parameter values, in cgs units; see the class docstring.
        **_ignored
            Any other model parameter values, ignored.

        Returns
        -------
        numpy.ndarray
            :math:`T_\mathrm{bump}(t)`, in Kelvin.
        """
        t_days = np.maximum(t, _T_BUMP_MIN_DAYS * SECONDS_PER_DAY) / SECONDS_PER_DAY
        return T_bump * t_days ** (-alpha_T)

    @classmethod
    def _components_cgs(
        cls, t: FloatArray, **parameters: CGSParameterValue
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        r"""
        Evaluate the bolometric luminosity and temperature of each component, with the Arnett integral done once.

        Parameters
        ----------
        t : numpy.ndarray
            Time since explosion, in seconds.
        **parameters
            This model's parameter values, in cgs units.

        Returns
        -------
        tuple of numpy.ndarray
            :math:`\ln L_\mathrm{SN}` in erg/s, :math:`T_\mathrm{SN}` in K, :math:`\ln L_\mathrm{bump}` in erg/s and
            :math:`T_\mathrm{bump}` in K, in that order.
        """
        L_sn, T_sn = ArnettDecaySED._luminosity_and_temperature_cgs(
            t, **{name: parameters[name] for name in cls._ARNETT_PARAMETERS}
        )
        ln_L_bump = BazinLightcurve._eval(
            t,
            amplitude=parameters["A_bump"],
            t0=parameters["t0_bump"],
            rise=parameters["tau_rise"],
            fall=parameters["tau_fall"],
        )
        with np.errstate(divide="ignore"):
            ln_L_sn = np.log(L_sn)
        return ln_L_sn, T_sn, ln_L_bump, cls._bump_temperature_cgs(t, **parameters)

    @classmethod
    def temperature(cls, t: u.Quantity, **parameters: u.Quantity) -> tuple[u.Quantity, u.Quantity]:
        r"""
        Return :math:`T_\mathrm{SN}(t)` and :math:`T_\mathrm{bump}(t)`, the temperature of each component.

        The two blackbodies are summed, so there is no single photospheric temperature.

        Parameters
        ----------
        t : ~astropy.units.Quantity
            Time since explosion.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        tuple of ~astropy.units.Quantity
            :math:`T_\mathrm{SN}(t)` and :math:`T_\mathrm{bump}(t)`, in Kelvin.
        """
        cgs_parameters = {name: to_cgs_value(value) for name, value in parameters.items()}
        _, T_sn, _, T_bump = cls._components_cgs(t.cgs.value, **cgs_parameters)
        return T_sn * u.K, T_bump * u.K

    # -------------------------------------- #
    # Bolometric Luminosity: L_bol(t)         #
    # -------------------------------------- #
    @classmethod
    def _eval_bolometric(cls, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        :math:`\ln L_\mathrm{bol}(t) = \ln\left[L_\mathrm{SN}(t) + L_\mathrm{bump}(t)\right]`, exactly.

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
        ln_L_sn, _, ln_L_bump, _ = cls._components_cgs(t, **parameters)
        return np.logaddexp(ln_L_sn, ln_L_bump)

    # -------------------------------------- #
    # Spectral Luminosity: L_nu(nu, t)        #
    # -------------------------------------- #
    @classmethod
    def _eval(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        :math:`\ln L_\nu(\nu, t)`, the sum of the two blackbody components.

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
        ln_L_sn, T_sn, ln_L_bump, T_bump = cls._components_cgs(t, **parameters)
        return np.logaddexp(
            ln_L_sn + BlackbodySpectrum._eval(nu, temperature=T_sn),
            ln_L_bump + BlackbodySpectrum._eval(nu, temperature=T_bump),
        )

    @classmethod
    def _eval_spectrum(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        :math:`\ln S(\nu, t) = \ln L_\nu - \ln L_\mathrm{bol}`, with the Arnett integral evaluated once.

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
        ln_L_sn, T_sn, ln_L_bump, T_bump = cls._components_cgs(t, **parameters)
        ln_L_nu = np.logaddexp(
            ln_L_sn + BlackbodySpectrum._eval(nu, temperature=T_sn),
            ln_L_bump + BlackbodySpectrum._eval(nu, temperature=T_bump),
        )
        return ln_L_nu - np.logaddexp(ln_L_sn, ln_L_bump)
