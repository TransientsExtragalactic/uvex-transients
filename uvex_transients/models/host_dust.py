r"""
Host-galaxy dust as part of an SED model.

`HostDustMixin` adds two parameters, the V-band attenuation :math:`A_V` and the total-to-selective ratio
:math:`R_V`, to a `~uvex_transients.models.core.base.SpectralModel` and multiplies its rest-frame spectral
luminosity by the attenuation of a reddening law (G23 by default, see `HostDustMixin.DUST_LAW`),

.. math::

    L_\nu^\mathrm{obs}(\nu, t) = L_\nu^\mathrm{int}(\nu, t)\,10^{-0.4\,A(\nu)},
    \qquad A(\nu) = A_V\,\frac{A(\nu)}{A(V)}\bigg|_{R_V}.

The dust is *part of the model*, not a line-of-sight correction. Milky Way foreground dust is a different
thing and remains the ``ebv``/``log_attenuation`` argument of the photometry methods (see
`uvex_transients.dust`).
"""

from typing import ClassVar

import numpy as np

from uvex_transients.dust import host_log_attenuation_cgs
from uvex_transients.models._constants import LOG_EV_TO_ERG, LOG_H_CGS
from uvex_transients.models._typing import CGSParameterValue, FloatArray
from uvex_transients.models.core.parameters import Parameter
from uvex_transients.models.core.priors import ExponentialPrior, TruncatedNormalPrior

__all__ = ["HostDustMixin"]


class HostDustMixin:
    r"""
    Mixin that puts host-galaxy dust attenuation into an SED model.

    Put it *before* the SED in the bases (``class Foo(HostDustMixin, SomeSED)``) and add
    :attr:`HOST_DUST_PARAMETERS` to the subclass's ``_DEFAULT_PARAMETERS``. The wrapped SED never sees
    ``A_V`` and ``R_V``: they are removed before its own methods are called.

    .. rubric:: What the dust is, and what that means for :math:`L_\mathrm{bol}`

    The attenuation is applied to the rest-frame :math:`L_\nu`, so it enters everything derived from the SED:
    photometry, peak band luminosities and the detection horizon. It is deliberately not a separate
    correction. For a supernova we do not know the dust conditions that shaped the light we receive, so the
    model describes the *observed and inferred* luminosity of the event, dust included, rather than the
    intrinsic one. Consequently :math:`L_\mathrm{bol}(t)` is the integral of the attenuated
    :math:`L_\nu` (this is what keeps ``eval_bolometric`` consistent with ``eval`` and the normalized spectrum
    integrating to 1), and it is *lower* than the wrapped SED's own bolometric luminosity by the fraction
    absorbed. Set ``A_V=0`` to recover the intrinsic SED exactly.

    The bolometric integral has no closed form here. It is evaluated on a fixed, log-spaced grid in
    :math:`h\nu` (:attr:`_LOG_HNU_EV_GRID`) with the trapezoidal rule in :math:`\ln\nu`, in chunks to bound memory.

    Attributes
    ----------
    DUST_LAW : str
        Name of the reddening law, as registered in `uvex_transients.dust` (``"g23"``). The law must be affine
        in :math:`1/R_V`, which is what lets :math:`R_V` differ from draw to draw.
    HOST_DUST_PARAMETERS : dict of str to Parameter
        ``A_V`` (exponential, mean 0.4 mag) and ``R_V`` (normal, :math:`2.70 \pm 0.25`, truncated to the 2.3 to
        5.6 range that G23 is defined on). The :math:`R_V` width is that of the quoted uncertainty on the mean,
        less than the 95% upper limit of 0.61 on the population scatter.
    """

    DUST_LAW: ClassVar[str] = "g23"
    """str: Reddening law of the host dust, looked up in `uvex_transients.dust`. Change it per subclass."""

    HOST_DUST_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "A_V": Parameter(
            prior=ExponentialPrior(scale=0.4),
            scale=1.0,
            description="V-band attenuation by dust in the host galaxy, in magnitudes.",
            latex=r"A_V",
        ),
        "R_V": Parameter(
            prior=TruncatedNormalPrior(mean=2.7, sigma=0.25, lower=2.3, upper=5.6),
            scale=1.0,
            description="Total-to-selective extinction ratio of the host dust.",
            latex=r"R_V",
        ),
    }

    _HOST_DUST_NAMES: ClassVar[tuple[str, ...]] = ("A_V", "R_V")

    _LOG_HNU_EV_GRID: ClassVar[FloatArray] = np.log(np.geomspace(2e-2, 3e2, 300))
    """numpy.ndarray: Natural log of the :math:`h\\nu` grid, in eV, of the bolometric integral."""

    _BOLOMETRIC_CHUNK: ClassVar[int] = 4000
    """int: Number of (time, parameter) realizations integrated at once, to bound the memory of the grid."""

    # -------------------------------------- #
    # Helpers                                #
    # -------------------------------------- #
    @classmethod
    def _split_host_dust(
        cls, parameters: dict[str, CGSParameterValue]
    ) -> tuple[dict[str, CGSParameterValue], dict[str, CGSParameterValue]]:
        """Split the cgs parameters into the wrapped SED's own and the dust's ``A_V`` and ``R_V``."""
        dust = {name: parameters[name] for name in cls._HOST_DUST_NAMES}
        rest = {name: value for name, value in parameters.items() if name not in cls._HOST_DUST_NAMES}
        return rest, dust

    @classmethod
    def _luminosity_and_temperature_cgs(cls, t: FloatArray, **parameters: CGSParameterValue):  # type: ignore[override]
        """Return the wrapped SED's :math:`L(t)` and :math:`T(t)`, which do not depend on the dust parameters."""
        # The wrapped SED's own methods call this with the dust parameters already removed, others with them.
        rest = {name: value for name, value in parameters.items() if name not in cls._HOST_DUST_NAMES}
        return super()._luminosity_and_temperature_cgs(t, **rest)  # type: ignore[misc]

    # -------------------------------------- #
    # Spectral Luminosity: L_nu(nu, t)        #
    # -------------------------------------- #
    @classmethod
    def _eval(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        :math:`\log L_\nu(\nu, t)` of the wrapped SED plus the host-dust attenuation at the rest-frame ``nu``.

        Parameters
        ----------
        nu : numpy.ndarray
            Rest-frame frequency, in Hz.
        t : numpy.ndarray
            Time since explosion, in seconds.
        **parameters
            This model's parameter values, in cgs units, including ``A_V`` and ``R_V``.

        Returns
        -------
        numpy.ndarray
            The natural log of the attenuated :math:`L_\nu(\nu, t)`, in erg/s/Hz.
        """
        rest, dust = cls._split_host_dust(parameters)
        intrinsic = super()._eval(nu, t, **rest)  # type: ignore[misc]
        return intrinsic + host_log_attenuation_cgs(nu, dust["A_V"], dust["R_V"], cls.DUST_LAW)

    # -------------------------------------- #
    # Bolometric Luminosity: L_bol(t)         #
    # -------------------------------------- #
    @classmethod
    def _eval_bolometric(cls, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        :math:`\log L_\mathrm{bol}(t)` of the attenuated SED, by quadrature of :meth:`_eval` over frequency.

        Parameters
        ----------
        t : numpy.ndarray
            Time since explosion, in seconds, broadcastable against ``parameters``.
        **parameters
            This model's parameter values, in cgs units, including ``A_V`` and ``R_V``.

        Returns
        -------
        numpy.ndarray
            The natural log of :math:`L_\mathrm{bol}(t)`, in erg/s. ``-inf`` where the SED is dark.
        """
        t_grid, *param_arrays = np.broadcast_arrays(np.asarray(t, dtype=np.float64), *parameters.values())
        names = list(parameters)
        flat_t = t_grid.ravel()
        flat_params = [np.ravel(array) for array in param_arrays]

        log_hnu = cls._LOG_HNU_EV_GRID
        nu_grid = np.exp(log_hnu + LOG_EV_TO_ERG - LOG_H_CGS)[:, np.newaxis]

        integral = np.empty(flat_t.shape)
        for start in range(0, flat_t.size, cls._BOLOMETRIC_CHUNK):
            chunk = slice(start, start + cls._BOLOMETRIC_CHUNK)
            log_L_nu = cls._eval(
                nu_grid,
                flat_t[chunk][np.newaxis, :],
                **{name: value[chunk][np.newaxis, :] for name, value in zip(names, flat_params)},
            )
            # With d(ln nu) the integrand is L_nu * nu, integrated by the trapezoidal rule in ln(nu).
            integrand = np.where(np.isnan(log_L_nu), 0.0, np.exp(log_L_nu) * nu_grid)
            integral[chunk] = np.trapezoid(integrand, x=log_hnu, axis=0)

        with np.errstate(divide="ignore"):
            return np.log(integral).reshape(t_grid.shape)

    # -------------------------------------- #
    # Normalized Spectral Shape: S(nu, t)    #
    # -------------------------------------- #
    @classmethod
    def _eval_spectrum(cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        :math:`\log S(\nu, t) = \log L_\nu - \log L_\mathrm{bol}`, normalized to 1 for the attenuated SED.

        Parameters
        ----------
        nu : numpy.ndarray
            Rest-frame frequency, in Hz.
        t : numpy.ndarray
            Time since explosion, in seconds.
        **parameters
            This model's parameter values, in cgs units, including ``A_V`` and ``R_V``.

        Returns
        -------
        numpy.ndarray
            The natural log of the normalized spectral shape, in 1/Hz.
        """
        return cls._eval(nu, t, **parameters) - cls._eval_bolometric(t, **parameters)
