"""
The transient that stands in for one point of a recovery map.

A recovery map is over a population's peak luminosity and half-decline time, so a point of it needs a transient with
just those two free quantities. `_CanonicalTransient` is that transient: a fast-rise, exponential-decay light curve with
a power-law spectrum, whose shape, spectral index and band edges are fixed. `set_point` places it at a luminosity and
a half-decline time, and it is otherwise an ordinary `~uvex_transients.transients.base.ExtragalacticTransient`, at a
constant rate of one event per Mpc^3 per year, so the usual event sampling and SNR screening run on it unchanged (see
`~uvex_transients.phase_space.evaluator.RecoveryEvaluator`). Any other SED is placed on the same map by measuring the
same two quantities on it (`~uvex_transients.models.core.base.SpectralModel.peak_luminosity` and ``decline_time``).

A decaying light curve has no end, so an event is followed until it has faded by `fade_mag` magnitudes, which sets the
window of observations that can see it.

The module also holds the frequency limits of the spectrum and `fred_time_scales`. The class name starts with an
underscore and the module is not one the CLI imports, which keeps it out of ``full_run.yaml`` and the checks that
iterate over shipped classes.
"""

from typing import Union

import numpy as np
from astropy import units as u
from astropy.units import Quantity
from numpy.typing import NDArray
from scipy.optimize import brentq

from uvex_transients.models._util_functions import powerlaw_shape_integral_cgs
from uvex_transients.models.core.base import ComposedSpectralModel
from uvex_transients.models.lightcurves.generic import FREDLightcurve
from uvex_transients.models.spectra.powerlaw import PowerLawSpectrum
from uvex_transients.transients.base import ExtragalacticTransient
from uvex_transients.utils import config

__all__ = ["FREQUENCY_MAX", "FREQUENCY_MIN", "fred_time_scales"]

# ================================== #
# Spectrum                           #
# ================================== #
#: Rest-frame extent of the power-law spectrum: 3 micron to 30 A. The red end is far beyond anything UVEX sees, and the
#: blue end lies blueward of the FUV band out to z of about 40, which is where an event leaves the model.
FREQUENCY_MIN = 1e14 * u.Hz
FREQUENCY_MAX = 1e17 * u.Hz

#: Frequency at which `_CanonicalTransient.set_point` takes the spectral luminosity, unless one is given.
_REFERENCE_FREQUENCY = 1e15 * u.Hz


# ================================== #
# Light curve                        #
# ================================== #
def fred_time_scales(rise_ratio: float, fade_mag: float) -> tuple[float, float]:
    r"""
    Return the half-decline time and the fade time of a FRED light curve, in units of its decay time :math:`\tau_2`.

    With :math:`\tau_1=r\tau_2` and times in units of :math:`\tau_2`, the light curve is
    :math:`g(t)=\exp(2\sqrt r - r/t - t)`, with its peak, :math:`g=1`, at :math:`t=\sqrt r`.

    Parameters
    ----------
    rise_ratio : float
        The ratio :math:`r=\tau_1/\tau_2`, positive.
    fade_mag : float
        The depth, in magnitudes below the peak, that the fade time is the time to reach.

    Returns
    -------
    half_time : float
        The time from the peak until :math:`g` falls to 1/2, in units of :math:`\tau_2`.
    fade_time : float
        The time since explosion at which :math:`g` has fallen by `fade_mag` magnitudes, in units of :math:`\tau_2`.
    """
    peak = np.sqrt(rise_ratio)

    # `g` only falls after the peak, so each crossing is bracketed between the peak and a time far past the fade
    half_level, fade_level = 0.5, 10.0 ** (-0.4 * fade_mag)
    half_crossing, fade_crossing = (
        brentq(lambda t, level=level: np.exp(2.0 * peak - rise_ratio / t - t) - level, peak, 200.0 + 3.0 * fade_mag)
        for level in (half_level, fade_level)
    )
    return half_crossing - peak, fade_crossing


class _FREDPowerLawSED(ComposedSpectralModel):
    """
    A `FREDLightcurve` times a `PowerLawSpectrum`.

    `_CanonicalTransient` pins every parameter but the amplitude and the two timescales: the spectral index, the
    reference frequency and the band edges. ``amplitude`` is the peak bolometric luminosity between the edges.
    """

    _LIGHTCURVE_CLASS = FREDLightcurve
    _SPECTRUM_CLASS = PowerLawSpectrum


# ================================== #
# Transient                          #
# ================================== #
class _CanonicalTransient(ExtragalacticTransient):
    r"""
    A FRED light curve with a power-law spectrum, at a constant comoving rate of one event per Mpc^3 per year.

    `set_point` places it at a peak spectral luminosity and a half-decline time, and redshifts are drawn
    volume-weighted out to the `redshift_limit` it sets.

    Parameters
    ----------
    cosmology : ~astropy.cosmology.Cosmology, optional
        Defaults to the configured cosmology.
    spectral_index : float, optional
        The index :math:`\alpha` of :math:`L_\nu\\propto\nu^\alpha`, which sets the K-correction, a flux factor of
        :math:`(1+z)^{1+\alpha}`. If `None` (the default), uses ``config["phase_space.transient.spectral_index"]``.
    reference_frequency : ~astropy.units.Quantity, optional
        The frequency at which `set_point` takes the spectral luminosity. It fixes what a luminosity means and does
        not change the spectrum. The default is 1e15 Hz.
    rise_ratio : float, optional
        The ratio :math:`\tau_1/\tau_2` of the rise to the decay time. If `None` (the default), uses
        ``config["phase_space.transient.rise_ratio"]``.
    fade_mag : float, optional
        How far below its peak, in magnitudes, an event is followed. If `None` (the default), uses
        ``config["phase_space.transient.fade_mag"]``.
    """

    DEFAULT_MODEL = _FREDPowerLawSED
    DEFAULT_DURATION = 100 * u.day
    DEFAULT_Z_LIM = 1.0
    HAS_STORED_HORIZON_CURVE = False

    def __init__(
        self,
        cosmology=None,
        spectral_index: float | None = None,
        reference_frequency: Quantity | None = None,
        rise_ratio: float | None = None,
        fade_mag: float | None = None,
    ):
        super().__init__(cosmology=cosmology)

        # Anything not given comes from the configuration
        if spectral_index is None:
            spectral_index = config["phase_space.transient.spectral_index"]
        if rise_ratio is None:
            rise_ratio = config["phase_space.transient.rise_ratio"]
        if fade_mag is None:
            fade_mag = config["phase_space.transient.fade_mag"]
        if reference_frequency is None:
            reference_frequency = _REFERENCE_FREQUENCY
        if rise_ratio <= 0.0 or fade_mag <= 0.0:
            raise ValueError(f"Need rise_ratio > 0 and fade_mag > 0, got {rise_ratio!r} and {fade_mag!r}.")

        self.spectral_index = float(spectral_index)
        self.reference_frequency = reference_frequency.to(u.Hz)
        self.rise_ratio = float(rise_ratio)
        self.fade_mag = float(fade_mag)

        # With the rise ratio fixed the shape is A g(t / decay), so the half-decline time is a constant times the
        # decay time, and the window an event is followed over is a constant times the half-decline time
        half_time, fade_time = fred_time_scales(self.rise_ratio, self.fade_mag)
        self._half_time_per_decay = half_time
        self.window_factor = fade_time / half_time
        """float: How long an event is followed, in half-decline times, in its rest frame."""

        # Pin the spectrum. Only the amplitude and the two timescales move, and `set_point` moves them
        self.sed["spectral_index"].fix(self.spectral_index)
        self.sed["reference_frequency"].fix(self.reference_frequency)
        self.sed["frequency_min"].fix(FREQUENCY_MIN)
        self.sed["frequency_max"].fix(FREQUENCY_MAX)

    # -------------------------------------- #
    # Rate                                   #
    # -------------------------------------- #
    @property
    def rate(self) -> Quantity:
        """~astropy.units.Quantity: One event per cubic megaparsec per year, so counts are volume-times."""
        return 1.0 * u.Mpc**-3 * u.yr**-1

    def rate_shape(self, z: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
        """
        Return the constant rate shape, ones the shape of `z`.

        Parameters
        ----------
        z : float or numpy.ndarray
            Redshift(s).

        Returns
        -------
        float or numpy.ndarray
            Ones, the same shape as `z`.
        """
        z = np.asarray(z)
        shape = np.ones_like(z, dtype=np.float64)
        return shape if z.ndim > 0 else shape.item()

    # -------------------------------------- #
    # Point of the map                       #
    # -------------------------------------- #
    def set_point(self, luminosity: Quantity, half_time: Quantity, redshift_limit: float) -> None:
        r"""
        Move to a point of the map.

        Parameters
        ----------
        luminosity : ~astropy.units.Quantity
            The spectral luminosity :math:`L_\nu` at the peak, at the `reference_frequency`, in erg/s/Hz or
            equivalent.
        half_time : ~astropy.units.Quantity
            The rest-frame time :math:`t_{1/2}` from the peak until the luminosity has fallen to half.
        redshift_limit : float
            The farthest redshift to place events at. The window of observations that can see an event is the longest
            observed time it takes one to fade, :math:`(1+z_\mathrm{lim})\,t_{1/2}` times `window_factor`.
        """
        decay = half_time.to(u.day) / self._half_time_per_decay

        # The spectrum S(nu) = (1/nu_0) (nu/nu_0)^alpha has the dimensionless integral N over the band, so the
        # spectral luminosity at nu_0 is L_bol / (nu_0 N), and the amplitude that gives the luminosity asked for is
        # the inverse
        shape_integral = powerlaw_shape_integral_cgs(
            self.spectral_index,
            self.reference_frequency.to_value(u.Hz),
            FREQUENCY_MIN.to_value(u.Hz),
            FREQUENCY_MAX.to_value(u.Hz),
        )
        self.sed["amplitude"].fix((luminosity * self.reference_frequency * shape_integral).to(u.erg / u.s))
        self.sed["decay"].fix(decay)
        self.sed["rise"].fix(self.rise_ratio * decay)

        # The survey windows follow the light curve out to its fade, stretched by time dilation at the farthest event
        self.redshift_limit = float(redshift_limit)
        self.duration_limit = (1.0 + redshift_limit) * self.window_factor * half_time.to(u.day)
