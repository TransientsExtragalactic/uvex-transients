r"""
A flat-spectrum, top-hat transient: one point of a luminosity-duration recovery map.

`_TopHatTransient` is an `~uvex_transients.transients.base.ExtragalacticTransient` whose SED is a constant
spectral luminosity :math:`L_\nu` for a rest-frame duration :math:`\Delta t`, with the same value at every frequency. It
exists so that the ordinary machinery (`SurveySimulator.sample_N_events`, `SurveySimulator.filter_by_snr`) can
be run on a single point of the map unchanged; `set_point` moves the one instance between points.

The volumetric rate is fixed at one event per :math:`\mathrm{Mpc}^3\,\mathrm{yr}`. The rate does not matter for a
fixed-count sample. It makes the expected number of events over the survey, at this rate, the effective
volume-time :math:`\mathcal V T` in :math:`\mathrm{Mpc}^3\,\mathrm{yr}`, which scales to any other rate by
multiplication.

The name starts with an underscore, and the module is not in the CLI's list of known transient modules, so the
class stays out of ``full_run.yaml`` and the checks that iterate over shipped classes.
"""

from typing import Union

import numpy as np
from astropy import units as u
from astropy.units import Quantity
from numpy.typing import NDArray

from uvex_transients.models.core.base import ComposedSpectralModel
from uvex_transients.models.lightcurves.generic import TopHatLightcurve
from uvex_transients.models.spectra.powerlaw import PowerLawSpectrum
from uvex_transients.transients.base import ExtragalacticTransient

__all__ = ["BANDWIDTH", "FREQUENCY_MAX", "FREQUENCY_MIN"]

FREQUENCY_MIN = 1e14 * u.Hz
"""~astropy.units.Quantity: Lower edge of the flat spectrum (3 micron). Far redder than anything UVEX sees."""

FREQUENCY_MAX = 1e17 * u.Hz
"""~astropy.units.Quantity: Upper edge of the flat spectrum (30 A), blueward of the FUV band out to z of about 40."""

BANDWIDTH = FREQUENCY_MAX - FREQUENCY_MIN
r"""~astropy.units.Quantity: Width of the flat spectrum, which converts :math:`L_\nu` to a bolometric amplitude."""


class _FlatTopHatSED(ComposedSpectralModel):
    r"""
    :math:`L_\nu(\nu, t) = L_\mathrm{bol}\,\Theta(\Delta t - t) / (\nu_\mathrm{max}-\nu_\mathrm{min})`.

    A `TopHatLightcurve` times a `PowerLawSpectrum` of index zero. The spectral index and the band edges are pinned by
    `_TopHatTransient`; ``amplitude`` is the bolometric luminosity, ``duration`` the rest-frame plateau length.
    """

    _LIGHTCURVE_CLASS = TopHatLightcurve
    _SPECTRUM_CLASS = PowerLawSpectrum


class _TopHatTransient(ExtragalacticTransient):
    r"""
    A top-hat light curve with a flat spectrum, at a constant comoving rate of one event per Mpc^3 per year.

    Use `set_point` to choose the spectral luminosity and the rest-frame duration. Redshifts are drawn
    from the volume-weighted distribution out to `redshift_limit`, set by `set_point`.
    """

    DEFAULT_MODEL = _FlatTopHatSED
    DEFAULT_DURATION = 100 * u.day
    DEFAULT_Z_LIM = 1.0
    HAS_STORED_HORIZON_CURVE = False

    def __init__(self, cosmology=None):
        super().__init__(cosmology=cosmology)
        self.sed["spectral_index"].fix(0.0)
        self.sed["reference_frequency"].fix(1e15 * u.Hz)
        self.sed["frequency_min"].fix(FREQUENCY_MIN)
        self.sed["frequency_max"].fix(FREQUENCY_MAX)

    @property
    def rate(self) -> Quantity:
        """~astropy.units.Quantity: One event per cubic megaparsec per year, so counts are volume-times."""
        return 1.0 * u.Mpc**-3 * u.yr**-1

    def rate_shape(self, z: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
        """
        Return the constant rate shape.

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

    def set_point(self, luminosity: Quantity, duration: Quantity, redshift_limit: float) -> None:
        r"""
        Move to a point of the map.

        Parameters
        ----------
        luminosity : ~astropy.units.Quantity
            The flat spectral luminosity :math:`L_\nu`, in erg/s/Hz or equivalent.
        duration : ~astropy.units.Quantity
            The rest-frame duration :math:`\Delta t` of the plateau.
        redshift_limit : float
            The farthest redshift to place events at. The window that decides which observations can see an event
            is set to the longest observed duration, :math:`(1+z_\mathrm{lim})\Delta t`.
        """
        self.sed["amplitude"].fix((luminosity * BANDWIDTH).to(u.erg / u.s))
        self.sed["duration"].fix(duration.to(u.day))
        self.redshift_limit = float(redshift_limit)
        self.duration_limit = (1.0 + redshift_limit) * duration.to(u.day)
