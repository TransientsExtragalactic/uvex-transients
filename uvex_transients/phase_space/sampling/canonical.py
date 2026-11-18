r"""
The canonical sampler: a fast-rise, exponential-decay transient with a power-law spectrum.

:class:`~uvex_transients.phase_space.sampling.canonical.CanonicalSampler` is the sampler of the canonical recovery map.
A point of its space is the base-10 logarithm of
:math:`L_\mathrm{band}` in erg/s and of :math:`t_{1/2}` in days. :math:`L_\mathrm{band}` is the peak rest-frame
luminosity emitted in the band, :math:`L_\nu\Delta\nu` for the equivalent bandwidth :math:`\Delta\nu` of the band
with :math:`L_\nu` taken at its effective frequency, and :math:`t_{1/2}` is the rest-frame time from the peak until the
luminosity has fallen to half. Any other SED is placed on the same map by measuring the same two quantities on it
(:meth:`~uvex_transients.models.core.base.SpectralModel.peak_luminosity` and ``decline_time``). One sampler makes the
map of one band.

The transient that stands in for one point of the map is
:class:`~uvex_transients.phase_space.sampling.canonical._CanonicalTransient`. A recovery map is over a population's peak
luminosity and half-decline time, so a point needs a transient with just those two free quantities: a FRED light curve
with a power-law spectrum, whose shape, spectral index and band edges are fixed. `~_CanonicalTransient.set_point` places
it at a luminosity and a half-decline time, and it is otherwise an ordinary
:class:`~uvex_transients.transients.base.ExtragalacticTransient`, at a constant rate of one event per Mpc^3 per year, so
the usual event sampling and SNR screening run on it unchanged. A decaying light curve has no end, so an event is
followed until it has faded by `fade_mag` magnitudes, which sets the window of observations that can see it. The class
name starts with an underscore and the module is not one the CLI imports, which keeps it out of ``full_run.yaml`` and
the checks that iterate over shipped classes.

:class:`~uvex_transients.phase_space.sampling.canonical.RedshiftLimit` is the redshift out to which a power-law source
of a given luminosity is brighter than a depth, tabulated once and inverted by interpolation.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Union

import numpy as np
from astropy import units as u
from astropy.cosmology import Cosmology
from astropy.table import Table
from astropy.units import Quantity
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from uvex_transients.models._util_functions import powerlaw_shape_integral_cgs
from uvex_transients.models.core.base import ComposedSpectralModel
from uvex_transients.models.lightcurves.generic import FREDLightcurve
from uvex_transients.models.spectra.powerlaw import PowerLawSpectrum
from uvex_transients.phase_space._config import configured
from uvex_transients.phase_space.sampling.core import SurveySampler
from uvex_transients.phase_space.space import Axis, Space
from uvex_transients.surveys.base import SurveySchedule
from uvex_transients.transients.base import ExtragalacticTransient
from uvex_transients.utils import config, logger
from uvex_transients.utils.cosmology import get_cosmology
from uvex_transients.utils.phot_utils import effective_frequency, equivalent_bandwidth

__all__ = ["FREQUENCY_MAX", "FREQUENCY_MIN", "CanonicalSampler", "RedshiftLimit", "fred_time_scales"]


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
    A FRED light curve times a power-law spectrum, as one spectral model.

    It is a :class:`~uvex_transients.models.lightcurves.generic.FREDLightcurve` times a
    :class:`~uvex_transients.models.spectra.powerlaw.PowerLawSpectrum`.
    :class:`~uvex_transients.phase_space.sampling.canonical._CanonicalTransient` pins every parameter but the amplitude
    and the two timescales: the spectral index, the reference frequency and the band edges. ``amplitude`` is the peak
    bolometric luminosity between the edges.
    """

    _LIGHTCURVE_CLASS = FREDLightcurve
    _SPECTRUM_CLASS = PowerLawSpectrum


# ================================== #
# Transient                          #
# ================================== #
class _CanonicalTransient(ExtragalacticTransient):
    r"""
    A FRED light curve with a power-law spectrum, at a constant comoving rate of one event per Mpc^3 per year.

    :meth:`~uvex_transients.phase_space.sampling.canonical._CanonicalTransient.set_point` places it at a peak spectral
    luminosity and a half-decline time, and redshifts are drawn volume-weighted out to the
    :attr:`~uvex_transients.transients.base.ExtragalacticTransient.redshift_limit` it sets.

    Parameters
    ----------
    cosmology : ~astropy.cosmology.Cosmology, optional
        Defaults to the configured cosmology.
    spectral_index : float, optional
        The index :math:`\alpha` of :math:`L_\nu\\propto\nu^\alpha`, which sets the K-correction, a flux factor of
        :math:`(1+z)^{1+\alpha}`. If `None` (the default), uses ``config["phase_space.canonical.spectral_index"]``.
    reference_frequency : ~astropy.units.Quantity, optional
        The frequency at which :meth:`~uvex_transients.phase_space.sampling.canonical._CanonicalTransient.set_point`
        takes the spectral luminosity. It fixes what a luminosity means and does not change the spectrum. The default is
        1e15 Hz.
    rise_ratio : float, optional
        The ratio :math:`\tau_1/\tau_2` of the rise to the decay time. If `None` (the default), uses
        ``config["phase_space.canonical.rise_ratio"]``.
    fade_mag : float, optional
        How far below its peak, in magnitudes, an event is followed. If `None` (the default), uses
        ``config["phase_space.canonical.fade_mag"]``.
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
            spectral_index = config["phase_space.canonical.spectral_index"]
        if rise_ratio is None:
            rise_ratio = config["phase_space.canonical.rise_ratio"]
        if fade_mag is None:
            fade_mag = config["phase_space.canonical.fade_mag"]
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


# ============================================ #
# The canonical sampler                        #
# ============================================ #

#: Absolute AB magnitude of a source of spectral luminosity L_nu, M = -2.5 log10(L_nu / (4 pi (10 pc)^2)) - 48.6, is
#: M = _M_ZERO - 2.5 log10(L_nu / (erg/s/Hz)).
_M_ZERO = -2.5 * np.log10(1.0 / (4.0 * np.pi * (10.0 * u.pc).to_value(u.cm) ** 2)) - 48.6


# ============================================ #
# Redshift Limit                               #
# ============================================ #
class RedshiftLimit:
    r"""
    The redshift out to which a power-law source of a given luminosity is brighter than a depth.

    For a spectrum :math:`L_\nu\propto\nu^\alpha` the AB magnitude in a band at the observed frequency
    :math:`\nu_\mathrm{obs}` is

    .. math::

        m = -2.5\log_{10}\frac{(1+z)^{1+\alpha}\,L_\nu(\nu_\mathrm{obs})}{4\pi d_L^2(z)} - 48.6,

    where :math:`(1+z)^{1+\alpha}` is the K-correction: one factor of :math:`1+z` is the bandwidth compression and
    :math:`(1+z)^\alpha` the shift to the bluer rest-frame frequency :math:`(1+z)\nu_\mathrm{obs}`. The luminosity at
    which :math:`m` equals the depth is tabulated once on a redshift grid, increases with :math:`z`, and is inverted by
    interpolation, so a call is a table lookup. The band is taken as the single frequency :math:`\nu_\mathrm{obs}` at
    which the luminosity is quoted, which is accurate enough for a limit that only bounds where events are drawn.

    With a `bandwidth` the luminosity is the one in the band, :math:`L_\mathrm{band}=L_\nu(\nu_\mathrm{obs})\Delta\nu`
    in erg/s (see :func:`~uvex_transients.utils.phot_utils.equivalent_bandwidth`), the same table shifted by
    :math:`\log_{10}\Delta\nu`.

    Parameters
    ----------
    mag_limit : float
        The AB depth. Events fainter than it, at their own redshift, are never drawn.
    cosmology : ~astropy.cosmology.Cosmology, optional
        Defaults to the configured cosmology.
    z_min, z_max : float, optional
        The range of the table. A luminosity whose limit lies outside it is clipped to the nearest end.
    n_grid : int, optional
        Number of log-spaced redshifts in the table. The default is 512.
    bandwidth : ~astropy.units.Quantity, optional
        The equivalent width of the band, in frequency. If given, luminosities are :math:`L_\mathrm{band}` in erg/s;
        if omitted, the spectral luminosity :math:`L_\nu` in erg/s/Hz.
    spectral_index : float, optional
        The index :math:`\alpha` of the spectrum. The default of 0 is flat.

    Raises
    ------
    ValueError
        If the luminosity that reaches the depth does not increase with redshift, which happens for a spectrum rising
        steeply to the blue (an index of about 3 or more): such a source has no redshift limit.
    """

    def __init__(
        self,
        mag_limit: float,
        cosmology: Cosmology | None = None,
        z_min: float = 1e-3,
        z_max: float = 10.0,
        n_grid: int = 512,
        bandwidth: Quantity | None = None,
        spectral_index: float = 0.0,
    ):
        cosmology = get_cosmology(cosmology)
        self.mag_limit = float(mag_limit)
        self.spectral_index = float(spectral_index)
        self.z_min, self.z_max = float(z_min), float(z_max)

        # The spectral luminosity that gives the limiting flux density at every redshift of the table
        z = np.geomspace(self.z_min, self.z_max, n_grid)
        d_l = cosmology.luminosity_distance(z).to_value(u.cm)
        flux = 10.0 ** (-0.4 * (self.mag_limit + 48.6))
        k_correction = (1.0 + z) ** (1.0 + self.spectral_index)

        # Shifted to the luminosity in the band, if there is a bandwidth to shift it by
        log_offset = 0.0 if bandwidth is None else float(np.log10(bandwidth.to_value(u.Hz)))
        self._log_luminosity = np.log10(4.0 * np.pi * d_l**2 * flux / k_correction) + log_offset
        self._log_z = np.log10(z)

        # The table is inverted by interpolation, so it has to be monotonic
        if np.any(np.diff(self._log_luminosity) <= 0.0):
            raise ValueError(
                f"A source of spectral index {self.spectral_index} needs no more luminosity to reach the depth at a "
                f"greater redshift somewhere below z={self.z_max:g}, so it has no redshift limit; the spectrum is too "
                "blue."
            )

    def __call__(self, log_luminosity: ArrayLike) -> NDArray[np.float64]:
        r"""
        Return the redshift limit of a luminosity.

        Parameters
        ----------
        log_luminosity : array-like
            :math:`\log_{10}` of the luminosity in the band in erg/s, or, without a `bandwidth`, of the spectral
            luminosity in erg/s/Hz.

        Returns
        -------
        numpy.ndarray
            The redshift at which the source reaches the depth, clipped to ``[z_min, z_max]``.
        """
        return 10.0 ** np.interp(np.asarray(log_luminosity, dtype=np.float64), self._log_luminosity, self._log_z)

    @property
    def log_luminosity_max(self) -> float:
        """float: The log10 luminosity above which the limit is clipped to ``z_max``, not set by depth."""
        return float(self._log_luminosity[-1])

    def is_clipped(self, log_luminosity: ArrayLike) -> NDArray[np.bool_]:
        """
        Return whether the limit of each luminosity lies beyond the table, so that it was clipped.

        Such a limit is one that :meth:`~uvex_transients.phase_space.sampling.canonical.RedshiftLimit.__call__` held at
        the end of its table.

        Parameters
        ----------
        log_luminosity : array-like
            The log10 luminosity, in the same units as for
            :meth:`~uvex_transients.phase_space.sampling.canonical.RedshiftLimit.__call__`.

        Returns
        -------
        numpy.ndarray
            Boolean array, true where the luminosity is outside the range the table covers.
        """
        log_luminosity = np.asarray(log_luminosity, dtype=np.float64)
        return (log_luminosity < self._log_luminosity[0]) | (log_luminosity > self._log_luminosity[-1])


# ============================================ #
# Canonical sampler                            #
# ============================================ #
class CanonicalSampler(SurveySampler, kind="canonical"):
    r"""
    Count the recoveries of a canonical transient at points of the luminosity-half-decline-time plane.

    The transient has a fast-rise, exponential-decay light curve and a power-law spectrum (see
    :class:`~uvex_transients.phase_space.sampling.canonical._CanonicalTransient`). A point is
    ``x = (log10 L_band / (erg/s), log10 t_half / day)``. :math:`L_\mathrm{band}` is the peak rest-frame luminosity
    emitted in the band, :math:`L_\nu\Delta\nu` for the equivalent bandwidth :math:`\Delta\nu` of the band with
    :math:`L_\nu` taken at its effective frequency. One sampler makes the map of one band.

    Both axes are derived quantities, not parameters of the SED, so this sampler places a point itself instead of
    binding axes to parameters.

    Every argument that is `None` (the default) takes ``config["phase_space.canonical.<name>"]`` for the shape of the
    transient, and what :class:`~uvex_transients.phase_space.sampling.core.SurveySettings` resolves for the rest.

    Parameters
    ----------
    schedule : ~uvex_transients.surveys.base.SurveySchedule
        The survey to recover events against.
    mission : str or m4opt.missions.Mission, optional
        The mission (detector and background), or the name of one.
    spectral_index : float, optional
        The index :math:`\alpha` of the spectrum, :math:`L_\nu\propto\nu^\alpha`, which sets the K-correction
        :math:`(1+z)^{1+\alpha}`. A spectrum rising steeply to the blue (about 3) has no redshift limit and raises
        `ValueError`.
    rise_ratio : float, optional
        The ratio of the rise to the decay time of the light curve.
    fade_mag : float, optional
        How far below its peak, in magnitudes, an event is followed by the survey.
    cosmology : ~astropy.cosmology.Cosmology, optional
        Defaults to the configured cosmology.
    schedule_name : str, optional
        The registry name of `schedule`, recorded in
        :attr:`~uvex_transients.phase_space.sampling.core.SurveySampler.settings` so that
        :meth:`~uvex_transients.phase_space.sampling.canonical.CanonicalSampler.from_settings` can fetch the same
        schedule.
    **survey
        What counts as a recovery and how a point is simulated: ``band`` (the one that recovers events and in which the
        luminosity is measured), ``snr_threshold``, ``n_visits``, ``exclude_first_visit_detections``, ``detection``,
        ``mag_limit``, ``time_bins``, ``nside`` and ``order``. See
        :class:`~uvex_transients.phase_space.sampling.core.SurveySampler` and
        :class:`~uvex_transients.phase_space.sampling.core.SurveySettings`.
    """

    TRANSIENT_KEY = "canonical"

    def __init__(
        self,
        schedule: SurveySchedule,
        mission=None,
        *,
        spectral_index: float | None = None,
        rise_ratio: float | None = None,
        fade_mag: float | None = None,
        cosmology: Cosmology | None = None,
        schedule_name: str | None = None,
        **survey,
    ):
        super().__init__(schedule, mission, cosmology=cosmology, schedule_name=schedule_name, **survey)
        self.spectral_index = float(configured(spectral_index, "canonical.spectral_index"))

        # The scale of the luminosity axis: the luminosity in the band, from the spectral one at the band's
        # effective frequency
        bandpass = self.mission.detector.bandpasses[self.band]
        self.reference_frequency = effective_frequency(bandpass)
        self.bandwidth = equivalent_bandwidth(bandpass, self.spectral_index, self.reference_frequency)

        # The shape of the transient is recorded beside what the survey sampler records
        shape = {
            "spectral_index": self.spectral_index,
            "rise_ratio": float(configured(rise_ratio, "canonical.rise_ratio")),
            "fade_mag": float(configured(fade_mag, "canonical.fade_mag")),
        }

        # The redshift limit comes from the depth alone. Its table ends where the band leaves the spectrum, which the
        # model sets at FREQUENCY_MAX in the rest frame, so that no event is drawn where it would be unseen
        redshift_edge = float(FREQUENCY_MAX / self.reference_frequency) - 1.0
        self._limit = RedshiftLimit(
            self.mag_limit,
            self.cosmology,
            z_min=config["phase_space.numerics.z_table_min"],
            z_max=redshift_edge,
            n_grid=config["phase_space.numerics.z_table_points"],
            bandwidth=self.bandwidth,
            spectral_index=self.spectral_index,
        )

        space = Space(
            [
                Axis(
                    "log_luminosity",
                    *config["phase_space.canonical.log_luminosity"],
                    label=rf"$L_\mathrm{{{self.band}}}$",
                    unit="erg/s",
                ),
                Axis(
                    "log_half_time",
                    *config["phase_space.canonical.log_half_time"],
                    label=r"rest-frame half-decline time $t_{1/2}$",
                    unit="day",
                ),
            ]
        )
        self._configure(space, redshift_edge, **shape)

    # -------------------------------------- #
    # The figure and its overlay             #
    # -------------------------------------- #
    HAS_OVERLAY = True

    def measure_overlay(self, n_draws: int | None = None, seed: int = 0) -> Table:
        """
        Measure the shipped transient classes in the band of the map, to shade on its figure.

        Parameters
        ----------
        n_draws : int, optional
            SED draws per class. By default ``config["phase_space.overlay.n_draws"]``.
        seed : int, optional
            The seed of the draws.

        Returns
        -------
        astropy.table.Table
            One row per draw, in the axes of the map, with the class of each (see
            :func:`~uvex_transients.phase_space.analysis.overlay.stack_measurements`).
        """
        from uvex_transients.phase_space.analysis.overlay import measure_class, shipped_classes, stack_measurements

        bandpass = self.mission.detector.bandpasses[self.band]
        return stack_measurements(
            [measure_class(cls, bandpass, n_draws=n_draws, rng=seed) for cls in shipped_classes().values()]
        )

    @classmethod
    def draw_map(
        cls,
        grid: Table,
        points: Table,
        path: str | Path,
        title: str,
        *,
        overlay: Table | None = None,
        show_points: bool = False,
        fields: Sequence[str] | None = None,
    ):
        """
        Draw the canonical map, with its four panels and the shipped classes shaded if there is an overlay.

        With `fields` the panels are the ones asked for, drawn by the default figure.

        Parameters
        ----------
        grid : astropy.table.Table
            The fitted map (``map.ecsv``).
        points : astropy.table.Table
            The measured points (``points.ecsv``).
        path : str or pathlib.Path
            Where to save the figure.
        title : str
            The suptitle.
        overlay : astropy.table.Table, optional
            The measurements of :meth:`~uvex_transients.phase_space.sampling.canonical.CanonicalSampler.measure_overlay`
            (``overlay.ecsv``).
        show_points : bool, optional
            Whether to mark the measured points.
        fields : sequence of str, optional
            The columns to draw, one panel each, in place of the four of the canonical figure.

        Returns
        -------
        matplotlib.figure.Figure
            The figure, which the caller closes.
        """
        if fields is not None:
            return super().draw_map(grid, points, path, title, overlay=overlay, show_points=show_points, fields=fields)

        from uvex_transients.phase_space.analysis.overlay import split_measurements
        from uvex_transients.phase_space.plotting import plot_map

        overlays = None if overlay is None else split_measurements(overlay)
        return plot_map(grid, points, path, title, overlays=overlays, show_points=show_points)

    @classmethod
    def annotate_map(cls, ax, overlay: Table):
        """
        Shade the region of the map that each shipped class occupies, each in its own colour.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes of a panel, in the plotted coordinates.
        overlay : astropy.table.Table
            The measurements of
            :meth:`~uvex_transients.phase_space.sampling.canonical.CanonicalSampler.measure_overlay`.

        Returns
        -------
        list of matplotlib.patches.Patch
            A legend handle for each class that has a measured draw.
        """
        from uvex_transients.phase_space.analysis.overlay import split_measurements
        from uvex_transients.phase_space.plotting.core import OVERLAY_COLORS, draw_overlay

        handles = []
        for k, measurement in enumerate(split_measurements(overlay)):
            handle = draw_overlay(np.array([ax]), measurement, OVERLAY_COLORS[k % len(OVERLAY_COLORS)])
            if handle is not None:
                handles.append(handle)
        return handles

    # -------------------------------------- #
    # Recording and rebuilding               #
    # -------------------------------------- #
    @property
    def table_meta(self) -> dict:
        r"""
        dict: What a figure of this sampler's maps needs to label the luminosity axis.

        ``band``, the unit of the luminosity, and ``absolute_magnitude_zero_point``, the :math:`Z` for which the
        absolute AB magnitude of a source of luminosity :math:`L_\mathrm{band}` is
        :math:`M = Z - 2.5\log_{10}L_\mathrm{band}`. Also the shape of the canonical transient the axes are measured
        on (``spectral_index``, ``rise_ratio``, ``fade_mag``), which
        :func:`~uvex_transients.phase_space.analysis.compare.difference_map` checks before it compares two maps.
        """
        bandwidth_hz = float(self.bandwidth.to_value(u.Hz))
        return {
            "band": self.band,
            "luminosity_unit": "erg/s",
            "spectral_index": self.spectral_index,
            "rise_ratio": self._settings["rise_ratio"],
            "fade_mag": self._settings["fade_mag"],
            "bandwidth_hz": bandwidth_hz,
            "absolute_magnitude_zero_point": float(_M_ZERO + 2.5 * np.log10(bandwidth_hz)),
        }

    @classmethod
    def from_settings(
        cls,
        settings: Mapping,
        *,
        schedule: SurveySchedule | None = None,
        mission=None,
        cosmology: Cosmology | None = None,
        **overrides,
    ) -> "CanonicalSampler":
        """
        Rebuild a sampler from the `settings` of another one.

        Parameters
        ----------
        settings : Mapping
            The `settings` of the sampler to rebuild.
        schedule : ~uvex_transients.surveys.base.SurveySchedule, optional
            The schedule. Needed unless `settings` names one in the registry.
        mission : str or m4opt.missions.Mission, optional
            The mission. Needed unless `settings` names one.
        cosmology : ~astropy.cosmology.Cosmology, optional
            By default the configured one, with a warning if its name is not the recorded one.
        **overrides
            Any other constructor argument, replacing the recorded value.

        Returns
        -------
        CanonicalSampler
            A sampler that makes the same measurement, and so reproduces the scales of the recorded one.

        Raises
        ------
        ValueError
            If no schedule or no mission can be found, or the settings are of a map made with the earlier top-hat
            light curve.
        """
        if "rise_ratio" not in settings:
            raise ValueError(
                "These settings are of a map made with a top-hat light curve, which can no longer be rebuilt; "
                "make the map again."
            )
        settings, schedule, mission, schedule_name = cls._rebuild_arguments(settings, schedule, mission, cosmology)
        settings.update(overrides)
        return cls(schedule, mission, cosmology=cosmology, schedule_name=schedule_name, **settings)

    def make_transient(self) -> _CanonicalTransient:
        """
        Return a transient of this sampler's spectrum, whose luminosities are taken at `reference_frequency`.

        Returns
        -------
        _CanonicalTransient
            A new instance, not yet placed at a point of the map.
        """
        return _CanonicalTransient(
            cosmology=self.cosmology,
            spectral_index=self.spectral_index,
            reference_frequency=self.reference_frequency,
            rise_ratio=self._settings["rise_ratio"],
            fade_mag=self._settings["fade_mag"],
        )

    # -------------------------------------- #
    # The point of the space                 #
    # -------------------------------------- #
    def redshift_limit(self, x: ArrayLike) -> NDArray[np.float64]:
        """
        Return the redshift out to which events are drawn at points of the space.

        It depends on the luminosity alone, so the half-decline time is not used.

        Parameters
        ----------
        x : array-like
            Points ``(log10 L_band, log10 t_half)``, shape ``(..., 2)``.

        Returns
        -------
        numpy.ndarray
            The redshift limit of each point, of shape ``(...)``.
        """
        return self._limit(self._points(x)[..., 0])

    def _place(self, x: NDArray[np.float64]) -> None:
        """
        Move the transient to ``(log10 L_band / (erg/s), log10 t_half / day)``.

        Parameters
        ----------
        x : numpy.ndarray
            The point, shape ``(2,)``.
        """
        log_luminosity, log_half_time = float(x[0]), float(x[1])
        z_limit = float(self.redshift_limit(x))

        # Every point brighter than this one is clipped too, so say so once
        if self._limit.is_clipped(log_luminosity) and not self._warned_clipped:
            self._warned_clipped = True
            logger.warning(
                "log10 L_band = %.2f is so bright that its redshift limit is beyond the one at which the band leaves "
                "the spectrum, so z_limit is clipped to %.3g. Every brighter point is clipped too, and its recovered "
                "volume-time is a lower bound.",
                log_luminosity,
                z_limit,
            )

        spectral_luminosity = 10.0**log_luminosity * u.erg / u.s / self.bandwidth
        self._transient.set_point(spectral_luminosity, 10.0**log_half_time * u.day, z_limit)

    def _log_window(self, z: ArrayLike, x: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Return the log10 of the window in days: the half-decline time, stretched to the fade and by time dilation.

        Parameters
        ----------
        z : array-like
            The redshift limit of each point.
        x : numpy.ndarray
            The points, shape ``(..., 2)``.

        Returns
        -------
        numpy.ndarray
            :math:`\log_{10}` of the window in days, of the shape of `z`.
        """
        return np.log10(1.0 + z) + np.log10(self._transient.window_factor) + x[..., 1]
