"""
Missions, including bandpass-downsampled "fast" variants of m4opt missions.

m4opt ships each mission's bandpasses as dense throughput tables (m4opt 3.1.0 gave the UVEX
bands 11000 wavelength samples each, up from 100). Every band-integrated quantity in this
package (`~uvex_transients.models.core.base.SpectralModel.flux_band`, the limiting-magnitude
and SNR screening cuts, and m4opt's own count rates) costs time and memory linear in that
sample count, so a fine table is expensive well beyond what a smooth SED can resolve.

:func:`downsample_bandpass` replaces a table by a much shorter one whose band integrals agree
with the original to a requested accuracy, for thermal-like spectra over a stated temperature
range. :func:`downsample_mission` applies it to every bandpass of an m4opt
`~m4opt.missions.Mission`, and :func:`get_mission` resolves mission names, including the
registered ``"uvex_fast"``, honoring the ``missions.downsample`` section of the configuration.

Notes
-----
Take ``n_spec`` blackbodies spaced log-uniformly in temperature from ``t_min`` to ``t_max``.
For each, form the integrands that a band average integrates, throughput times spectrum, both
as a flux-weighted and as a photon-weighted (extra ``1/nu``) integral, plus the throughput
itself (the denominator of the band average). Each curve is normalized to unit peak. Starting
from only the two end points, the point whose deviation from the straight chord between its
neighboring kept points is largest (taken over all curves) is added as a knot, recursively,
until every deviation is below a threshold ``eps``. The throughput values at the kept
frequencies are the downsampled table, and the table is interpolated linearly between them,
matching the trapezoidal quadrature used throughout the package.

``eps`` is not the accuracy. It is a tuning parameter that is bisected so that the *measured*
worst-case band magnitude error over a denser set of temperatures (which includes temperatures
between the optimization ones) stays within ``tol_mag``. The measured quantities are the
flux-weighted band average, the photon-weighted band average, and the absolute flux and photon
rates (the latter two are what m4opt's count rates see, and are what the mass rescale below
corrects).

The kept table is finally rescaled by a single constant so its integrated transmission,
``int T dnu``, equals that of the original exactly. Band averages are ratios, so the rescale
cancels in them, but absolute count rates carry the mass error directly.

Accuracy is only established for smooth, thermal-like spectra between ``t_min`` and ``t_max``.
A spectrum with sharp features, or one cooler than ``t_min``, can exceed ``tol_mag``.
"""

import dataclasses
import functools

import m4opt.missions
import numpy as np
from astropy import constants as const
from astropy import units as u
from m4opt.missions import Mission
from numpy.typing import NDArray
from synphot import SpectralElement
from synphot.models import Empirical1D

from .utils.config import config
from .utils.log import logger

__all__ = [
    "FAST_MISSIONS",
    "downsample_bandpass",
    "downsample_mission",
    "get_mission",
    "list_missions",
]

#: Registered downsampled missions: name -> the `m4opt.missions` attribute it is built from.
FAST_MISSIONS: dict[str, str] = {"uvex_fast": "uvex"}

# Fallbacks for configuration files written before the ``missions.downsample`` section existed.
_DEFAULT_DOWNSAMPLE_SETTINGS = {
    "enabled": False,
    "tol_mag": 1.0e-3,
    "t_min": 1500.0,
    "t_max": 50000.0,
    "n_spec": 12,
}

# Bisection search range for the knot threshold ``eps`` (log10), and the number of halvings.
_LOG_EPS_RANGE = (-8.0, 0.0)
_N_BISECTIONS = 24

_H = const.h.cgs.value
_C = const.c.cgs.value
_K_B = const.k_B.cgs.value


def _downsample_settings() -> dict:
    """
    Read the ``missions.downsample`` configuration, falling back to defaults for missing keys.

    Returns
    -------
    dict
        The keys ``enabled``, ``tol_mag``, ``t_min``, ``t_max`` and ``n_spec``.
    """
    settings = dict(_DEFAULT_DOWNSAMPLE_SETTINGS)
    for key in settings:
        try:
            settings[key] = config[f"missions.downsample.{key}"]
        except KeyError:
            pass
    return settings


def _planck_nu(nu: NDArray, temperature: NDArray) -> NDArray:
    """
    Blackbody specific intensity ``B_nu`` in cgs.

    Parameters
    ----------
    nu : numpy.ndarray
        Frequencies in Hz, shape ``(N,)``.
    temperature : numpy.ndarray
        Temperatures in K, shape ``(M,)``.

    Returns
    -------
    numpy.ndarray
        ``B_nu``, shape ``(M, N)``.
    """
    x = _H * nu[None, :] / (_K_B * temperature[:, None])
    return 2.0 * _H * nu[None, :] ** 3 / _C**2 / np.expm1(x)


def _band_integrals(nu: NDArray, throughput: NDArray, temperatures: NDArray) -> NDArray:
    """
    Band integrals of blackbodies through a throughput table (trapezoid rule in frequency).

    Parameters
    ----------
    nu : numpy.ndarray
        Strictly increasing frequencies in Hz, shape ``(N,)``.
    throughput : numpy.ndarray
        Throughput at each frequency, shape ``(N,)``.
    temperatures : numpy.ndarray
        Blackbody temperatures in K, shape ``(M,)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(4, M)``: the flux-weighted band average ``int B T dnu / int T dnu``, the
        photon-weighted band average ``int B T/nu dnu / int T/nu dnu``, and the absolute
        integrals ``int B T dnu`` and ``int B T/nu dnu`` (no denominator).
    """
    spectra = _planck_nu(nu, temperatures)
    inverse_nu = 1.0 / nu
    flux = np.trapezoid(spectra * throughput, nu, axis=-1)
    photons = np.trapezoid(spectra * throughput * inverse_nu, nu, axis=-1)
    flux_ratio = flux / np.trapezoid(throughput, nu)
    photon_ratio = photons / np.trapezoid(throughput * inverse_nu, nu)
    return np.stack([flux_ratio, photon_ratio, flux, photons])


def _mag_error(approximate: NDArray, reference: NDArray) -> float:
    """
    Worst-case magnitude difference between two sets of band integrals.

    Parameters
    ----------
    approximate, reference : numpy.ndarray
        Band integrals as returned by `_band_integrals`.

    Returns
    -------
    float
        ``max |2.5 log10(approximate / reference)|``, or ``inf`` if the ratio is not a finite
        positive number anywhere.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = approximate / reference
    if not np.all(np.isfinite(ratio) & (ratio > 0)):
        return np.inf
    return float(np.max(np.abs(2.5 * np.log10(ratio))))


def _integrand_curves(nu: NDArray, throughput: NDArray, temperatures: NDArray) -> NDArray:
    """
    Build the curves whose piecewise-linear approximation drives the knot selection.

    Parameters
    ----------
    nu : numpy.ndarray
        Strictly increasing frequencies in Hz, shape ``(N,)``.
    throughput : numpy.ndarray
        Throughput at each frequency, shape ``(N,)``.
    temperatures : numpy.ndarray
        The optimization temperatures in K, shape ``(M,)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(2 * (M + 1), N)``: for the flux and photon weightings, the throughput itself
        and its product with each blackbody, each normalized to unit peak.
    """
    spectra = _planck_nu(nu, temperatures)
    flux_curves = np.concatenate([throughput[None, :], throughput * spectra])
    photon_curves = flux_curves / nu
    curves = np.concatenate([flux_curves, photon_curves])
    peak = curves.max(axis=-1, keepdims=True)
    return curves / np.where(peak > 0, peak, 1.0)


def _select_knots(nu: NDArray, curves: NDArray, eps: float) -> NDArray:
    """
    Greedy knot insertion: refine until every curve is within `eps` of its piecewise-linear fit.

    Parameters
    ----------
    nu : numpy.ndarray
        Strictly increasing frequencies, shape ``(N,)``.
    curves : numpy.ndarray
        Curves sampled at `nu`, shape ``(K, N)``.
    eps : float
        Largest allowed deviation of any curve from the chord between kept points.

    Returns
    -------
    numpy.ndarray
        Sorted indices into `nu` of the kept points. Always includes both end points.
    """
    keep = np.zeros(nu.size, dtype=bool)
    keep[[0, -1]] = True
    stack = [(0, nu.size - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2:
            continue
        fraction = (nu[lo + 1 : hi] - nu[lo]) / (nu[hi] - nu[lo])
        chord = curves[:, [lo]] + (curves[:, [hi]] - curves[:, [lo]]) * fraction
        deviation = np.abs(curves[:, lo + 1 : hi] - chord).max(axis=0)
        worst = int(np.argmax(deviation))
        if deviation[worst] > eps:
            knot = lo + 1 + worst
            keep[knot] = True
            stack.append((lo, knot))
            stack.append((knot, hi))
    return np.flatnonzero(keep)


def downsample_bandpass(
    bandpass: SpectralElement,
    *,
    t_min: float | None = None,
    t_max: float | None = None,
    n_spec: int | None = None,
    tol_mag: float | None = None,
    rescale: bool = True,
) -> SpectralElement:
    """
    Replace a dense bandpass table by a short one with matching thermal band integrals.

    See the module docstring for the algorithm. The returned bandpass is linearly interpolated
    between the kept points. Its ``meta["downsampling"]`` records the parameters, the point
    counts, the achieved worst-case magnitude error, and the mass rescale factor.

    Parameters
    ----------
    bandpass : ~synphot.SpectralElement
        The bandpass to downsample. Its ``waveset`` and values at the waveset define the table.
    t_min, t_max : float, optional
        Temperature range in K over which the downsampled bandpass must reproduce blackbody band
        integrals. Default to ``config["missions.downsample.t_min"]`` and ``t_max``.
    n_spec : int, optional
        Number of log-uniformly spaced blackbodies the knot selection is optimized on. The error
        is verified on a denser set that includes temperatures between these. Defaults to
        ``config["missions.downsample.n_spec"]``.
    tol_mag : float, optional
        Largest tolerated magnitude error in any band integral over the verification set.
        Defaults to ``config["missions.downsample.tol_mag"]``.
    rescale : bool, optional
        If True (default), scale the kept values so the integrated transmission ``int T dnu``
        equals the original's.

    Returns
    -------
    ~synphot.SpectralElement
        The downsampled bandpass, or `bandpass` itself if no reduction meets `tol_mag`.

    Raises
    ------
    ValueError
        If ``t_min`` and ``t_max`` are not positive with ``t_min < t_max``, ``n_spec < 2``, or
        ``tol_mag <= 0``.
    """
    settings = _downsample_settings()
    t_min = settings["t_min"] if t_min is None else t_min
    t_max = settings["t_max"] if t_max is None else t_max
    n_spec = settings["n_spec"] if n_spec is None else n_spec
    tol_mag = settings["tol_mag"] if tol_mag is None else tol_mag

    if not 0 < t_min < t_max:
        raise ValueError(f"Need 0 < t_min < t_max, got t_min={t_min}, t_max={t_max}.")
    if n_spec < 2:
        raise ValueError(f"'n_spec' must be at least 2, got {n_spec}.")
    if tol_mag <= 0:
        raise ValueError(f"'tol_mag' must be positive, got {tol_mag}.")

    wave = bandpass.waveset.to(u.AA)
    throughput = np.asarray(bandpass(wave).to_value(u.dimensionless_unscaled), dtype=float)
    nu_all = wave.to_value(u.Hz, equivalencies=u.spectral())

    # Work in increasing frequency, the order the trapezoid rules in the package integrate in.
    order = np.argsort(nu_all)
    wave, nu, throughput = wave[order], nu_all[order], throughput[order]
    n_points = nu.size
    if n_points < 3 or not throughput.max() > 0:
        return bandpass

    optimization_temperatures = np.geomspace(t_min, t_max, n_spec)
    # Every optimization temperature plus three between each adjacent pair.
    check_temperatures = np.geomspace(t_min, t_max, 4 * (n_spec - 1) + 1)

    curves = _integrand_curves(nu, throughput, optimization_temperatures)
    reference = _band_integrals(nu, throughput, check_temperatures)
    mass = np.trapezoid(throughput, nu)

    def evaluate(eps):
        """
        Knot selection at threshold `eps`, with its rescaled values, rescale factor and measured error.

        Parameters
        ----------
        eps : float
            Knot threshold for `_select_knots`.

        Returns
        -------
        tuple
            ``(keep, values, scale, error)``: kept indices, their (rescaled) throughput, the rescale
            factor applied, and the worst-case magnitude error over the verification temperatures.
        """
        keep = _select_knots(nu, curves, eps)
        values = throughput[keep]
        scale = 1.0
        if rescale:
            scale = mass / np.trapezoid(values, nu[keep])
            values = values * scale
        error = _mag_error(_band_integrals(nu[keep], values, check_temperatures), reference)
        return keep, values, scale, error

    best = None
    log_lo, log_hi = _LOG_EPS_RANGE
    for _ in range(_N_BISECTIONS):
        log_mid = 0.5 * (log_lo + log_hi)
        candidate = evaluate(10.0**log_mid)
        if candidate[3] <= tol_mag:
            log_lo, best = log_mid, candidate
        else:
            log_hi = log_mid

    if best is None or best[0].size >= n_points:
        logger.warning(
            "Could not downsample the bandpass table (%d points) to within %.2g mag; keeping it unchanged.",
            n_points,
            tol_mag,
        )
        return bandpass

    keep, values, scale, error = best
    downsampled = SpectralElement(Empirical1D, points=wave[keep][::-1], lookup_table=values[::-1])
    downsampled.meta["downsampling"] = {
        "n_original": int(n_points),
        "n_downsampled": int(keep.size),
        "tol_mag": float(tol_mag),
        "achieved_mag_error": float(error),
        "mass_rescale": float(scale),
        "t_min": float(t_min),
        "t_max": float(t_max),
        "n_spec": int(n_spec),
    }
    logger.info(
        "Downsampled a bandpass table from %d to %d points (worst-case %.2g mag over %.0f-%.0f K, mass rescale %.6f).",
        n_points,
        keep.size,
        error,
        t_min,
        t_max,
        scale,
    )
    return downsampled


def downsample_mission(mission: Mission, *, name: str | None = None, **kwargs) -> Mission:
    """
    Copy a mission with every detector bandpass downsampled.

    The original mission is not modified. Everything other than the bandpasses (field of view,
    background, noise terms, scheduling model) is shared with `mission`.

    Parameters
    ----------
    mission : ~m4opt.missions.Mission
        The mission to copy.
    name : str, optional
        Name of the copy. Defaults to ``f"{mission.name}_fast"``.
    **kwargs
        Passed to :func:`downsample_bandpass` (``t_min``, ``t_max``, ``n_spec``, ``tol_mag``,
        ``rescale``).

    Returns
    -------
    ~m4opt.missions.Mission
        The downsampled copy.

    Raises
    ------
    TypeError
        If a detector bandpass is not a `~synphot.SpectralElement`.
    """
    bandpasses = {}
    for band, bandpass in mission.detector.bandpasses.items():
        if not isinstance(bandpass, SpectralElement):
            raise TypeError(
                f"Bandpass {band!r} of mission {mission.name!r} is a {type(bandpass)}, not a SpectralElement."
            )
        bandpasses[band] = downsample_bandpass(bandpass, **kwargs)
    detector = dataclasses.replace(mission.detector, bandpasses=bandpasses)
    return dataclasses.replace(mission, name=f"{mission.name}_fast" if name is None else name, detector=detector)


@functools.cache
def _fast_mission(base: str, tol_mag: float, t_min: float, t_max: float, n_spec: int) -> Mission:
    """Build (once per distinct setting) the downsampled copy of the m4opt mission `base`."""
    return downsample_mission(
        getattr(m4opt.missions, base),
        name=f"{base}_fast",
        tol_mag=tol_mag,
        t_min=t_min,
        t_max=t_max,
        n_spec=n_spec,
    )


def list_missions() -> list[str]:
    """
    List the names accepted by :func:`get_mission`.

    Returns
    -------
    list of str
        Every `~m4opt.missions.Mission` in `m4opt.missions` plus the registered `FAST_MISSIONS`.
    """
    available = {attr for attr, value in vars(m4opt.missions).items() if isinstance(value, Mission)}
    return sorted(available | set(FAST_MISSIONS))


def get_mission(name: str, *, downsample: bool | None = None) -> Mission:
    """
    Resolve a mission name to a `~m4opt.missions.Mission`, optionally with downsampled bandpasses.

    A registered fast name (``"uvex_fast"``) is always downsampled. For any other name,
    `downsample` decides, and ``None`` defers to ``config["missions.downsample.enabled"]``. The
    downsampling parameters always come from the ``missions.downsample`` configuration section.
    Downsampled missions are built once per distinct set of parameters and cached.

    Parameters
    ----------
    name : str
        An attribute name of `m4opt.missions`, or a key of `FAST_MISSIONS`.
    downsample : bool, optional
        Force downsampling on or off for a non-fast name. Ignored for a fast name.

    Returns
    -------
    ~m4opt.missions.Mission
        The resolved mission.

    Raises
    ------
    ValueError
        If `name` is not a known mission.
    """
    base = FAST_MISSIONS.get(name, name)
    mission = getattr(m4opt.missions, base, None)
    if not isinstance(mission, Mission):
        raise ValueError(f"Unknown mission {name!r}; available: {list_missions()}.")

    settings = _downsample_settings()
    if name in FAST_MISSIONS or (settings["enabled"] if downsample is None else downsample):
        return _fast_mission(base, settings["tol_mag"], settings["t_min"], settings["t_max"], settings["n_spec"])
    return mission


def __getattr__(name: str) -> Mission:
    """Build registered fast missions on first access, so ``from uvex_transients.missions import uvex_fast`` works."""
    if name in FAST_MISSIONS:
        return get_mission(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List the module attributes, including the lazily built fast missions."""
    return sorted([*globals(), *FAST_MISSIONS])
