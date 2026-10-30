"""Photometry utilities: bandpasses as arrays, AB magnitudes, and integrating a spectrum over bands."""

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from astropy import units as u
from astropy.units import Quantity
from numpy.typing import ArrayLike, NDArray
from synphot import SpectralElement

__all__ = [
    "AB_MAG_ZERO_POINT",
    "D_LN_NU",
    "band_integration_weights",
    "band_luminosity_curve",
    "band_weight_matrix",
    "bandpass_table",
    "default_chunk_size",
    "effective_frequency",
    "equivalent_bandwidth",
    "flux_to_ab_mag",
    "hz_per_unit",
    "integrate_chunk",
    "photon_weighted_band_average",
    "relative_bandpass_table",
    "resolve_bandpasses",
    "restframe_frequency_grid",
]

AB_MAG_ZERO_POINT: float = 3631e-23
"""float: The zero-point flux density of the AB magnitude system, in erg/s/cm^2/Hz."""


# --------------------------------------------------------------------------- #
# Magnitudes and units                                                        #
# --------------------------------------------------------------------------- #
def flux_to_ab_mag(flux_density: ArrayLike) -> NDArray[np.float64]:
    r"""
    Convert a flux density to an AB magnitude.

    Parameters
    ----------
    flux_density : array-like
        Flux density in erg/s/cm^2/Hz. Zero is allowed.

    Returns
    -------
    numpy.ndarray
        :math:`-2.5\\log_{10}(F_\nu/F_0)` for the AB zero point :math:`F_0`. Zero flux is an infinitely faint
        magnitude, not an error.
    """
    with np.errstate(divide="ignore"):
        return -2.5 * np.log10(flux_density / AB_MAG_ZERO_POINT)


def hz_per_unit(unit: str | u.UnitBase, *, is_wavelength: bool) -> float:
    r"""
    Return the coefficient converting a bare number in ``unit`` to a frequency in Hz.

    Resolved once, up front -- not on every element of every model
    evaluation -- so that letting a model's spectral axis be expressed in
    an arbitrary (self-consistent) unit costs one extra multiply or divide
    per evaluation, not a :class:`~astropy.units.Quantity` conversion in
    the hot loop.

    Parameters
    ----------
    unit : str or ~astropy.units.UnitBase
        A wavelength unit if ``is_wavelength``, otherwise a frequency unit.
    is_wavelength : bool
        Whether ``unit`` is a wavelength unit (``True``) or a frequency
        unit (``False``).

    Returns
    -------
    float
        If ``is_wavelength`` is ``False``, :math:`\nu_\mathrm{Hz} =
        \mathrm{coefficient} \times x`; if ``True``, :math:`\nu_\mathrm{Hz}
        = \mathrm{coefficient} / x` (frequency is inversely, not linearly,
        related to wavelength). Either way, this is that coefficient.

    Raises
    ------
    ValueError
        If ``unit`` is not a wavelength/frequency unit matching
        ``is_wavelength``.

    Examples
    --------
    >>> hz_per_unit(u.THz, is_wavelength=False)
    1000000000000.0
    """
    resolved = u.Unit(unit)
    expected_type = "length" if is_wavelength else "frequency"
    if resolved.physical_type != expected_type:
        kind = "wavelength" if is_wavelength else "frequency"
        raise ValueError(f"{unit!r} is not a {kind} unit.")

    return resolved.to(u.Hz, equivalencies=u.spectral())


# --------------------------------------------------------------------------- #
# Bandpasses as arrays                                                        #
# --------------------------------------------------------------------------- #
def bandpass_table(bandpass: SpectralElement) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Return a bandpass's frequencies and its throughput there, in the order of its wavelength table.

    Parameters
    ----------
    bandpass : ~synphot.SpectralElement
        The band.

    Returns
    -------
    nu : numpy.ndarray
        Frequency samples in Hz.
    throughput : numpy.ndarray
        The dimensionless throughput at each.
    """
    wave = bandpass.waveset
    return wave.to_value(u.Hz, equivalencies=u.spectral()), bandpass(wave).to_value(u.dimensionless_unscaled)


def _sorted_by_frequency(nu: NDArray, throughput: NDArray) -> tuple[NDArray, NDArray]:
    """Order a band's samples by increasing frequency."""
    order = np.argsort(nu)
    return nu[order], throughput[order]


def _throughput_on_grid(nu_grid: NDArray, nu_band: NDArray, throughput: NDArray, z: float = 0.0) -> NDArray:
    """
    Sample a band's throughput, as seen from redshift `z`, on a rest-frame frequency grid.

    Photons that arrive in the band left the source at ``nu * (1 + z)``, so the rest-frame frequency ``nu_grid`` is
    sampled at ``nu_grid / (1 + z)`` of the band. Outside the tabulated band the throughput is zero.
    """
    return np.interp(nu_grid / (1.0 + z), nu_band, throughput, left=0.0, right=0.0)


def relative_bandpass_table(bandpass: SpectralElement) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Return a bandpass's frequencies, increasing, and its throughput there relative to the peak.

    Parameters
    ----------
    bandpass : ~synphot.SpectralElement
        The band.

    Returns
    -------
    nu : numpy.ndarray
        Frequency samples in Hz, in increasing order.
    throughput : numpy.ndarray
        The throughput at each, divided by its maximum.
    """
    nu, throughput = bandpass_table(bandpass)
    nu_sorted, throughput_sorted = _sorted_by_frequency(nu, throughput)
    return nu_sorted, throughput_sorted / throughput.max()


def resolve_bandpasses(
    bandpasses: Mapping[str, SpectralElement] | Sequence[SpectralElement],
) -> tuple[tuple[str, ...], list[tuple[np.ndarray, np.ndarray]]]:
    """
    Reduce bandpasses to the ``(frequency, throughput)`` arrays that band integration needs.

    Parameters
    ----------
    bandpasses : Mapping[str, ~synphot.SpectralElement] or sequence of ~synphot.SpectralElement
        The bands. A sequence is named ``band0``, ``band1``, and so on.

    Returns
    -------
    names : tuple of str
        Band names, in the order of `grids`.
    grids : list of tuple of numpy.ndarray
        For each band, its frequency samples in Hz and the dimensionless throughput at each.

    Raises
    ------
    ValueError
        If `bandpasses` is empty.
    """
    if isinstance(bandpasses, Mapping):
        names = tuple(bandpasses)
        elements = list(bandpasses.values())
    else:
        elements = list(bandpasses)
        names = tuple(f"band{i}" for i in range(len(elements)))
    if not elements:
        raise ValueError("'bandpasses' must contain at least one bandpass.")
    return names, [bandpass_table(element) for element in elements]


def effective_frequency(bandpass: SpectralElement) -> Quantity:
    r"""
    Return the throughput-weighted mean frequency of a bandpass: :math:`\int\nu\,T\,d\nu/\int T\,d\nu`.

    Parameters
    ----------
    bandpass : synphot.SpectralElement
        One of a detector's bandpasses.

    Returns
    -------
    ~astropy.units.Quantity
        The frequency, in Hz.
    """
    nu, throughput = relative_bandpass_table(bandpass)
    return float(np.trapezoid(nu * throughput, nu) / np.trapezoid(throughput, nu)) * u.Hz


def equivalent_bandwidth(
    bandpass: SpectralElement, spectral_index: float = 0.0, reference_frequency: Quantity | None = None
) -> Quantity:
    r"""
    Return the equivalent width, in frequency, of a bandpass.

    The width is :math:`\int(\nu/\nu_0)^\alpha\,T(\nu)/T_\mathrm{max}\,d\nu`.

    For a spectrum :math:`L_\nu=L_{\nu_0}(\nu/\nu_0)^\alpha` the luminosity in the band is :math:`L_{\nu_0}\,\Delta\nu`
    erg/s, with :math:`\Delta\nu` this width. Without an index it is the width of a top hat of the bandpass's peak
    throughput that passes as much, which is the same at every :math:`\nu_0`. It is independent of how efficient the
    detector is.

    Parameters
    ----------
    bandpass : synphot.SpectralElement
        One of a detector's bandpasses.
    spectral_index : float, optional
        The index :math:`\alpha`. The default of 0 is a flat spectrum.
    reference_frequency : ~astropy.units.Quantity, optional
        The frequency :math:`\nu_0` at which the spectral luminosity is quoted. The default is the `effective_frequency`
        of the bandpass. It does not matter for a flat spectrum.

    Returns
    -------
    ~astropy.units.Quantity
        The width, in Hz.
    """
    nu, throughput = relative_bandpass_table(bandpass)
    nu_0 = effective_frequency(bandpass) if reference_frequency is None else reference_frequency
    return float(np.trapezoid((nu / nu_0.to_value(u.Hz)) ** spectral_index * throughput, nu)) * u.Hz


# --------------------------------------------------------------------------- #
# Integrating over a band                                                     #
# --------------------------------------------------------------------------- #
D_LN_NU = 0.002
"""float: Default spacing of the rest-frame frequency grid in :math:`\\ln\\nu`.

0.2% is about 50 samples across a UVEX band, and changes the peak magnitudes by less than the bandpass tables'
own downsampling error (about 1e-3 mag)."""

# Target size of the largest intermediate array. Larger chunks cost cache, smaller ones Python overhead.
_CHUNK_BYTES = 32_000_000


def photon_weighted_band_average(flux_density: NDArray, nu: NDArray, throughput: NDArray) -> NDArray:
    r"""
    Average a flux density over a band, weighting by photon counts: :math:`\int F_\nu T\,d\nu/\nu\,/\int T\,d\nu/\nu`.

    This is the same average `band_weight_matrix` evaluates on a rest-frame grid for many redshifts at once. Here it
    is a trapezoid rule on the band's own samples, for one set of observed flux densities.

    Parameters
    ----------
    flux_density : numpy.ndarray
        The flux density at each band sample, shape ``(..., F)``.
    nu : numpy.ndarray
        The band's frequencies, shape ``(F,)``, increasing.
    throughput : numpy.ndarray
        The throughput at each frequency, shape ``(F,)``.

    Returns
    -------
    numpy.ndarray
        The band-averaged flux density, shape ``(...)``.
    """
    photon_throughput = throughput / nu
    numerator = np.trapezoid(flux_density * photon_throughput, nu, axis=-1)
    denominator = np.trapezoid(photon_throughput, nu)
    return numerator / denominator


def restframe_frequency_grid(
    band_grids: Sequence[tuple[NDArray, NDArray]], z_grid: NDArray, d_ln_nu: float = D_LN_NU
) -> tuple[NDArray, float]:
    r"""
    Build the uniform-in-:math:`\ln\nu` rest-frame grid that covers every band at every redshift.

    Parameters
    ----------
    band_grids : sequence of tuple of numpy.ndarray
        Each band's observed frequency samples (Hz) and throughput, as from
        :func:`resolve_bandpasses`.
    z_grid : numpy.ndarray
        Redshifts, increasing.
    d_ln_nu : float, optional
        Requested spacing in :math:`\ln\nu`. The actual spacing is slightly smaller, so the grid ends exactly on its
        limits.

    Returns
    -------
    nu_rest : numpy.ndarray
        Rest-frame frequencies in Hz.
    step : float
        The actual spacing in :math:`\ln\nu`.
    """
    lo, hi = np.inf, 0.0
    for nu, throughput in band_grids:
        support = nu[throughput > 0]
        if support.size == 0:
            raise ValueError("A bandpass has no positive throughput.")
        lo, hi = min(lo, support.min()), max(hi, support.max())
    # One extra sample either side, so that the throughput's edges are resolved.
    u_lo = np.log(lo * (1.0 + z_grid[0])) - d_ln_nu
    u_hi = np.log(hi * (1.0 + z_grid[-1])) + d_ln_nu
    n = int(np.ceil((u_hi - u_lo) / d_ln_nu)) + 1
    u_grid = np.linspace(u_lo, u_hi, n)
    return np.exp(u_grid), (u_hi - u_lo) / (n - 1)


def band_weight_matrix(nu_rest: NDArray, band_grids: Sequence[tuple[NDArray, NDArray]], z_grid: NDArray) -> NDArray:
    r"""
    Build the weights that turn rest-frame luminosity densities into band-averaged ones, for every band and redshift.

    Parameters
    ----------
    nu_rest : numpy.ndarray
        Uniform-in-:math:`\ln\nu` rest-frame grid, from :func:`restframe_frequency_grid`.
    band_grids : sequence of tuple of numpy.ndarray
        Each band's observed frequency samples (Hz) and throughput.
    z_grid : numpy.ndarray
        Redshifts.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(nu_rest), len(band_grids), len(z_grid))``. Each ``[:, b, j]`` column sums to one.
    """
    weights = np.empty((nu_rest.size, len(band_grids), z_grid.size))
    for b, (nu, throughput) in enumerate(band_grids):
        nu_sorted, throughput_sorted = _sorted_by_frequency(nu, throughput)
        for j, z in enumerate(z_grid):
            column = _throughput_on_grid(nu_rest, nu_sorted, throughput_sorted, z)
            weights[:, b, j] = column / column.sum()
    return weights


def band_integration_weights(
    bandpass: SpectralElement, d_ln_nu: float = D_LN_NU
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""
    Build the frequency grid and weights that integrate a luminosity density over a band.

    The grid is uniform in :math:`\ln\nu` and covers the band, and the weights are
    :math:`\tilde T(\nu_k)\,\nu_k\,\Delta\ln\nu`, so that :math:`\sum_k w_k L_\nu(\nu_k)` is
    :math:`\int L_\nu\,\tilde T\,d\nu`. The weights of a flat spectrum add up to the `equivalent_bandwidth`.

    Parameters
    ----------
    bandpass : synphot.SpectralElement
        One of a detector's bandpasses.
    d_ln_nu : float, optional
        Spacing of the grid in :math:`\ln\nu`. The default is the one the detection-horizon calculation uses.

    Returns
    -------
    nu : numpy.ndarray
        Frequencies in Hz, increasing.
    weights : numpy.ndarray
        The weight of each frequency, in Hz.
    """
    nu_table, throughput = relative_bandpass_table(bandpass)
    nu, step = restframe_frequency_grid([(nu_table, throughput)], np.zeros(1), d_ln_nu)
    return nu, _throughput_on_grid(nu, nu_table, throughput) * nu * step


def default_chunk_size(n_draws: int, n_time: int, n_frequency: int) -> int:
    """
    Choose how many draws to evaluate per step, so that the largest intermediate array is about 32 MB.

    Parameters
    ----------
    n_draws : int
        Number of draws.
    n_time, n_frequency : int
        Sizes of the time and frequency grids.

    Returns
    -------
    int
        Draws per chunk, between 1 and `n_draws`.
    """
    return int(np.clip(_CHUNK_BYTES // (8 * n_time * n_frequency), 1, n_draws))


def integrate_chunk(
    eval_log_luminosity: Callable[..., NDArray],
    parameters: Mapping[str, NDArray],
    start: int,
    stop: int,
    nu_axis: NDArray,
    t_axis: NDArray,
    weights: NDArray,
    support: NDArray,
) -> NDArray:
    r"""
    Evaluate the SED for some draws and apply frequency weights.

    Parameters
    ----------
    eval_log_luminosity : callable
        ``eval_log_luminosity(nu, t, **parameters)``: the natural log of the rest-frame :math:`L_\nu` in cgs; for a
        `~uvex_transients.models.core.base.SpectralModel` its ``_eval``.
    parameters : Mapping[str, numpy.ndarray]
        The model parameters in cgs units, each of shape ``(N,)``.
    start, stop : int
        The draws to evaluate, ``parameters[...][start:stop]``.
    nu_axis, t_axis : numpy.ndarray
        Rest-frame frequencies (Hz) and times (s), shaped ``(1, 1, F)`` and ``(1, T, 1)`` to broadcast against the
        draws.
    weights : numpy.ndarray
        Shape ``(F, K)``: one column of weights over frequency per output.
    support : numpy.ndarray
        Shape ``(F, K)``, ``weights > 0``: which frequencies each output integrates over.

    Returns
    -------
    numpy.ndarray
        Shape ``(stop - start, T, K)``. ``NaN`` where the model is ``NaN`` at a frequency that output integrates over.
    """
    n_rows, n_time, n_frequency = stop - start, t_axis.shape[1], nu_axis.shape[2]
    chunk = {name: value[start:stop, np.newaxis, np.newaxis] for name, value in parameters.items()}

    log_luminosity = np.broadcast_to(eval_log_luminosity(nu_axis, t_axis, **chunk), (n_rows, n_time, n_frequency))
    luminosity = np.exp(log_luminosity).reshape(n_rows * n_time, n_frequency)
    missing = np.isnan(luminosity)
    has_missing = missing.any()
    if has_missing:
        luminosity[missing] = 0.0
    flux = (luminosity @ weights).reshape(n_rows, n_time, weights.shape[1])
    if has_missing:
        # An epoch is invalid in an output if any frequency that output integrates over is missing.
        invalid = (missing.astype(np.float64) @ support.astype(np.float64)) > 0
        flux[invalid.reshape(flux.shape)] = np.nan
    return flux


def band_luminosity_curve(
    eval_log_luminosity: Callable[..., NDArray],
    parameters: Mapping[str, NDArray],
    t: NDArray,
    bandpass: SpectralElement,
    *,
    d_ln_nu: float = D_LN_NU,
) -> NDArray[np.float64]:
    r"""
    Evaluate the rest-frame luminosity in a band for each of many draws, on a grid of times.

    Parameters
    ----------
    eval_log_luminosity : callable
        ``eval_log_luminosity(nu, t, **parameters)``: the natural log of the rest-frame :math:`L_\nu` in cgs, for
        frequencies ``nu`` (Hz) and times ``t`` (s) that broadcast against the parameters.
    parameters : Mapping[str, numpy.ndarray]
        The parameters in cgs units, each of shape ``(N,)``, one entry per draw.
    t : numpy.ndarray
        Rest-frame times in seconds, shape ``(T,)``.
    bandpass : synphot.SpectralElement
        The band that is integrated over; see `band_integration_weights`.
    d_ln_nu : float, optional
        Spacing of the frequency grid in :math:`\ln\nu`.

    Returns
    -------
    numpy.ndarray
        :math:`L_\mathrm{band}` in erg/s, shape ``(N, T)``. ``NaN`` where the model is ``NaN`` at a frequency the
        band covers.
    """
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    parameters = {name: np.atleast_1d(value) for name, value in parameters.items()}
    n_draws = len(next(iter(parameters.values())))
    nu, weights = band_integration_weights(bandpass, d_ln_nu)
    weights = weights[:, np.newaxis]
    support = weights > 0

    chunk = default_chunk_size(n_draws, t.size, nu.size)
    nu_axis, t_axis = nu[np.newaxis, np.newaxis, :], t[np.newaxis, :, np.newaxis]
    out = np.empty((n_draws, t.size))
    for start in range(0, n_draws, chunk):
        stop = min(start + chunk, n_draws)
        out[start:stop] = integrate_chunk(
            eval_log_luminosity, parameters, start, stop, nu_axis, t_axis, weights, support
        )[..., 0]
    return out
