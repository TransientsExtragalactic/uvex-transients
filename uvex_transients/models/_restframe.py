r"""
Band peak magnitudes of many redshifts from one rest-frame evaluation.

The detection-horizon calculation needs, for every parameter draw, the brightest point of its light curve in each
band at every redshift of a grid. Evaluating the observed flux afresh at each redshift repeats the expensive part,
the SED, on a frequency grid that only differs from the last one by a shift. This module evaluates the rest-frame
luminosity density :math:`L_\nu(\nu', t)` once per draw, on a grid in rest-frame time :math:`t` and logarithmic
rest-frame frequency :math:`u = \ln\nu'`, and turns it into band fluxes at every redshift with one matrix product.

The band-averaged flux density used throughout the package is

.. math::

    \bar F_\nu = \frac{\int F_\nu(\nu)\,T(\nu)\,d\nu/\nu}{\int T(\nu)\,d\nu/\nu},
    \qquad F_\nu(\nu) = \frac{(1+z)\,L_\nu\big(\nu(1+z),\,t\big)}{4\pi D_L^2},

and :math:`d\nu/\nu` is unchanged by the substitution :math:`\nu' = \nu(1+z)`. So on the uniform :math:`u` grid

.. math::

    \bar F_\nu(z) = \frac{1+z}{4\pi D_L^2} \sum_k L_\nu(\nu'_k, t)\,\tilde W_k(z),
    \qquad
    \tilde W_k(z) = \frac{T\big(\nu'_k/(1+z)\big)}{\sum_j T\big(\nu'_j/(1+z)\big)},

which is a product of the ``(draws * times, frequencies)`` luminosity array with a ``(frequencies, bands * redshifts)``
weight matrix that does not depend on the draw. Because the weights sum to one, a spectrum that is flat in
:math:`L_\nu` is recovered exactly at any grid spacing.
"""

import warnings
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from uvex_transients.models._constants import AB_MAG_ZERO_POINT
from uvex_transients.utils import logger

__all__ = ["band_weight_matrix", "peak_band_magnitudes", "restframe_frequency_grid"]

D_LN_NU = 0.002
"""float: Default spacing of the rest-frame frequency grid in :math:`\\ln\\nu`.

0.2% is about 50 samples across a UVEX band, and changes the peak magnitudes by less than the bandpass tables'
own downsampling error (about 1e-3 mag)."""

# Target size of the largest intermediate array. Larger chunks cost cache, smaller ones Python overhead.
_CHUNK_BYTES = 32_000_000


def restframe_frequency_grid(
    band_grids: Sequence[tuple[NDArray, NDArray]], z_grid: NDArray, d_ln_nu: float = D_LN_NU
) -> tuple[NDArray, float]:
    r"""
    Build the uniform-in-:math:`\ln\nu` rest-frame grid that covers every band at every redshift.

    Parameters
    ----------
    band_grids : sequence of tuple of numpy.ndarray
        Each band's observed frequency samples (Hz) and throughput, as from
        :func:`~uvex_transients.models.core._tolerance.resolve_bandpasses`.
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
        order = np.argsort(nu)
        nu_sorted, throughput_sorted = nu[order], throughput[order]
        for j, z in enumerate(z_grid):
            # Photons arriving in the band left the source at the shifted frequency. Outside the tabulated band the
            # throughput is zero, as in the observed-frame integral.
            column = np.interp(nu_rest / (1.0 + z), nu_sorted, throughput_sorted, left=0.0, right=0.0)
            weights[:, b, j] = column / column.sum()
    return weights


def peak_band_magnitudes(
    eval_log_luminosity: Callable[..., NDArray],
    parameters: Mapping[str, NDArray],
    z_grid: NDArray,
    luminosity_distance: NDArray,
    band_grids: Sequence[tuple[NDArray, NDArray]],
    t_rest: NDArray,
    *,
    chunk_size: int | None = None,
    d_ln_nu: float = D_LN_NU,
    progress: bool = True,
) -> NDArray:
    r"""
    Brightest AB magnitude over time of each draw, in each band, at each redshift.

    Parameters
    ----------
    eval_log_luminosity : callable
        ``eval_log_luminosity(nu, t, **parameters)``: the natural log of the rest-frame :math:`L_\nu` in cgs, for
        frequencies ``nu`` (Hz) and times ``t`` (s) that broadcast against the parameters. For a
        `~uvex_transients.models.core.base.SpectralModel` this is its ``_eval``.
    parameters : Mapping[str, numpy.ndarray]
        The model parameters in cgs units, each of shape ``(N,)``, one entry per draw.
    z_grid : numpy.ndarray
        Redshifts, shape ``(Z,)``, increasing and positive.
    luminosity_distance : numpy.ndarray
        Luminosity distance at each redshift in cm, shape ``(Z,)``.
    band_grids : sequence of tuple of numpy.ndarray
        Each band's observed frequency samples (Hz) and throughput.
    t_rest : numpy.ndarray
        Rest-frame times since explosion in seconds, shape ``(T,)``.
    chunk_size : int, optional
        Draws evaluated per step. By default it is chosen so that the largest intermediate array is about 32 MB.
    d_ln_nu : float, optional
        Spacing of the rest-frame frequency grid in :math:`\ln\nu`; see :func:`restframe_frequency_grid`.
    progress : bool, optional
        Whether to show a progress bar.

    Returns
    -------
    numpy.ndarray
        Shape ``(N, B, Z)``. ``NaN`` where the model has no valid epoch for the draw, redshift and band, which is
        any epoch where the model returns ``NaN`` at a frequency the band receives light from.
    """
    n_draws = len(next(iter(parameters.values())))
    nu_rest, _ = restframe_frequency_grid(band_grids, z_grid, d_ln_nu)
    n_bands, n_z = len(band_grids), z_grid.size
    weights = band_weight_matrix(nu_rest, band_grids, z_grid).reshape(nu_rest.size, n_bands * n_z)
    support = weights > 0  # which frequencies each (band, redshift) integrates over
    # (1 + z) / (4 pi D_L^2), and the AB zero point, are applied after the sum over frequency.
    prefactor = (1.0 + z_grid) / (4.0 * np.pi * np.asarray(luminosity_distance) ** 2)

    if chunk_size is None:
        chunk_size = int(np.clip(_CHUNK_BYTES // (8 * t_rest.size * nu_rest.size), 1, n_draws))
    n_chunks = -(-n_draws // chunk_size)

    nu_axis, t_axis = nu_rest[np.newaxis, np.newaxis, :], t_rest[np.newaxis, :, np.newaxis]
    peak_flux = np.empty((n_draws, n_bands, n_z))
    with (
        tqdm(total=n_chunks, desc="Evaluating peak magnitudes", unit="chunk", disable=not progress) as pbar,
        logging_redirect_tqdm(loggers=[logger]),
    ):
        for start in range(0, n_draws, chunk_size):
            stop = min(start + chunk_size, n_draws)
            n_rows = stop - start
            chunk = {name: value[start:stop, np.newaxis, np.newaxis] for name, value in parameters.items()}

            log_luminosity = np.broadcast_to(
                eval_log_luminosity(nu_axis, t_axis, **chunk), (n_rows, t_rest.size, nu_rest.size)
            )
            luminosity = np.exp(log_luminosity).reshape(n_rows * t_rest.size, nu_rest.size)
            missing = np.isnan(luminosity)
            has_missing = missing.any()
            if has_missing:
                luminosity[missing] = 0.0
            flux = (luminosity @ weights).reshape(n_rows, t_rest.size, n_bands * n_z)
            if has_missing:
                # An epoch is invalid in a band at a redshift if any frequency the band integrates over is missing.
                invalid = (missing.astype(np.float64) @ support.astype(np.float64)) > 0
                flux[invalid.reshape(flux.shape)] = np.nan

            # The brightest valid epoch; a draw with no valid epoch at all stays NaN.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # "All-NaN slice"
                peak_flux[start:stop] = np.nanmax(flux, axis=1).reshape(n_rows, n_bands, n_z)
            pbar.update(1)

    # Zero flux is an infinitely faint magnitude, not an error.
    with np.errstate(divide="ignore"):
        return -2.5 * np.log10(peak_flux * prefactor / AB_MAG_ZERO_POINT)
