"""Shared helpers for tests of band photometry."""

import numpy as np
from astropy import units as u


def effective_flux_jy(spectrum, bandpass) -> np.ndarray:
    r"""
    Photon-weighted band-averaged flux density of `spectrum` through `bandpass`, in Jy.

    :math:`\int F_\nu T\,d\nu/\nu / \int T\,d\nu/\nu`, by trapezoidal quadrature over the bandpass's own
    wavelength samples. This is deliberately independent of both m4opt's ``countrate`` and the package's
    ``flux_band``, so that agreement with either is a real check.

    Parameters
    ----------
    spectrum : ~synphot.SourceSpectrum
        The spectrum, possibly with a leading batch axis ahead of its wavelength axis.
    bandpass : ~synphot.SpectralElement
        The bandpass.

    Returns
    -------
    numpy.ndarray
        The effective flux density in Jy, one entry per batch element of `spectrum`.
    """
    wave = bandpass.waveset
    nu = wave.to_value(u.Hz, equivalencies=u.spectral())
    flux = np.reshape(spectrum(wave, flux_unit=u.Jy).to_value(u.Jy), (-1, len(wave)))
    photon_throughput = np.asarray(bandpass(wave).to_value(u.dimensionless_unscaled)) / nu
    # `nu` is decreasing along an increasing wavelength grid, which flips the sign of both integrals alike.
    return np.trapezoid(flux * photon_throughput, nu, axis=-1) / np.trapezoid(photon_throughput, nu)
