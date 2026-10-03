"""
Numerical coercion and unit-conversion utilities for :mod:`uvex_transients.models`.

This module provides small helpers for normalizing user-facing numerical
inputs before they are passed to model calculations. Inputs may be ordinary
NumPy-compatible values or :class:`astropy.units.Quantity` objects; outputs
are plain NumPy arrays with units removed.

Unit-aware helpers explicitly convert quantities before stripping their units.
Unitless numerical inputs are assumed to already be expressed in the requested
units.
"""

from collections.abc import Callable, Mapping
from dataclasses import replace

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.modeling import Model
from astropy.table import QTable, vstack
from astropy.time import Time
from astropy.units import Quantity
from m4opt.synphot import Detector, observing
from numpy.typing import DTypeLike, NDArray
from synphot import SourceSpectrum
from synphot import units as synphot_units

from uvex_transients.models._typing import FloatArray, FloatResult, PhysicalInput, RNGInput, UnitLike
from uvex_transients.utils import config, get_rng
from uvex_transients.utils.keyed_noise import keyed_standard_normal

# ------------------------------------------ #
# Internal Unit Conventions                  #
# ------------------------------------------ #
# To avoid explicit / lengthy units in the code, we define a set of standard units for
# outputs from the uvex_transients.models library.
_SPEC_FLUX_UNIT: u.Unit = u.erg / u.s / u.Hz / u.cm**2
_BOL_FLUX_UNIT: u.Unit = u.erg / u.s / u.cm**2
_SPEC_LUM_UNIT: u.Unit = u.erg / u.s / u.Hz
_BOL_LUM_UNIT: u.Unit = u.erg / u.s

# The SED shape unit is the unit used to parameterize the shape of a raw spectrum.
_SED_SHAPE_UNIT: u.Unit = u.Hz**-1

# Defaults for `observer_location`/`obstime` when a photometry caller doesn't supply real ones.
# Correct as long as `background` doesn't actually depend on either (true of
# `m4opt.synphot.background.GalacticBackground`, false of
# `ZodiacalBackground`/`EarthshineBackground`).
_PLACEHOLDER_OBSERVER_LOCATION = EarthLocation(0 * u.m, 0 * u.m, 0 * u.m)
_PLACEHOLDER_OBSTIME = Time("2000-01-01T00:00:00", scale="utc")

# The substitute for an exactly-zero flux in a background-only measurement.
# `Detector.get_snr` correctly returns a finite `snr` of exactly 0 for a source
# with zero flux, but `measure_bands` then computes
# `flux_err = true_flux / snr` -- a 0/0 division at exact zero, even though the
# real (background-limited) noise level is perfectly well defined there. This
# value is many orders of magnitude below any physically meaningful flux, so
# using it in place of an exact zero only avoids that floating-point
# singularity; it never contributes a measurable source-shot-noise term of its
# own (verified numerically stable, with the recovered `flux_err` converged to
# the true background-limited noise, from 1e-3 Jy down to 1e-40 Jy).
_NEGLIGIBLE_FLUX_JY = 1e-30

# ------------------------------------------ #
# Unit Coercion Functions                    #
# ------------------------------------------ #
# These functions are each concerned with manipulating numpy / astropy arrays and quantities
# into one another and are used heavily in the uvex_transients.models module to coerce unit-bearing user
# input to raw numerical internal arrays.


def ensure_numpy_array(
    value: PhysicalInput,
    dtype: DTypeLike | None = None,
) -> NDArray:
    """
    Coerce a numerical input to a NumPy array.

    Astropy quantities are accepted only when they are dimensionless. Their
    units are stripped before conversion. All other inputs are passed directly
    to :func:`numpy.asarray`.

    The function follows normal NumPy coercion semantics: scalar inputs become
    zero-dimensional arrays, existing arrays are reused when possible, and the
    dtype is preserved unless an explicit ``dtype`` is requested.

    Parameters
    ----------
    value : array_like or ~astropy.units.Quantity
        Numerical input to convert. Quantity inputs must be convertible to
        :attr:`astropy.units.dimensionless_unscaled`.
    dtype : numpy.dtype-like, optional
        Desired dtype of the returned array. If omitted, NumPy infers or
        preserves the dtype.

    Returns
    -------
    numpy.ndarray
        NumPy representation of ``value``.

    Raises
    ------
    astropy.units.UnitConversionError
        If ``value`` is a dimensional quantity.

    Examples
    --------
    >>> ensure_numpy_array([1, 2, 3])
    array([1, 2, 3])

    >>> ensure_numpy_array([1, 2, 3], dtype=np.float64)
    array([1., 2., 3.])

    >>> ensure_numpy_array(2.0 * u.dimensionless_unscaled)
    array(2.)
    """
    if isinstance(value, u.Quantity):
        value = value.to_value(u.dimensionless_unscaled)

    return np.asarray(value, dtype=dtype)


def to_cgs_value(value: PhysicalInput) -> FloatArray:
    """
    Convert a numerical input to unit-stripped CGS values.

    Quantity inputs are converted to their corresponding CGS representation
    before their units are removed. Unitless inputs are assumed to already be
    expressed in CGS units.

    Parameters
    ----------
    value : array_like or ~astropy.units.Quantity
        Numerical input to convert.

    Returns
    -------
    numpy.ndarray
        Numerical values expressed in the appropriate CGS units, with dtype
        ``float64``.

    Examples
    --------
    >>> to_cgs_value(1.0 * u.km)
    array(100000.)

    >>> to_cgs_value([1.0, 2.0] * u.kg)
    array([1000., 2000.])

    >>> to_cgs_value([1.0, 2.0])
    array([1., 2.])
    """
    if isinstance(value, u.Quantity):
        value = value.cgs.value

    return np.asarray(value, dtype=np.float64)


def ensure_in_units(
    value: PhysicalInput,
    unit: UnitLike,
) -> FloatArray:
    """
    Convert a numerical input to unit-stripped values in a specified unit.

    Quantity inputs are converted to ``unit`` before their units are removed.
    Unitless inputs are assumed to already be expressed in ``unit`` and are
    therefore unchanged apart from conversion to a ``float64`` NumPy array.

    If ``unit`` is ``None``, the requested unit is interpreted as
    :attr:`astropy.units.dimensionless_unscaled`.

    Parameters
    ----------
    value : array_like or ~astropy.units.Quantity
        Numerical input to convert.
    unit : str, ~astropy.units.UnitBase, or None
        Unit in which the returned numerical values should be expressed.
        ``None`` denotes a dimensionless value.

    Returns
    -------
    numpy.ndarray
        Unit-stripped numerical values expressed in ``unit``, with dtype
        ``float64``.

    Raises
    ------
    ValueError
        If ``unit`` cannot be interpreted as a valid Astropy unit.
    astropy.units.UnitConversionError
        If ``value`` is a quantity incompatible with ``unit``.

    Examples
    --------
    >>> ensure_in_units(1500.0 * u.m, u.km)
    array(1.5)

    >>> ensure_in_units([1.0, 2.0] * u.MHz, "Hz")
    array([1000000., 2000000.])

    >>> ensure_in_units([1.0, 2.0], u.s)
    array([1., 2.])
    """
    unit = u.dimensionless_unscaled if unit is None else u.Unit(unit)

    if isinstance(value, u.Quantity):
        value = value.to_value(unit)

    return np.asarray(value, dtype=np.float64)


def hz_per_unit(unit: UnitLike, *, is_wavelength: bool) -> float:
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


def convert_CI_to_fractional(
    value: FloatArray, upper: FloatArray, lower: FloatArray
) -> tuple[FloatResult, FloatResult]:
    """
    Convert an asymmetric confidence interval into fractional errors.

    Parameters
    ----------
    value : array_like
        The central (best-fit) value.
    upper : array_like
        The upper bound of the confidence interval.
    lower : array_like
        The lower bound of the confidence interval.

    Returns
    -------
    tuple of array_like
        The ``(lower, upper)`` fractional errors, i.e. ``(value - lower) /
        value`` and ``(upper - value) / value``.
    """
    return (value - lower) / value, (upper - value) / value


# ------------------------------------------ #
# Astropy Model Construction                 #
# ------------------------------------------ #
# At the m4opt.synphot level, operations are performed on Synphot / Astropy Model objects,
# which are not immediately compatible with the machinery of the SpectralModel class. This
# helper lets a caller wrap a plain broadcasting kernel as such a Model on demand.
def model_class_from_kernel(
    name: str,
    inputs: dict[str, u.UnitBase],
    outputs: dict[str, u.UnitBase],
    evaluate: Callable[..., FloatResult],
) -> type[Model]:
    """
    Dynamically build a parameterless :class:`~astropy.modeling.Model` subclass around a plain broadcasting kernel.

    Deliberately does not use astropy's formal :class:`~astropy.modeling.Parameter`
    machinery -- any state ``evaluate`` needs (e.g. a model's SED
    parameter values, batched or not) must already be bound into it by
    the caller (typically via closure). This is the same pattern used by
    ``m4opt.synphot._extrinsic.ScaleFactor``: because no ``Parameter`` is
    involved, :mod:`synphot`'s ``n_models=1`` restriction never applies,
    and any leading batch axes on the bound state broadcast straight
    through :meth:`~astropy.modeling.Model.evaluate` against whatever
    shape this model is called with.

    Parameters
    ----------
    name : str
        Class name for the generated :class:`~astropy.modeling.Model` subclass.
    inputs : dict of str to ~astropy.units.UnitBase
        ``{name: unit}`` for each positional input, in call order.
    outputs : dict of str to ~astropy.units.UnitBase
        ``{name: unit}`` for this model's output. Exactly one entry.
    evaluate : callable
        ``evaluate(*input_values) -> FloatResult``, operating on
        unit-stripped, broadcastable NumPy arrays -- e.g. a closure over
        one of a :class:`~uvex_transients.models.core.base.SpectralModel` subclass's
        ``*_cgs`` classmethods. Bound directly as a ``staticmethod`` on
        the generated class (no ``self``).

    Returns
    -------
    type
        A fresh :class:`~astropy.modeling.Model` subclass (not an
        instance) with ``n_inputs``/``n_outputs``/``inputs``/``outputs``/
        ``input_units``/``return_units`` populated from ``inputs``/``outputs``.
    """
    if len(outputs) != 1:
        raise NotImplementedError("Only single-output models are currently supported.")
    (output_name,) = outputs
    input_units = tuple(inputs.values())

    def _strip_units(*values: FloatArray) -> FloatResult:
        # `Model.__call__` converts a Quantity input to the declared unit
        # but -- unlike a bare-number call -- does *not* strip it down to a
        # plain array before handing it to `evaluate`; without this, a
        # Quantity would silently ride along through `evaluate`'s kernel
        # arithmetic, corrupting whatever unit `_process_output_units`
        # later tries to attach to the result. Bare-number calls pass
        # through `to_value` untouched (it's a no-op for non-Quantities).
        return evaluate(
            *(
                value.to_value(unit) if isinstance(value, u.Quantity) else value
                for value, unit in zip(values, input_units)
            )
        )

    def _prepare_outputs(self, broadcasted_shapes, *outputs, **kwargs):
        # `evaluate` already returns whatever shape its bound batch state
        # (broadcast against the call's `x`/`t`) actually produces --
        # unlike a normal Parameter-based model, astropy's shape inference
        # here only sees the *declared* inputs, not any batch axes bound
        # into `evaluate` via closure. The default `prepare_outputs` still
        # applies correctly for the common case (an ordinary, non-batched
        # call, where it turns a size-1 array into a proper scalar); it
        # only needs to be skipped when the closure-bound batch shape isn't
        # visible in the declared inputs' broadcast shape, which makes its
        # unguarded `output.item()`/`.reshape()` raise.
        try:
            return self._prepare_outputs_single_model(outputs, broadcasted_shapes)
        except ValueError:
            return outputs

    return type(
        name,
        (Model,),
        {
            "n_inputs": len(inputs),
            "n_outputs": 1,
            "inputs": tuple(inputs),
            "outputs": (output_name,),
            "input_units": dict(inputs),
            "return_units": dict(outputs),
            # synphot -- and any other caller passing bare floats -- expects
            # them to be interpreted in `inputs`' declared units; mirror
            # `astropy.modeling.physical_models.BlackBody`.
            "_input_units_allow_dimensionless": True,
            "evaluate": staticmethod(_strip_units),
            "prepare_outputs": _prepare_outputs,
        },
    )


# ------------------------------------------ #
# Detector-Aware (Noisy) Photometry           #
# ------------------------------------------ #
# The shared machinery behind `SpectralModel.simulate_photometry` (a real, time-dependent
# SED) and `simulate_flat_photometry` (a flat placeholder flux, no SED at all) -- only how
# the underlying `~synphot.SourceSpectrum` gets built differs between the two; everything
# from validating `bands`/`sys_err`/`coord` through the per-band noise realization and
# table assembly is identical, which is why it lives here rather than being duplicated.
def resolve_bands(detector: Detector, bands: list | None, sys_err: float | Mapping[str, float] | None) -> list[str]:
    """
    Validate a photometry call's ``bands`` and ``sys_err`` against a detector.

    Parameters
    ----------
    detector : ~m4opt.synphot.Detector
        Supplies the available bandpasses.
    bands : list, optional
        The requested bands. `None` means every bandpass `detector` has.
    sys_err : float or Mapping[str, float], optional
        A systematic floor. If a mapping, it must have an entry for every requested band.

    Returns
    -------
    list of str
        The band names to measure.

    Raises
    ------
    ValueError
        If `bands` names a bandpass `detector` doesn't have, or `sys_err` is a mapping
        missing one of the bands.
    """
    band_names = list(detector.bandpasses) if bands is None else list(bands)
    unknown = [band for band in band_names if band not in detector.bandpasses]
    if unknown:
        raise ValueError(f"Unknown bandpass(es) {unknown}; available: {list(detector.bandpasses)}.")
    if isinstance(sys_err, Mapping):
        missing = [band for band in band_names if band not in sys_err]
        if missing:
            raise ValueError(f"'sys_err' is missing entries for band(s) {missing}.")
    return band_names


def normalize_times(t: PhysicalInput, exptime: Quantity) -> tuple[Quantity, Quantity]:
    """
    Turn a photometry call's ``t`` and ``exptime`` into matching 1-d arrays.

    Parameters
    ----------
    t : array-like or ~astropy.units.Quantity
        Time(s) since explosion; a scalar becomes shape ``(1,)``.
    exptime : ~astropy.units.Quantity
        Exposure duration(s), scalar (applied to every entry of `t`) or shape matching `t`.

    Returns
    -------
    t : ~astropy.units.Quantity
        Shape ``(N,)``.
    exptime : ~astropy.units.Quantity
        Shape ``(N,)``.

    Raises
    ------
    ValueError
        If `exptime` is neither scalar nor shaped like `t`.
    """
    t = np.atleast_1d(u.Quantity(t))
    exptime = u.Quantity(exptime)
    if exptime.isscalar:
        exptime = np.broadcast_to(exptime, t.shape, subok=True)
    elif exptime.shape != t.shape:
        raise ValueError(f"'exptime' must be scalar or match 't' shape {t.shape}, got {exptime.shape}.")
    return t, exptime


def measure_bands(
    detector: Detector,
    band_names: list[str],
    exptime: Quantity,
    spectra: SourceSpectrum,
    coord: SkyCoord,
    observer_location: EarthLocation,
    obstime: Time,
    *,
    sys_err: float | Mapping[str, float] | None = None,
    rng: RNGInput = None,
    noise_seed: int | NDArray | None = None,
    noise_observation_keys: NDArray | None = None,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Measure a batch of spectra with a detector: one noisy measurement per epoch and band.

    The single place the detector noise model is turned into measurements, shared by the
    per-event photometry tables and the survey-wide detection cuts so both always agree.
    For each epoch and band, `~m4opt.synphot.Detector.get_snr` gives the expected SNR of the
    true spectrum (source and sky Poisson noise plus detector read and dark noise). The
    one-sigma uncertainty is the true flux over that SNR, the measured flux is the true flux
    plus a Gaussian draw of one uncertainty, and the measured SNR is that flux over the
    uncertainty.

    Every epoch can have its own position, observer location and time, so many events can
    be measured at once.

    Parameters
    ----------
    detector : ~m4opt.synphot.Detector
        Supplies the bandpasses and noise terms, including the background.
    band_names : list of str
        The bands to measure, in output order.
    exptime : ~astropy.units.Quantity
        Exposure duration of each epoch, shape ``(N,)``.
    spectra : ~synphot.SourceSpectrum
        The true spectrum of each epoch, with a trailing batch axis reserved for wavelength.
    coord : ~astropy.coordinates.SkyCoord
        Position of each epoch, scalar or shape ``(N,)``.
    observer_location : ~astropy.coordinates.EarthLocation
        Observer location of each epoch, scalar or shape ``(N,)``.
    obstime : ~astropy.time.Time
        Observation time of each epoch, scalar or shape ``(N,)``.
    sys_err : float or Mapping[str, float], optional
        A per-band systematic calibration error floor, in magnitudes, combined in quadrature
        with the shot-noise uncertainty before any noise is drawn. A mapping must have an
        entry for every band in `band_names`. `None` adds nothing.
    rng : numpy.random.Generator, int, or None
        Random-number source for the noise; see :func:`~uvex_transients.utils.get_rng`.
        Ignored when `noise_seed` is given.
    noise_seed : int or numpy.ndarray, optional
        If given, noise is drawn with `~uvex_transients.utils.keyed_noise.keyed_standard_normal`
        instead of `rng`: each measurement's draw depends only on its `noise_seed` (a scalar,
        or one per epoch), its entry of `noise_observation_keys`, and the band's position in
        `detector.bandpasses`. The same measurement then gets the same noise whichever other
        epochs are measured with it.
    noise_observation_keys : numpy.ndarray, optional
        ``uint64`` key per epoch (see `~uvex_transients.utils.keyed_noise.time_key`). Required
        when `noise_seed` is given.

    Returns
    -------
    flux : numpy.ndarray
        Measured flux density in Jy, shape ``(n_bands, N)``.
    flux_err : numpy.ndarray
        One-sigma uncertainty in Jy, same shape.
    snr : numpy.ndarray
        Measured SNR, ``flux / flux_err``, same shape.
    snr_expected : numpy.ndarray
        Noiseless SNR, the true flux over ``flux_err``, same shape. It sets the size of the
        noise, and is what the linearized ``mag_err`` is built from.

    Raises
    ------
    ValueError
        If `noise_seed` is given without a `noise_observation_keys` of shape ``(N,)``.

    Notes
    -----
    `flux`, `flux_err` and `snr` are ``nan`` wherever the expected SNR is not finite and positive.
    """
    if noise_seed is not None and (noise_observation_keys is None or np.shape(noise_observation_keys) != exptime.shape):
        raise ValueError(f"'noise_observation_keys' must have shape {exptime.shape} when 'noise_seed' is given.")

    rng = get_rng(rng)
    detector_bands = list(detector.bandpasses)
    flux, flux_err, snr, snr_expected = [], [], [], []

    with observing(observer_location, coord, obstime):
        for band in band_names:
            expected = u.Quantity(detector.get_snr(exptime, spectra, band)).to_value(u.dimensionless_unscaled)
            true_flux = np.squeeze(spectra(detector.bandpasses[band].pivot(), flux_unit=u.Jy).to_value(u.Jy), axis=-1)

            with np.errstate(invalid="ignore", divide="ignore"):
                valid = np.isfinite(expected) & (expected > 0)
                err = true_flux / np.where(valid, expected, np.nan)
                band_sys_err = sys_err[band] if isinstance(sys_err, Mapping) else sys_err
                if band_sys_err:
                    # `sys_err` is a fixed fractional-magnitude floor; convert to a fractional
                    # flux error (exact for the same small-error limit `mag_err = 2.5 / (ln(10) *
                    # snr)` assumes) and combine it in quadrature with the shot-noise error,
                    # before anything is drawn, so the noise itself carries the systematic scatter
                    # and not just a wider reported error bar.
                    err = np.hypot(err, np.abs(true_flux) * band_sys_err * np.log(10) / 2.5)
                    expected = np.where(valid, true_flux / err, expected)

                scale = np.where(valid, np.abs(err), 1.0)
                if noise_seed is not None:
                    noise = keyed_standard_normal(noise_seed, noise_observation_keys, detector_bands.index(band))
                    draw = true_flux + scale * noise
                else:
                    draw = rng.normal(true_flux, scale)
                draw = np.where(valid, draw, np.nan)

                flux.append(draw)
                flux_err.append(err)
                snr.append(np.where(valid, draw / np.abs(err), np.nan))
                snr_expected.append(expected)

    return tuple(np.stack(values) for values in (flux, flux_err, snr, snr_expected))


def photometry_table(
    t: Quantity,
    exptime: Quantity,
    band_names: list[str],
    flux: NDArray,
    flux_err: NDArray,
    snr: NDArray,
    snr_expected: NDArray,
    n_sigma: float | None = None,
    in_model: NDArray | None = None,
) -> QTable:
    r"""
    Lay `measure_bands`'s measurements out as a photometry table, with magnitudes and bounds.

    Parameters
    ----------
    t : ~astropy.units.Quantity
        Time of each epoch, shape ``(N,)``.
    exptime : ~astropy.units.Quantity
        Exposure duration of each epoch, shape ``(N,)``.
    band_names : list of str
        The measured bands, in the order of the first axis of the arrays below.
    flux, flux_err, snr, snr_expected : numpy.ndarray
        The four outputs of `measure_bands`, each of shape ``(len(band_names), N)``.
    n_sigma : float, optional
        Width, in multiples of ``flux_err``, of the ``flux_upper``/``flux_lower``/
        ``mag_upper``/``mag_lower`` interval. If `None` (the default), uses
        ``config["simulation.detection_n_sigma"]`` (5 out of the box).
    in_model : numpy.ndarray, optional
        Boolean, shape ``(N,)``: whether each epoch was evaluated against a transient model
        (as opposed to background only). If given, it becomes the ``in_model`` column.

    Returns
    -------
    astropy.table.QTable
        One row per (time, band), sorted by ``t`` then ``band``, with columns ``t``,
        ``exptime``, ``band``, ``snr`` (the measured SNR, ``flux / flux_err``),
        ``flux``/``flux_err`` (Jy), ``flux_upper``/``flux_lower`` (Jy, ``flux ± n_sigma*flux_err``),
        ``ab_mag``/``mag_err``, ``mag_upper``/``mag_lower`` -- the ``n_sigma`` interval
        transformed to magnitude, brighter bound first -- and ``in_model`` if given. ``mag_err`` is the linearized error
        ``2.5 / (ln(10) * snr_expected)``. See
        :meth:`~uvex_transients.simulation.event.Event.simulate_photometry` for the exact
        semantics of every column.
    """
    if n_sigma is None:
        n_sigma = config["simulation.detection_n_sigma"]

    tables = []
    for b, band in enumerate(band_names):
        with np.errstate(invalid="ignore", divide="ignore"):
            flux_upper = flux[b] + n_sigma * flux_err[b]
            flux_lower = flux[b] - n_sigma * flux_err[b]
            band_table = QTable()
            band_table["t"] = t
            band_table["exptime"] = exptime
            band_table["band"] = np.full(len(t), band)
            band_table["snr"] = snr[b]
            band_table["flux"] = flux[b] * u.Jy
            band_table["flux_err"] = flux_err[b] * u.Jy
            band_table["flux_upper"] = flux_upper * u.Jy
            band_table["flux_lower"] = flux_lower * u.Jy
            band_table["ab_mag"] = np.where(flux[b] > 0, (flux[b] * u.Jy).to_value(u.ABmag), np.nan)
            band_table["mag_err"] = np.where(
                np.isfinite(flux_err[b]), 2.5 / (np.log(10) * np.abs(snr_expected[b])), np.nan
            )
            band_table["mag_upper"] = np.where(flux_lower > 0, (flux_lower * u.Jy).to_value(u.ABmag), np.nan)
            band_table["mag_lower"] = np.where(flux_upper > 0, (flux_upper * u.Jy).to_value(u.ABmag), np.nan)
            if in_model is not None:
                band_table["in_model"] = in_model
        tables.append(band_table)

    table = vstack(tables)
    return table[np.lexsort((table["band"], table["t"].to_value(t.unit)))]


def _flat_flux_source_spectrum(flux: Quantity) -> SourceSpectrum:
    """
    Build a `~synphot.SourceSpectrum` with a flat (wavelength-independent) flux density.

    `flux` (shape ``(N,)``) gets a trailing wavelength batch axis added internally,
    matching the ``t[:, np.newaxis]`` convention
    `~uvex_transients.models.core.base.SpectralModel.as_source_spectrum` callers use, so
    the returned spectrum broadcasts the same way against a bandpass's wavelength grid
    -- one (constant) flux value per entry of `flux`, not one shared scalar.

    Parameters
    ----------
    flux : ~astropy.units.Quantity
        Flux density, in units convertible to Jy, shape ``(N,)``.

    Returns
    -------
    ~synphot.SourceSpectrum
        Callable as ``spectrum(wave)``, returning `flux` (converted to synphot's
        internal photon flux) at every wavelength, unchanged with wavelength.
    """
    flux_jy = ensure_in_units(flux, u.Jy)[:, np.newaxis]

    def evaluate(wave: FloatArray) -> FloatResult:
        """
        Return `flux_jy`, converted to PHOTLAM at `wave`.

        Parameters
        ----------
        wave : numpy.ndarray
            Wavelength(s), in Angstrom (unit-stripped, per `model_class_from_kernel`'s
            calling convention).

        Returns
        -------
        numpy.ndarray
            `flux_jy`, converted to PHOTLAM at `wave` -- constant with wavelength.
        """
        return synphot_units.convert_flux(wave * u.AA, flux_jy * u.Jy, synphot_units.PHOTLAM).value

    model_class = model_class_from_kernel(
        "_FlatFluxSourceSpectrumModel",
        inputs={"wave": u.AA},
        outputs={"y": synphot_units.PHOTLAM},
        evaluate=evaluate,
    )
    return SourceSpectrum(model_class())


def simulate_flat_photometry(
    t: PhysicalInput,
    exptime: Quantity,
    detector: Detector,
    coord: SkyCoord,
    *,
    flux: Quantity = 0 * u.Jy,
    background: SourceSpectrum | None = None,
    bands: list | None = None,
    observer_location: EarthLocation | None = None,
    obstime: Time | None = None,
    n_sigma: float | None = None,
    sys_err: float | Mapping[str, float] | None = None,
    rng: RNGInput = None,
    noise_seed: int | NDArray | None = None,
    noise_observation_keys: NDArray | None = None,
) -> QTable:
    """
    Simulate noisy detector photometry of a flat (wavelength-independent) flux, with no `SpectralModel`.

    `flux=0` (the default) simulates a pure background measurement: the detector noise
    (sky background plus dark and read noise, against a true source flux of zero) still
    varies realistically per band and observation, computed the same way as a real
    detection. A nonzero, uniform `flux` also works, e.g. to simulate photometry of a known
    non-transient point source.

    Parameters
    ----------
    t : array-like or ~astropy.units.Quantity
        Time(s), shape ``(N,)`` (or scalar, promoted to shape ``(1,)``). These need not be
        times since any explosion; `flux` doesn't depend on `t` at all.
    exptime : ~astropy.units.Quantity
        Exposure duration(s), scalar (applied to every entry of `t`) or shape matching `t`.
    detector : ~m4opt.synphot.Detector
        Supplies bandpasses, collecting area, plate scale, and detector noise terms.
    coord : ~astropy.coordinates.SkyCoord
        Sky position of the target, scalar or shape ``(N,)``.
    flux : ~astropy.units.Quantity, optional
        Flux density, in units convertible to Jy. Scalar (the same at every time, the default:
        ``0 * u.Jy``, i.e. no source) or shape matching `t`.
    background : ~synphot.SourceSpectrum, optional
        Sky background to simulate against for this call. If `None` (the default), `detector`'s
        own ``background`` is used.
    bands : list, optional
        Which of `detector`'s bandpasses to evaluate. Defaults to every bandpass.
    observer_location : ~astropy.coordinates.EarthLocation, optional
        Defaults to a fixed placeholder, correct whenever `background` doesn't depend on it.
    obstime : ~astropy.time.Time, optional
        Defaults to a fixed placeholder, correct whenever `background` doesn't depend on it.
    n_sigma : float, optional
        See `photometry_table`.
    sys_err : float or Mapping[str, float], optional
        See `measure_bands`.
    rng : numpy.random.Generator, int, or None
        See `measure_bands`.
    noise_seed, noise_observation_keys : optional
        See `measure_bands`.

    Returns
    -------
    astropy.table.QTable
        See `photometry_table`.

    Raises
    ------
    ValueError
        If `bands` or `sys_err` doesn't match `detector`, `exptime` doesn't match `t`, or
        `noise_seed` is given without matching `noise_observation_keys`.
    """
    if background is not None:
        detector = replace(detector, background=background)
    band_names = resolve_bands(detector, bands, sys_err)
    t, exptime = normalize_times(t, exptime)

    # `np.where` (rather than a scalar `if flux == 0`) so a per-observation `flux` array can mix
    # genuine non-detections (0) with real nonzero entries: only the exact zeros get nudged to
    # `_NEGLIGIBLE_FLUX_JY`; see that constant's comment for why.
    flux_jy = ensure_in_units(flux, u.Jy)
    flux_jy = np.broadcast_to(np.where(flux_jy == 0, _NEGLIGIBLE_FLUX_JY, flux_jy), t.shape)

    measurements = measure_bands(
        detector,
        band_names,
        exptime,
        _flat_flux_source_spectrum(flux_jy * u.Jy),
        coord,
        _PLACEHOLDER_OBSERVER_LOCATION if observer_location is None else observer_location,
        _PLACEHOLDER_OBSTIME if obstime is None else obstime,
        sys_err=sys_err,
        rng=rng,
        noise_seed=noise_seed,
        noise_observation_keys=noise_observation_keys,
    )
    return photometry_table(t, exptime, band_names, *measurements, n_sigma=n_sigma)
