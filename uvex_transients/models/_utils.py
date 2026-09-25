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

# `simulate_detector_photometry`'s defaults for `observer_location`/`obstime` when the
# caller doesn't supply real ones -- correct as long as `background` doesn't actually
# depend on either (true of `m4opt.synphot.background.GalacticBackground`, false of
# `ZodiacalBackground`/`EarthshineBackground`).
_PLACEHOLDER_OBSERVER_LOCATION = EarthLocation(0 * u.m, 0 * u.m, 0 * u.m)
_PLACEHOLDER_OBSTIME = Time("2000-01-01T00:00:00", scale="utc")

# `simulate_flat_photometry`'s substitute for an exactly-zero requested flux.
# `Detector.get_snr` correctly returns a finite `snr` of exactly 0 for a source
# with zero flux, but `simulate_detector_photometry` then computes
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
def simulate_detector_photometry(
    t: PhysicalInput,
    exptime: Quantity,
    detector: Detector,
    coord: SkyCoord,
    spectra_factory: Callable[[Quantity], SourceSpectrum],
    *,
    background: SourceSpectrum | None = None,
    bands: list | None = None,
    observer_location: EarthLocation | None = None,
    obstime: Time | None = None,
    n_sigma: float | None = None,
    sys_err: float | Mapping[str, float] | None = None,
    rng: RNGInput = None,
) -> QTable:
    r"""
    Simulate noisy detector photometry of a `~synphot.SourceSpectrum` at given time(s).

    Builds the actual noisy measurement -- one Gaussian realization of the true flux at
    each requested time and band, at that time/band's own implied uncertainty, via
    `~m4opt.synphot.Detector.get_snr` (the same OIR CCD equation
    `~astropy.stats.signal_to_noise_oir_ccd` implements) -- once `spectra_factory` has
    supplied the spectrum to measure. `bands`/`sys_err`/`coord`/`t`/`exptime` are
    validated, and `t`/`exptime` normalized, *before* `spectra_factory` is ever called,
    so an unknown band name (or any other validation failure) raises without first
    paying for whatever `spectra_factory` would have done to build a spectrum -- the
    same order `SpectralModel.simulate_photometry` used before this was extracted from
    it.

    Parameters
    ----------
    t : array-like or Quantity
        Time(s) since explosion, shape ``(N,)`` (or scalar, promoted to shape ``(1,)``)
        -- one entry per requested observation.
    exptime : ~astropy.units.Quantity
        Exposure duration(s), scalar (applied to every entry of `t`) or shape matching
        `t`.
    detector : ~m4opt.synphot.Detector
        Supplies bandpasses, collecting area, plate scale, and detector noise terms.
        Its own ``background`` is used only if `background` isn't given.
    coord : ~astropy.coordinates.SkyCoord
        Scalar sky position of the target.
    spectra_factory : callable
        ``spectra_factory(t) -> SourceSpectrum``, called exactly once, after
        validation, with `t` already normalized to shape ``(N,)`` -- responsible for
        adding whatever trailing wavelength batch axis its own `SourceSpectrum`
        construction needs (e.g. ``t[:, np.newaxis]``).
    background : ~synphot.SourceSpectrum, optional
        Sky background surface brightness to simulate against for this call. If `None`
        (the default), `detector`'s own ``background`` is used.
    bands : list, optional
        Which of `detector`'s bandpasses to evaluate. Defaults to every bandpass
        `detector` has.
    observer_location : ~astropy.coordinates.EarthLocation, optional
        Defaults to a fixed placeholder location, correct whenever `background`
        doesn't depend on it.
    obstime : ~astropy.time.Time, optional
        Defaults to a fixed placeholder epoch, correct whenever `background` doesn't
        depend on it.
    n_sigma : float, optional
        Width, in multiples of ``flux_err``, of the ``flux_upper``/``flux_lower``/
        ``mag_upper``/``mag_lower`` interval. If `None` (the default), uses
        ``config["simulation.detection_n_sigma"]`` (5 out of the box).
    sys_err : float or Mapping[str, float], optional
        A per-band systematic calibration error floor, in magnitudes, combined in
        quadrature with the shot-noise uncertainty. A bare `float` applies the same
        floor to every band in `bands`; a mapping must have an entry for every band in
        `bands`. If `None` (the default), no systematic floor is added.
    rng : numpy.random.Generator, int, or None
        Random-number source for the noise realization; see
        :func:`~uvex_transients.utils.get_rng`.

    Returns
    -------
    astropy.table.QTable
        One row per (time, band), sorted by ``t`` then ``band``, with columns ``t``,
        ``exptime``, ``band``, ``snr``, ``flux``/``flux_err`` (Jy), ``flux_upper``/
        ``flux_lower`` (Jy, ``flux ± n_sigma*flux_err``), ``ab_mag``/``mag_err``, and
        ``mag_upper``/``mag_lower`` -- the ``n_sigma`` interval transformed to
        magnitude, brighter bound first. See
        :meth:`~uvex_transients.simulation.event.Event.simulate_photometry` for the
        exact semantics of every column.

    Raises
    ------
    ValueError
        If `coord` is not scalar, if `bands` contains a name `detector` doesn't have,
        if `exptime` is neither scalar nor shaped like `t`, or if `sys_err` is a
        mapping missing an entry for one of `bands`.
    """
    if n_sigma is None:
        n_sigma = config["simulation.detection_n_sigma"]

    if background is not None:
        detector = replace(detector, background=background)

    band_names = list(detector.bandpasses) if bands is None else list(bands)
    unknown = [band for band in band_names if band not in detector.bandpasses]
    if unknown:
        raise ValueError(f"Unknown bandpass(es) {unknown}; available: {list(detector.bandpasses)}.")

    if isinstance(sys_err, Mapping):
        missing_sys_err = [band for band in band_names if band not in sys_err]
        if missing_sys_err:
            raise ValueError(f"'sys_err' is missing entries for band(s) {missing_sys_err}.")

    if not coord.isscalar:
        raise ValueError("Parameter 'coord' must be a scalar SkyCoord.")

    t = np.atleast_1d(u.Quantity(t))
    exptime = u.Quantity(exptime)
    if exptime.isscalar:
        exptime = np.broadcast_to(exptime, t.shape, subok=True)
    elif exptime.shape != t.shape:
        raise ValueError(f"'exptime' must be scalar or match 't' shape {t.shape}, got {exptime.shape}.")
    n_obs = t.shape[0]

    if observer_location is None:
        observer_location = _PLACEHOLDER_OBSERVER_LOCATION
    if obstime is None:
        obstime = _PLACEHOLDER_OBSTIME

    rng = get_rng(rng)
    spectra = spectra_factory(t)

    tables = []

    with observing(observer_location, coord, obstime):
        for band in band_names:
            snr = detector.get_snr(exptime, spectra, band)
            band_sys_err = sys_err[band] if isinstance(sys_err, Mapping) else sys_err

            pivot = detector.bandpasses[band].pivot()
            with np.errstate(invalid="ignore", divide="ignore"):
                true_flux = np.squeeze(spectra(pivot, flux_unit=u.Jy).to_value(u.Jy), axis=-1)

                valid = np.isfinite(snr) & (snr > 0)
                safe_snr = np.where(valid, snr, np.nan)
                flux_err = true_flux / safe_snr
                reported_snr = snr
                if band_sys_err:
                    # `sys_err` is a fixed fractional-magnitude floor; convert to a
                    # fractional flux error (exact for the same small-error limit
                    # `mag_err = 2.5 / (ln(10) * snr)` already assumes) and combine
                    # in quadrature with the shot-noise flux error above, before
                    # anything is drawn from it -- so the noise realization itself
                    # carries the systematic scatter, not just a wider reported bar
                    # around an unchanged draw.
                    flux_err = np.hypot(flux_err, np.abs(true_flux) * band_sys_err * np.log(10) / 2.5)
                    reported_snr = np.where(valid, true_flux / flux_err, snr)
                flux = rng.normal(true_flux, np.where(valid, np.abs(flux_err), 1.0))
                flux = np.where(valid, flux, np.nan)
                mag_err = np.where(valid, 2.5 / (np.log(10) * np.abs(reported_snr)), np.nan)
                mag = np.where(flux > 0, (flux * u.Jy).to_value(u.ABmag), np.nan)

                flux_upper = flux + n_sigma * flux_err
                flux_lower = flux - n_sigma * flux_err
                mag_lower = np.where(flux_upper > 0, (flux_upper * u.Jy).to_value(u.ABmag), np.nan)
                mag_upper = np.where(flux_lower > 0, (flux_lower * u.Jy).to_value(u.ABmag), np.nan)

            band_table = QTable()
            band_table["t"] = t
            band_table["exptime"] = exptime
            band_table["band"] = np.full(n_obs, band)
            band_table["snr"] = reported_snr
            band_table["flux"] = flux * u.Jy
            band_table["flux_err"] = flux_err * u.Jy
            band_table["flux_upper"] = flux_upper * u.Jy
            band_table["flux_lower"] = flux_lower * u.Jy
            band_table["ab_mag"] = mag
            band_table["mag_err"] = mag_err
            band_table["mag_upper"] = mag_upper
            band_table["mag_lower"] = mag_lower
            tables.append(band_table)

    table = vstack(tables)
    order = np.lexsort((table["band"], table["t"].to_value(t.unit)))
    return table[order]


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
) -> QTable:
    r"""
    Simulate noisy detector photometry of a flat (wavelength-independent) flux, with no `SpectralModel`.

    The non-model counterpart to
    :meth:`~uvex_transients.models.core.base.SpectralModel.simulate_photometry`: every
    time in `t` gets the same constant-in-wavelength `flux`, rather than a
    `SpectralModel`'s time-dependent SED, so no `~uvex_transients.models.core._base.SpectralModel._eval`
    is ever called. That's the point of this function: `flux=0` (the default)
    simulates a pure background/non-detection measurement -- the detector noise
    (sky background plus detector dark/read noise, against a true source flux of
    zero) still varies realistically per band/observation, computed the same way as
    a real detection -- at times where the underlying `SpectralModel` isn't valid
    (e.g. before an event's explosion time, or after its light curve's calibrated
    domain ends), instead of extrapolating that model's `_eval` outside its intended
    domain, which for some light curve shapes (e.g. those with a ``1/t`` early-time
    singularity) diverges rather than merely being physically wrong.

    A nonzero, uniform `flux` also works, e.g. to simulate photometry of a known
    non-transient point source.

    Parameters
    ----------
    t : array-like or Quantity
        Time(s), shape ``(N,)`` (or scalar, promoted to shape ``(1,)``) -- one entry
        per requested observation. Unlike `SpectralModel.simulate_photometry`, these
        need not be times since any explosion; `flux` doesn't depend on `t` at all.
    exptime : ~astropy.units.Quantity
        Exposure duration(s), scalar (applied to every entry of `t`) or shape matching
        `t`.
    detector : ~m4opt.synphot.Detector
        Supplies bandpasses, collecting area, plate scale, and detector noise terms.
    coord : ~astropy.coordinates.SkyCoord
        Scalar sky position of the target.
    flux : ~astropy.units.Quantity, optional
        Flux density, in units convertible to Jy. Scalar (the same flux at every time
        in `t`, the default: ``0 * u.Jy``, i.e. no source) or shape matching `t`.
    background : ~synphot.SourceSpectrum, optional
        See :func:`simulate_detector_photometry`.
    bands : list, optional
        See :func:`simulate_detector_photometry`.
    observer_location : ~astropy.coordinates.EarthLocation, optional
        See :func:`simulate_detector_photometry`.
    obstime : ~astropy.time.Time, optional
        See :func:`simulate_detector_photometry`.
    n_sigma : float, optional
        See :func:`simulate_detector_photometry`.
    sys_err : float or Mapping[str, float], optional
        See :func:`simulate_detector_photometry`.
    rng : numpy.random.Generator, int, or None
        See :func:`simulate_detector_photometry`.

    Returns
    -------
    astropy.table.QTable
        See :func:`simulate_detector_photometry`.
    """
    # `np.where` (rather than a scalar `if flux == 0`) so a per-observation `flux`
    # array can mix genuine non-detections (0) with real nonzero flux entries --
    # only the exact zeros get nudged to `_NEGLIGIBLE_FLUX_JY`; see that constant's
    # docstring for why.
    flux_jy = ensure_in_units(flux, u.Jy)
    flux_jy = np.where(flux_jy == 0, _NEGLIGIBLE_FLUX_JY, flux_jy)

    def spectra_factory(t_norm: Quantity) -> SourceSpectrum:
        """
        Build the flat-flux `~synphot.SourceSpectrum` for `t_norm`'s normalized shape.

        Parameters
        ----------
        t_norm : ~astropy.units.Quantity
            `t`, already validated/normalized to shape ``(N,)`` by
            `simulate_detector_photometry`.

        Returns
        -------
        ~synphot.SourceSpectrum
            The flat-flux spectrum to simulate photometry of.
        """
        flux_per_obs = flux_jy if flux_jy.shape == t_norm.shape else np.broadcast_to(flux_jy, t_norm.shape)
        return _flat_flux_source_spectrum(flux_per_obs * u.Jy)

    return simulate_detector_photometry(
        t,
        exptime,
        detector,
        coord,
        spectra_factory,
        background=background,
        bands=bands,
        observer_location=observer_location,
        obstime=obstime,
        n_sigma=n_sigma,
        sys_err=sys_err,
        rng=rng,
    )
