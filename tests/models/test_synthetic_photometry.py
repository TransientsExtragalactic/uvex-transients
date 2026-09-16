"""
This test suite verifies that the model module can plug into the synphot machinery.
"""

from functools import partial

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from m4opt.missions._uvex import uvex
from m4opt.skygrid import _geodesic
from m4opt.synphot import observing
from m4opt.synphot.background import GalacticBackground

from uvex_transients.dust import log_attenuation
from uvex_transients.models import VanVelzenTDESED, VillarCoolingBlackbodySED


def test_synthetic_photometry_batches_events_and_times():
    """
    Per-event, per-time SNR light curves via a batched `SourceSpectrum`, round-tripped against `flux`.

    Mirrors the batching pattern a real scheduling run uses -- one parameter
    realization per sky-grid event, a light curve of several times since
    explosion per event -- at a size small enough to run as a fast unit
    test. Unlike `test_synthetic_photometry_with_batched_dust_extinction`
    below, this does not multiply in dust extinction via a separate
    `DustExtinction` `SpectralElement`, since batching a per-event SED
    through that path is a known, separate limitation -- see that test's
    docstring for the batching-safe workaround.
    """
    rng = np.random.default_rng(0)

    n_events = 5
    n_times = 4
    redshift = 0.1
    band = "FUV"
    exptime = 900 * u.s

    model = VillarCoolingBlackbodySED()

    # One parameter realization per event, kept as plain (n_events,) arrays;
    # each consumer below adds however many trailing axes *it* needs to keep
    # the event axis from colliding with its own batch axes (time,
    # wavelength).
    raw_params = model.sample_parameters(n_events, rng=rng)
    t_since_explosion = np.linspace(1, 200, n_times) * u.day

    # --- Band-averaged magnitude light curves, via the model's own
    # vectorized quadrature (not synphot's Observation/effstim, which
    # chokes on batched spectra).
    bandpass = uvex.detector.bandpasses[band]
    wave = bandpass.waveset
    nu = wave.to(u.Hz, equivalencies=u.spectral())
    throughput = bandpass(wave)

    mag_params = {name: value[:, np.newaxis] for name, value in raw_params.items()}
    mag = model.mag_band(nu, throughput, t_since_explosion, redshift=redshift, **mag_params)
    assert mag.shape == (n_events, n_times)
    assert np.all(np.isfinite(mag.value))

    # --- SNR light curves, via a batched `SourceSpectrum` fed to the
    # detector. `as_source_spectrum` does not auto-insert axes, so every
    # batch axis needs its own manually reserved trailing axis: two here
    # (time, then wavelength) for the parameters, one (wavelength) for `t`.
    source_params = {name: value[:, np.newaxis, np.newaxis] for name, value in raw_params.items()}
    t_batched = t_since_explosion[:, np.newaxis]
    source_spectra = model.as_source_spectrum(t=t_batched, redshift=redshift, **source_params)

    base_time = Time("2025-01-01T00:00:00", scale="utc")
    obs_times = base_time + t_since_explosion
    locations = uvex.observer_location(obs_times)

    with observing(locations, uvex.skygrid[0], obs_times):
        snr = uvex.detector.get_snr(exptime, source_spectra, band)

    assert snr.shape == (n_events, n_times)
    assert np.all(np.isfinite(snr))
    assert np.all(snr > 0)

    # --- Round trip: sample the batched `SourceSpectrum` -- the same object
    # just fed to `get_snr` -- at the bandpass wavelength grid, then confirm
    # one (event, time, wavelength) slot of that batch matches `flux`
    # computed directly and independently for that same single realization.
    # This is what actually confirms that broadcasting several extra batch
    # axes through `as_source_spectrum` lines up each event/time slot with
    # the right parameter values, rather than merely checking finiteness.
    #
    # A bare scalar frequency trips an astropy `Model.__call__` bug
    # unrelated to this (it assumes a scalar input broadcasts to a scalar
    # output, which doesn't hold once extra batch axes are bound into the
    # model via closure) -- sample the whole wavelength grid instead and
    # index into it.
    i_event, i_time, i_freq = 2, 1, len(nu) // 2
    scalar_params = {name: value[i_event] for name, value in raw_params.items()}
    expected = model.flux(nu[i_freq], t_since_explosion[i_time], redshift=redshift, **scalar_params)
    actual = source_spectra(nu, flux_unit=expected.unit)[i_event, i_time, i_freq]
    np.testing.assert_allclose(actual.value, expected.value, rtol=1e-6)


def test_synthetic_photometry_with_batched_dust_extinction():
    """
    Per-event dust extinction via `as_source_spectrum`'s `log_attenuation`, fully batched.

    Multiplying a batched `SourceSpectrum` by a separate `DustExtinction()`
    `SpectralElement` routes through `m4opt.synphot._math.countrate`'s
    per-Ebv interpolation shortcut, which chokes on a per-event batch axis
    (a known, separate m4opt limitation -- see
    `test_synthetic_photometry_batches_events_and_times`'s docstring).
    Instead, `uvex_transients.dust.log_attenuation` (bound to each event's
    own E(B-V) via `functools.partial`) is folded directly into the flux --
    as plain NumPy arithmetic inside `as_source_spectrum`'s own evaluation
    kernel -- before the `SourceSpectrum` is ever built. That keeps the
    whole thing one ordinary broadcast, so it batches over events exactly as
    cleanly as every other parameter already does in
    `test_synthetic_photometry_batches_events_and_times`.
    """
    rng = np.random.default_rng(0)

    skygrid = _geodesic.for_subdivision(21, 4, "icosahedron")
    n_events = len(skygrid)

    model = VanVelzenTDESED()

    parameters = {name: value[:, np.newaxis] for name, value in model.sample_parameters(n_events, rng=rng).items()}
    # One E(B-V) per event. `dust.log_attenuation` derives its own output
    # shape from `Ebv`'s shape directly (no manually reserved trailing
    # axis needed here, unlike `parameters` above), so a flat (n_events,)
    # array already broadcasts correctly against wavelength.
    ebv = rng.uniform(0.0, 0.3, size=n_events)

    time_since_explosion = 10 * u.day
    obstime = Time("2025-01-01T00:00:00", scale="utc")
    redshift = 0.05

    spectra_with_dust = model.as_source_spectrum(
        time_since_explosion,
        redshift=redshift,
        log_attenuation=partial(log_attenuation, Ebv=ebv),
        **parameters,
    )
    spectra_without_dust = model.as_source_spectrum(time_since_explosion, redshift=redshift, **parameters)

    with observing(uvex.observer_location(obstime), skygrid, obstime):
        snr_with_dust = uvex.detector.get_snr(900 * u.s, spectra_with_dust, "FUV")
        snr_without_dust = uvex.detector.get_snr(900 * u.s, spectra_without_dust, "FUV")

    assert snr_with_dust.shape == (n_events,)
    assert np.all(np.isfinite(snr_with_dust))
    assert np.all(snr_with_dust >= 0)
    # Every event has EBV > 0, so extinction should only ever suppress flux.
    assert np.all(snr_with_dust <= snr_without_dust)

    # Cross-check one event/wavelength slot against a manual, unbatched
    # computation, the same way `test_synthetic_photometry_batches_events_and_times`
    # confirms broadcasting actually lines up event slots with the right values.
    i_event, i_freq = 3, 5
    wave = uvex.detector.bandpasses["FUV"].waveset
    scalar_params = {name: value[i_event, 0] for name, value in parameters.items()}
    scalar_spectrum = VanVelzenTDESED().as_source_spectrum(
        time_since_explosion,
        redshift=redshift,
        log_attenuation=partial(log_attenuation, Ebv=ebv[i_event]),
        **scalar_params,
    )
    expected = scalar_spectrum(wave[i_freq])
    actual = spectra_with_dust(wave)[i_event, i_freq]
    np.testing.assert_allclose(actual.value, expected.value, rtol=1e-6)


# =========================================================================== #
# SpectralModel.simulate_photometry                                          #
# =========================================================================== #
# Unlike the tests above, none of these go through a `SurveySchedule` or
# `Event` -- `simulate_photometry` is meant to work for a target of
# opportunity, given only a sky position and whatever times/exposures the
# caller wants evaluated.


def test_simulate_photometry_matches_manual_get_snr():
    """
    Batched over several times and every band, cross-checked cell-by-cell.

    Checked against an independent, unbatched `as_source_spectrum`/`Detector.get_snr`
    call -- the same public primitives `simulate_photometry` is built from (mirrors
    `test_event.py::test_simulate_photometry_batches_observations_and_bands`, just
    with no `SurveySchedule`/`Event` in the loop at all).
    """
    model = VanVelzenTDESED()
    coord = SkyCoord(ra=150 * u.deg, dec=20 * u.deg)
    redshift = 0.05
    params = {name: value[0] for name, value in model.sample_parameters(1, rng=1).items()}

    t = np.sort(np.random.default_rng(0).uniform(1, 150, 6)) * u.day
    exptime = 900 * u.s
    band_names = list(uvex.detector.bandpasses)

    base_time = Time("2025-01-01T00:00:00", scale="utc")
    obstime = base_time + t
    observer_location = uvex.observer_location(obstime)

    phot = model.simulate_photometry(
        t,
        exptime,
        uvex.detector,
        coord,
        observer_location=observer_location,
        obstime=obstime,
        redshift=redshift,
        rng=0,
        **params,
    )

    assert len(phot) == len(t) * len(band_names)
    assert set(phot["band"]) == set(band_names)
    assert np.all(np.isfinite(phot["snr"]))

    for band in band_names:
        for i, t_i in enumerate(t):
            spectrum = model.as_source_spectrum(t_i, redshift=redshift, **params)
            with observing(observer_location[i], coord, obstime[i]):
                expected_snr = uvex.detector.get_snr(exptime, spectrum, band)
            row = phot[(phot["band"] == band) & (phot["t"] == t_i)]
            assert len(row) == 1
            np.testing.assert_allclose(row["snr"][0], expected_snr, rtol=1e-6)


def test_simulate_photometry_scalar_t_and_exptime():
    """A scalar `t`/`exptime` (a single requested observation) works without any manual reshaping."""
    model = VanVelzenTDESED()
    coord = SkyCoord(ra=30 * u.deg, dec=5 * u.deg)
    params = {name: value[0] for name, value in model.sample_parameters(1, rng=2).items()}

    phot = model.simulate_photometry(
        10 * u.day,
        900 * u.s,
        uvex.detector,
        coord,
        background=GalacticBackground(),
        redshift=0.05,
        rng=0,
        **params,
    )

    assert len(phot) == len(uvex.detector.bandpasses)
    assert np.all(phot["t"] == 10 * u.day)
    assert np.all(np.isfinite(phot["snr"]))


def test_simulate_photometry_background_override_is_explicit_and_effective():
    """
    `background`, not `detector.background`, decides what sky background is simulated.

    The caller never has to reconstruct or mutate `detector` to change it. With
    `background=GalacticBackground()` (no zodiacal term), SNR must be identical
    regardless of `obstime`, since `GalacticBackground` only depends on sky position.
    Left at the default (`background=None`, i.e. `uvex.detector`'s own
    Galactic + Zodiacal background), SNR must actually change with `obstime`, which
    both confirms the override in the first branch really took effect and that the
    method's `obstime` placeholder default is inert only for backgrounds that don't
    use it.
    """
    model = VanVelzenTDESED()
    coord = SkyCoord(ra=200 * u.deg, dec=-10 * u.deg)
    params = {name: value[0] for name, value in model.sample_parameters(1, rng=3).items()}

    obstime_a = Time("2025-01-01T00:00:00", scale="utc")
    obstime_b = Time("2025-07-01T00:00:00", scale="utc")

    def snr_at(obstime, background):
        phot = model.simulate_photometry(
            10 * u.day,
            900 * u.s,
            uvex.detector,
            coord,
            background=background,
            obstime=obstime,
            redshift=0.05,
            rng=0,
            **params,
        )
        return phot["snr"][0]

    snr_galactic_a = snr_at(obstime_a, GalacticBackground())
    snr_galactic_b = snr_at(obstime_b, GalacticBackground())
    np.testing.assert_allclose(snr_galactic_a, snr_galactic_b, rtol=1e-10)

    snr_default_a = snr_at(obstime_a, None)
    snr_default_b = snr_at(obstime_b, None)
    assert not np.isclose(snr_default_a, snr_default_b, rtol=1e-6)
    assert not np.isclose(snr_default_a, snr_galactic_a, rtol=1e-6)


def test_simulate_photometry_unknown_band_raises():
    """A `bands` entry `detector` doesn't have raises, regardless of `background`."""
    model = VanVelzenTDESED()
    coord = SkyCoord(ra=10 * u.deg, dec=-10 * u.deg)
    params = {name: value[0] for name, value in model.sample_parameters(1, rng=4).items()}

    with pytest.raises(ValueError, match="Unknown bandpass"):
        model.simulate_photometry(
            10 * u.day,
            900 * u.s,
            uvex.detector,
            coord,
            bands=["not-a-real-band"],
            redshift=0.05,
            **params,
        )


def test_simulate_photometry_requires_scalar_coord():
    """A non-scalar `coord` raises rather than silently doing something batched-looking."""
    model = VanVelzenTDESED()
    coord = SkyCoord(ra=[10, 20] * u.deg, dec=[-10, -20] * u.deg)
    params = {name: value[0] for name, value in model.sample_parameters(1, rng=5).items()}

    with pytest.raises(ValueError, match="scalar"):
        model.simulate_photometry(10 * u.day, 900 * u.s, uvex.detector, coord, redshift=0.05, **params)


def test_simulate_photometry_exptime_shape_mismatch_raises():
    """`exptime` must be scalar or exactly match `t`'s shape."""
    model = VanVelzenTDESED()
    coord = SkyCoord(ra=10 * u.deg, dec=-10 * u.deg)
    params = {name: value[0] for name, value in model.sample_parameters(1, rng=6).items()}

    t = [1, 5, 10] * u.day
    exptime = [900, 900] * u.s  # wrong length

    with pytest.raises(ValueError, match="exptime"):
        model.simulate_photometry(t, exptime, uvex.detector, coord, redshift=0.05, **params)


def test_simulate_photometry_sys_err_widens_uncertainty_and_scatter():
    """
    `sys_err` inflates `snr`/`flux_err`/`mag_err` in quadrature and is drawn into `flux` itself.

    Checked two ways: analytically, that `mag_err` with `sys_err` matches
    `sqrt(mag_err_without**2 + sys_err**2)` combined in quadrature; and empirically,
    over many independent draws (different `rng` per draw, same `t`), that the sample
    standard deviation of `flux` widens along with it -- confirming the systematic
    floor is folded into the noise realization, not just reported as a wider bar
    around an unchanged draw.
    """
    model = VanVelzenTDESED()
    coord = SkyCoord(ra=80 * u.deg, dec=15 * u.deg)
    params = {name: value[0] for name, value in model.sample_parameters(1, rng=8).items()}
    sys_err = 0.05

    def simulate(rng):
        return model.simulate_photometry(
            10 * u.day,
            900 * u.s,
            uvex.detector,
            coord,
            bands=["FUV"],
            background=GalacticBackground(),
            redshift=0.05,
            sys_err=sys_err,
            rng=rng,
            **params,
        )

    def simulate_baseline(rng):
        return model.simulate_photometry(
            10 * u.day,
            900 * u.s,
            uvex.detector,
            coord,
            bands=["FUV"],
            background=GalacticBackground(),
            redshift=0.05,
            rng=rng,
            **params,
        )

    phot_without = simulate_baseline(rng=0)
    phot_with = simulate(rng=0)

    expected_mag_err = np.hypot(phot_without["mag_err"][0], sys_err)
    np.testing.assert_allclose(phot_with["mag_err"][0], expected_mag_err, rtol=1e-6)
    assert phot_with["snr"][0] < phot_without["snr"][0]
    assert phot_with["flux_err"][0] > phot_without["flux_err"][0]

    fluxes_without = np.array([simulate_baseline(rng=i)["flux"][0].value for i in range(50)])
    fluxes_with = np.array([simulate(rng=i)["flux"][0].value for i in range(50)])
    assert np.std(fluxes_with) > np.std(fluxes_without)


def test_simulate_photometry_sys_err_mapping_is_per_band():
    """A `Mapping` applies a different floor per band, matching the equivalent per-band scalar calls."""
    model = VanVelzenTDESED()
    coord = SkyCoord(ra=80 * u.deg, dec=15 * u.deg)
    params = {name: value[0] for name, value in model.sample_parameters(1, rng=9).items()}
    sys_err = {"FUV": 0.01, "NUV": 0.05}

    phot = model.simulate_photometry(
        10 * u.day,
        900 * u.s,
        uvex.detector,
        coord,
        background=GalacticBackground(),
        redshift=0.05,
        sys_err=sys_err,
        rng=0,
        **params,
    )

    for band, floor in sys_err.items():
        phot_scalar = model.simulate_photometry(
            10 * u.day,
            900 * u.s,
            uvex.detector,
            coord,
            bands=[band],
            background=GalacticBackground(),
            redshift=0.05,
            sys_err=floor,
            rng=0,
            **params,
        )
        row = phot[phot["band"] == band]
        np.testing.assert_allclose(row["mag_err"][0], phot_scalar["mag_err"][0], rtol=1e-10)


def test_simulate_photometry_sys_err_mapping_missing_band_raises():
    """A `Mapping` missing an entry for a requested band raises, rather than silently using zero."""
    model = VanVelzenTDESED()
    coord = SkyCoord(ra=80 * u.deg, dec=15 * u.deg)
    params = {name: value[0] for name, value in model.sample_parameters(1, rng=10).items()}

    with pytest.raises(ValueError, match="sys_err"):
        model.simulate_photometry(
            10 * u.day,
            900 * u.s,
            uvex.detector,
            coord,
            bands=["FUV", "NUV"],
            background=GalacticBackground(),
            redshift=0.05,
            sys_err={"FUV": 0.01},
            **params,
        )


def test_simulate_photometry_detector_is_not_mutated():
    """`background=` must never leak into the caller's own `detector` object."""
    model = VanVelzenTDESED()
    coord = SkyCoord(ra=10 * u.deg, dec=-10 * u.deg)
    params = {name: value[0] for name, value in model.sample_parameters(1, rng=7).items()}
    original_background = uvex.detector.background

    model.simulate_photometry(
        10 * u.day,
        900 * u.s,
        uvex.detector,
        coord,
        background=GalacticBackground(),
        redshift=0.05,
        **params,
    )

    assert uvex.detector.background is original_background
