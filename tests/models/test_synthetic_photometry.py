"""
This test suite verifies that the model module can plug into the synphot machinery.
"""

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from m4opt.missions._uvex import uvex
from m4opt.skygrid import _geodesic
from m4opt.synphot import observing
from m4opt.synphot.background import GalacticBackground

from uvex_transients.models import VanVelzenTDESED, VillarCoolingBlackbodySED
from uvex_transients.utils.keyed_noise import keyed_standard_normal, time_key


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
    Per-event dust extinction via `as_source_spectrum`'s `ebv`, fully batched.

    Multiplying a batched `SourceSpectrum` by a separate `DustExtinction()`
    `SpectralElement` routes through `m4opt.synphot._math.countrate`'s
    per-Ebv interpolation shortcut, which chokes on a per-event batch axis
    (a known, separate m4opt limitation -- see
    `test_synthetic_photometry_batches_events_and_times`'s docstring).
    Instead, `as_source_spectrum` resolves `ebv` (each event's own E(B-V)) to
    `uvex_transients.dust.log_attenuation` internally and folds it directly into
    the flux -- as plain NumPy arithmetic inside its own evaluation kernel --
    before the `SourceSpectrum` is ever built. That keeps the whole thing one
    ordinary broadcast, so it batches over events exactly as cleanly as every
    other parameter already does in `test_synthetic_photometry_batches_events_and_times`.
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
        ebv=ebv,
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
        ebv=ebv[i_event],
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
            true_flux = float(np.squeeze(spectrum(uvex.detector.bandpasses[band].pivot(), flux_unit=u.Jy).value))
            row = phot[(phot["band"] == band) & (phot["t"] == t_i)]
            assert len(row) == 1
            # The uncertainty is the true flux over the detector's expected SNR.
            np.testing.assert_allclose(row["flux_err"][0].to_value(u.Jy), true_flux / expected_snr, rtol=1e-6)
            # The reported `snr` is the measured one: the noisy flux over its uncertainty.
            measured_snr = row["flux"][0].to_value(u.Jy) / row["flux_err"][0].to_value(u.Jy)
            np.testing.assert_allclose(row["snr"][0], measured_snr, rtol=1e-10)


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
        return phot["flux_err"][0].to_value(u.Jy)  # the noise level, set by the background

    snr_galactic_a = snr_at(obstime_a, GalacticBackground())
    snr_galactic_b = snr_at(obstime_b, GalacticBackground())
    np.testing.assert_allclose(snr_galactic_a, snr_galactic_b, rtol=1e-10)

    snr_default_a = snr_at(obstime_a, None)
    snr_default_b = snr_at(obstime_b, None)
    assert not np.isclose(snr_default_a, snr_default_b, rtol=1e-6, atol=0)
    assert not np.isclose(snr_default_a, snr_galactic_a, rtol=1e-6, atol=0)


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


def test_simulate_photometry_per_epoch_arrays_must_match_t():
    """A per-epoch `coord` (or observer location, or time) must be scalar or shaped like `t`."""
    model = VanVelzenTDESED()
    coord = SkyCoord(ra=[10, 20] * u.deg, dec=[-10, -20] * u.deg)
    params = {name: value[0] for name, value in model.sample_parameters(1, rng=5).items()}

    with pytest.raises(ValueError, match="coord"):
        model.simulate_photometry([1, 5, 10] * u.day, 900 * u.s, uvex.detector, coord, redshift=0.05, **params)


def test_simulate_photometry_is_vectorized_over_events():
    """
    One call over epochs of different events equals separate calls, one per event.

    Each epoch has its own position, redshift, model parameters and noise seed; the keyed noise
    means an epoch's measurement cannot depend on which other epochs are in the call.
    """
    model = VanVelzenTDESED()
    sampled = model.sample_parameters(2, rng=7)
    coords = SkyCoord(ra=[150, 30] * u.deg, dec=[20, 5] * u.deg)
    redshift = np.array([0.05, 0.08])
    seeds = np.array([11, 22], dtype=np.uint64)

    # Three epochs of event 0 and two of event 1, interleaved in time. Some are before the explosion,
    # so each event has both modelled and background-only epochs, at its own position.
    event = np.array([0, 1, 0, 1, 0])
    t = np.array([2, -3, 10, 15, -1]) * u.day
    obstime = Time("2025-01-01T00:00:00", scale="utc") + t
    location = uvex.observer_location(obstime)
    keys = time_key(obstime)

    batched = model.simulate_photometry(
        t,
        900 * u.s,
        uvex.detector,
        coords[event],
        observer_location=location,
        obstime=obstime,
        redshift=redshift[event],
        noise_seed=seeds[event],
        noise_observation_keys=keys,
        **{name: value[event] for name, value in sampled.items()},
    )

    for i in (0, 1):
        rows = np.flatnonzero(event == i)
        single = model.simulate_photometry(
            t[rows],
            900 * u.s,
            uvex.detector,
            coords[i],
            observer_location=location[rows],
            obstime=obstime[rows],
            redshift=redshift[i],
            noise_seed=int(seeds[i]),
            noise_observation_keys=keys[rows],
            **{name: value[i] for name, value in sampled.items()},
        )
        for band in uvex.detector.bandpasses:
            for column in ("flux", "flux_err", "snr"):
                mine = np.isin(batched["t"].to_value(u.day), t[rows].to_value(u.day))
                got = batched[(batched["band"] == band) & mine][column]
                want = single[single["band"] == band][column]
                np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


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


# =========================================================================== #
# SpectralModel.measure_photometry                                           #
# =========================================================================== #
def _measure_setup(n):
    """A TDE model, a sky position, and ``n`` epochs spread over 200 days, with their observing geometry."""
    model = VanVelzenTDESED()
    params = {name: value[0] for name, value in model.sample_parameters(1, rng=8).items()}
    t = np.linspace(1, 200, n) * u.day
    obstime = Time("2025-01-01T00:00:00", scale="utc") + t
    return model, params, SkyCoord(ra=150 * u.deg, dec=20 * u.deg), t, obstime, uvex.observer_location(obstime)


def test_measure_photometry_noise_is_exactly_the_keyed_draw():
    """With one band, measured minus expected SNR is `keyed_standard_normal` of (seed, start time, band)."""
    model, params, coord, t, obstime, location = _measure_setup(400)
    band = list(uvex.detector.bandpasses)[-1]

    flux, flux_err, snr, snr_expected = model.measure_photometry(
        t,
        900 * u.s,
        uvex.detector,
        coord,
        bands=[band],
        observer_location=location,
        obstime=obstime,
        redshift=0.05,
        noise_seed=42,
        noise_observation_keys=time_key(obstime),
        **params,
    )

    assert np.all(np.isfinite(snr))
    draw = keyed_standard_normal(42, time_key(obstime), list(uvex.detector.bandpasses).index(band))
    np.testing.assert_allclose(snr[0] - snr_expected[0], draw, atol=1e-6)
    np.testing.assert_allclose(flux[0] / flux_err[0], snr[0], rtol=1e-10)

    # Across many epochs those draws are standard normal.
    assert abs(draw.mean()) < 5 / np.sqrt(len(draw))
    assert abs(draw.std() - 1) < 5 / np.sqrt(2 * len(draw))


def test_measure_photometry_distinct_seeds_give_distinct_noise():
    """Two events measured at the same instants do not share a noise draw."""
    model, params, coord, t, obstime, location = _measure_setup(50)
    common = dict(
        observer_location=location,
        obstime=obstime,
        redshift=0.05,
        noise_observation_keys=time_key(obstime),
        bands=[list(uvex.detector.bandpasses)[0]],
        **params,
    )
    *_, snr_a, expected_a = model.measure_photometry(t, 900 * u.s, uvex.detector, coord, noise_seed=1, **common)
    *_, snr_b, expected_b = model.measure_photometry(t, 900 * u.s, uvex.detector, coord, noise_seed=2, **common)

    noise_a, noise_b = (snr_a - expected_a)[0], (snr_b - expected_b)[0]
    assert not np.any(np.isclose(noise_a, noise_b))


def test_pre_explosion_epochs_are_background_only_by_default():
    """Epochs with ``t < 0`` are never evaluated against the model: expected SNR is nil, flagged not `in_model`."""
    model, params, coord, _, _, _ = _measure_setup(3)
    t = [-20, -5, 10] * u.day

    phot = model.simulate_photometry(t, 900 * u.s, uvex.detector, coord, redshift=0.05, rng=0, **params)

    assert phot["in_model"][phot["t"] < 0 * u.day].tolist() == [False] * 2 * len(uvex.detector.bandpasses)
    assert np.all(phot["in_model"][phot["t"] > 0 * u.day])

    _, _, _, snr_expected = model.measure_photometry(t, 900 * u.s, uvex.detector, coord, redshift=0.05, rng=0, **params)
    assert np.all(np.abs(snr_expected[:, :2]) < 1e-6)  # pure background: no source
    assert np.all(snr_expected[:, 2] > 1)  # the one real epoch is a real measurement


def test_in_model_overrides_the_default():
    """A caller can mark a ``t >= 0`` epoch as background (e.g. an exposure running past its window)."""
    model, params, coord, _, _, _ = _measure_setup(3)
    t = [5, 10, 15] * u.day

    _, _, _, snr_expected = model.measure_photometry(
        t, 900 * u.s, uvex.detector, coord, redshift=0.05, in_model=[True, False, True], rng=0, **params
    )
    assert np.all(np.abs(snr_expected[:, 1]) < 1e-6)
    assert np.all(snr_expected[:, [0, 2]] > 1)

    with pytest.raises(ValueError, match="in_model"):
        model.measure_photometry(t, 900 * u.s, uvex.detector, coord, redshift=0.05, in_model=[True], **params)
