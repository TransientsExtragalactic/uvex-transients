"""Tests for `ExtragalacticTransient.sample_events_on_healpix_grid`'s `sampling_factor` option."""

import astropy_healpix as ah
import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time

from uvex_transients.transients.TDEs import TidalDisruptionEvent
from uvex_transients.utils import get_seed_sequence

_KWARGS = {
    "t_start": Time("2030-01-01"),
    "duration": 30 * u.day,
    "pixel_ids": np.arange(0, 3000),
    "seed": 11,
}
_NSIDE = 16


@pytest.fixture(scope="module")
def transient():
    return TidalDisruptionEvent()


def _expected_events(transient, duration):
    """Mean of the Poisson event count over `_KWARGS`' pixels for a window of `duration`."""
    solid_angle = len(_KWARGS["pixel_ids"]) * ah.nside_to_pixel_area(_NSIDE)
    return (transient.integrated_rate * solid_angle * duration).to_value(u.dimensionless_unscaled)


def _counts(transient, duration, sampling_factor, n_draws, seed=0):
    """`n_draws` independent event counts at `sampling_factor`, from `sample_event_count`."""
    solid_angle = len(_KWARGS["pixel_ids"]) * ah.nside_to_pixel_area(_NSIDE)
    seeds = np.random.SeedSequence(seed).spawn(n_draws)
    return np.array(
        [
            transient.sample_event_count(
                solid_angle=solid_angle, duration=duration, seed=s, sampling_factor=sampling_factor
            )
            for s in seeds
        ]
    )


@pytest.mark.parametrize("factor", [0.05, 0.5, 1.0, 2.0])
def test_event_count_is_poisson_with_the_scaled_mean(transient, factor):
    """The count is Poisson, with a mean `sampling_factor` times the physical expectation."""
    n_draws = 400
    mean = factor * _expected_events(transient, 30 * u.day)
    counts = _counts(transient, 30 * u.day, factor, n_draws)

    # The sample mean is within 4 standard errors of the scaled expectation, and the variance ~ the mean.
    assert abs(counts.mean() - mean) < 4 * np.sqrt(mean / n_draws)
    assert counts.var(ddof=1) == pytest.approx(mean, rel=0.4)


def test_none_is_the_physical_rate(transient):
    """`sampling_factor=None` and `1.0` are the same draw as not passing it at all."""
    default = transient.sample_events_on_healpix_grid(_NSIDE, **_KWARGS)
    none = transient.sample_events_on_healpix_grid(_NSIDE, sampling_factor=None, **_KWARGS)
    one = transient.sample_events_on_healpix_grid(_NSIDE, sampling_factor=1.0, **_KWARGS)
    np.testing.assert_array_equal(default["parameter_seed"], none["parameter_seed"])
    np.testing.assert_array_equal(default["parameter_seed"], one["parameter_seed"])


@pytest.mark.parametrize("factor", [0.1, 3.0])
def test_table_length_is_the_scaled_count_draw(transient, factor):
    """The table holds exactly the events the scaled Poisson count asks for, no more and no fewer."""
    table = transient.sample_events_on_healpix_grid(_NSIDE, sampling_factor=factor, **_KWARGS)
    count_seed = get_seed_sequence(_KWARGS["seed"]).spawn(6)[0]
    solid_angle = len(_KWARGS["pixel_ids"]) * ah.nside_to_pixel_area(_NSIDE)
    n = transient.sample_event_count(
        solid_angle=solid_angle, duration=_KWARGS["duration"], seed=count_seed, sampling_factor=factor
    )
    assert len(table) == n > 0
    assert len(np.unique(table["parameter_seed"])) == n


def test_oversampling_draws_more_events_than_the_physical_rate(transient):
    """A `sampling_factor` above 1 oversamples."""
    physical = _counts(transient, 30 * u.day, 1.0, 200).mean()
    doubled = _counts(transient, 30 * u.day, 2.0, 200).mean()
    assert doubled / physical == pytest.approx(2.0, rel=0.1)


def test_sampling_factor_is_reproducible(transient):
    first = transient.sample_events_on_healpix_grid(_NSIDE, sampling_factor=0.5, **_KWARGS)
    second = transient.sample_events_on_healpix_grid(_NSIDE, sampling_factor=0.5, **_KWARGS)
    np.testing.assert_array_equal(first["parameter_seed"], second["parameter_seed"])
    np.testing.assert_array_equal(first["redshift"], second["redshift"])


def test_uniform_weight_recovers_a_bin_dependent_selection(transient):
    """
    The type-wide weight ``expected / n_generated`` is unbiased when the selected fraction varies by bin.

    A per-bin round-up of the kept count (``ceil(N / k)``, the earlier ``downsample``) failed this: bins
    with few events kept all of them, so events from different bins stood for different numbers of real
    ones and one weight over- or under-counted whichever bins the selection favoured. Thinning the
    Poisson rate itself keeps the inclusion probability the same everywhere. The footprint is small, so
    that bins hold only a handful of events at these factors (the regime of a rare type), where the
    round-up was biased by several percent.
    """
    solid_angle = 30 * ah.nside_to_pixel_area(_NSIDE)
    durations = np.linspace(1, 60, 24) * u.day  # bins holding very different numbers of events
    expected = np.array(
        [(transient.integrated_rate * solid_angle * d).to_value(u.dimensionless_unscaled) for d in durations]
    )
    # A selection that favours the bins holding the most events (as a larger observed footprint would).
    selected_fraction = np.linspace(0.02, 0.9, len(durations))
    truth = (expected * selected_fraction).sum()

    n_draws = 1000

    def z_score(estimates):
        return (estimates.mean() - truth) / (estimates.std(ddof=1) / np.sqrt(len(estimates)))

    for factor in (0.05, 0.1):
        thinned, round_up = [], []
        for draw in range(n_draws):
            rng = np.random.default_rng(draw)
            seeds = np.random.SeedSequence(1000 + draw).spawn(len(durations))
            n_generated = np.array(
                [
                    transient.sample_event_count(solid_angle=solid_angle, duration=d, seed=s, sampling_factor=factor)
                    for d, s in zip(durations, seeds)
                ]
            )
            n_selected = rng.binomial(n_generated, selected_fraction).sum()
            thinned.append(expected.sum() * n_selected / n_generated.sum())  # expected / n_generated, per event

            # The earlier scheme, as a negative control: a round-up on the physical draw.
            n_kept = -(-rng.poisson(expected) // round(1 / factor))
            round_up.append(expected.sum() * rng.binomial(n_kept, selected_fraction).sum() / n_kept.sum())

        assert abs(z_score(np.array(thinned))) < 4, factor
        assert abs(z_score(np.array(round_up))) > 4, factor  # the setup can tell the two schemes apart


@pytest.mark.parametrize("bad", [0, -2, -0.5, True, False, float("nan"), float("inf"), "2"])
def test_bad_sampling_factor_raises(transient, bad):
    with pytest.raises(ValueError, match="sampling_factor"):
        transient.sample_events_on_healpix_grid(_NSIDE, sampling_factor=bad, **_KWARGS)
    with pytest.raises(ValueError, match="sampling_factor"):
        transient.sample_event_count(solid_angle=1 * u.sr, duration=1 * u.day, seed=0, sampling_factor=bad)
