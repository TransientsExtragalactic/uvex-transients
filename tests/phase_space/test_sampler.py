"""Tests for `PhaseSpaceSampler`, on synthetic recovery surfaces with a known truth."""

import numpy as np
import pytest
from scipy.special import expit

pytest.importorskip("sklearn")

from uvex_transients.phase_space.sampler import PhaseSpaceSampler, _logit_data  # noqa: E402

BOUNDS = [(0.0, 1.0), (0.0, 1.0)]
NAMES = ("a", "b")


def truth(x):
    """A sharp, curved edge that saturates at 0.9 and falls to a small floor."""
    a, b = x[..., 0], x[..., 1]
    return np.clip(0.9 * expit(10.0 * (a - 0.5 - 0.2 * np.sin(3.0 * b))), 1e-5, None)


def make_evaluator(seed=0, chunk=1000, k_target=30, n_max=20000):
    """Return a binomial evaluator that tops up in chunks, like `RecoveryEvaluator`."""
    rng = np.random.default_rng(seed)

    def evaluate(x):
        p = float(truth(np.asarray(x)))
        k = n = 0
        while n < n_max and k < k_target:
            n += chunk
            k += rng.binomial(chunk, p)
        return {"k": k, "n": n, "extra": 2.5}

    return evaluate


def make_sampler(**kwargs):
    defaults = {
        "names": NAMES,
        "n_initial": 12,
        "batch_size": 3,
        "max_points": 24,
        "tolerance": 0.0,
        "n_candidates": 256,
        "n_restarts": 0,
        "seed": 1,
    }
    defaults.update(kwargs)
    return PhaseSpaceSampler(make_evaluator(), BOUNDS, **defaults)


def test_logit_data_is_finite_at_the_extremes_and_matches_the_delta_method():
    y, variance = _logit_data(np.array([0, 50, 1000]), np.array([1000, 100, 1000]))
    assert np.all(np.isfinite(y)) and np.all(np.isfinite(variance))
    p_hat = 50.5 / 101.0
    assert y[1] == pytest.approx(np.log(p_hat / (1 - p_hat)))
    assert variance[1] == pytest.approx(1.0 / (101.0 * p_hat * (1 - p_hat)))
    # At small p the logit noise is the Poisson relative error of the count, about 1/k.
    assert variance[0] == pytest.approx(1.0 / (1001.0 * (0.5 / 1001.0) * (1 - 0.5 / 1001.0)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bounds": [(0, 1), (1, 1)]},
        {"bounds": [(0, 1)]},
        {"names": ("a", "a")},
        {"acquisition": "nonsense"},
        {"floor": 0.0},
        {"batch_size": 0},
    ],
)
def test_invalid_arguments_raise(kwargs):
    arguments = {"evaluate": make_evaluator(), "bounds": BOUNDS, "names": NAMES}
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        PhaseSpaceSampler(**arguments)


def test_evaluate_must_report_k_and_n():
    sampler = PhaseSpaceSampler(lambda x: {"k": 1}, BOUNDS, names=NAMES, n_initial=4)
    with pytest.raises(KeyError, match="'n'"):
        sampler.run()


@pytest.mark.parametrize("acquisition", ["variance", "imse"])
def test_run_stops_at_max_points_and_records_every_entry(acquisition):
    sampler = make_sampler(acquisition=acquisition).run()
    table = sampler.table
    assert len(sampler) == len(table) == 24
    assert set(table.colnames) == {"a", "b", "k", "n", "extra"}
    assert np.all(table["k"] <= table["n"])
    lo, hi = np.array(BOUNDS).T
    points = np.column_stack([table["a"], table["b"]])
    assert np.all((points >= lo) & (points <= hi))
    assert table.meta["names"] == list(NAMES)


@pytest.mark.parametrize("acquisition", ["variance", "imse"])
def test_the_fitted_map_recovers_a_known_surface(acquisition):
    sampler = make_sampler(acquisition=acquisition, n_initial=16, max_points=40).run()
    grid = np.stack(np.meshgrid(np.linspace(0, 1, 25), np.linspace(0, 1, 25)), axis=-1).reshape(-1, 2)
    fitted = sampler.recovery(grid)["p"]
    error = np.sqrt(np.mean((np.log(fitted + 1e-3) - np.log(truth(grid) + 1e-3)) ** 2))
    assert error < 0.3


def test_the_interval_brackets_the_median():
    sampler = make_sampler(max_points=14).run()
    result = sampler.recovery(np.random.default_rng(0).random((20, 2)))
    assert np.all(result["p_lo"] <= result["p"]) and np.all(result["p"] <= result["p_hi"])
    assert np.all(result["sigma_logit"] > 0)


def test_a_batch_has_distinct_points_inside_the_bounds():
    sampler = make_sampler(max_points=12)
    sampler.run()
    points, uncertainty = sampler.propose(5)
    assert points.shape == (5, 2)
    assert len({tuple(point) for point in points}) == 5
    assert np.all((points >= 0.0) & (points <= 1.0))
    assert uncertainty > 0


def test_a_loose_tolerance_stops_after_the_initial_design():
    sampler = make_sampler(tolerance=1e6, max_points=60).run()
    assert len(sampler) == 12


def test_save_and_resume_continue_from_the_stored_points(tmp_path):
    first = make_sampler(max_points=12).run()
    path = tmp_path / "points.ecsv"
    first.save(path)

    resumed = PhaseSpaceSampler.load(
        path,
        make_evaluator(seed=5),
        n_initial=12,
        batch_size=3,
        max_points=18,
        tolerance=0.0,
        n_candidates=256,
        n_restarts=0,
    )
    assert len(resumed) == 12
    assert resumed.names == NAMES
    np.testing.assert_allclose(resumed.bounds, np.array(BOUNDS))
    resumed.run()
    assert len(resumed) == 18
    # The stored points are kept as they were.
    np.testing.assert_array_equal(resumed.table["k"][:12], first.table["k"])


# ---------------------------------------------------------------------------- #
# Saving and loading                                                           #
# ---------------------------------------------------------------------------- #
def test_the_file_records_the_settings_and_load_restores_them(tmp_path):
    sampler = make_sampler(max_points=12, tolerance=0.5, acquisition="variance", floor=2e-3)
    sampler.run()
    path = tmp_path / "points.ecsv"
    sampler.save(path)

    loaded = PhaseSpaceSampler.load(path)
    assert loaded.settings == sampler.settings
    assert loaded.names == NAMES
    assert len(loaded) == 12
    # Nothing was recorded about an evaluator, because this one does not describe itself.
    assert "evaluator" not in sampler.table.meta


def test_a_loaded_sampler_predicts_but_cannot_simulate(tmp_path):
    sampler = make_sampler(max_points=12).run()
    path = tmp_path / "points.ecsv"
    sampler.save(path)

    loaded = PhaseSpaceSampler.load(path, n_restarts=0, seed=1)
    x = np.random.default_rng(0).random((10, 2))
    np.testing.assert_allclose(loaded.recovery(x)["p"], sampler.recovery(x)["p"], rtol=1e-6)
    assert loaded.evaluate_recovery_map(x[:, 0], x[:, 1]).shape == (10,)

    with pytest.raises(RuntimeError, match="no evaluator"):
        loaded.max_points = 14
        loaded.run()


def test_maps_that_need_scales_say_so_without_an_evaluator():
    sampler = make_sampler(max_points=12).run()
    for method in (
        sampler.evaluate_efficiency_map,
        sampler.evaluate_recovered_volume_time_map,
        sampler.evaluate_coverage_map,
    ):
        with pytest.raises(RuntimeError, match="needs an evaluator"):
            method(0.5, 0.5)
    with pytest.raises(RuntimeError, match="needs an evaluator"):
        sampler.map_table(5)


def test_the_recovery_map_broadcasts_and_the_interval_brackets_the_median():
    sampler = make_sampler(max_points=12).run()
    log_l, log_dt = np.linspace(0.1, 0.9, 4)[:, None], np.linspace(0.1, 0.9, 3)[None, :]
    median, lower, upper = sampler.evaluate_recovery_map(log_l, log_dt, interval=True)
    assert median.shape == lower.shape == upper.shape == (4, 3)
    assert np.all(lower <= median) and np.all(median <= upper)
    np.testing.assert_array_equal(sampler.evaluate_recovery_map(log_l, log_dt), median)
    assert sampler.evaluate_uncertainty_map(log_l, log_dt).shape == (4, 3)
    assert np.ndim(sampler.evaluate_recovery_map(0.3, 0.4)) == 0
