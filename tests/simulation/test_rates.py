"""Tests for `uvex_transients.simulation.rates.estimate_yield`."""

import numpy as np
import pytest
from astropy.table import QTable

from uvex_transients.simulation._stats import clopper_pearson_interval
from uvex_transients.simulation.rates import estimate_yield
from uvex_transients.utils import config


def _summary(types, **meta):
    table = QTable()
    table["transient_type"] = np.asarray(types)
    table["x"] = np.arange(len(types))
    table.meta.update(
        {
            "n_pre_cut": {"a": 100, "b": 40, "c": 10},
            "expected_events": {"a": 1000.0, "b": 200.0, "c": 50.0},
            "rate_ci": {"a": [0.5, 2.0], "b": [1.0, 1.0], "c": [0.8, 1.5]},
            **meta,
        }
    )
    return table


@pytest.fixture
def summary():
    # a: 6 rows, b: 3 rows, c: none (generated but never detected)
    return _summary(["a"] * 6 + ["b"] * 3)


def _row(result, name):
    return result[result["transient_type"] == name][0]


def test_one_row_per_generated_type_including_empty_ones(summary):
    result = estimate_yield(summary)
    assert list(result["transient_type"]) == ["a", "b", "c"]
    assert list(result["n_pre_cut"]) == [100, 40, 10]
    assert list(result["n_selected"]) == [6, 3, 0]


def test_fraction_and_expected_events(summary):
    a = _row(estimate_yield(summary), "a")
    assert a["fraction"] == pytest.approx(0.06)
    assert a["intrinsic_events"] == 1000.0
    assert a["expected_events"] == pytest.approx(60.0)


def test_confidence_defaults_to_the_configured_value(summary):
    default = config["simulation.default_confidence"]
    assert estimate_yield(summary).meta["confidence"] == default
    assert estimate_yield(summary, confidence=None).meta["confidence"] == default


def test_binomial_bounds_are_clopper_pearson_propagated_through_mu0(summary):
    result = estimate_yield(summary, confidence=0.8)
    lower, upper = clopper_pearson_interval(6, 100, 0.8)
    a = _row(result, "a")
    assert a["fraction_lower"] == pytest.approx(lower)
    assert a["fraction_upper"] == pytest.approx(upper)
    assert a["expected_events_binom_lower"] == pytest.approx(1000.0 * lower)
    assert a["expected_events_binom_upper"] == pytest.approx(1000.0 * upper)
    assert result.meta["confidence"] == 0.8


def test_rate_bounds_scale_the_point_estimate_by_the_rate_factors(summary):
    result = estimate_yield(summary)
    a, b = _row(result, "a"), _row(result, "b")
    assert a["expected_events_rate_lower"] == pytest.approx(0.5 * a["expected_events"])
    assert a["expected_events_rate_upper"] == pytest.approx(2.0 * a["expected_events"])
    assert b["expected_events_rate_lower"] == pytest.approx(b["expected_events"])


def test_a_type_with_no_selected_rows_gets_zero_with_a_nonzero_upper_bound(summary):
    c = _row(estimate_yield(summary), "c")
    assert c["fraction"] == 0.0
    assert c["expected_events"] == 0.0
    assert c["fraction_lower"] == 0.0
    assert c["fraction_upper"] > 0.0
    assert c["expected_events_binom_upper"] > 0.0


def test_mask_counts_only_selected_rows(summary):
    selection = np.asarray(summary["x"]) % 2 == 0  # rows 0, 2, 4 (all a) and 6, 8 (b)
    result = estimate_yield(summary, mask=selection)
    assert list(result["n_selected"]) == [3, 2, 0]
    assert _row(result, "a")["expected_events"] == pytest.approx(30.0)


def test_denominator_ignores_how_the_table_was_sliced(summary):
    """`n_pre_cut` is the generation count, so cutting rows changes only the numerator."""
    sliced = summary[np.asarray(summary["x"]) >= 3]
    assert sliced.meta["n_pre_cut"] == summary.meta["n_pre_cut"]
    result = estimate_yield(sliced)
    assert list(result["n_pre_cut"]) == [100, 40, 10]
    assert list(result["n_selected"]) == [3, 3, 0]


def test_transient_types_restricts_and_validates(summary):
    result = estimate_yield(summary, transient_types=["b"])
    assert list(result["transient_type"]) == ["b"]
    with pytest.raises(ValueError, match="Unknown transient type"):
        estimate_yield(summary, transient_types=["nope"])


@pytest.mark.parametrize("key", ["n_pre_cut", "expected_events", "rate_ci"])
def test_missing_meta_raises(summary, key):
    del summary.meta[key]
    with pytest.raises(ValueError, match=key):
        estimate_yield(summary)


@pytest.mark.parametrize("selection", [np.ones(3, dtype=bool), np.arange(9)])
def test_mask_must_be_a_boolean_array_of_the_right_length(summary, selection):
    with pytest.raises(ValueError, match="'mask'"):
        estimate_yield(summary, mask=selection)


def test_more_selected_rows_than_generated_events_raises():
    table = _summary(["c"] * 11)
    with pytest.raises(ValueError, match="only 10 were generated"):
        estimate_yield(table)
