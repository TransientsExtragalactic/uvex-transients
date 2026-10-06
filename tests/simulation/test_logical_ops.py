"""Tests for `uvex_transients.simulation.logical_ops`."""

from dataclasses import replace

import numpy as np
import pytest

from uvex_transients.simulation.logical_ops import LOGICAL_OP_ARITY, LOGICAL_OPS, difference, intersection, union

from .test_core import _make_catalog


def _subset(catalog, ids):
    """Build a new catalog restricted to the given `event_id`s, preserving row order in `catalog`."""
    mask = np.isin(np.asarray(catalog.table["event_id"]), list(ids))
    return type(catalog)(
        table=catalog.table[mask],
        nside=catalog.nside,
        order=catalog.order,
        time_bins=catalog.time_bins,
        seed=catalog.seed,
        downsample=catalog.downsample,
        pre_cut_counts=catalog.pre_cut_counts,
    )


class _DummyTransient:
    """A stand-in with just the `.cosmology` attribute `_make_catalog` needs."""

    from astropy.cosmology import Planck18 as cosmology


def _assert_metadata_carried_from_first(result, first):
    """Every `logical_ops` function must carry `catalogs[0]`'s own metadata through unchanged."""
    assert result.nside == first.nside
    assert result.order == first.order
    assert np.array_equal(result.time_bins.jd, first.time_bins.jd)
    assert result.seed == first.seed
    assert result.downsample == first.downsample
    assert result.pre_cut_counts == first.pre_cut_counts


# --------------------------------------------------------------------------- #
# union                                                                       #
# --------------------------------------------------------------------------- #
def test_union_deduplicates_by_event_id(hot_spot):
    """`union` combines every input's rows, keeping each `event_id` only once."""
    catalog, *_ = _make_catalog(_DummyTransient(), hot_spot, n_events=10, seed=1)

    a = _subset(catalog, range(0, 6))
    b = _subset(catalog, range(4, 10))

    combined = union([a, b])
    assert set(np.asarray(combined.table["event_id"])) == set(range(10))
    assert len(combined) == 10
    _assert_metadata_carried_from_first(combined, a)


def test_union_is_nary(hot_spot):
    """`union` accepts more than two inputs at once, not just a pair."""
    catalog, *_ = _make_catalog(_DummyTransient(), hot_spot, n_events=12, seed=6)

    a = _subset(catalog, range(0, 4))
    b = _subset(catalog, range(3, 8))
    c = _subset(catalog, range(6, 12))

    combined = union([a, b, c])
    assert set(np.asarray(combined.table["event_id"])) == set(range(12))
    assert len(combined) == 12


def test_union_requires_at_least_one_input():
    """`union` raises on an empty input list."""
    with pytest.raises(ValueError, match="at least one"):
        union([])


# --------------------------------------------------------------------------- #
# intersection                                                                #
# --------------------------------------------------------------------------- #
def test_intersection_keeps_only_ids_in_every_input(hot_spot):
    """`intersection` keeps only rows whose `event_id` is present in every input catalog."""
    catalog, *_ = _make_catalog(_DummyTransient(), hot_spot, n_events=10, seed=2)

    a = _subset(catalog, range(0, 7))
    b = _subset(catalog, range(3, 10))

    result = intersection([a, b])
    assert set(np.asarray(result.table["event_id"])) == set(range(3, 7))
    _assert_metadata_carried_from_first(result, a)


def test_intersection_is_nary(hot_spot):
    """`intersection` narrows correctly across three or more inputs, not just a pair."""
    catalog, *_ = _make_catalog(_DummyTransient(), hot_spot, n_events=10, seed=7)

    a = _subset(catalog, range(0, 8))
    b = _subset(catalog, range(2, 10))
    c = _subset(catalog, range(1, 6))

    result = intersection([a, b, c])
    assert set(np.asarray(result.table["event_id"])) == {2, 3, 4, 5}


def test_intersection_requires_at_least_one_input():
    """`intersection` raises on an empty input list."""
    with pytest.raises(ValueError, match="at least one"):
        intersection([])


# --------------------------------------------------------------------------- #
# difference                                                                  #
# --------------------------------------------------------------------------- #
def test_difference_keeps_a_minus_b(hot_spot):
    """`difference` keeps `a`'s rows whose `event_id` does not appear in `b`."""
    catalog, *_ = _make_catalog(_DummyTransient(), hot_spot, n_events=10, seed=3)

    a = _subset(catalog, range(0, 7))
    b = _subset(catalog, range(3, 10))

    result = difference([a, b])
    assert set(np.asarray(result.table["event_id"])) == set(range(0, 3))
    _assert_metadata_carried_from_first(result, a)


def test_difference_requires_exactly_two_inputs(hot_spot):
    """`difference` raises for anything other than exactly 2 inputs."""
    catalog, *_ = _make_catalog(_DummyTransient(), hot_spot, n_events=3, seed=4)
    with pytest.raises(ValueError, match="exactly 2"):
        difference([])
    with pytest.raises(ValueError, match="exactly 2"):
        difference([catalog])
    with pytest.raises(ValueError, match="exactly 2"):
        difference([catalog, catalog, catalog])


# --------------------------------------------------------------------------- #
# shared validation (_check_compatible)                                      #
# --------------------------------------------------------------------------- #
def test_mismatched_nside_raises(hot_spot):
    """Combining catalogs with different `nside` raises rather than silently comparing across them."""
    catalog, *_ = _make_catalog(_DummyTransient(), hot_spot, n_events=3, seed=5)
    other = type(catalog)(
        table=catalog.table.copy(),
        nside=catalog.nside * 2,
        order=catalog.order,
        time_bins=catalog.time_bins,
        seed=catalog.seed,
        downsample=catalog.downsample,
    )
    with pytest.raises(ValueError, match=r"same \(nside, order\)"):
        union([catalog, other])


def test_mismatched_order_raises(hot_spot):
    """Combining catalogs with different `order` raises rather than silently comparing across them."""
    catalog, *_ = _make_catalog(_DummyTransient(), hot_spot, n_events=3, seed=8)
    other_order = "ring" if catalog.order == "nested" else "nested"
    other = type(catalog)(
        table=catalog.table.copy(),
        nside=catalog.nside,
        order=other_order,
        time_bins=catalog.time_bins,
        seed=catalog.seed,
        downsample=catalog.downsample,
    )
    with pytest.raises(ValueError, match=r"same \(nside, order\)"):
        difference([catalog, other])


def _with_counts(catalog, counts):
    return replace(catalog, pre_cut_counts=counts)


@pytest.mark.parametrize("op", [union, intersection, difference])
def test_every_op_carries_pre_cut_counts(hot_spot, op):
    """No set operation changes the generation-time counts, whichever rows it keeps."""
    catalog, *_ = _make_catalog(_DummyTransient(), hot_spot, n_events=10, seed=2)
    catalog = _with_counts(catalog, {"tde": 1000})
    result = op([_subset(catalog, range(0, 7)), _subset(catalog, range(5, 10))])
    assert result.pre_cut_counts == {"tde": 1000}


@pytest.mark.parametrize("op", [union, intersection, difference])
def test_inputs_with_different_pre_cut_counts_raise(hot_spot, op):
    """Catalogs cut from different generation runs cannot share a denominator, so combining them raises."""
    catalog, *_ = _make_catalog(_DummyTransient(), hot_spot, n_events=6, seed=3)
    a = _with_counts(_subset(catalog, range(0, 4)), {"tde": 100})
    b = _with_counts(_subset(catalog, range(2, 6)), {"tde": 200})
    with pytest.raises(ValueError, match="pre_cut_counts"):
        op([a, b])


def test_union_takes_the_counts_from_whichever_input_has_them(hot_spot):
    catalog, *_ = _make_catalog(_DummyTransient(), hot_spot, n_events=6, seed=4)
    a = _subset(catalog, range(0, 4))
    b = _with_counts(_subset(catalog, range(2, 6)), {"tde": 100})
    assert a.pre_cut_counts is None
    assert union([a, b]).pre_cut_counts == {"tde": 100}


# --------------------------------------------------------------------------- #
# LOGICAL_OPS / LOGICAL_OP_ARITY registries                                  #
# --------------------------------------------------------------------------- #
def test_logical_ops_maps_names_to_the_right_functions():
    """`LOGICAL_OPS` dispatches each name to the matching module-level function."""
    assert LOGICAL_OPS == {"union": union, "intersection": intersection, "difference": difference}


def test_logical_op_arity_matches_registered_ops():
    """`LOGICAL_OP_ARITY` names exactly the ops in `LOGICAL_OPS`."""
    assert set(LOGICAL_OP_ARITY) == set(LOGICAL_OPS)


def test_logical_op_arity_bounds():
    """`union`/`intersection` are n-ary (unbounded above); `difference` is fixed at exactly 2."""
    assert LOGICAL_OP_ARITY["union"] == (1, None)
    assert LOGICAL_OP_ARITY["intersection"] == (2, None)
    assert LOGICAL_OP_ARITY["difference"] == (2, 2)
