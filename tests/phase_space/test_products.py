"""Tests for `Estimate`, `Product`, `Scale` and `Weighted`."""

import numpy as np
import pytest

from uvex_transients.phase_space.samplers import Estimate, Product, Scale, Weighted


@pytest.fixture
def points():
    """Five points of a two dimensional space."""
    return np.column_stack([np.linspace(0.0, 1.0, 5), np.linspace(2.0, 4.0, 5)])


@pytest.fixture
def estimate():
    p = np.array([0.1, 0.2, 0.4, 0.6, 0.8])
    return Estimate(p, p - 0.05, p + 0.05)


# ---------------------------------------------------------------------------- #
# Estimate                                                                     #
# ---------------------------------------------------------------------------- #
def test_an_estimate_holds_a_value_and_its_interval_as_floats(estimate):
    assert estimate.shape == (5,)
    assert estimate.p.dtype == np.float64
    np.testing.assert_allclose(estimate.p_hi - estimate.p_lo, 0.1)
    assert Estimate(0.5, 0.4, 0.6).shape == ()
    assert Estimate([1, 0], [1, 0], [1, 0]).p.dtype == np.float64


def test_an_estimate_needs_one_shape_for_its_three_arrays():
    with pytest.raises(ValueError, match="one shape"):
        Estimate([0.5, 0.5], [0.4], [0.6, 0.6])


def test_an_estimate_is_immutable(estimate):
    with pytest.raises(AttributeError):
        estimate.p = np.zeros(5)


# ---------------------------------------------------------------------------- #
# Product                                                                      #
# ---------------------------------------------------------------------------- #
def test_a_product_cannot_be_instantiated_without_saying_how_it_is_made():
    assert Product.__abstractmethods__ == {"__call__"}
    with pytest.raises(TypeError, match="abstract"):
        Product("x")


@pytest.mark.parametrize("name", ["", "two words", "1x", None, 3])
def test_a_product_needs_an_identifier_for_a_name(name):
    with pytest.raises(ValueError, match="identifier"):
        Scale(name, lambda x: x[..., 0])


def test_a_product_keeps_its_name_label_and_unit():
    product = Weighted("yield", lambda x: x[..., 0], label=r"$Y$", unit="Mpc3 yr")
    assert (product.name, product.label, product.unit) == ("yield", r"$Y$", "Mpc3 yr")
    assert repr(product) == "Weighted('yield')"
    assert Scale("z", lambda x: x[..., 0]).label == ""


# ---------------------------------------------------------------------------- #
# Scale                                                                        #
# ---------------------------------------------------------------------------- #
def test_a_scale_is_a_quantity_of_the_point_alone(points, estimate):
    scale = Scale("total", lambda x: x.sum(axis=-1))
    columns = scale(points, estimate)
    assert list(columns) == ["total"]
    np.testing.assert_allclose(columns["total"], points.sum(axis=-1))

    # The estimate is not involved: it only says how many points there are
    other = Estimate(np.zeros(5), np.zeros(5), np.zeros(5))
    np.testing.assert_array_equal(scale(points, other)["total"], columns["total"])


def test_a_scale_that_is_a_constant_is_laid_out_over_the_points(points, estimate):
    columns = Scale("one", lambda x: 1.0)(points, estimate)
    np.testing.assert_array_equal(columns["one"], np.ones(5))
    # A copy that can be written to, not a read-only view of the broadcast constant
    columns["one"][0] = 2.0
    assert columns["one"][0] == 2.0


# ---------------------------------------------------------------------------- #
# Weighted                                                                     #
# ---------------------------------------------------------------------------- #
def test_a_weighted_product_is_the_estimate_times_the_weight_with_its_interval(points, estimate):
    product = Weighted("recovered", lambda x: 10.0 ** x[..., 1])
    columns = product(points, estimate)
    assert list(columns) == ["recovered", "recovered_lo", "recovered_hi"]
    weight = 10.0 ** points[:, 1]
    np.testing.assert_allclose(columns["recovered"], estimate.p * weight)
    np.testing.assert_allclose(columns["recovered_lo"], estimate.p_lo * weight)
    np.testing.assert_allclose(columns["recovered_hi"], estimate.p_hi * weight)
    assert np.all(columns["recovered_lo"] <= columns["recovered"]) and np.all(
        columns["recovered"] <= columns["recovered_hi"]
    )


def test_a_weighted_product_works_for_points_of_any_leading_shape():
    p = np.full((3, 4), 0.5)
    estimate = Estimate(p, p - 0.1, p + 0.1)
    x = np.random.default_rng(0).random((3, 4, 2))
    columns = Weighted("w", lambda points: points[..., 0] + 1.0)(x, estimate)
    assert columns["w"].shape == (3, 4)
    np.testing.assert_allclose(columns["w"], 0.5 * (x[..., 0] + 1.0))


def test_a_weighted_product_refuses_a_negative_weight(points, estimate):
    with pytest.raises(ValueError, match="must not be negative"):
        Weighted("w", lambda x: x[..., 0] - 0.5)(points, estimate)


def test_points_and_estimate_have_to_describe_the_same_points(points):
    short = Estimate(np.zeros(3), np.zeros(3), np.zeros(3))
    for product in (Scale("s", lambda x: x[..., 0]), Weighted("w", lambda x: x[..., 0])):
        with pytest.raises(ValueError, match="do not describe the same points"):
            product(points, short)
        with pytest.raises(ValueError, match="do not describe the same points"):
            product(np.float64(1.0), short)


# ---------------------------------------------------------------------------- #
# A product is free to make its outputs however it needs                       #
# ---------------------------------------------------------------------------- #
class _ExpectedCount(Product):
    """The recovered yield times a rate, and the number of events that fall in a region, from one estimate."""

    def __init__(self, rate):
        super().__init__("expected", label="expected count", unit="")
        self._rate = rate

    def __call__(self, x, estimate):
        weight = self._rate * (1.0 + x[..., 0])
        return {
            "expected": estimate.p * weight,
            "expected_lo": estimate.p_lo * weight,
            "expected_hi": estimate.p_hi * weight,
            "expected_width": (estimate.p_hi - estimate.p_lo) * weight,
        }


def test_a_product_can_create_any_columns_from_the_points_and_the_estimate(points, estimate):
    """The rate-weighted yield and an extra column of its own: what a fixed fraction-times-scale could not express."""
    columns = _ExpectedCount(rate=3.0)(points, estimate)
    assert set(columns) == {"expected", "expected_lo", "expected_hi", "expected_width"}
    np.testing.assert_allclose(columns["expected_width"], 0.1 * 3.0 * (1.0 + points[:, 0]))
    np.testing.assert_allclose(columns["expected"], estimate.p * 3.0 * (1.0 + points[:, 0]))


def test_a_scale_and_a_weighted_product_can_share_a_function(points, estimate):
    """A weighted product of a quantity and the quantity itself agree where the fraction is one."""

    def volume(x):
        return 100.0 * (1.0 + x[..., 0])

    ones = Estimate(np.ones(5), np.ones(5), np.ones(5))
    np.testing.assert_allclose(Weighted("v", volume)(points, ones)["v"], Scale("v", volume)(points, estimate)["v"])
