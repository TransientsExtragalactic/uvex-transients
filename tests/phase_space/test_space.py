"""Tests for `Axis` and `Space`."""

import numpy as np
import pytest
from scipy.stats import qmc

from uvex_transients.phase_space.space import Axis, Space


@pytest.fixture
def axes():
    return [
        Axis("log_luminosity", 36.5, 45.0, label=r"$L$", unit="erg/s"),
        Axis("log_half_time", -1.5, 2.5, label=r"$t_{1/2}$", unit="day"),
    ]


@pytest.fixture
def space(axes):
    return Space(axes)


# ---------------------------------------------------------------------------- #
# Axis                                                                         #
# ---------------------------------------------------------------------------- #
def test_an_axis_converts_its_range_to_floats_and_defaults_to_a_log_axis():
    axis = Axis("x", 1, 3)
    assert axis.bounds == (1.0, 3.0) and isinstance(axis.lo, float)
    assert axis.scale == "log10" and axis.label == "" and axis.unit == ""


@pytest.mark.parametrize("name", ["", "log L", "1x", 3, None])
def test_an_axis_needs_an_identifier_for_a_name(name):
    with pytest.raises(ValueError, match="identifier"):
        Axis(name, 0.0, 1.0)


@pytest.mark.parametrize("lo, hi", [(1.0, 1.0), (2.0, 1.0), (0.0, np.inf), (np.nan, 1.0), (-np.inf, 0.0)])
def test_an_axis_needs_a_finite_increasing_range(lo, hi):
    with pytest.raises(ValueError, match="finite range"):
        Axis("x", lo, hi)


def test_an_axis_needs_a_known_scale():
    with pytest.raises(ValueError, match="scale"):
        Axis("x", 0.0, 1.0, scale="ln")


def test_an_axis_is_immutable_and_round_trips_through_a_dict():
    axis = Axis("x", 0.0, 1.0, label=r"$x$", unit="cm", scale="linear")
    with pytest.raises(AttributeError):
        axis.lo = 2.0
    assert Axis.from_dict(axis.to_dict()) == axis


# ---------------------------------------------------------------------------- #
# Space                                                                        #
# ---------------------------------------------------------------------------- #
def test_a_space_describes_its_axes(space, axes):
    assert space.ndim == len(space) == 2
    assert space.names == ("log_luminosity", "log_half_time")
    assert space.axes == tuple(axes)
    np.testing.assert_array_equal(space.bounds, [[36.5, 45.0], [-1.5, 2.5]])
    assert space["log_half_time"] is axes[1] and space[0] is axes[0]
    assert space.index("log_half_time") == 1
    assert list(space) == axes


def test_a_space_does_not_expose_its_bounds_for_editing(space):
    space.bounds[0, 0] = 0.0
    assert space.bounds[0, 0] == 36.5


def test_a_space_needs_axes_with_distinct_names():
    with pytest.raises(ValueError, match="at least one"):
        Space([])
    with pytest.raises(ValueError, match="distinct"):
        Space([Axis("x", 0, 1), Axis("x", 0, 2)])
    with pytest.raises(TypeError, match="Axis"):
        Space([("x", 0, 1)])


def test_an_unknown_axis_is_named_in_the_error(space):
    with pytest.raises(KeyError, match="log_luminosity"):
        space.index("z")


def test_spaces_compare_by_their_axes(space, axes):
    assert space == Space(axes)
    assert hash(space) == hash(Space(axes))
    assert space != Space(axes[:1])
    assert space != Space([axes[0], Axis("log_half_time", -1.5, 2.0)])
    assert "log_luminosity=[36.5, 45]" in repr(space)


def test_a_space_round_trips_through_a_dict(space):
    assert Space.from_dict(space.to_dict()) == space


def test_a_space_can_have_any_number_of_axes():
    three = Space([Axis("a", 0, 1), Axis("b", 10, 20), Axis("c", -5, 5, scale="linear")])
    assert three.ndim == 3
    u = three.to_unit([[0.5, 15.0, 0.0], [1.0, 10.0, -5.0]])
    np.testing.assert_allclose(u, [[0.5, 0.5, 0.5], [1.0, 0.0, 0.0]])


# ---------------------------------------------------------------------------- #
# The unit cube                                                                #
# ---------------------------------------------------------------------------- #
def test_the_unit_cube_maps_the_corners_to_the_corners(space):
    np.testing.assert_allclose(space.to_unit(space.bounds[:, 0]), [0.0, 0.0])
    np.testing.assert_allclose(space.to_unit(space.bounds[:, 1]), [1.0, 1.0])
    np.testing.assert_allclose(space.from_unit([0.0, 1.0]), [36.5, 2.5])


@pytest.mark.parametrize("shape", [(2,), (7, 2), (3, 4, 2)])
def test_points_round_trip_through_the_unit_cube_for_any_leading_shape(space, shape):
    u = np.random.default_rng(0).random(shape)
    x = space.from_unit(u)
    assert x.shape == shape
    np.testing.assert_allclose(space.to_unit(x), u, rtol=1e-12)


def test_points_with_the_wrong_number_of_coordinates_are_refused(space):
    for bad in (np.zeros(3), np.zeros((4, 3)), np.float64(1.0)):
        with pytest.raises(ValueError, match="2 coordinates"):
            space.to_unit(bad)
    with pytest.raises(ValueError, match="2 coordinates"):
        space.from_unit(np.zeros((5, 1)))


def test_the_unit_cube_is_the_arithmetic_the_sampler_has_always_used(space):
    """The moves are ``(x - lo) / (hi - lo)`` and its inverse, evaluated in that order, so that a map is unchanged."""
    lo, hi = space.bounds[:, 0], space.bounds[:, 1]
    x = np.random.default_rng(1).uniform([36.5, -1.5], [45.0, 2.5], size=(11, 2))
    np.testing.assert_array_equal(space.to_unit(x), (x - lo) / (hi - lo))
    u = np.random.default_rng(2).random((11, 2))
    np.testing.assert_array_equal(space.from_unit(u), lo + u * (hi - lo))


def test_contains_includes_the_edges(space):
    inside = space.contains([[40.0, 0.0], [36.5, -1.5], [45.0, 2.5], [36.4, 0.0], [40.0, 2.6]])
    assert inside.tolist() == [True, True, True, False, False]
    assert space.contains([40.0, 0.0]).shape == ()


# ---------------------------------------------------------------------------- #
# Designs                                                                      #
# ---------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [1, 5, 8, 33])
def test_a_sobol_design_has_n_points_in_the_unit_cube(space, n):
    design = space.sobol(n, seed=3)
    assert design.shape == (n, 2)
    assert np.all((design >= 0.0) & (design <= 1.0))


def test_a_sobol_design_follows_the_dimension_of_the_space():
    assert Space([Axis(name, 0, 1) for name in "abcd"]).sobol(8, seed=0).shape == (8, 4)
    assert Space([Axis("a", 0, 1)]).sobol(8, seed=0).shape == (8, 1)


def test_a_sobol_design_is_reproducible_from_its_seed(space):
    np.testing.assert_array_equal(space.sobol(16, seed=5), space.sobol(16, seed=5))
    assert not np.array_equal(space.sobol(16, seed=5), space.sobol(16, seed=6))


def test_a_sobol_design_does_not_advance_a_seed_sequence(space):
    """Scipy spawns from the sequence a generator was made from, which would change the next design."""
    child = np.random.SeedSequence(9).spawn(1)[0]
    first = space.sobol(16, seed=child)
    assert child.n_children_spawned == 0
    np.testing.assert_array_equal(space.sobol(16, seed=child), first)


def test_a_sobol_design_is_the_one_the_sampler_has_always_drawn(space):
    """The same scrambled design the sampler's `_sobol` draws from a freshly spawned child seed, for two axes."""

    def fresh_child():
        return np.random.SeedSequence(7).spawn(1)[0]

    sobol = qmc.Sobol(d=2, scramble=True, seed=np.random.default_rng(fresh_child()))
    expected = sobol.random_base2(int(np.ceil(np.log2(10))))[:10]
    np.testing.assert_array_equal(space.sobol(10, seed=fresh_child()), expected)


@pytest.mark.parametrize("n", [0, -3, 2.5])
def test_a_sobol_design_needs_a_positive_integer_count(space, n):
    with pytest.raises(ValueError, match="positive integer"):
        space.sobol(n)


# ---------------------------------------------------------------------------- #
# Grids                                                                        #
# ---------------------------------------------------------------------------- #
def test_a_grid_is_the_ij_meshgrid_with_the_first_axis_slowest(space):
    grid = space.grid(3)
    assert grid.shape == (9, 2)
    lines = [np.linspace(36.5, 45.0, 3), np.linspace(-1.5, 2.5, 3)]
    mesh = np.meshgrid(*lines, indexing="ij")
    np.testing.assert_array_equal(grid[:, 0], mesh[0].ravel())
    np.testing.assert_array_equal(grid[:, 1], mesh[1].ravel())
    np.testing.assert_array_equal(grid[:, 0].reshape(3, 3)[:, 0], lines[0])


def test_a_grid_can_have_a_size_for_each_axis_and_its_own_rectangle(space):
    grid = space.grid([2, 4], bounds=[[40.0, 41.0], [0.0, 3.0]])
    assert grid.shape == (8, 2)
    np.testing.assert_allclose(np.unique(grid[:, 0]), [40.0, 41.0])
    np.testing.assert_allclose(np.unique(grid[:, 1]), [0.0, 1.0, 2.0, 3.0])


def test_a_grid_over_more_axes_has_a_point_for_every_combination():
    three = Space([Axis("a", 0, 1), Axis("b", 0, 1), Axis("c", 0, 1)])
    assert three.grid(4).shape == (64, 3)


@pytest.mark.parametrize("size", [1, 0, [3], [3, 3, 3], 2.5, [3, 1]])
def test_a_grid_needs_an_integer_of_at_least_two_for_every_axis(space, size):
    with pytest.raises(ValueError, match="size"):
        space.grid(size)


def test_a_grid_needs_bounds_for_every_axis(space):
    with pytest.raises(ValueError, match="bounds"):
        space.grid(3, bounds=[[0.0, 1.0]])


# ---------------------------------------------------------------------------- #
# Changing the ranges                                                          #
# ---------------------------------------------------------------------------- #
def test_with_bounds_keeps_everything_but_the_ranges(space):
    narrower = space.with_bounds([[38.0, 44.0], [0.0, 2.0]])
    np.testing.assert_array_equal(narrower.bounds, [[38.0, 44.0], [0.0, 2.0]])
    assert narrower.names == space.names
    assert [a.label for a in narrower] == [a.label for a in space]
    assert [a.unit for a in narrower] == [a.unit for a in space]
    np.testing.assert_array_equal(space.bounds, [[36.5, 45.0], [-1.5, 2.5]])


def test_with_bounds_refuses_bad_ranges(space):
    with pytest.raises(ValueError, match="shape"):
        space.with_bounds([[0.0, 1.0]])
    with pytest.raises(ValueError, match="finite range"):
        space.with_bounds([[2.0, 1.0], [0.0, 1.0]])
