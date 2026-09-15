"""Tests for the `TransientBase.registry` auto-registration mechanism."""

from astropy import units as u

from uvex_transients.models.tdes import VanVelzenTDESED
from uvex_transients.transients.base import TransientBase


def test_concrete_subclass_registers_itself():
    """Defining a concrete `TransientBase` subclass adds it to `TransientBase.registry()` by name."""

    class _ThrowawayTransient(TransientBase):
        DEFAULT_MODEL = VanVelzenTDESED
        DEFAULT_DURATION = 1 * u.day

    registry = TransientBase.registry()
    assert registry["_ThrowawayTransient"] is _ThrowawayTransient


def test_registry_is_a_copy():
    """`registry()` returns a snapshot, not a live view -- mutating it can't corrupt the real registry."""
    registry = TransientBase.registry()
    registry["not_real"] = object()
    assert "not_real" not in TransientBase.registry()
