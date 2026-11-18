"""Read phase-space settings from the package configuration, in one place."""

from collections.abc import Sequence
from typing import Any

from uvex_transients.utils import config

__all__ = ["configured", "default"]


def default(section: str, key: str, kinds: Sequence[str] = ()) -> Any:
    """
    Return the configured default of a setting for a kind of sampler.

    A section named for a kind of sampler (``phase_space.canonical``, ``phase_space.breakout``) holds the settings of
    that kind, and a key in it takes the place of the key of the same name in the shared section, for that kind only.
    The kinds are tried in order, so a kind built on another (see
    :meth:`Sampler.kinds <uvex_transients.phase_space.sampling.base.Sampler.kinds>`) takes its own value, then its
    parent's, then the shared one.

    Parameters
    ----------
    section : str
        The shared section the setting belongs to, such as ``"detection"`` or ``"simulation"``.
    key : str
        The setting.
    kinds : sequence of str, optional
        The kinds to look in first, most specific first.

    Returns
    -------
    Any
        ``config["phase_space.<kind>.<key>"]`` for the first kind that has it, else
        ``config["phase_space.<section>.<key>"]``.

    Raises
    ------
    KeyError
        If neither a kind nor the section has the setting.
    """
    for kind in kinds:
        try:
            return config[f"phase_space.{kind}.{key}"]
        except KeyError:
            continue
    return config[f"phase_space.{section}.{key}"]


def configured(value: Any, key: str) -> Any:
    """
    Return `value`, or the configured default if it is `None`.

    Parameters
    ----------
    value : Any
        The argument the caller gave, or `None` for "use the default".
    key : str
        The setting, relative to the ``phase_space`` section of the configuration, e.g. ``"campaign.n_initial"``.

    Returns
    -------
    Any
        `value` if it is not `None`, else ``config["phase_space.<key>"]``.
    """
    return config[f"phase_space.{key}"] if value is None else value
