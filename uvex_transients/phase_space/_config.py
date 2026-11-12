"""Read a phase-space setting from the package configuration unless the caller gave it."""

from typing import Any

from uvex_transients.utils import config

__all__ = ["configured"]


def configured(value: Any, key: str) -> Any:
    """
    Return `value`, or the configured default if it is `None`.

    Parameters
    ----------
    value : Any
        The argument the caller gave, or `None` for "use the default".
    key : str
        The setting, relative to the ``phase_space`` section of the configuration, e.g. ``"sampler.n_initial"``.

    Returns
    -------
    Any
        `value` if it is not `None`, else ``config["phase_space.<key>"]``.
    """
    return config[f"phase_space.{key}"] if value is None else value
