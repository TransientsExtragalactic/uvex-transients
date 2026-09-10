"""Logging setup for uvex_transients.

Configures the package-wide logger (:attr:`logger`), the main entry point for
logging messages across the codebase. Behavior -- level and message format --
is controlled by the central configuration
(:attr:`~uvex_transients.utils.config.config`) under the ``system.logging.main``
namespace.
"""

import logging
from typing import Union

from .config import config

logger = logging.getLogger("uvex_transients")
"""logging.Logger: The main logger for the uvex_transients package."""

logger.propagate = False  # Avoid duplicate logs via the root logger.


def configure_logging(level: Union[int, str, None] = None) -> None:
    """(Re)configure `logger` from `config`, optionally overriding its level.

    Parameters
    ----------
    level : int, str, or None, optional
        Logging level to use instead of ``config["system.logging.main.level"]``
        (e.g. ``logging.DEBUG`` or ``"DEBUG"``).
    """
    resolved_level = level if level is not None else config["system.logging.main.level"]
    if isinstance(resolved_level, str):
        resolved_level = getattr(logging, resolved_level)
    logger.setLevel(resolved_level)

    # Don't stack handlers across repeated calls.
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(config["system.logging.main.format"]))
    logger.addHandler(handler)


configure_logging()
