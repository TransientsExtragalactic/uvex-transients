"""
Global configuration system for uvex_transients.

This module provides a hierarchical, YAML-backed configuration manager.
Configuration values are accessed via dot-separated keys (e.g.
``config["physics.default_cosmology"]``); changes are written back to disk
immediately (autosave).

Configuration files are located using the following precedence:

    1. Environment variable ``$UVEX_TRANSIENTS_CONFIG`` (if set)
    2. Local project file ``.uvex_transientsrc`` in the current working directory
    3. User-specific config at ``~/.config/uvex_transients/config.yaml``
    4. Package default config distributed with uvex_transients

Use :attr:`config` to access the active configuration; it behaves like a
nested dictionary with automatic loading and saving.
"""

import os
from collections.abc import MutableMapping
from pathlib import Path
from typing import Union

from platformdirs import user_config_dir

from .io_utils import config_yaml


class ConfigManager(MutableMapping):
    """
    Hierarchical configuration manager with dot-separated keys and optional autosave.

    Parameters
    ----------
    path : str or Path
        Path to the backing YAML configuration file.
    autosave : bool, optional
        If True (default), write changes to disk immediately.
    """

    def __init__(self, path: Union[str, Path], autosave: bool = True):
        """
        Load the configuration at `path`, or start empty if it doesn't exist yet.

        Parameters
        ----------
        path : str or Path
            Path to the backing YAML configuration file.
        autosave : bool, optional
            If True (default), write changes to disk immediately.
        """
        self._path = Path(path).expanduser().resolve()
        self._autosave = autosave
        self._data = self._load()

    def _load(self) -> dict:
        """
        Load configuration data from the YAML file.

        Returns
        -------
        dict
            The parsed configuration, or ``{}`` if the file doesn't exist.
        """
        if not self._path.exists():
            return {}
        with open(self._path) as f:
            return config_yaml.load(f) or {}

    def _save(self) -> None:
        """Save configuration data to the YAML file."""
        with open(self._path, "w") as f:
            config_yaml.dump(self._data, f)

    def _traverse(self, key: str, create_missing: bool = False):
        """
        Navigate nested dictionaries using a dot-separated key, returning ``(parent, last_key)``.

        Parameters
        ----------
        key : str
            Dot-separated key, e.g. ``"physics.default_cosmology"``.
        create_missing : bool, optional
            If True, create empty intermediate dicts for missing path segments
            rather than raising. Default False.

        Returns
        -------
        tuple of (dict, str)
            The parent dict holding the final key segment, and that segment itself.

        Raises
        ------
        KeyError
            If an intermediate key segment is missing and `create_missing` is False.
        """
        keys = key.split(".")
        node = self._data
        for k in keys[:-1]:
            if k not in node:
                if create_missing:
                    node[k] = {}
                else:
                    raise KeyError(f"{k!r} not found in config.")
            node = node[k]
        return node, keys[-1]

    def __getitem__(self, key: str):
        """
        Look up a dot-separated key.

        Parameters
        ----------
        key : str
            Dot-separated key, e.g. ``"physics.default_cosmology"``.

        Returns
        -------
        object
            The stored value.
        """
        node, final_key = self._traverse(key)
        return node[final_key]

    def __setitem__(self, key: str, value) -> None:
        """
        Set a dot-separated key, creating intermediate levels as needed.

        Parameters
        ----------
        key : str
            Dot-separated key, e.g. ``"physics.default_cosmology"``.
        value : object
            The value to store.
        """
        node, final_key = self._traverse(key, create_missing=True)
        node[final_key] = value
        if self._autosave:
            self._save()

    def __delitem__(self, key: str) -> None:
        """
        Delete a dot-separated key.

        Parameters
        ----------
        key : str
            Dot-separated key, e.g. ``"physics.default_cosmology"``.
        """
        node, final_key = self._traverse(key)
        del node[final_key]
        if self._autosave:
            self._save()

    def __iter__(self):
        """
        Iterate over this configuration's top-level keys.

        Returns
        -------
        Iterator of str
            An iterator over the top-level keys.
        """
        return iter(self._data)

    def __len__(self) -> int:
        """
        Return the number of top-level keys.

        Returns
        -------
        int
            ``len(self._data)``.
        """
        return len(self._data)

    def __repr__(self) -> str:
        """
        Return a one-line representation showing the backing path and current data.

        Returns
        -------
        str
            ``<ConfigManager path=... data=...>``.
        """
        return f"<ConfigManager path={self._path} data={self._data}>"


# Cache to avoid reloading on every `get_config()` call.
__CONFIG__ = None


def get_config() -> ConfigManager:
    """
    Return the global uvex_transients configuration, following the standard precedence.

    Returns
    -------
    ConfigManager
        The global configuration instance (cached across calls).

    Raises
    ------
    OSError
        If no configuration file is found at any of the standard locations.
    """
    global __CONFIG__
    if __CONFIG__ is not None:
        return __CONFIG__

    # 1. Environment override
    # 2. Project-local file
    # 3. User-global config
    # 4. Package defaults
    candidates = []

    env_path = os.environ.get("UVEX_TRANSIENTS_CONFIG")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.append(Path.cwd() / ".uvex_transientsrc")
    candidates.append(Path(user_config_dir("uvex_transients")) / "config.yaml")
    candidates.append(Path(__file__).parents[1] / "config.yaml")

    for path in candidates:
        if path.exists():
            __CONFIG__ = ConfigManager(path)
            break
    else:
        raise OSError(f"Missing default configuration file at {candidates[-1]}. Was the install corrupted?")

    return __CONFIG__


config = get_config()
"""ConfigManager: The global uvex_transients configuration instance."""
