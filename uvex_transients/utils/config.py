"""uvex_transients global configuration system.

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
    """Hierarchical configuration manager with dot-separated keys and optional autosave.

    Parameters
    ----------
    path : str or Path
        Path to the backing YAML configuration file.
    autosave : bool, optional
        If True (default), write changes to disk immediately.
    """

    def __init__(self, path: Union[str, Path], autosave: bool = True):
        self._path = Path(path).expanduser().resolve()
        self._autosave = autosave
        self._data = self._load()

    def _load(self) -> dict:
        """Load configuration data from the YAML file."""
        if not self._path.exists():
            return {}
        with open(self._path) as f:
            return config_yaml.load(f) or {}

    def _save(self) -> None:
        """Save configuration data to the YAML file."""
        with open(self._path, "w") as f:
            config_yaml.dump(self._data, f)

    def _traverse(self, key: str, create_missing: bool = False):
        """Navigate nested dictionaries using a dot-separated key, returning ``(parent, last_key)``."""
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
        node, final_key = self._traverse(key)
        return node[final_key]

    def __setitem__(self, key: str, value) -> None:
        node, final_key = self._traverse(key, create_missing=True)
        node[final_key] = value
        if self._autosave:
            self._save()

    def __delitem__(self, key: str) -> None:
        node, final_key = self._traverse(key)
        del node[final_key]
        if self._autosave:
            self._save()

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"<ConfigManager path={self._path} data={self._data}>"


# Cache to avoid reloading on every `get_config()` call.
__CONFIG__ = None


def get_config() -> ConfigManager:
    """Return the global uvex_transients configuration, following the standard precedence."""
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
