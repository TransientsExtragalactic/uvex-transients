"""YAML (de)serialization helpers for CLI run-configs.

The only non-trivial extension needed here is a ``!prior`` tag, resolving a mapping
node (``type: <DISTRIBUTION_NAME>`` plus that `~uvex_transients.models.core.priors.Prior`
subclass's own dataclass fields) to a live `Prior` instance -- e.g.::

    !prior {type: power_law, alpha: -1.5, lower: 1.0e+42, upper: 1.0e+45}

Built directly on top of `~uvex_transients.utils.io_utils.get_config_yaml`, so a run-config
also gets the package config's ``!astropy_cosmology`` tag for free (e.g. for a per-transient
``cosmology:`` override), without mutating the shared instance package config loading uses.
"""

from dataclasses import fields

import numpy as np
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

from uvex_transients.models.core.priors import Prior
from uvex_transients.utils.io_utils import get_config_yaml

_PRIOR_TAG = "!prior"


def _construct_prior(constructor, node) -> Prior:
    """Resolve a ``!prior`` node to a `Prior` instance, validating against its registry/fields."""
    line = node.start_mark.line + 1

    data = CommentedMap()
    constructor.construct_mapping(node, maptyp=data, deep=True)

    try:
        distribution_name = data.pop("type")
    except KeyError:
        raise ValueError(
            f"'!prior' node at line {line} is missing required key 'type' (one of {sorted(Prior.registry())})."
        ) from None

    registry = Prior.registry()
    try:
        prior_cls = registry[distribution_name]
    except KeyError:
        raise ValueError(
            f"Unknown prior type {distribution_name!r} at line {line}; available: {sorted(registry)}."
        ) from None

    valid_names = {field.name for field in fields(prior_cls) if field.init}
    unknown = sorted(set(data) - valid_names)
    if unknown:
        raise ValueError(
            f"{prior_cls.__name__} (line {line}) got unknown parameter(s) {unknown}; "
            f"valid parameters are {sorted(valid_names)}."
        )

    # Sequence-valued fields (e.g. `DiscretePrior.values`/`.probabilities`) arrive from
    # YAML as plain lists; `Prior._validate`/`_logpdf` implementations expect arrays.
    kwargs = {name: (np.asarray(value) if isinstance(value, (list, tuple)) else value) for name, value in data.items()}

    try:
        return prior_cls(**kwargs)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Failed to construct {prior_cls.__name__} at line {line}: {exc}") from exc


def get_run_yaml() -> YAML:
    """Return a `ruamel.yaml.YAML` instance that understands ``!prior`` (and ``!astropy_cosmology``)."""
    yaml = get_config_yaml()
    yaml.constructor.add_constructor(_PRIOR_TAG, _construct_prior)
    return yaml


run_yaml = get_run_yaml()
"""ruamel.yaml.YAML: The YAML (de)serializer used to load CLI run-configs."""
