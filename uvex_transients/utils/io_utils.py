"""YAML (de)serialization helpers for the package configuration.

The only non-trivial extension needed here is a ``!astropy_cosmology`` tag so
``config.yaml`` can name one of Astropy's built-in cosmologies (e.g.
``Planck18``) and have it load as a live `~astropy.cosmology.FLRW` instance
rather than a bare string -- see
:func:`uvex_transients.models._cosmology.get_cosmology`.
"""

from astropy.cosmology import FLRW, FlatLambdaCDM
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

_COSMOLOGY_TAG = "!astropy_cosmology"


def _represent_cosmology(representer, obj: FLRW):
    """Serialize a named built-in cosmology as ``{name: <registry name>}``."""
    if obj.name is None:
        raise ValueError("Only named cosmologies (e.g. 'Planck18') can be serialized to YAML.")
    return representer.represent_mapping(_COSMOLOGY_TAG, {"name": obj.name})


def _construct_cosmology(constructor, node) -> FLRW:
    """Resolve a ``!astropy_cosmology`` node to the named entry in `astropy.cosmology`."""
    from astropy import cosmology

    data = CommentedMap()
    constructor.construct_mapping(node, maptyp=data, deep=True)

    name = data["name"]
    try:
        return getattr(cosmology, name)
    except AttributeError as exc:
        raise ValueError(f"Unknown built-in astropy cosmology: {name!r}") from exc


def get_config_yaml() -> YAML:
    """Return a `ruamel.yaml.YAML` instance that understands ``!astropy_cosmology``."""
    yaml = YAML(typ="rt")

    # Registered for both the concrete `FlatLambdaCDM` (what all of Astropy's
    # named realizations, e.g. `Planck18`, actually are) and the abstract
    # `FLRW` base, since ruamel dispatches representers by exact type.
    for cosmology_type in (FlatLambdaCDM, FLRW):
        yaml.representer.add_representer(cosmology_type, _represent_cosmology)
    yaml.constructor.add_constructor(_COSMOLOGY_TAG, _construct_cosmology)

    return yaml


config_yaml = get_config_yaml()
"""ruamel.yaml.YAML: The YAML (de)serializer used to load/save the package configuration."""
