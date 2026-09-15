"""Tests for `uvex_transients.cli.yaml_tags`'s ``!prior`` tag."""

import io

import numpy as np
import pytest
from astropy.cosmology import FLRW

from uvex_transients.cli.yaml_tags import get_run_yaml
from uvex_transients.models.core.priors import DiscretePrior, PowerLawPrior


def _load(doc: str):
    """Parse a YAML string with the run-config loader and return its top-level mapping."""
    return get_run_yaml().load(io.StringIO(doc))


def test_prior_tag_resolves_to_the_right_prior_instance():
    """A ``!prior`` node with a known `type:` builds the matching `Prior` subclass."""
    data = _load("x: !prior {type: power_law, alpha: -1.5, lower: 1.0e+42, upper: 1.0e+45}\n")
    assert data["x"] == PowerLawPrior(alpha=-1.5, lower=1.0e42, upper=1.0e45)


def test_prior_tag_converts_sequence_fields_to_arrays():
    """`DiscretePrior`'s `values`/`probabilities` arrive from YAML as plain lists but sample fine."""
    data = _load("x: !prior {type: discrete, values: [1, 2, 3], probabilities: [0.2, 0.3, 0.5]}\n")
    prior = data["x"]
    assert isinstance(prior, DiscretePrior)
    assert np.array_equal(prior.values, np.array([1, 2, 3]))


def test_prior_tag_and_cosmology_tag_coexist():
    """`get_run_yaml` carries the package config's ``!astropy_cosmology`` tag alongside ``!prior``."""
    data = _load("x: !astropy_cosmology {name: Planck18}\n")
    assert isinstance(data["x"], FLRW)


def test_prior_tag_missing_type_raises():
    """A ``!prior`` node with no `type:` key raises, listing the valid type names."""
    with pytest.raises(ValueError, match="missing required key 'type'"):
        _load("x: !prior {lower: 0, upper: 1}\n")


def test_prior_tag_unknown_type_raises():
    """A ``!prior`` node with an unregistered `type:` raises, listing the valid ones."""
    with pytest.raises(ValueError, match="Unknown prior type 'bogus'"):
        _load("x: !prior {type: bogus}\n")


def test_prior_tag_unknown_parameter_raises():
    """A ``!prior`` node with an extra, unrecognized kwarg raises, listing the valid ones."""
    with pytest.raises(ValueError, match="unknown parameter"):
        _load("x: !prior {type: uniform, lower: 0, upper: 1, extra: 5}\n")
