"""
Set operations between two or more named `EventCatalog` objects, keyed by ``event_id``.

Unlike `~uvex_transients.simulation.core.cut` (a predicate that narrows *one* catalog),
these combine several already-produced catalogs -- e.g. the events a magnitude cut kept
intersected with the events an independent SNR-based branch kept, or the events one
branch caught that another didn't. `LOGICAL_OPS` is a small, fixed dict rather than a
`@cut`/`@action`-style open registry: these three set operations are the whole space of
what "combine catalogs by event_id" can mean, not something users are expected to add
bespoke variants of.
"""

import numpy as np
from astropy.table import vstack

from .event_catalog import EventCatalog


def _check_compatible(catalogs: list[EventCatalog]) -> None:
    """
    Raise if `catalogs` don't share the same `nside`/`order`.

    Comparing ``event_id`` across incompatible HEALPix resolutions is a config error,
    not something to silently allow.

    Parameters
    ----------
    catalogs : list of EventCatalog
        The catalogs a `logical_op` is about to combine.

    Raises
    ------
    ValueError
        If `catalogs` is empty, or any two disagree on `nside`/`order`.
    """
    if not catalogs:
        raise ValueError("logical_op requires at least one input catalog.")
    first = catalogs[0]
    for other in catalogs[1:]:
        if other.nside != first.nside or other.order != first.order:
            raise ValueError(
                f"logical_op inputs must share the same (nside, order); got ({first.nside}, {first.order}) "
                f"and ({other.nside}, {other.order})."
            )


def union(catalogs: list[EventCatalog]) -> EventCatalog:
    """
    Combine two or more `EventCatalog` objects, deduplicating by ``event_id``.

    Parameters
    ----------
    catalogs : list of EventCatalog
        One or more catalogs to combine. Order doesn't matter (union is commutative);
        where the same ``event_id`` appears in more than one input, the first
        occurrence (in `catalogs` order) is kept.

    Returns
    -------
    EventCatalog
        A new catalog over the union of every input's rows, carrying `catalogs[0]`'s own
        `nside`/`order`/`time_bins`/`seed`/`downsample`.

    Raises
    ------
    ValueError
        If `catalogs` is empty, or any two disagree on `nside`/`order`.
    """
    _check_compatible(catalogs)
    first = catalogs[0]

    seen: set[int] = set()
    keep_masks = []
    for catalog in catalogs:
        ids = np.asarray(catalog.table["event_id"])
        mask = np.array([eid not in seen for eid in ids])
        seen.update(ids[mask].tolist())
        keep_masks.append(mask)

    combined = vstack([catalog.table[mask] for catalog, mask in zip(catalogs, keep_masks)], metadata_conflicts="silent")
    return EventCatalog(
        table=combined,
        nside=first.nside,
        order=first.order,
        time_bins=first.time_bins,
        seed=first.seed,
        downsample=first.downsample,
    )


def intersection(catalogs: list[EventCatalog]) -> EventCatalog:
    """
    Restrict `catalogs[0]` to events (by ``event_id``) present in every other input too.

    Parameters
    ----------
    catalogs : list of EventCatalog
        Two or more catalogs to intersect. Order doesn't affect the resulting *rows*
        (intersection is commutative), only which input's own columns/metadata the
        result carries (`catalogs[0]`'s).

    Returns
    -------
    EventCatalog
        A new catalog over `catalogs[0]`'s rows whose ``event_id`` also appears in every
        other input.

    Raises
    ------
    ValueError
        If `catalogs` is empty, or any two disagree on `nside`/`order`.
    """
    _check_compatible(catalogs)
    first = catalogs[0]

    common = set(np.asarray(first.table["event_id"]).tolist())
    for catalog in catalogs[1:]:
        common &= set(np.asarray(catalog.table["event_id"]).tolist())

    mask = np.isin(np.asarray(first.table["event_id"]), list(common))
    return EventCatalog(
        table=first.table[mask],
        nside=first.nside,
        order=first.order,
        time_bins=first.time_bins,
        seed=first.seed,
        downsample=first.downsample,
    )


def difference(catalogs: list[EventCatalog]) -> EventCatalog:
    """
    Restrict `catalogs[0]` to events (by ``event_id``) *not* present in `catalogs[1]`.

    Parameters
    ----------
    catalogs : list of EventCatalog
        Exactly two catalogs, ``[a, b]``; the result is ``a`` minus ``b`` (unlike
        `union`/`intersection`, this is order-sensitive, so no n-ary form is offered).

    Returns
    -------
    EventCatalog
        A new catalog over `catalogs[0]`'s rows whose ``event_id`` does *not* appear in
        `catalogs[1]`.

    Raises
    ------
    ValueError
        If `catalogs` doesn't have exactly two entries, or they disagree on `nside`/`order`.
    """
    if len(catalogs) != 2:
        raise ValueError(f"'difference' requires exactly 2 input catalogs, got {len(catalogs)}.")
    _check_compatible(catalogs)
    first, second = catalogs

    exclude = set(np.asarray(second.table["event_id"]).tolist())
    mask = ~np.isin(np.asarray(first.table["event_id"]), list(exclude))
    return EventCatalog(
        table=first.table[mask],
        nside=first.nside,
        order=first.order,
        time_bins=first.time_bins,
        seed=first.seed,
        downsample=first.downsample,
    )


#: ``{op name: implementation}``, dispatched by name from a config-driven ``logical_op`` step
#: (see `uvex_transients.cli.steps`). Every implementation takes ``list[EventCatalog]`` and
#: returns a single `EventCatalog`; `union`/`intersection` accept any number of inputs (n-ary,
#: since both are associative/commutative), `difference` requires exactly two (``a - b`` isn't
#: well-defined for more).
LOGICAL_OPS = {"union": union, "intersection": intersection, "difference": difference}

#: Minimum/maximum number of inputs each op in `LOGICAL_OPS` accepts, for upfront arity validation.
LOGICAL_OP_ARITY = {"union": (1, None), "intersection": (2, None), "difference": (2, 2)}
