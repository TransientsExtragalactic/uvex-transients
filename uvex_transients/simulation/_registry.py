"""
A small, reusable "tag a method, collect it into a class-level registry" pattern.

`~uvex_transients.simulation.core.SurveySimulator` uses one instance of this for its
``@cut``-tagged screening methods and a second, independent instance for its
``@action``-tagged pipeline actions (see `uvex_transients.simulation.core`). Both share
the same collection behavior.
"""

from collections.abc import Callable


def make_tagged_registry(tag_attr: str, registry_attr: str) -> tuple[Callable[[str], Callable], type]:
    """
    Build a matched ``(decorator, metaclass)`` pair implementing the tag-and-collect pattern.

    Parameters
    ----------
    tag_attr : str
        The attribute name the decorator stashes the registry key under (e.g.
        ``"_cut_name"``), on the tagged method itself.
    registry_attr : str
        The class attribute name the metaclass collects every tagged method's key into
        (e.g. ``"_CUT_REGISTRY"``), as ``{key: method_name}``.

    Returns
    -------
    decorator : callable
        A decorator factory: ``decorator(name)`` tags a method with `name` and returns
        it unchanged.
    metaclass : type
        A metaclass that, on class creation, walks the whole MRO and collects every
        `tag_attr`-tagged method into `registry_attr`.
    """

    def decorator(name: str):
        """
        Mark a method as reachable under `name` in the registry this decorator feeds.

        Parameters
        ----------
        name : str
            The registry key this method should be reachable under.

        Returns
        -------
        Callable
            A decorator that tags the method with `name` and returns it unchanged.
        """

        def _tag(method):
            setattr(method, tag_attr, name)
            return method

        return _tag

    class _TaggedRegistryMeta(type):
        """Metaclass collecting every `tag_attr`-tagged method (across the whole MRO) into `registry_attr`."""

        def __new__(mcls, name, bases, namespace, **kwargs):
            cls = super().__new__(mcls, name, bases, namespace, **kwargs)
            registry: dict[str, str] = {}
            for base in reversed(cls.__mro__):
                for attr_name, attr in vars(base).items():
                    key = getattr(attr, tag_attr, None)
                    if key is not None:
                        registry[key] = attr_name
            setattr(cls, registry_attr, registry)
            return cls

    return decorator, _TaggedRegistryMeta


def combine_metaclasses(*metaclasses: type) -> type:
    """
    Build one metaclass whose MRO includes every metaclass in `metaclasses`.

    Parameters
    ----------
    *metaclasses : type
        The metaclasses to combine, most-specific first (mirroring how they'd be
        listed in a ``class Foo(Base1, Base2, metaclass=...)`` statement).

    Returns
    -------
    type
        A new metaclass subclassing every one of `metaclasses`, so a class using it
        triggers each metaclass's own `__new__` (via normal cooperative `super()`
        chaining) exactly once.
    """
    return type("_CombinedRegistryMeta", metaclasses, {})
