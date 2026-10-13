"""
Discovery of the simulation "runs" that the docs' Report page links to.

Each run is one directory of ``report_examples/``, which holds its analysis scripts (a
sphinx-gallery source tree), its ``GALLERY_HEADER.rst``, and a ``run.yaml`` describing it:

.. code-block:: yaml

    title: Infant Supernovae          # card and gallery title
    summary: One or two sentences.    # card blurb
    config: configs/infant_sne.yaml   # the run-config in the repository, shown on the card
    simulation: infant_sne            # get_results(simulation=...) name; omit for the primary run
    icon: zap                         # an octicon name for the card
    order: 2                          # card order, ascending (default 100)
    tags: [Type Ia, Type IIP]         # the populations simulated, shown as badges on the card

Adding a run to the Report is therefore a new directory with those three files, plus its scripts. Both
``conf.py`` (to register the galleries) and ``_report_gen.py`` (to draw the cards) read them through
`discover_runs`, so neither needs editing.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

__all__ = ["RunSpec", "discover_runs"]


@dataclass(frozen=True)
class RunSpec:
    """One simulation run on the Report page."""

    key: str
    """str: The directory name, which is also the run's URL segment (``report/<key>/index``)."""

    title: str
    """str: Display name."""

    summary: str
    """str: A sentence or two for the run's card."""

    source_dir: Path
    """pathlib.Path: The run's directory under ``report_examples/``."""

    config: str | None = None
    """str or None: Repository path of the run-config."""

    simulation: str | None = None
    """str or None: The ``simulation`` name for ``get_results``, `None` for the primary run."""

    icon: str = "graph"
    """str: Octicon name drawn on the card."""

    order: int = 100
    """int: Position among the cards, ascending."""

    tags: tuple[str, ...] = field(default_factory=tuple)
    """tuple of str: The populations the run simulates."""

    @property
    def is_primary(self) -> bool:
        """bool: Whether this is the release's primary run, the one the Report's summary table comes from."""
        return self.simulation is None

    @property
    def n_analyses(self) -> int:
        """int: How many example scripts the run's gallery holds, in all of its subsections."""
        return sum(1 for _ in self.source_dir.rglob("plot_*.py"))


def discover_runs(examples_dir: str | Path) -> list[RunSpec]:
    """
    Find every run under `examples_dir`, in card order.

    Parameters
    ----------
    examples_dir : str or pathlib.Path
        The ``report_examples`` directory. Subdirectories without a ``run.yaml`` are ignored.

    Returns
    -------
    list of RunSpec
        The runs sorted by their ``order`` and then their key.

    Raises
    ------
    ValueError
        If a ``run.yaml`` lacks a ``title`` or ``summary``.
    """
    runs = []
    for path in sorted(Path(examples_dir).glob("*/run.yaml")):
        meta = yaml.safe_load(path.read_text()) or {}
        missing = [name for name in ("title", "summary") if name not in meta]
        if missing:
            raise ValueError(f"{path} lacks {missing}.")
        runs.append(
            RunSpec(
                key=path.parent.name,
                title=str(meta["title"]),
                summary=" ".join(str(meta["summary"]).split()),
                source_dir=path.parent,
                config=meta.get("config"),
                simulation=meta.get("simulation"),
                icon=meta.get("icon", RunSpec.icon),
                order=int(meta.get("order", RunSpec.order)),
                tags=tuple(str(tag) for tag in meta.get("tags", ())),
            )
        )
    return sorted(runs, key=lambda run: (run.order, run.key))
