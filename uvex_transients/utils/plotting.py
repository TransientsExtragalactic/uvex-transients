r"""
Shared plotting helpers for the "detection funnel" figures used throughout the docs and notebooks.

A detection funnel shows how many events of a population survive each progressively more expensive
screening stage (sampled -> magnitude-limited -> SNR-detected; see
`~uvex_transients.simulation.core.SurveySimulator`). Each stage count carries two independent
sources of uncertainty that are otherwise easy to conflate into a single, misleadingly precise bar:

- **MC (statistical) uncertainty**: every stage after the first is a binomial subsample of the raw
  Monte Carlo draws in the first ("sampled") stage, so its uncertainty is exactly the same
  Clopper-Pearson interval `~uvex_transients.simulation.event_catalog.EventCatalog.compute_detection_efficiency`
  and `~uvex_transients.simulation.event_catalog.EventCatalog.compute_yield_summary` already use for
  detection efficiency, propagated back into count units.
- **Rate (systematic) uncertainty**: the population's overall normalization, from
  `~uvex_transients.transients.base.ExtragalacticTransient.RATE_CI`. Because it is a pure
  multiplicative scale on the underlying rate, it applies identically to every stage's point
  estimate -- unlike the MC uncertainty, it does not shrink as later stages winnow the sample down.

`compute_funnel_bounds` computes both; `plot_detection_funnel` draws them, together with the bars
themselves, as two visually distinct error layers on the same `matplotlib.axes.Axes`.
"""

from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from numpy.typing import NDArray

from ..simulation._stats import clopper_pearson_interval

__all__ = ["compute_funnel_bounds", "plot_detection_funnel", "add_funnel_legend"]


def compute_funnel_bounds(
    stage_counts: Sequence[int],
    rate_ci: tuple[float, float] | None = None,
    confidence: float = 0.9,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""
    Compute per-stage MC and rate bounds for a detection-funnel's raw (undownsampled) counts.

    ``stage_counts[0]`` is taken as the funnel's binomial denominator :math:`n` -- the number of raw
    Monte Carlo draws in the "sampled" stage -- and every later stage's count :math:`k_i` is treated
    as a binomial subsample of it, with a `clopper_pearson_interval` bound on :math:`k_i/n` converted
    back into count units. This matches `~uvex_transients.simulation.event_catalog.EventCatalog.compute_yield_summary`'s
    own convention (feasible draws as :math:`n`, a later cut's count as :math:`k`), just applied to
    every stage rather than only the final one.

    `rate_ci` (an `~uvex_transients.transients.base.ExtragalacticTransient.RATE_CI`-style
    ``(lower, upper)`` multiplicative pair, or `None`) is applied uniformly to every stage's point
    estimate, since it rescales the population's overall rate normalization rather than any stage's
    selection efficiency.

    Parameters
    ----------
    stage_counts : sequence of int
        Raw (undownsampled) event counts at each funnel stage, in order, e.g.
        ``(len(catalog), len(mag_filtered), len(detected))``. Each entry must not exceed
        ``stage_counts[0]``.
    rate_ci : tuple of float, optional
        ``(lower, upper)`` multiplicative bounds, as `ExtragalacticTransient.RATE_CI`. `None`
        (the default) collapses the rate bounds to the point estimate, i.e. ``(1.0, 1.0)``.
    confidence : float, optional
        Confidence level for the Clopper-Pearson MC bounds. The default is ``0.9``.

    Returns
    -------
    mc_lower : numpy.ndarray
        Per-stage Clopper-Pearson lower bound, in the same (undownsampled) count units.
    mc_upper : numpy.ndarray
        Per-stage Clopper-Pearson upper bound, in the same (undownsampled) count units.
    rate_lower : numpy.ndarray
        Per-stage rate-normalization lower bound, in the same (undownsampled) count units.
    rate_upper : numpy.ndarray
        Per-stage rate-normalization upper bound, in the same (undownsampled) count units.

    Notes
    -----
    To plot alongside a downsample-rescaled ``counts`` array, multiply all four returned arrays by
    the same downsample factor first.
    """
    n = int(stage_counts[0])
    mc_lower = np.empty(len(stage_counts))
    mc_upper = np.empty(len(stage_counts))
    for i, k in enumerate(stage_counts):
        lower, upper = clopper_pearson_interval(int(k), n, confidence)
        mc_lower[i] = lower * n
        mc_upper[i] = upper * n

    lower_factor, upper_factor = rate_ci if rate_ci is not None else (1.0, 1.0)
    counts = np.asarray(stage_counts, dtype=float)
    rate_lower = counts * lower_factor
    rate_upper = counts * upper_factor

    return mc_lower, mc_upper, rate_lower, rate_upper


def plot_detection_funnel(
    ax: Axes,
    x: NDArray[np.float64],
    counts: NDArray[np.float64],
    mc_lower: NDArray[np.float64],
    mc_upper: NDArray[np.float64],
    rate_lower: NDArray[np.float64],
    rate_upper: NDArray[np.float64],
    color: str | Sequence[str],
    width: float = 0.8,
    label: str | None = None,
) -> None:
    r"""
    Draw one funnel's bars plus its MC and rate uncertainty layers onto `ax`.

    The two uncertainty sources are drawn as visually distinct layers rather than combined into one
    interval, since they behave differently (MC uncertainty narrows at later stages as the binomial
    denominator's share grows more certain; rate uncertainty is a constant fractional band on every
    stage) and conflating them would hide that:

    - **Rate (systematic)** uncertainty is drawn first, as a wide, pale-filled rectangle in `color`
      spanning the full bar width from `rate_lower` to `rate_upper`, outlined with an opaque edge
      in the same color -- a shaded envelope rather than an error bar, since it applies uniformly
      to every stage and reads more like a systematic "this whole bar could be scaled by ..." than
      a per-point measurement uncertainty. The opaque outline keeps both endpoints -- including the
      lower bound, which a capless translucent line alone tends to fade into the axes background --
      legible at a glance.
    - **MC (statistical)** uncertainty is drawn on top, as a narrow black error bar with caps -- the
      conventional per-point measurement-uncertainty treatment, layered over the systematic band so
      both remain legible at once.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw onto.
    x : numpy.ndarray
        Bar center positions, one per stage.
    counts : numpy.ndarray
        Bar heights (downsample-rescaled stage counts), one per stage.
    mc_lower, mc_upper, rate_lower, rate_upper : numpy.ndarray
        Bound arrays from `compute_funnel_bounds`, rescaled by the same downsample factor as
        `counts`.
    color : str or sequence of str
        Bar (and rate-band) color: either one color shared by every bar, or one color per bar
        (e.g. one per funnel stage, or one per transient type in a grouped funnel), matched
        positionally against `x`.
    width : float, optional
        Bar width, forwarded to `~matplotlib.axes.Axes.bar`. The default is ``0.8``.
    label : str, optional
        Legend label for the bars.
    """
    colors = [color] * len(x) if isinstance(color, str) else list(color)

    ax.bar(x, counts, width=width, color=colors, label=label, zorder=2)
    for xi, rlo, rhi, c in zip(x, rate_lower, rate_upper, colors):
        ax.bar(
            xi,
            rhi - rlo,
            width=width,
            bottom=rlo,
            facecolor=to_rgba(c, alpha=0.25),
            edgecolor=to_rgba(c, alpha=0.9),
            linewidth=1.2,
            zorder=1,
        )
    ax.errorbar(
        x,
        counts,
        yerr=[counts - mc_lower, mc_upper - counts],
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=3,
        capthick=1.2,
        zorder=3,
        label="_nolegend_",
    )


def add_funnel_legend(ax: Axes, loc: str = "best") -> None:
    r"""
    Add legend entries explaining `plot_detection_funnel`'s two uncertainty layers.

    Appends two proxy handles -- a black capped error bar labeled "MC (statistical)
    uncertainty" and a pale, outlined gray patch labeled "Rate (systematic) uncertainty" -- to whatever
    handles/labels `ax` already has (e.g. one legend entry per bar color/label from a grouped
    funnel), then redraws the legend with all of them together. The proxies use a neutral gray
    rather than matching any one bar's color, since both uncertainty layers have the same meaning
    for every bar regardless of its own color.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add the legend to; normally the same `ax` passed to one or more
        `plot_detection_funnel` calls.
    loc : str, optional
        Forwarded to `~matplotlib.axes.Axes.legend`. The default is ``"best"``.
    """
    handles, labels = ax.get_legend_handles_labels()

    stat_handle = ax.errorbar([], [], yerr=[[1]], fmt="none", ecolor="black", elinewidth=1.2, capsize=3, capthick=1.2)
    rate_handle = Patch(facecolor=to_rgba("0.5", alpha=0.25), edgecolor=to_rgba("0.5", alpha=0.9), linewidth=1.2)

    handles += [stat_handle, rate_handle]
    labels += ["MC (statistical) uncertainty", "Rate (systematic) uncertainty"]

    ax.legend(handles, labels, loc=loc)
