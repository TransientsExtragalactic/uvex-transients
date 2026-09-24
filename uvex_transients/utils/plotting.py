r"""
Shared plotting infrastructure for uvex_transients.

Two layers live here:

- **General infrastructure** (`resolve_fig_axes`, `set_plot_style`, `get_default_cmap`, `get_cmap`,
  `get_band_color`) that every plotting function in the package and docs gallery is expected to go
  through, so that figure size, style, and color choices come from one place --
  ``config["plotting.*"]`` -- rather than being repeated (and drifting) at each call site.
- **Reusable generators** for the plot shapes that recur throughout the docs gallery: a full-sky
  HEALPix map plus its pooled histogram (`plot_healpix_map`/`plot_histogram`, both driven by
  `~uvex_transients.surveys.base.SurveySchedule`'s per-pixel diagnostics), a per-band light curve of
  theory curve + SNR-thresholded detections + upper limits (`plot_band_light_curve`, driven by
  `~uvex_transients.models.core.base.SpectralModel.simulate_photometry`), and the "detection funnel"
  figures below.

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

import astropy.units as u
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from numpy.typing import NDArray

from ..simulation._stats import clopper_pearson_interval
from . import resolve_healpix_resolution
from .config import config

__all__ = [
    "resolve_fig_axes",
    "set_plot_style",
    "get_default_cmap",
    "get_cmap",
    "get_band_color",
    "plot_healpix_map",
    "plot_histogram",
    "plot_band_light_curve",
    "compute_funnel_bounds",
    "plot_detection_funnel",
    "add_funnel_legend",
    "plot_rate_bars",
]


# ============================================================================== #
# General Infrastructure                                                         #
# ============================================================================== #
def resolve_fig_axes(
    fig: Figure | None = None,
    axes: Axes | None = None,
    fig_size: tuple | None = None,
    dpi: float | None = None,
    subplot_kw: dict | None = None,
) -> tuple[Figure, Axes]:
    """
    Resolve a ``(figure, axes)`` pair from any combination of already-provided pieces.

    Every plotting function in the package takes optional ``fig``/``axes`` arguments and starts by
    calling this, so callers can either let a function create its own figure or hand it axes already
    embedded in a larger layout (e.g. one panel of a `~matplotlib.pyplot.subplots` grid), without the
    function needing its own branching for the two cases.

    Parameters
    ----------
    fig : matplotlib.figure.Figure, optional
        An existing figure. If `None`, one is created (unless `axes` is given, in which case its
        parent figure is used).
    axes : matplotlib.axes.Axes, optional
        Existing axes to draw onto. If `None`, new axes are created on `fig`.
    fig_size : tuple, optional
        ``(width, height)`` in inches for a newly created figure, forwarded to
        `~matplotlib.pyplot.subplots`. Defaults to ``config["plotting.default_figsize"]``.
    dpi : float, optional
        Resolution for a newly created figure, forwarded to `~matplotlib.pyplot.subplots`. Defaults
        to ``config["plotting.dpi"]`` (the same default `set_plot_style` applies globally via
        ``rcParams["figure.dpi"]``); pass explicitly to override it for one figure, e.g. a
        higher-resolution figure meant to be saved as a small thumbnail.
    subplot_kw : dict, optional
        Forwarded to `~matplotlib.pyplot.subplots` when new axes are created, e.g.
        ``{"projection": "aitoff"}``. Ignored when `axes` is already given.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The resolved figure.
    axes : matplotlib.axes.Axes
        The resolved axes.
    """
    import matplotlib.pyplot as plt

    if fig_size is None:
        fig_size = config["plotting.default_figsize"]
    if dpi is None:
        dpi = config["plotting.dpi"]

    if fig is None and axes is None:
        fig, axes = plt.subplots(figsize=fig_size, dpi=dpi, subplot_kw=subplot_kw)
    elif fig is not None and axes is None:
        axes = fig.gca()
    elif fig is None and axes is not None:
        fig = axes.figure

    return fig, axes


def set_plot_style() -> None:
    """
    Apply the package's default `matplotlib` style, from ``config["plotting.*"]``.

    Sets figure resolution, tick appearance (inward-pointing major/minor ticks on all four sides,
    the conventional look for a publication-style plot), and optional LaTeX text rendering. Every
    plotting function that creates its own figure calls this first, so a single edit to
    ``config["plotting.*"]`` (or a project-local override; see
    `~uvex_transients.utils.config.get_config`) is enough to restyle every plot in the package and
    docs gallery consistently.
    """
    import matplotlib.pyplot as plt

    plt.rcParams["figure.dpi"] = config["plotting.dpi"]
    plt.rcParams["text.usetex"] = config["plotting.use_tex"]
    if config["plotting.use_tex"]:
        plt.rcParams["text.latex.preamble"] = config["plotting.latex_preamble"]

    plt.rcParams["xtick.major.size"] = 8
    plt.rcParams["xtick.minor.size"] = 5
    plt.rcParams["ytick.major.size"] = 8
    plt.rcParams["ytick.minor.size"] = 5
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"


def get_default_cmap() -> Colormap:
    """
    Return the package's default colormap, ``config["plotting.default_cmap"]``.

    Returns
    -------
    matplotlib.colors.Colormap
        The default colormap.
    """
    import matplotlib as mpl

    return mpl.colormaps[config["plotting.default_cmap"]]


def get_cmap(cmap: str | Colormap) -> Colormap:
    """
    Resolve a colormap name (or an already-resolved colormap) to a `~matplotlib.colors.Colormap`.

    Parameters
    ----------
    cmap : str or matplotlib.colors.Colormap
        Either the name of a registered Matplotlib colormap, or a `Colormap` instance (returned
        unchanged).

    Returns
    -------
    matplotlib.colors.Colormap
        The resolved colormap.

    Raises
    ------
    TypeError
        If `cmap` is neither a `str` nor a `Colormap`.
    ValueError
        If `cmap` is a `str` but names no registered colormap.
    """
    import matplotlib as mpl

    if isinstance(cmap, Colormap):
        return cmap

    if isinstance(cmap, str):
        try:
            return mpl.colormaps[cmap]
        except KeyError as exc:
            raise ValueError(f"Unknown colormap name {cmap!r}.") from exc

    raise TypeError("cmap must be either a matplotlib.colors.Colormap instance or a string colormap name.")


def get_band_color(band: str) -> str:
    """
    Look up a photometric band's plotting color, from ``config["plotting.band_colors"]``.

    Every gallery example and report figure that plots multiple bands on the same axes (e.g. UVEX's
    ``FUV``/``NUV`` alongside Rubin's ``u``/``g``/``r``/``i``/``z``/``y``) shares this single mapping,
    so a given band always reads as the same color everywhere it appears in the docs.

    Parameters
    ----------
    band : str
        Band name, e.g. ``"FUV"`` or ``"r"``.

    Returns
    -------
    str
        A hex color string. Bands not present in ``config["plotting.band_colors"]`` fall back to a
        color sampled from `get_default_cmap`, keyed by a hash of `band` so the same unknown band name
        always maps to the same color within a run.
    """
    band_colors = config["plotting.band_colors"]
    if band in band_colors:
        return band_colors[band]

    cmap = get_default_cmap()
    return to_rgba(cmap(hash(band) % 997 / 997))


# ============================================================================== #
# Reusable Generators                                                            #
# ============================================================================== #
def plot_healpix_map(
    values: NDArray,
    *,
    nside: int | None = None,
    order: str | None = None,
    title: str | None = None,
    cbar_label: str | None = None,
    cmap: str | Colormap | None = None,
    fig: Figure | None = None,
    ax: Axes | None = None,
    fig_size: tuple = (10, 5.5),
    s: float = 4,
) -> tuple[Figure, Axes]:
    r"""
    Aitoff-projected scatter of a full-sky HEALPix map, log-color-scaled.

    Non-positive and non-finite pixels are dropped rather than plotted, since every diagnostic this
    is meant for (a `~uvex_transients.surveys.base.SurveySchedule` per-pixel count, separation, or
    duration) is strictly positive, and the color scale is always logarithmic.

    Parameters
    ----------
    values : array-like
        Per-pixel values, ordered to match a HEALPix map at `nside`/`order` (i.e. length
        ``12 * nside**2``).
    nside, order : int, str, optional
        HEALPix resolution/ordering for `values`; see
        `~uvex_transients.utils.resolve_healpix_resolution` for the shared default.
    title : str, optional
        Axes title.
    cbar_label : str, optional
        Colorbar label.
    cmap : str or matplotlib.colors.Colormap, optional
        Defaults to `get_default_cmap`.
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes, optional
        Existing figure/axes to draw onto, via `resolve_fig_axes`; `ax`, if given, must already carry
        an ``"aitoff"`` projection.
    fig_size : tuple, optional
        Passed to `resolve_fig_axes` when creating a new figure.
    s : float, optional
        Marker size, forwarded to `~matplotlib.axes.Axes.scatter`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The resolved figure.
    ax : matplotlib.axes.Axes
        The aitoff-projected axes the map was drawn onto.
    """
    import astropy_healpix as ah
    from matplotlib.colors import LogNorm

    set_plot_style()
    nside, order = resolve_healpix_resolution(nside, order)
    resolved_cmap = get_cmap(cmap) if cmap is not None else get_default_cmap()

    hpx = ah.HEALPix(nside=nside, order=order, frame="icrs")
    lon, lat = hpx.healpix_to_lonlat(np.arange(hpx.npix))

    values = np.asarray(values, dtype=float)
    values = np.where(values > 0, values, np.nan)
    valid = np.isfinite(values)

    fig, ax = resolve_fig_axes(fig, ax, fig_size, subplot_kw={"projection": "aitoff"})
    sc = ax.scatter(
        lon[valid].wrap_at(180 * u.deg).radian,
        lat[valid].radian,
        c=values[valid],
        cmap=resolved_cmap,
        norm=LogNorm(vmin=np.min(values[valid]), vmax=np.max(values[valid])),
        s=s,
        rasterized=True,
    )
    ax.grid(True)
    fig.colorbar(sc, label=cbar_label, pad=0.05, shrink=0.7)
    if title is not None:
        ax.set_title(title)
    return fig, ax


def plot_histogram(
    values: NDArray,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = "Pixels",
    n_bins: int = 50,
    color: str | None = None,
    fig: Figure | None = None,
    ax: Axes | None = None,
    fig_size: tuple = (9, 5.5),
) -> tuple[Figure, Axes]:
    r"""
    Log-binned histogram of a strictly positive quantity, e.g. one pooled across a HEALPix map's pixels.

    Parameters
    ----------
    values : array-like
        Values to histogram. Non-positive and non-finite entries are dropped before binning.
    title : str, optional
        Axes title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label. The default, ``"Pixels"``, suits a per-pixel diagnostic; pass e.g.
        ``"Pairs of visits"`` for a per-pair one.
    n_bins : int, optional
        Number of log-spaced bins between `values`' min and max. The default is ``50``.
    color : str, optional
        Bar color. Defaults to a fixed point on `get_default_cmap`, so histograms across a page share
        one consistent color drawn from the same palette as any accompanying `plot_healpix_map` calls.
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes, optional
        Existing figure/axes to draw onto, via `resolve_fig_axes`.
    fig_size : tuple, optional
        Passed to `resolve_fig_axes` when creating a new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The resolved figure.
    ax : matplotlib.axes.Axes
        The axes the histogram was drawn onto.
    """
    set_plot_style()
    color = color if color is not None else to_rgba(get_default_cmap()(0.55))

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]

    fig, ax = resolve_fig_axes(fig, ax, fig_size)
    ax.hist(values, bins=np.geomspace(values.min(), values.max(), n_bins), color=color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    return fig, ax


def plot_band_light_curve(
    ax: Axes,
    band: str,
    t_obs: u.Quantity,
    phot,
    *,
    t_theory: u.Quantity | None = None,
    theory_mag: u.Quantity | NDArray | None = None,
    snr_threshold: float = 5.0,
    color: str | None = None,
    err_scale: float = 1.0,
    marker: str = "s",
    label: str | None = None,
) -> None:
    r"""
    Draw one photometric band's theory curve, SNR-detected points, and upper limits onto `ax`.

    This is the light-curve convention shared by every simulated-photometry gallery example: a
    faint, semi-transparent noiseless theory curve; SNR-detected visits as filled points with
    symmetric error bars; and fainter visits as downward-pointing open upper limits, from a
    `~uvex_transients.models.core.base.SpectralModel.simulate_photometry` table's asymmetric
    ``mag_lower``/``mag_upper`` bounds.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw onto.
    band : str
        Band name to select from `phot`'s ``"band"`` column, and (absent an explicit `color`) to look
        up via `get_band_color`.
    t_obs : astropy.units.Quantity
        Observation times, one per row of `phot` (not necessarily a column of `phot` itself, e.g. a
        precomputed "time since explosion").
    phot : astropy.table.Table
        A `~uvex_transients.models.core.base.SpectralModel.simulate_photometry`-style table with
        ``"band"``, ``"ab_mag"``, ``"snr"``, and ``"mag_err"`` columns; ``"mag_lower"``/``"mag_upper"``
        are used for upper limits when present, and upper limits are skipped otherwise.
    t_theory : astropy.units.Quantity, optional
        Time grid for the noiseless theory curve. Both `t_theory` and `theory_mag` must be given to
        draw it; the curve is omitted otherwise.
    theory_mag : astropy.units.Quantity or array-like, optional
        Noiseless AB magnitude at `t_theory`.
    snr_threshold : float, optional
        SNR above which a visit is drawn as a detection rather than an upper limit. The default is
        ``5.0``.
    color : str, optional
        Overrides `get_band_color(band)`.
    err_scale : float, optional
        Multiplicative factor applied to ``phot["mag_err"]`` for detected points' error bars (some
        galleries plot a wider, more conservative bar than the raw 1-sigma value). The default is
        ``1.0``.
    marker : str, optional
        Marker for detected points. The default is ``"s"``.
    label : str, optional
        Legend label for the detected points (upper limits are always unlabeled, since they share the
        detection's color and would otherwise duplicate its legend entry).
    """
    color = color if color is not None else get_band_color(band)

    if t_theory is not None and theory_mag is not None:
        theory_values = theory_mag.value if hasattr(theory_mag, "value") else np.asarray(theory_mag)
        ax.plot(u.Quantity(t_theory).to_value(u.day), theory_values, color=color, lw=1.5, alpha=0.6)

    in_band = np.isfinite(phot["ab_mag"]) & (np.asarray(phot["band"]) == band)
    detected = in_band & (phot["snr"] > snr_threshold)
    upper_limits = in_band & (phot["snr"] <= snr_threshold)

    t_days = u.Quantity(t_obs).to_value(u.day)

    if np.any(detected):
        ax.errorbar(
            t_days[detected],
            phot["ab_mag"][detected],
            yerr=err_scale * phot["mag_err"][detected],
            marker=marker,
            mfc=color,
            mec="k",
            ecolor=color,
            linestyle="none",
            label=label,
        )

    has_bounds = "mag_lower" in phot.colnames and "mag_upper" in phot.colnames
    if np.any(upper_limits) and has_bounds:
        ax.errorbar(
            t_days[upper_limits],
            phot["ab_mag"][upper_limits],
            yerr=[
                phot["mag_upper"][upper_limits] - phot["ab_mag"][upper_limits],
                np.abs(phot["mag_lower"][upper_limits] - phot["ab_mag"][upper_limits]),
            ],
            marker="v",
            mfc="w",
            mec=color,
            ecolor=color,
            linestyle="none",
        )


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
    color: str | Sequence[str] | None = None,
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
    color : str or sequence of str, optional
        Bar (and rate-band) color: either one color shared by every bar, or one color per bar
        (e.g. one per funnel stage, or one per transient type in a grouped funnel), matched
        positionally against `x`. Defaults to ``config["plotting.funnel_color"]``.
    width : float, optional
        Bar width, forwarded to `~matplotlib.axes.Axes.bar`. The default is ``0.8``.
    label : str, optional
        Legend label for the bars.
    """
    if color is None:
        color = config["plotting.funnel_color"]
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


def plot_rate_bars(
    ax: Axes,
    categories: Sequence[str],
    visible_counts: Sequence[int],
    n_samples: int,
    all_sky_rate: u.Quantity,
    *,
    rate_ci: tuple[float, float] | None = None,
    confidence: float = 0.9,
    color: str | Sequence[str] | None = None,
    width: float = 0.6,
    rate_unit: u.Unit = 1 / u.yr,
    label: str | None = None,
) -> NDArray[np.float64]:
    r"""
    Bar chart of an all-sky rate estimate per category (e.g. band), with MC + rate uncertainty.

    Each category's rate is estimated as the fraction of `n_samples` Monte Carlo draws that satisfy
    some visibility criterion (e.g. "peak apparent magnitude below the survey's limit"), scaled by
    the population's `all_sky_rate`. That is exactly `compute_funnel_bounds`'s "sampled ->
    screened" convention with `n_samples` playing the role of its shared binomial denominator, so the
    same two uncertainty layers `plot_detection_funnel` draws for a detection funnel apply here too:
    MC (statistical) uncertainty from treating each category's count as a Clopper-Pearson binomial
    subsample of `n_samples`, and rate (systematic) uncertainty from `rate_ci` (see
    `~uvex_transients.transients.base.ExtragalacticTransient.RATE_CI`), which scales every category's
    point estimate by the same factor rather than shrinking with the sample.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw onto.
    categories : sequence of str
        Bar labels, e.g. band names.
    visible_counts : sequence of int
        Number of the `n_samples` draws satisfying each category's visibility criterion, e.g.
        ``np.count_nonzero(peak_mag < limit)`` per band.
    n_samples : int
        Total Monte Carlo draws each `visible_counts` entry is a subsample of.
    all_sky_rate : astropy.units.Quantity
        The population's all-sky rate (`~uvex_transients.transients.base.ExtragalacticTransient.all_sky_rate`);
        each bar height is ``visible_counts[i] / n_samples * all_sky_rate``.
    rate_ci : tuple of float, optional
        ``(lower, upper)`` multiplicative rate bounds; see `compute_funnel_bounds`. `None` (the
        default) omits the rate (systematic) uncertainty band.
    confidence : float, optional
        Confidence level for the MC (statistical) uncertainty. The default is ``0.9``.
    color : str or sequence of str, optional
        Forwarded to `plot_detection_funnel`.
    width : float, optional
        Bar width. The default is ``0.6``.
    rate_unit : astropy.units.Unit, optional
        Unit the bars (and the returned array) are expressed in. The default is ``1 / u.yr``.
    label : str, optional
        Legend label for the bars themselves.

    Returns
    -------
    numpy.ndarray
        The plotted bar heights (visible rates), in `rate_unit`.

    See Also
    --------
    add_funnel_legend : Adds legend entries explaining the two uncertainty layers.
    """
    set_plot_style()

    stage_counts = [n_samples, *visible_counts]
    mc_lower, mc_upper, rate_lower, rate_upper = compute_funnel_bounds(
        stage_counts, rate_ci=rate_ci, confidence=confidence
    )
    scale = (all_sky_rate / n_samples).to_value(rate_unit)

    x = np.arange(len(categories))
    rates = np.asarray(visible_counts, dtype=float) * scale

    plot_detection_funnel(
        ax,
        x=x,
        counts=rates,
        mc_lower=mc_lower[1:] * scale,
        mc_upper=mc_upper[1:] * scale,
        rate_lower=rate_lower[1:] * scale,
        rate_upper=rate_upper[1:] * scale,
        color=color,
        width=width,
        label=label,
    )
    ax.set_xticks(x, categories)

    return rates
