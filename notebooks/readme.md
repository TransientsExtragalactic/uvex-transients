# UVEX Transients Visualization Notebooks

---

The notebooks in this directory are intended to act as plug-and-play scripts for
visualizing different elements of the UVEX transients data. Because many of these are fairly complex
visualizations, we place them into notebooks to make them easier to modify / customize without
a bunch of boilerplate code in the main codebase.

Below are the various notebooks available in this directory and a brief
description of what they do.

## Single-Transient Visualizations

- [`single_transient/footprint_and_lightcurve_animation.ipynb`](single_transient/footprint_and_lightcurve_animation.ipynb):

   For a **single transient**, this notebook produces a visualization of the UVEX fields around the event site during
   a specific time window along with the transient's lightcurve. The static visualization here provides
   an overlay of the total exposure time of the UVEX fields around the event site along with the complete lightcurve.
   The animated version shows a time-lapse of the UVEX observations on top of the transient's brightening and decline. On the
   other panel, the lightcurve is drawn with a moving vertical line to indicate the current time in the animation.

## Simulation-Result Visualizations

- [`simulation_results/all_sky_transients.ipynb`](simulation_results/all_sky_transients.ipynb):

   Turns a simulation run into a movie of the whole survey. An all-sky Aitoff map of the UVEX fields builds up
   observation by observation, shaded by the number of visits, while every detected transient appears as a dot,
   colored by type, that brightens and fades over its lifetime and gets a rim when it is first detected. Next to the map,
   one vertical bar per transient type fills with the expected number of detections. It needs only the run's
   `event_summary.ecsv` (downloaded from a release by default) and the survey schedule. Each dot lasts from the
   explosion to the later of the last detection and a per-type minimum duration set at the top of the notebook, which
   keeps fast events on screen long enough to see.

- [`simulation_results/transient_redshift_distribution.ipynb`](simulation_results/transient_redshift_distribution.ipynb):

   Plots the redshift distribution of each transient type that UVEX is expected to detect, as the expected number of
   real events per redshift bin. A combined figure puts every type on one set of axes (translucent overlapping
   distributions, cumulative distributions, or one offset row per type), and a second figure gives each type its own
   panel. Every bin carries two separate uncertainties: the Monte Carlo (Clopper-Pearson) interval, with an upper limit
   in empty bins, and the correlated volumetric-rate band. It also builds a table of every bin that it can save. It
   needs only the run's `event_summary.ecsv` (downloaded from a release by default).

All three notebooks write their figures, animations and tables to the repo-level `results/` directory by default
(see [Data Handling](#data-handling)).


# Developer Instructions

---

In an effort to standardize these notebooks in a manner which makes them easy to manipulate and
to work with, we have a few guidelines for the generation of new notebooks.

## Notebook Structure

The notebook structure should follow the following format:

1. **Overview**: Describes the purpose of the notebook, what it does, any caveats, etc.
2. **Setup and Configuration**: This section should contain any imports, configuration settings, downloads, etc.

   - Various user-modifiable settings should be named in ``ALL_CAPS`` and provided in a single cell with
     comments describing what they do. This makes it easy for users to modify the notebook without
     having to dig through the code.
   - The standard **plot style** cell (see [Plot Style](#plot-style)) goes right after the imports, so that every
     notebook looks the same.
3. **Data Loading**: This section should contain the code to load the data needed for the notebook. This may include
   downloading data from a remote source, loading from local files, or generating synthetic data.
4. **Data Processing**: This section should contain the code to process the data into a format suitable for visualization. This may include filtering, aggregating, or transforming the data.
5. **Visualization**: This section should contain the code to generate the visualizations. This may include creating plots, animations, or interactive visualizations.

## Conventions

### Data Handling

All notebooks should declare in the **overview** what data they require. These should be standard outputs from
the simulation pipeline or other parts of the UVEX Transients code. The user should be able to supply these items as part of the
setup of the notebook. In cases where it is possible, the notebook should use the available downloader to download a default dataset if none is
provided.

Notebooks write figures, animations and tables to the repo-level `results/` directory (gitignored) by default, set by an
`OUTPUT_DIR` constant in the setup section (`Path("../../results")`, relative to the notebook's own directory).

### Plot Style

Every notebook starts from the same plot style: a publication look (AAS two-column-ish, LaTeX Computer Modern text) with
the UVEX palette. Copy the cell below, unchanged, into the setup section of a new notebook, right after the imports.
It also defines the palette constants (`UVEX_BLUE`, `UVEX_VIOLET`, ...), a registered colormap named `"uvex"`, and
a small `tex()` helper that escapes the characters LaTeX treats specially (`_ % & #`), so that plain text such as
`iip_sne` or `90%` can go into labels when LaTeX is on.

The palette comes from the UVEX logo, whose gradient runs from blue at the top to violet at the bottom:

| Name | Color | Used for |
| --- | --- | --- |
| `UVEX_BLUE` | `#72AEFF` | the blue end of the logo gradient; the pale high end of the colormap |
| `UVEX_VIOLET` | `#AF6EFF` | the violet end of the logo gradient |
| `UVEX_BLUE_DEEP` | `#3B87EB` | first color of the default cycle; thin lines and markers on white |
| `UVEX_VIOLET_DEEP` | `#9742ED` | second color of the default cycle |
| `UVEX_NAVY` | `#0A0B1E` | dark background for images and maps |

The default color cycle is the two deep logo hues followed by four accents (teal, amber, pink, slate) that stay apart
from them and from each other, since a cycle of only blues and violets cannot be told apart. The sequential colormap
`"uvex"` runs from indigo through violet and blue to pale, with steadily rising lightness, so it also reads in
grayscale. When a quantity has a natural ordering by wavelength, use the gradient: for example, the shorter-wavelength
FUV band is violet and the NUV band is blue.

LaTeX is used when `latex` and `dvipng` are found on the path, and matplotlib's own Computer Modern look-alike
otherwise, so the notebooks still run without a TeX installation. Size figures for about 3.5 inches (one column) or
7 to 8 inches (full width), because the font sizes are set for that.

```python
# ---------------------------------------------------------------------------------------------- #
# UVEX Transients notebook style. Copy this cell unchanged into every notebook (see the readme), #
# so that all of them look the same.                                                             #
# ---------------------------------------------------------------------------------------------- #
import shutil

import matplotlib as mpl
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

# The UVEX logo gradient runs from blue (top) to violet (bottom). These are its colors, sampled from the logo files.
UVEX_BLUE = "#72AEFF"
UVEX_VIOLET = "#AF6EFF"
# The same hues, deepened so that thin lines and small markers stay legible on a white background.
UVEX_BLUE_DEEP = "#3B87EB"
UVEX_VIOLET_DEEP = "#9742ED"
UVEX_NAVY = "#0A0B1E"  # A dark background for images and maps.

# Default color cycle: the two logo hues first, then accents that stay apart from them and from each other.
UVEX_CYCLE = [UVEX_BLUE_DEEP, UVEX_VIOLET_DEEP, "#13A89E", "#F2A93B", "#E5509A", "#475569"]

# Sequential colormap for images and maps, "uvex": indigo (low) through violet and blue to pale (high). Its lightness
# rises steadily, so it also reads in grayscale. The low end is lighter than UVEX_NAVY, so faint pixels stay visible.
UVEX_CMAP = LinearSegmentedColormap.from_list("uvex", ["#2A2078", "#4B2BA8", UVEX_VIOLET, UVEX_BLUE, "#EAF4FF"])
mpl.colormaps.register(UVEX_CMAP, name="uvex", force=True)

# Text is set with LaTeX when a TeX installation is found, and with matplotlib's own Computer Modern look-alike if not.
USE_LATEX = shutil.which("latex") is not None and shutil.which("dvipng") is not None

mpl.rcParams.update(
    {
        # --- Publication style (AAS two-column-ish, LaTeX/Computer Modern) ---
        "text.usetex": USE_LATEX,
        "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.major.size": 3,
        "ytick.minor.size": 1.5,
        "lines.linewidth": 1.0,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        # --- UVEX colors ---
        "axes.prop_cycle": cycler(color=UVEX_CYCLE),
        "image.cmap": "uvex",
        "legend.frameon": False,
    }
)


def tex(text):
    """Escape the characters LaTeX treats specially (_ % & #), so plain text can go into labels when LaTeX is on."""
    if mpl.rcParams["text.usetex"]:
        for char in "_%&#":
            text = text.replace(char, "\\" + char)
    return text
```
