# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import matplotlib  # noqa: F401
# -- Project information -----------------------------------------------------

project = "UVEX Transients"
copyright = "2026, Eliza Diggins"
author = "Eliza Diggins"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.graphviz",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "sphinx.ext.doctest",
    "matplotlib.sphinxext.plot_directive",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["../_templates"]
bibtex_bibfiles = ["docs_bib.bib"]
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["transients/_page_template.rst", "user_guide/_page_template.rst"]

# -- Options for HTML output -------------------------------------------------
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

# Set the HTML theme to "pydata_sphinx_theme" and configure theme options for the project.
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "logo": {
        "text": "UVEX Transients",
        "image_light": "images/uvex_logo_dark.png",
        "image_dark": "images/uvex_logo.png",
        "alt_text": "UVEX Transients",
    },
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/TransientsExtragalactic/uvex-transients",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-square-github",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        }
    ],
}

html_favicon = "images/trex_logo.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

napoleon_use_param = True
napoleon_preprocess_types = True

# Suppress toc.not_included warnings for autosummary-generated attribute pages
# that are referenced by the class pages but not explicitly in a toctree.
suppress_warnings = ["toc.not_included"]

# Configure the sphinx galleries. These are contained in the
# /examples gallery.
sphinx_gallery_conf = {
    "examples_dirs": [
        "./galleries/schedules",
        "./galleries/simulating",
        "./galleries/custom_transients",
    ],
    "gallery_dirs": [
        "auto_examples/schedules",
        "auto_examples/simulating",
        "auto_examples/custom_transients",
    ],
    # Do not abort the build if an individual gallery example fails.
    "abort_on_example_error": False,
}
