# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Make the package importable for autodoc (src layout).
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------

project = "DERIVFLOW-FINANCE"
author = "Jeevan B A"
copyright = "2026, Jeevan B A"

# Read the version from installed package metadata, with a safe fallback.
try:
    from importlib.metadata import version as _pkg_version

    release = _pkg_version("derivflow-finance")
except Exception:  # pragma: no cover - metadata may be unavailable on RTD
    release = "0.0.0"

# The short X.Y version.
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Support both reStructuredText and MyST Markdown sources.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- autodoc / autosummary ---------------------------------------------------

autodoc_typehints = "description"
autodoc_member_order = "bysource"
autosummary_generate = True

# Mock the heavy scientific/optional stack so autodoc can import the package
# without those dependencies installed (e.g. on Read the Docs). Only the
# package itself needs to be importable for signatures and docstrings.
autodoc_mock_imports = [
    "numba",
    "yfinance",
    "plotly",
    "matplotlib",
    "sklearn",
    "scikit-learn",
    "seaborn",
    "pytz",
    "pandas",
    "scipy",
    "numpy",
]

# -- Napoleon (NumPy-style docstrings) ---------------------------------------

napoleon_numpy_docstring = True
napoleon_google_docstring = False

# -- MyST parser -------------------------------------------------------------

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
]

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}
