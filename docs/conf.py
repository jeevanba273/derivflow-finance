"""Sphinx configuration for the derivflow documentation."""

import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

sys.path.insert(0, os.path.abspath("../src"))

project = "derivflow"
author = "Jeevan B A"
copyright = "2025-2026, Jeevan B A"

try:
    release = _pkg_version("derivflow-finance")
except PackageNotFoundError:  # building from a source tree without installing
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# Only the optional extras are mocked; numpy and scipy are installed with the package.
autodoc_mock_imports = ["yfinance", "pandas", "plotly", "matplotlib"]
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "show-inheritance": True}

napoleon_numpy_docstring = True
napoleon_google_docstring = False

myst_enable_extensions = ["dollarmath", "amsmath"]
myst_heading_anchors = 3

html_theme = "sphinx_rtd_theme"
html_title = f"derivflow {release}"
html_static_path = ["_static"]
