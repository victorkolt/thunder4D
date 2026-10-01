# Sphinx configuration for the thunder4d documentation
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "THUNDER4D"
author = "THUNDER4D contributors"
copyright = "2026, THUNDER4D contributors"
try:
    from thunder4d import __version__ as release
except Exception:  # documentation must build even without numpy installed
    release = "0.1.0"
version = release

extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",   # writes .nojekyll so GitHub Pages serves _static/
    "sphinx.ext.viewcode",
]
templates_path = []
exclude_patterns = []
default_role = "math"
primary_domain = "py"
highlight_language = "python3"

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = f"THUNDER4D {release} documentation"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "style_external_links": True,
}
html_show_sourcelink = True
numfig = True
