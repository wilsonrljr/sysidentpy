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

import sphinx_glpi_theme
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))
# sys.setrecursionlimit(1500)


# -- Project information -----------------------------------------------------

project = 'sysidentpy'
copyright = '2020, Wilson Rocha, Luan Pascoal, Samuel Oliveira, Samir Martins'
author = 'Wilson Rocha, Luan Pascoal, Samuel Oliveira, Samir Martins'

# The full version, including alpha/beta/rc tags
release = '0.1.3'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

autodoc_member_order = "bysource"

html_theme_options = {
    "description": "System Identification in Python",
    "body_text_align": "left",
    "github_user": "wilsonrljr",
    "github_repo": "sysidentpy",
    "show_relbars": True,
    "page_width": "80%",
    "github_button": True,
}

add_function_parentheses = True

add_module_names = True

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.viewcode',
              'sphinx.ext.napoleon',
              # 'sphinx_autodoc_typehints',
              "sphinx.ext.todo",
              'sphinx.ext.mathjax',
              # 'numpydoc',
              "nbsphinx",
              "sphinx.ext.intersphinx",
              "IPython.sphinxext.ipython_console_highlighting",
              'sphinx.ext.githubpages',
              ]

source_suffix = ".rst"

master_doc = "index"

# Napoleon settings
# napoleon_google_docstring = False
# napoleon_numpy_docstring = True
# napoleon_include_init_with_doc = False
# napoleon_include_private_with_doc = False
# napoleon_include_special_with_doc = False
# napoleon_use_admonition_for_examples = False
# napoleon_use_admonition_for_notes = False
# napoleon_use_admonition_for_references = False
# napoleon_use_ivar = False
# napoleon_use_param = True
# napoleon_use_rtype = True


latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    'pointsize': '11pt',

    # Additional stuff for the LaTeX preamble.
    'preamble': r'''
        \usepackage{charter}
        % \usepackage[defaultsans]{lato}
        % \usepackage{inconsolata}
    ''',
}
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = "glpi"

html_title = "SysIdentPy"

htmlhelp_basename = "sysidentpydoc"


html_theme_path = sphinx_glpi_theme.get_html_themes_path()

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme_options = {
    'logo_only': True,
    'navigation_depth': 5,
}
