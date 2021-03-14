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

# import sphinx_book_theme
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))
# sys.setrecursionlimit(1500)


# -- Project information -----------------------------------------------------

project = 'sysidentpy'
copyright = '2020, Wilson Rocha, Luan Pascoal, Samuel Oliveira, Samir Martins'
author = 'Wilson Rocha, Luan Pascoal, Samuel Oliveira, Samir Martins'

# The full version, including alpha/beta/rc tags
release = '0.1.5'

autodoc_member_order = "bysource"
add_function_parentheses = True
add_module_names = True

extensions = ['myst_nb',
              'sphinx.ext.autodoc',
              'sphinx.ext.viewcode',
              'sphinx.ext.napoleon',
              "sphinx.ext.todo",
              'sphinx.ext.mathjax',
              # 'nbsphinx',
              'sphinx.ext.intersphinx',
              'IPython.sphinxext.ipython_console_highlighting',
              'sphinx.ext.githubpages',
              'sphinx_copybutton',
              'sphinx_togglebutton',
              # "ablog",
              ]

source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
    '.myst': 'myst-nb',
}

master_doc = "index"
# myst_admonition_enable = True
# myst_deflist_enable = True
myst_url_schemes = ("http", "https", "mailto")
jupyter_execute_notebooks = "cache"

# Napoleon settings
napoleon_google_docstring = False

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
html_theme = "sphinx_book_theme"
html_title = "NARMAX models"
html_logo = sys.path[0] + "\images\sysidentpy-logo.png"
html_baseurl = "http://sysidentpy.org"

htmlhelp_basename = "sysidentpydoc"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

html_theme_options = {
    "repository_url": "https://github.com/wilsonrljr/sysidentpy/",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "use_edit_page_button": True,
    "repository_branch": "master",
    "path_to_docs": "docs",
    "launch_buttons": {
        "colab_url": "https://colab.research.google.com",
        "notebook_interface": "jupyterlab"
    },
}
