#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PyREx documentation build configuration file, created by
# sphinx-quickstart on Thu Aug 24 15:12:35 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# Grab information about package without loading package
about = {}
with open(os.path.join(os.path.join("..", "pyrex"), "__about__.py")) as f:
    exec(f.read(), about)


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'numpydoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.extlinks',
            #   'fulltoc',
            #   'sphinx.ext.todo',
            #   'sphinx.ext.coverage',
              ]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
module_name = about['__modulename__']
project = about['__fullname__']
copyright = about['__copyright__']
author = about['__author__']
VERSION = about['__version__']
description = about['__description__']

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = VERSION[:VERSION.rindex('.')]
# The full version, including alpha/beta/rc tags.
release = VERSION

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
# todo_include_todos = True


# default_role = 'any'
default_role = 'autolink'


# -- Options for docstrings -----------------------------------------------

# For some reason this is necessary to suppress unknown document errors
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_class_members_toctree = False

# Generate referenced documentation
autosummary_generate = True


# Auto-fill arxiv and doi links in docstrings
extlinks = {
    'arxiv': ('https://arxiv.org/abs/%s', 'arXiv:'),
    'doi': ('https://dx.doi.org/%s', 'DOI:'),
}


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages. See the documentation for
# a list of builtin themes.
#
html_theme = 'scipy'
html_theme_path = ['_theme']
# html_theme = 'alabaster'

# Theme options are theme-specific and customize the look and feel of a theme
# further. For a list of options available for each theme, see the
# documentation.

html_theme_options = {
    "edit_link": False,
    "sidebar": "right",
    "scipy_org_logo": True, # edited to PyREx logo in layout.html
    "rootlinks": [],
    "description": description+"\nVersion "+VERSION,
    "github_user": "bhokansonfasig",
    "github_repo": "pyrex",
    # "logo": "logo.png",
    "nav_depth": 2,
    "extra_nav_links": {
        "Source (GitHub)": "https://github.com/bhokansonfasig/pyrex",
        "Report an Issue": "https://github.com/bhokansonfasig/pyrex/issues",
    },
}
# html_logo = '_static/logo.png'

# Sidebar navigation
html_sidebars = {
    '**': [
        'about.html',
        'searchbox.html',
        'navigation.html',
    ]
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Add searchtools.js script to make searching work. Looks like it was left out
# of the search.html page for the scipy theme, but this adds it back.
html_js_files = ['searchtools.js']

html_domain_indices = True



# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = project+'doc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',

    'classoptions': ',openany,oneside'
}

# File containing project logo for title page
latex_logo = './_static/logo_wide.png'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, project+'.tex', project+' Documentation',
     author, 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, module_name, project+' Documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, project, project+' Documentation',
     author, project, description,
     'Miscellaneous'),
]


# -- Options for autodoc --------------------------------------------------

# Sort member functions as they are in the source code
# autodoc_member_order = 'bysource'

# Flags automatically added
# autodoc_default_flags = ['members', 'show-inheritance',
#                          'inherited-members']

# Special controls for processing the docstrings
# def custom_process_docstring(app, what, name, obj, options, lines):
#     if what=="attribute" and "Particle" in name:
#         lines.clear()

# def custom_skip_init(app, what, name, obj, skip, options):
#     if name=="__init__":
#         return True
#     return skip

# def setup(app):
#     app.connect('autodoc-process-docstring', custom_process_docstring)
#     app.connect('autodoc-skip-member', custom_skip_init)
