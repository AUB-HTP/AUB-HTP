from aub_htp import __version__ as package_version
project = "AUB-HTP"
copyright = '2026, AUB-HTP'
author = 'Ahmad El Hajj'
release = package_version
root_doc = "index"

extensions = [
   'sphinx.ext.duration',
   'sphinx.ext.doctest',
   'sphinx.ext.autodoc',
   'sphinx.ext.autosummary',
   'sphinx.ext.mathjax',
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

templates_path = ['_templates']
exclude_patterns = []


html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "show_nav_level": 1,
    "show_toc_level": 2,
    "secondary_sidebar_items": ["page-toc"],
}

html_static_path = ['_static']
