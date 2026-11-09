import importlib
import importlib.metadata
import inspect
from pathlib import Path

import aub_htp

project = "AUB-HTP"
copyright = '2026, AUB-HTP'
author = 'Ahmad El Hajj'
release = importlib.metadata.version("aub-htp")

extensions = [
   'sphinx.ext.duration',
   'sphinx.ext.doctest',
   'sphinx.ext.autodoc',
   'sphinx.ext.autosummary',
   'sphinx.ext.mathjax',
]

# api_stubs.rst only lists the pages for autosummary to generate; it is not
# part of the docs, so the sidebar labels can be set by the toctrees below.
autosummary_generate = ["api_stubs.rst"]
exclude_patterns = ["api_stubs.rst"]


def write_page(name, lines):
    page = Path(__file__).with_name(f"{name}.rst")
    text = "\n".join(lines)
    if not page.exists() or page.read_text() != text:
        page.write_text(text)


def autosummary(names, toctree=False):
    return ([".. autosummary::"] + (["   :toctree: ."] if toctree else []) + [""]
            + [f"   {name}" for name in names] + [""])


def write_api_pages():
    """Write the API pages from aub_htp.__all__.

    Every name gets its own page at the docs root (aub_htp.<name>.html).
    In the sidebar, names from private modules are listed directly, and the
    others under a dropdown per subpackage, labelled aub_htp.<subpackage>.<name>.
    """
    top, groups = [], {}
    for name in aub_htp.__all__:
        if name.startswith("_"):
            continue
        subpackage = getattr(aub_htp, name).__module__.split(".")[1]
        if subpackage.startswith("_"):
            top.append(name)
        else:
            groups.setdefault(subpackage, []).append(name)

    every_name = top + [name for names in groups.values() for name in names]
    write_page("api_stubs", [".. currentmodule:: aub_htp", ""] + autosummary(every_name, toctree=True))

    api = ["API Reference", "=============", "", ".. module:: aub_htp", "",
           ".. toctree::", "   :hidden:", ""]
    api += [f"   aub_htp.{name}" for name in top]
    api += [f"   aub_htp.{subpackage}" for subpackage in groups] + [""]
    api += autosummary(top)
    for subpackage, names in groups.items():
        module = f"aub_htp.{subpackage}"
        intro = (inspect.getdoc(importlib.import_module(module)) or "").split("\n\n")[0]
        api += [f".. rubric:: {module}", ""] + autosummary(names)
        write_page(module, [module, "=" * len(module), "", intro, "",
                            ".. currentmodule:: aub_htp", ""] + autosummary(names)
                   + [".. toctree::", "   :hidden:", ""]
                   + [f"   {module}.{name} <aub_htp.{name}>" for name in names] + [""])
    write_page("api", api)


write_api_pages()

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "show_nav_level": 1,
    "show_toc_level": 2,
    "secondary_sidebar_items": ["page-toc"],
}
