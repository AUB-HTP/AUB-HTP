import importlib.metadata
import inspect
from pathlib import Path

import aub_htp

SOURCE = Path(__file__).parent

project = "AUB-HTP"
copyright = "2026, AUB-HTP"
author = "Ahmad El Hajj"
release = importlib.metadata.version("aub-htp")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
]

# api_stubs.rst only feeds autosummary; it is not a page.
autosummary_generate = ["api_stubs.rst"]
exclude_patterns = ["api_stubs.rst"]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

html_theme = "pydata_sphinx_theme"
html_sidebars = {"install": []}
html_theme_options = {
    "show_nav_level": 1,
    "show_toc_level": 2,
    "secondary_sidebar_items": ["page-toc"],
}


def heading(text):
    return f"{text}\n{'=' * len(text)}"


def autosummary(names, toctree=False):
    option = "   :toctree: .\n" if toctree else ""
    entries = "\n".join(f"   {name}" for name in names)
    return f".. autosummary::\n{option}\n{entries}\n"


def hidden_toctree(entries):
    lines = "\n".join(f"   {entry}" for entry in entries)
    return f".. toctree::\n   :hidden:\n\n{lines}\n"


def intro(module):
    """First paragraph of the module docstring, unless it is a heading or directive."""
    first = (inspect.getdoc(module) or "").split("\n\n")[0]
    lines = first.splitlines()
    if first.startswith("..") or (len(lines) > 1 and set(lines[1]) <= set("=-~^")):
        return ""
    return first


def subpackage(obj):
    """Public subpackage of aub_htp that obj comes from, or None."""
    module = getattr(obj, "__module__", None) or getattr(obj, "__name__", "")
    parts = module.split(".")
    if len(parts) < 2 or parts[1].startswith("_"):
        return None
    return parts[1]


def write_page(name, text):
    page = SOURCE / f"{name}.rst"
    if not page.exists() or page.read_text() != text:
        page.write_text(text)


def write_api_pages(app):
    """Write api.rst and one page per subpackage from aub_htp.__all__."""
    top, groups = [], {}
    for name in aub_htp.__all__:
        if name.startswith("_"):
            continue
        group = subpackage(getattr(aub_htp, name))
        if group:
            groups.setdefault(group, []).append(name)
        else:
            top.append(name)

    names = top + [name for members in groups.values() for name in members]
    pages = {f"aub_htp.{name}" for name in names} | {f"aub_htp.{group}" for group in groups}
    for page in SOURCE.glob("aub_htp*.rst"):
        if page.stem not in pages:
            page.unlink()

    write_page("api_stubs", f".. currentmodule:: aub_htp\n\n{autosummary(names, toctree=True)}")

    toctree = hidden_toctree([f"aub_htp.{name}" for name in top] + [f"aub_htp.{group}" for group in groups])
    sections = "\n".join(
        f".. rubric:: aub_htp.{group}\n\n{autosummary(members)}" for group, members in groups.items()
    )
    write_page("api", f"""{heading("API Reference")}

{intro(aub_htp)}

.. module:: aub_htp

{toctree}
{autosummary(top)}
{sections}""")

    for group, members in groups.items():
        module = f"aub_htp.{group}"
        toctree = hidden_toctree(f"{module}.{name} <aub_htp.{name}>" for name in members)
        write_page(module, f"""{heading(module)}

{intro(importlib.import_module(module))}

.. currentmodule:: aub_htp

{autosummary(members)}
{toctree}""")


def setup(app):
    # Run before autosummary, which generates the stub pages on the same event.
    app.connect("builder-inited", write_api_pages, priority=400)
