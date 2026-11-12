"""Build the docs site: the home and install pages once at the root, and the API reference of the latest release of every maintenance/v* branch.

The package comes from each release tag; docs/source and Sphinx come from the current branch.

Usage: python docs/build_docs.py --base-url https://aub-htp.github.io/AUB-HTP
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path

DOCS_SOURCE = Path(__file__).resolve().parent / "source"
PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

SITE_CSS = ".nav-link.nav-external::after { content: none; }\n"

SHARED_CONF = """
html_static_path = ["_static"]
html_css_files = ["site.css"]
"""

ROOT_CONF = SHARED_CONF + """
exclude_patterns += ["api.rst", "aub_htp*.rst"]
html_title = f"{project} documentation"
"""

SERIES_CONF = SHARED_CONF + """
exclude_patterns += ["install.rst"]
html_theme_options["logo"] = {{"link": "{base}/index.html"}}
html_theme_options["switcher"] = {{"json_url": "{base}/switcher.json", "version_match": "{series}"}}
html_theme_options["navbar_end"] = ["version-switcher", "theme-switcher", "navbar-icon-links"]
html_theme_options["show_version_warning_banner"] = True
html_theme_options["check_switcher"] = False
"""

SERIES_INDEX = """AUB-HTP
=======

.. toctree::
   :hidden:

   AUB-HTP <{base}/index.html>
   Installation <{base}/install.html>
   api
"""


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def sphinx_requirements():
    """Sphinx and theme requirements from the dev dependencies."""
    dev = tomllib.loads(PYPROJECT.read_text())["dependency-groups"]["dev"]
    return [r for r in dev if r.startswith(("sphinx", "pydata-sphinx-theme"))]


def latest_releases():
    """Return [(series, tag)], newest series first, e.g. [("v1.1", "v1.1.1")]."""
    branches = git("branch", "-r", "--list", "origin/maintenance/v*").split()
    tags = git("tag", "--list", "v*").split()
    releases = []
    for branch in branches:
        series = branch.removeprefix("origin/maintenance/")
        matching = [t for t in tags if re.fullmatch(rf"{re.escape(series)}\.\d+", t)]
        if not matching:
            print(f"skip {series}: no release tag")
            continue
        tag = max(matching, key=lambda t: int(t.rsplit(".", 1)[1]))
        releases.append((series, tag))
    return sorted(releases, key=lambda r: [int(n) for n in r[0][1:].split(".")], reverse=True)


def build(tag, out, workdir, conf, pages):
    """Build the docs source into out, against the package at tag.

    conf is appended to conf.py and pages ({path: text}) are written over the source.
    """
    name = out.name
    tree = workdir / name
    git("worktree", "add", "--detach", str(tree), tag)
    try:
        source = tree / "docs" / "source"
        shutil.rmtree(source, ignore_errors=True)
        shutil.copytree(DOCS_SOURCE, source, ignore=shutil.ignore_patterns("api*.rst", "aub_htp*.rst"))
        conf_py = source / "conf.py"
        conf_py.write_text(conf_py.read_text() + conf)
        for path, text in pages.items():
            (source / path).parent.mkdir(parents=True, exist_ok=True)
            (source / path).write_text(text)
        print(f"build {name} from {tag}")
        with_args = []
        for requirement in [str(tree), *sphinx_requirements()]:
            with_args += ["--with", requirement]
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        subprocess.run(
            ["uv", "run", "--isolated", "--no-project", *with_args,
             "python", "-m", "sphinx", "-E", "-b", "html", "-d", str(workdir / "doctrees" / name),
             str(source), str(out)],
            check=True, cwd=workdir, env=env,
        )
    finally:
        git("worktree", "remove", "--force", str(tree))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="URL the site is served from")
    parser.add_argument("--out", default="docs/_build/site", help="output folder")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    out = Path(args.out).resolve()
    releases = latest_releases()
    if not releases:
        raise SystemExit("no releases with docs found")
    newest, newest_tag = releases[0]

    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    workdir = Path(tempfile.mkdtemp(prefix="docs-"))
    try:
        for series, tag in releases:
            pages = {"index.rst": SERIES_INDEX.format(base=base), "_static/site.css": SITE_CSS}
            build(tag, out / series, workdir, SERIES_CONF.format(base=base, series=series), pages)
            (out / series / "index.html").write_text(
                '<!doctype html><meta http-equiv="refresh" content="0; url=api.html">'
                '<a href="api.html">Go to the API reference</a>\n'
            )

        index = re.sub(r"^   api$", f"   API Reference <{base}/{newest}/api.html>",
                       (DOCS_SOURCE / "index.rst").read_text(), flags=re.M)
        build(newest_tag, out, workdir, ROOT_CONF, {"index.rst": index, "_static/site.css": SITE_CSS})
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    switcher = [
        {"name": tag, "version": series, "url": f"{base}/{series}/", "preferred": i == 0}
        for i, (series, tag) in enumerate(releases)
    ]
    (out / "switcher.json").write_text(json.dumps(switcher, indent=2))


if __name__ == "__main__":
    main()
