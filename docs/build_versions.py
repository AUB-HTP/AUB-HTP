"""Build the docs of the latest release of every maintenance/v* branch.

Each release's package is installed from its tag and documented with the docs
source and Sphinx of this branch, into <out>/<series>/ (e.g. v1.1/), with a
version switcher added. The newest series is the default.

Usage: python docs/build_versions.py --base-url https://aub-htp.github.io/AUB-HTP
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

# Appended to the copied conf.py of each release.
SWITCHER_CONF = """

html_theme_options["switcher"] = {{"json_url": "{json_url}", "version_match": "{series}"}}
html_theme_options["navbar_end"] = ["version-switcher", "theme-switcher", "navbar-icon-links"]
html_theme_options["show_version_warning_banner"] = True
html_theme_options["check_switcher"] = False
"""


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def sphinx_requirements():
    """Sphinx and theme requirements from this branch's dev dependencies."""
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


def build(series, tag, out, json_url, workdir):
    tree = workdir / series
    git("worktree", "add", "--detach", str(tree), tag)
    try:
        source = tree / "docs" / "source"
        shutil.rmtree(source, ignore_errors=True)
        shutil.copytree(DOCS_SOURCE, source, ignore=shutil.ignore_patterns("api*.rst", "aub_htp*.rst"))
        conf = source / "conf.py"
        conf.write_text(conf.read_text() + SWITCHER_CONF.format(json_url=json_url, series=series))
        print(f"build {series} from {tag}")
        # Fresh environment: the release's package plus this branch's Sphinx.
        with_args = []
        for requirement in [str(tree), *sphinx_requirements()]:
            with_args += ["--with", requirement]
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        subprocess.run(
            ["uv", "run", "--isolated", "--no-project", *with_args,
             "python", "-m", "sphinx", "-E", "-b", "html", str(source), str(out / series)],
            check=True, cwd=workdir, env=env,
        )
    finally:
        shutil.rmtree(tree, ignore_errors=True)
        git("worktree", "prune")


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

    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    workdir = Path(tempfile.mkdtemp(prefix="docs-"))
    for series, tag in releases:
        build(series, tag, out, f"{base}/switcher.json", workdir)
    shutil.rmtree(workdir, ignore_errors=True)

    switcher = [
        {"name": tag, "version": series, "url": f"{base}/{series}/", "preferred": i == 0}
        for i, (series, tag) in enumerate(releases)
    ]
    (out / "switcher.json").write_text(json.dumps(switcher, indent=2))

    newest = releases[0][0]
    (out / "index.html").write_text(
        f'<!doctype html><meta http-equiv="refresh" content="0; url={newest}/">'
        f'<a href="{newest}/">Go to the latest docs</a>\n'
    )


if __name__ == "__main__":
    main()
