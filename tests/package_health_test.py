import pytest
from pathlib import Path
import tomli
from aub_htp import __version__ as package_version


def test_version_consistency():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as pyproject_file:
        pyproject_data = tomli.load(pyproject_file)

    assert package_version == pyproject_data["project"]["version"]
