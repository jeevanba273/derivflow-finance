"""The installed version, ``derivflow_finance.__version__`` and pyproject.toml must agree."""

import re
from importlib.metadata import version
from pathlib import Path

import derivflow_finance

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_version_matches_installed_metadata():
    assert derivflow_finance.__version__ == version("derivflow-finance")


def test_version_matches_pyproject():
    text = PYPROJECT.read_text(encoding="utf-8")
    declared = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE).group(1)
    assert derivflow_finance.__version__ == declared


def test_package_info_reports_runtime_version():
    info = derivflow_finance.get_package_info()
    assert info["version"] == derivflow_finance.__version__
    assert info["name"] == "derivflow-finance"
