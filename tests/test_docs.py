"""Code blocks in README.md and docs/ run as written.

Printed output is compared with the ```text block under each code block only
when DERIVFLOW_CHECK_DOC_OUTPUT=1 (seeded Monte Carlo output can change in the
last printed digit across NumPy versions). Refresh outputs with
``python docs/snippets.py README.md docs/*.md docs/user-guide/*.md``.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs"))
import snippets  # noqa: E402

PAGES = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
NEEDS = {"visualization.md": ("plotly", "pandas")}


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(ROOT)))
def test_page_code_runs(page):
    for module in NEEDS.get(page.name, ()):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"{page.name} needs {module}")
    results = snippets.run_page(page)
    if os.environ.get("DERIVFLOW_CHECK_DOC_OUTPUT") == "1":
        for printed, expected, _ in results:
            if expected is not None:
                assert printed == expected
