"""Every example script and every notebook runs as written.

Scripts run in a subprocess. Notebook code cells are concatenated and run in a
subprocess as well (no Jupyter kernel needed). The market data notebook needs
internet access and is marked 'network'.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((ROOT / "examples").glob("*.py"))
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))


def _run(*args):
    cmd = [sys.executable, *map(str, args)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=ROOT)
    assert result.returncode == 0, result.stderr[-3000:]
    return result.stdout


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.name)
def test_example_runs(script):
    out = _run(script)
    assert out.strip()


def _notebook_code(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n\n".join(
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
    )


@pytest.mark.parametrize(
    "notebook",
    [
        pytest.param(nb, marks=pytest.mark.network) if "market_data" in nb.name else nb
        for nb in NOTEBOOKS
    ],
    ids=lambda p: p.name,
)
def test_notebook_runs(notebook):
    if "dashboards" in notebook.name:
        pytest.importorskip("plotly")
    out = _run("-c", _notebook_code(notebook))
    assert out.strip()
