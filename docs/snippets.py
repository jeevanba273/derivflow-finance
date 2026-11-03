"""
Run the Python code blocks of Markdown pages and refresh their printed output.

In a page, a ```python block may be followed by a ```text block holding what
the code prints. Blocks of one page run in order in a shared namespace. A
python block whose first line starts with "# not run" is skipped (used for
code that needs internet access).

    python docs/snippets.py README.md docs/user-guide/*.md          # rewrite outputs
    python docs/snippets.py --check README.md docs/user-guide/*.md  # compare only

tests/test_docs.py executes every block of README.md and docs/ (it compares
outputs only when DERIVFLOW_CHECK_DOC_OUTPUT=1, because seeded Monte Carlo
output can differ in the last digit between NumPy versions).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
import sys
from pathlib import Path

FENCE = re.compile(r"^```(\w+)[^\n]*\n(.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)


def blocks(text: str) -> list[tuple[str, str, int, int]]:
    """(language, body, start, end) of every fenced block."""
    return [(m.group(1), m.group(2), m.start(), m.end()) for m in FENCE.finditer(text)]


def run_page(path: Path) -> list[tuple[str, str | None, tuple[int, int] | None]]:
    """Execute the page; return (code, expected output or None, span of the output block)."""
    text = path.read_text(encoding="utf-8")
    found = blocks(text)
    namespace: dict = {"__name__": "__docs__"}
    results = []
    for i, (lang, body, _, end) in enumerate(found):
        if lang != "python" or body.lstrip().startswith("# not run"):
            continue
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(compile(body, f"{path}:block{i}", "exec"), namespace)
        expected, span = None, None
        if i + 1 < len(found):
            nlang, nbody, nstart, nend = found[i + 1]
            between = text[end:nstart]
            if nlang == "text" and between.strip() == "":
                expected, span = nbody, (nstart, nend)
        results.append((out.getvalue(), expected, span))
    return results


def update(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    changed = False
    for printed, expected, span in reversed(run_page(path)):
        if span is None or printed == expected:
            continue
        start, end = span
        text = text[:start] + "```text\n" + printed + "```" + text[end:]
        changed = True
    if changed:
        path.write_text(text, encoding="utf-8", newline="\n")
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("pages", nargs="+", type=Path)
    parser.add_argument("--check", action="store_true", help="report differences, do not write")
    args = parser.parse_args(argv)
    failed = False
    for page in args.pages:
        if args.check:
            for printed, expected, _ in run_page(page):
                if expected is not None and printed != expected:
                    print(f"{page}: output differs:\n--- expected\n{expected}--- got\n{printed}")
                    failed = True
        elif update(page):
            print(f"updated {page}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
