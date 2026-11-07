# Contributing to DerivFlow Finance

Bug reports, fixes and new tests are welcome. A wrong number is the most useful kind of bug
report: please include the inputs, the value you got, the value you expected and where the
expected value comes from (a paper, a textbook or another library).

## Development setup

```bash
git clone https://github.com/jeevanba273/derivflow-finance.git
cd derivflow-finance
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -e ".[dev,docs]"
```

`[dev]` installs the test tools, ruff, build, the `data` and `plot` extras, and the reference
libraries used by the cross-checks (QuantLib and py_vollib).

## Tests

```bash
pytest                       # full suite; network tests are skipped
pytest -m "not slow"         # skip the few longer Monte Carlo tests
pytest --run-network         # also run the tests that call Yahoo Finance
pytest --cov=derivflow_finance       # with coverage
```

The cross-check tests skip themselves when QuantLib or py_vollib is not installed. Tests must
not use the network unless they are marked `network`.

A change to a numerical routine should come with a test against a value that does not come
from this code: a published reference value, a QuantLib engine, a finite difference of an
independent price function, or a statistical check of a Monte Carlo estimate against a closed
form.

## Lint and format

```bash
ruff check .
ruff format .
mypy                         # type check; CI pins mypy 2.4.1 with scipy-stubs and pandas-stubs
```

## Documentation

```bash
sphinx-build -W -b html docs docs/_build/html
python docs/snippets.py README.md docs/*.md docs/user-guide/*.md   # refresh printed outputs
```

Code blocks in the README and the docs are executed by `tests/test_docs.py`. Any number in the
README, the docs, docstrings or the changelog must come from code in this repository; the
validation and benchmark tables are produced by:

```bash
python benchmarks/validation_report.py --output benchmarks/validation.md
python benchmarks/run_benchmarks.py --output benchmarks/results.md --json benchmarks/results.json
```

## Pull requests

- Branch off `main` and open a pull request into `main`.
- Keep commits small and focused, with plain messages that describe the change.
- Add tests, update the docs and add an entry under "Unreleased" in `CHANGELOG.md`.

## Releases

1. Set the new version in `pyproject.toml` and move the changelog entries under it.
2. Merge into `main`, then tag and push: `git tag vX.Y.Z && git push origin vX.Y.Z`.
3. `.github/workflows/publish.yml` builds the sdist and wheel, checks that the tag matches the
   package version and uploads to PyPI with Trusted Publishing (no API token is stored).

## Reporting issues

Use the GitHub issue templates. For security concerns follow `SECURITY.md` and report
privately.
