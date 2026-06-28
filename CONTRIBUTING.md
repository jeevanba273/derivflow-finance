# Contributing to DERIVFLOW-FINANCE

Thanks for your interest in improving DERIVFLOW-FINANCE. This guide covers the
development setup, testing, and release process.

## Development setup

```bash
git clone https://github.com/jeevanba273/derivflow-finance.git
cd derivflow-finance
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Unix: source .venv/bin/activate
pip install -e ".[dev,testing]"
```

Optional extras: `.[docs]` (Sphinx docs), `.[notebooks]` (Jupyter), `.[visualization]`.

## Running the tests

```bash
pytest                                   # full suite
pytest -m "not slow and not integration" # fast lane (what CI gates on)
pytest -m slow                           # Monte Carlo / Heston statistical tests
pytest -m integration                    # live yfinance tests (needs internet)
pytest --cov=derivflow --cov-report=term-missing
```

Markers: `slow` for Monte Carlo / Heston tests, `integration` for tests that hit the
network. Both are excluded from the fast lane.

## Code style and types

```bash
black .                 # formatting (line length 88)
flake8 src tests        # linting
mypy src/derivflow      # type checking
```

New public functions should carry NumPy-style docstrings (`Parameters:` / `Returns:`) and
type hints. Please do not add decorative emojis to source, output, or docs; mathematical
and Greek symbols are welcome where they aid clarity.

## Branching and pull requests

- Branch off `main`; never commit directly to `main`.
- Open a pull request into `main` and fill in the PR template.
- Keep commit messages professional and concise, written as a human developer would. Do
  not include AI/automation references or `Co-Authored-By` trailers (see `CLAUDE.md`).
- Add or update tests for behavior changes, update the docs, and add a `CHANGELOG.md`
  entry under the unreleased section.

## Releases

Versioning is driven by Git tags via `setuptools_scm` (the version is written to
`src/derivflow/_version.py`). To cut a release:

1. Update `CHANGELOG.md` with the new version and date.
2. Tag and push: `git tag vX.Y.Z && git push origin vX.Y.Z`.
3. The `publish.yml` workflow builds the distributions and uploads them to PyPI using
   Trusted Publishing (no API token required).

### One-time PyPI Trusted Publisher setup (maintainers)

On PyPI, open the `derivflow-finance` project, go to Publishing, and add a GitHub Actions
trusted publisher: owner `jeevanba273`, repository `derivflow-finance`, workflow
`publish.yml`, environment `pypi`. After this, tagged releases publish automatically with
no stored secret. If you prefer a token instead, store a project-scoped token as the
`PYPI_API_TOKEN` secret and pass it to `pypa/gh-action-pypi-publish`.

## Reporting issues

Use the GitHub issue templates for bug reports and feature requests. For security
concerns, follow `SECURITY.md` (report privately, do not open a public issue).
