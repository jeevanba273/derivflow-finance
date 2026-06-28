# Installation

DERIVFLOW-FINANCE requires **Python 3.8 or newer**.

## Install from PyPI

The package is published on PyPI as `derivflow-finance`:

```bash
pip install derivflow-finance
```

This installs the core scientific stack (NumPy, SciPy, pandas, Matplotlib,
Plotly, Numba, scikit-learn, yfinance) required for pricing, Greeks, volatility
modelling, market data, and visualization.

## Optional extras

Additional dependency groups are available as pip "extras":

| Extra | Purpose |
| --- | --- |
| `dev` | Development tooling: pytest, pytest-cov, black, flake8, mypy |
| `testing` | Test runners: pytest, pytest-cov, pytest-xdist |
| `docs` | Documentation build: sphinx, sphinx-rtd-theme, myst-parser |
| `visualization` | Charting stack: plotly, matplotlib, seaborn |
| `notebooks` | Jupyter environment: jupyter, notebook |

Install one or more extras with the bracket syntax, for example:

```bash
pip install "derivflow-finance[visualization]"
pip install "derivflow-finance[docs]"
```

## Development install

To work on the library from a local clone, install it in editable mode together
with the development and testing extras:

```bash
git clone https://github.com/jeevanba273/derivflow-finance.git
cd derivflow-finance
pip install -e ".[dev,testing]"
```

This gives you an editable checkout plus the formatters, linters, type checker,
and test runners used by the project.

## Verifying the installation

```python
import derivflow
print(derivflow.__version__)
```
