# DERIVFLOW-FINANCE

**DERIVFLOW-FINANCE** is an open-source derivatives pricing and risk-management
toolkit for quantitative finance. It packages institutional-grade analytics
behind a clean, importable Python API.

The library covers the full derivatives workflow:

- **Core pricing** &mdash; analytical Black-Scholes-Merton (with continuous
  dividend yield), binomial trees, and a Monte Carlo engine.
- **Greeks** &mdash; first-, second-, and third-order analytical Greeks plus a
  finite-difference engine for exotic instruments.
- **Exotic options** &mdash; barrier options (Reiner-Rubinstein closed forms and
  Monte Carlo) and Asian options (geometric closed form, arithmetic Monte Carlo
  with control variates).
- **Volatility** &mdash; volatility-surface construction and interpolation, plus a
  robust implied-volatility solver (Newton-Raphson with a Brent fallback).
- **Stochastic volatility** &mdash; the Heston model with Monte Carlo and Fourier
  pricing and market calibration.
- **Portfolio risk** &mdash; Greeks aggregation, parametric and Monte Carlo VaR,
  scenario analysis, and hedging.
- **Market data and visualization** &mdash; live option-chain access via
  `yfinance` and interactive Plotly dashboards.

```python
from derivflow import price_european_option

price = price_european_option(S=100, K=105, T=0.25, r=0.05, sigma=0.2, option_type="call")
print(f"Call price: {price:.4f}")
```

```{toctree}
:maxdepth: 2
:caption: Contents

installation
quickstart
theory
user-guide/index
api/index
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
