# Getting started

## Install

DerivFlow Finance needs Python 3.10 or later. The core package depends only on NumPy and SciPy.

```bash
pip install derivflow-finance
```

The package is imported as `derivflow_finance`:

```python
from derivflow_finance import price_european_option
```

Up to version 0.3 the import name was `derivflow`. In the 0.4 series `import derivflow` and
`from derivflow.<module> import <name>` still work and return the same objects, with a
`FutureWarning`; the old name will be removed in 0.5.0. To update code written for 0.3, replace
`derivflow` with `derivflow_finance` in its import statements. Anyone who prefers not to change
yet can pin `derivflow-finance<0.4`.

Optional extras:

| extra | adds | for |
|---|---|---|
| `[data]` | yfinance, pandas, tzdata | `derivflow_finance.utils.market_data` (Yahoo Finance) |
| `[plot]` | plotly, matplotlib, pandas | `derivflow_finance.visualization` and `VolatilitySurface.plot_surface` |
| `[validation]` | QuantLib, py_vollib | the cross-check tests and the benchmark baselines |
| `[docs]`, `[test]`, `[dev]` | Sphinx, pytest, ruff, build | building the docs and developing |

From a clone of the repository:

```bash
git clone https://github.com/jeevanba273/derivflow-finance.git
cd derivflow-finance
pip install -e ".[dev]"
pytest
```

## Conventions

Every pricer uses the same units:

- `S` spot and `K` strike, in the same currency;
- `T` time to expiry in years;
- `r` risk-free rate and `q` dividend yield, continuously compounded, as decimals (0.05 is 5%);
- `sigma` annualised volatility as a decimal (0.20 is 20%).

Greeks are reported per unit of spot (delta, gamma), per calendar day (theta), per volatility
point (vega) and per rate point (rho); see [Greeks](user-guide/greeks.md). The one exception is
`AsianOptions.sensitivity_analysis`, which reports vega and rho per unit and theta per year
(see [Exotics](user-guide/exotics.md)).

## A first example

Hull's worked example: a six-month option with S = 42, K = 40, r = 10% and sigma = 20%.

```python
from derivflow_finance import GreeksCalculator, implied_volatility, price_european_option

S, K, T, r, sigma = 42.0, 40.0, 0.5, 0.10, 0.20
call = price_european_option(S, K, T, r, sigma, "call")
put = price_european_option(S, K, T, r, sigma, "put")
print(f"call {call:.4f}, put {put:.4f}")

g = GreeksCalculator().calculate_greeks(S, K, T, r, sigma, "call")
print(f"delta {g.delta:.4f}, gamma {g.gamma:.4f}, theta {g.theta:.4f} per day")

print(f"implied volatility of the call price: {implied_volatility(call, S, K, T, r):.10f}")
```

```text
call 4.7594, put 0.8086
delta 0.7791, gamma 0.0500, theta -0.0125 per day
implied volatility of the call price: 0.2000000000
```

The rest of the guide is organised by area, starting with [pricing](user-guide/pricing.md).
More runnable scripts are in the `examples/` directory of the repository.
