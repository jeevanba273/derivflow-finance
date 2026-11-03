# derivflow

[![CI](https://github.com/jeevanba273/derivflow-finance/actions/workflows/ci.yml/badge.svg)](https://github.com/jeevanba273/derivflow-finance/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/derivflow-finance)](https://pypi.org/project/derivflow-finance/)
[![Python versions](https://img.shields.io/pypi/pyversions/derivflow-finance)](https://pypi.org/project/derivflow-finance/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

derivflow is a Python library for pricing equity-style options and measuring the risk of small
option books: the Black-Scholes-Merton closed form and Greeks, implied volatility, binomial
trees with early exercise, Monte Carlo with variance reduction, barrier and Asian options, the
Heston model, volatility surfaces and portfolio VaR. It is written for students, researchers and
practitioners who want short NumPy/SciPy code they can read, with results checked against
QuantLib, py_vollib and textbook values. The core depends only on NumPy and SciPy.

## Install

```bash
pip install derivflow-finance            # Python 3.10 or later
pip install "derivflow-finance[data]"    # optional: Yahoo Finance market data
pip install "derivflow-finance[plot]"    # optional: Plotly figures
```

Units throughout: T in years, r and q continuously compounded, sigma annualised. Theta is per
calendar day, vega and rho per point (0.01).

## Examples

Price, Greeks and implied volatility for Hull's worked example (S=42, K=40, six months,
r=10%, sigma=20%):

```python
from derivflow import GreeksCalculator, implied_volatility, price_european_option

call = price_european_option(42, 40, 0.5, 0.10, 0.20, "call")
put = price_european_option(42, 40, 0.5, 0.10, 0.20, "put")
g = GreeksCalculator().calculate_greeks(42, 40, 0.5, 0.10, 0.20, "call")
print(f"call {call:.4f}  put {put:.4f}  delta {g.delta:.4f}  vega {g.vega:.4f} per vol point")
print(f"implied vol from the call price: {implied_volatility(call, 42, 40, 0.5, 0.10):.10f}")
```

```text
call 4.7594  put 0.8086  delta 0.7791  vega 0.0881 per vol point
implied vol from the call price: 0.2000000000
```

An American put on a binomial tree, and a Monte Carlo price with its confidence interval:

```python
from derivflow import BinomialTree, MonteCarloEngine, black_scholes_price

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
american = BinomialTree(2000).price(S, K, T, r, sigma, "put", american=True)
european = black_scholes_price(S, K, T, r, sigma, "put")
print(f"American put {american:.4f}, European put {european:.4f}")

engine = MonteCarloEngine(200_000, random_seed=7)
mc = engine.price(S, K, T, r, sigma, "call", control_variate=True)
lo, hi = mc["confidence_interval"]
print(f"MC call {mc['price']:.4f} +- {mc['std_error']:.4f}, 95% CI [{lo:.4f}, {hi:.4f}]")
print(f"closed form {black_scholes_price(S, K, T, r, sigma):.4f}")
```

```text
American put 6.0900, European put 5.5735
MC call 10.4377 +- 0.0125, 95% CI [10.4131, 10.4622]
closed form 10.4506
```

A Heston smile, and the VaR of a two-index option book (illustrative positions):

```python
from derivflow import HestonModel, PortfolioRiskAnalyzer, implied_volatility

heston = HestonModel()
heston.set_parameters(v0=0.04, kappa=2.0, theta=0.04, sigma=0.35, rho=-0.7)  # v0, theta: variances
for K in (80.0, 100.0, 120.0):
    price = heston.analytic_price(100.0, K, 1.0, 0.03, "call")
    vol = implied_volatility(price, 100.0, K, 1.0, 0.03)
    print(f"K={K:.0f}: Heston call {price:.4f}, implied vol {vol:.4f}")

book = PortfolioRiskAnalyzer(risk_free_rate=0.04)
book.add_stock_position("INDEX_A", 1_000, 100.0, volatility=0.18)
book.add_option_position("INDEX_A", -20, 100.0, 95.0, 0.25, "put", volatility=0.22)
book.add_stock_position("INDEX_B", -500, 50.0, volatility=0.25)
book.set_correlation(["INDEX_A", "INDEX_B"], [[1.0, 0.6], [0.6, 1.0]])
var, es = book.calculate_var_parametric(0.99)
var_mc, es_mc = book.calculate_var_monte_carlo(0.99, num_sims=200_000, random_seed=1)
print(f"1-day 99% VaR, delta-normal: {var:,.0f} (ES {es:,.0f})")
print(f"1-day 99% VaR, full revaluation: {var_mc:,.0f} (ES {es_mc:,.0f})")
```

```text
K=80: Heston call 23.7459, implied vol 0.2299
K=100: Heston call 9.1746, implied vol 0.1938
K=120: Heston call 1.6863, implied vol 0.1642
1-day 99% VaR, delta-normal: 3,589 (ES 4,112)
1-day 99% VaR, full revaluation: 3,708 (ES 4,287)
```

More scripts are in [`examples/`](examples/) and the [user guide](https://derivflow-finance.readthedocs.io).

## Validation

Every result below is reproduced by `python benchmarks/validation_report.py` (full tables in
[docs/validation](https://derivflow-finance.readthedocs.io/en/latest/validation.html)) and
asserted with tolerances in the test suite.

| check | reference | result |
|---|---|---|
| Black-Scholes call/put, S=42 K=40 T=0.5 r=10% sigma=20% | Hull: 4.76 / 0.81 | 4.7594 / 0.8086 |
| Six price and Greek worked examples | Haug (2007), 4 decimals | all match |
| Heston calls, 5 strikes | Alan Lewis's reference prices | max abs diff 1.1e-12 |
| European prices and 5 Greeks, 144 cases | QuantLib AnalyticEuropeanEngine | max abs diff 1.8e-13 |
| Barrier options, 48 cases | QuantLib AnalyticBarrierEngine | max abs diff 2.1e-14 |
| Geometric Asian, continuous and 12 fixings | QuantLib analytic engines | max abs diff 1.4e-14 |
| Heston, 27 cases incl. Feller violated | QuantLib AnalyticHestonEngine | max abs diff 2.1e-10 |
| American options, 2000-step tree | QuantLib finite differences (2000 x 2000) | max abs diff 5.8e-4 |
| Implied volatility, 128 quotes | py_vollib (Let's Be Rational) | max abs diff 4.1e-12 |
| Put-call parity, 324 parameter sets | identity | max abs error 5.7e-14 |
| 95% Monte Carlo interval, 400 seeds | closed form | coverage 94.0% to 95.0% |

The test suite also checks every Greek against central finite differences, tree convergence to
Black-Scholes, American put >= European put, and that Monte Carlo variance reduction lowers the
standard error.

## Benchmarks

Measured with `python benchmarks/run_benchmarks.py` on an Intel Core Ultra 9 185H, 31.5 GB RAM,
Windows 11, Python 3.13.5, NumPy 2.5.4, SciPy 1.18.1, QuantLib 1.44, py_vollib 1.0.12;
single-threaded, median of 7 runs after a warm-up. Full tables, spreads and method:
[docs/benchmarks](https://derivflow-finance.readthedocs.io/en/latest/benchmarks.html).

| operation | derivflow | baseline |
|---|---|---|
| European price, 1,000,000 options vectorised | 9.2 million/s | NumPy/SciPy formula: 7.8 million/s |
| European price, scalar Python loop | 762,000/s | py_vollib 417,000/s; QuantLib via Python 292,000/s |
| Implied volatility, scalar | 12.4 us per quote | py_vollib 19.8 us; QuantLib via Python 12.9 us |
| Implied volatility, vectorised (99,825 quotes) | 4.2 us per quote | - |
| American put, 2000-step tree | 33.3 ms | QuantLib CRR tree 14.4 ms |
| Heston analytic call | 0.7 ms per price | QuantLib 0.4 ms |

QuantLib's rows are driven from Python one option at a time and include binding overhead, and
its tree uses a different up-probability, so those rows are not like-for-like. Where
derivflow is slower (trees, Heston) the table says so.

## Limitations

- Black-Scholes-Merton with flat r, q and sigma: no term structures, no discrete dividends, no
  calendars; ACT/365 for listed expiries.
- Barrier options: continuous monitoring, no rebates, no dividend yield, single barriers only.
  The Monte Carlo barrier is discretely monitored and differs from the closed form by the
  monitoring bias.
- Asian options: average-price only, no dividend yield; arithmetic averages by Monte Carlo.
- Heston: no jumps; calibration is a plain least-squares fit and parameters can be unstable.
- Volatility surfaces do not enforce absence of arbitrage; `check_arbitrage` tests one smile
  across strikes, not across expiries.
- Implied volatility is not identifiable when the time value is below about 1e-12 of the price
  (the solver raises) and is imprecise where vega is tiny.
- Portfolio VaR takes volatilities and correlations as inputs, has no volatility or rate risk,
  and assumes jointly lognormal underlyings over the horizon.
- Market data comes from Yahoo Finance through yfinance: delayed, sometimes stale or missing.
  Its implied volatilities use the European formula, while listed US equity options are
  American.

## Documentation

[derivflow-finance.readthedocs.io](https://derivflow-finance.readthedocs.io): getting started,
a user guide per area, the API reference, methods and references, validation and benchmarks.

## Further reading

- [Hedging a Portfolio with Index Options](https://financefluentblog.com/blog/hedging-portfolio/)
- [Butterfly Option Structures Explained](https://financefluentblog.com/blog/neutral-options/)
- [Credit Spreads and Iron Condors](https://financefluentblog.com/blog/income-oriented-spreads/)
- [Monte Carlo Value-at-Risk, Step by Step](https://financefluentblog.com/blog/monte-carlo-var/)
- [Risk Management Essentials](https://financefluentblog.com/blog/risk-management/)
- [Portfolio Construction: Theory and Process](https://financefluentblog.com/blog/portfolio-construction/)

## Contributing, citing, licence

Bug reports and pull requests are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). To cite
derivflow, use the metadata in [CITATION.cff](CITATION.cff). Released under the MIT licence.

For research and education; not investment advice.
