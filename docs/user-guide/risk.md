# Portfolio risk

`PortfolioRiskAnalyzer` holds stock and European option positions on one or more underlyings
and computes their value, Greeks, VaR and expected shortfall, component VaR, delta hedges and
scenario P&L.

Conventions:

- volatilities are annualised; VaR horizons are trading days, and h days have standard
  deviation sigma sqrt(h / 252) (`TRADING_DAYS_PER_YEAR = 252` everywhere in the package);
- option quantities are contracts of 100 units by default (`contract_multiplier`);
- VaR and expected shortfall are positive loss amounts in currency units;
- correlations between underlyings are an input (`set_correlation`); without one the
  underlyings are treated as uncorrelated;
- each option position keeps the risk-free rate the analyzer had when the position was added.

## A small book

```python
from derivflow import PortfolioRiskAnalyzer

book = PortfolioRiskAnalyzer(risk_free_rate=0.04)
book.add_stock_position("INDEX_A", 1_000, 100.0, volatility=0.18)
book.add_option_position("INDEX_A", -20, 100.0, 95.0, 0.25, "put", volatility=0.22)
book.add_stock_position("INDEX_B", -500, 50.0, volatility=0.25)
book.add_option_position("INDEX_B", 10, 50.0, 55.0, 0.5, "call", volatility=0.27)
book.set_correlation(["INDEX_A", "INDEX_B"], [[1.0, 0.6], [0.6, 1.0]])

print(f"value {book.calculate_portfolio_value():,.2f}")
print({k: round(v, 2) for k, v in book.dollar_delta_by_underlying().items()})
```

```text
value 73,445.46
{'INDEX_A': 154039.77, 'INDEX_B': -5876.54}
```

## VaR and expected shortfall

The delta-normal method (RiskMetrics, 1996) treats the 1-day P&L as normal with standard
deviation sqrt(e' Sigma e), where e are the dollar deltas and Sigma the daily covariance:
VaR = z_c s and ES = phi(z_c) / (1 - c) s. It ignores gamma, so for option books it is a
first-order estimate. The Monte Carlo method draws one correlated shock per underlying and
reprices every option.

```python
for c in (0.95, 0.99):
    var_p, es_p = book.calculate_var_parametric(c)
    var_mc, es_mc = book.calculate_var_monte_carlo(c, num_sims=200_000, random_seed=1)
    print(
        f"{c:.0%}: delta-normal VaR {var_p:8.2f} ES {es_p:8.2f} | Monte Carlo VaR {var_mc:8.2f} ES {es_mc:8.2f}"
    )

print("component 99% VaR", {k: round(v, 2) for k, v in book.var_contributions(0.99).items()})
```

```text
95%: delta-normal VaR  2784.31 ES  3491.63 | Monte Carlo VaR  2826.15 ES  3574.84
99%: delta-normal VaR  3937.90 ES  4511.51 | Monte Carlo VaR  4046.04 ES  4664.60
component 99% VaR {'INDEX_A': 4059.42, 'INDEX_B': -121.52}
```

The component (Euler) contributions add up to the delta-normal VaR. Because the delta-normal
method is linear, it can report more than the most a long option can lose, for example for an
option expiring within the horizon; use the Monte Carlo method for books with short-dated or
far-from-the-money options.

## Scenarios and hedges

```python
shock = book.scenario_analysis({"INDEX_A down 10%": {"INDEX_A": -0.10}})["INDEX_A down 10%"]
print(f"full revaluation P&L {shock.portfolio_pnl:,.2f}")
approx = book.scenario_analysis({"s": {"INDEX_A": -0.10}}, method="delta_gamma")["s"]
print(f"delta-gamma P&L      {approx.portfolio_pnl:,.2f}")
print(
    "shares of INDEX_A to sell for delta neutrality:",
    round(-book.calculate_hedge_ratio("INDEX_A")["hedge_quantity"], 2),
)
```

```text
full revaluation P&L -18,913.50
delta-gamma P&L      -18,410.95
shares of INDEX_A to sell for delta neutrality: 1540.4
```

## VaR and expected shortfall of a P&L sample

`derivflow.portfolio.risk_measures` works on any P&L sample (historical or simulated) and
gives closed forms for normal and Student-t P&L.

```python
import numpy as np

from derivflow import (
    expected_shortfall,
    normal_es,
    normal_var,
    student_t_es,
    student_t_var,
    value_at_risk,
)

pnl = np.random.default_rng(0).standard_t(4, 1_000_000)
print(f"sample   VaR {value_at_risk(pnl, 0.99):.4f}  ES {expected_shortfall(pnl, 0.99):.4f}")
print(f"t(4)     VaR {student_t_var(1.0, 4, 0.99):.4f}  ES {student_t_es(1.0, 4, 0.99):.4f}")
print(f"normal   VaR {normal_var(1.0, 0.99):.4f}  ES {normal_es(1.0, 0.99):.4f}")
```

```text
sample   VaR 3.7563  ES 5.2390
t(4)     VaR 3.7469  ES 5.2206
normal   VaR 2.3263  ES 2.6652
```

The empirical VaR is minus the (1 - c) sample quantile (NumPy's linear interpolation) and the
expected shortfall is minus the mean of the P&L at or below -VaR.

## Limitations

Volatilities and correlations are inputs, not estimated; the VaR has no volatility or rate
risk; returns are jointly lognormal over the horizon; options are priced with
Black-Scholes-Merton at their own implied volatility.
