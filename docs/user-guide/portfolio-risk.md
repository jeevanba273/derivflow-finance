# Portfolio Risk

`PortfolioRiskAnalyzer` aggregates positions into a portfolio and computes
institutional-grade risk metrics: aggregated Greeks, parametric (delta-normal)
and Monte Carlo Value at Risk, scenario analysis, and delta-hedging
recommendations. Positions are added with `add_stock_position` and
`add_option_position`; the `create_sample_portfolio` helper returns a populated
analyzer for experimentation.

```python
from derivflow import PortfolioRiskAnalyzer, create_sample_portfolio

portfolio = create_sample_portfolio()

value = portfolio.calculate_portfolio_value()
greeks = portfolio.calculate_portfolio_greeks()
var_95, es_95 = portfolio.calculate_var_parametric(confidence_level=0.95)

print(f"Portfolio value: {value:,.2f}")
print(f"Portfolio delta: {greeks['delta']:.2f}")
print(f"95% VaR (1-day): {var_95:,.2f}")

# Stress scenarios: {scenario_name: {symbol: pct_change}}
scenarios = {"Sell-off": {"AAPL": -0.10, "MSFT": -0.08}}
results = portfolio.scenario_analysis(scenarios)
print(results["Sell-off"].portfolio_pnl)

# Full report
report = portfolio.generate_risk_report()
print(report.var_95, report.portfolio_delta)
```

To build a portfolio from scratch:

```python
p = PortfolioRiskAnalyzer(risk_free_rate=0.05)
p.add_stock_position("AAPL", quantity=100, current_price=150.0, volatility=0.25)
p.add_option_position("AAPL", quantity=10, current_price=150.0, strike=155.0,
                      expiry=0.25, option_type="call", volatility=0.25)
```

**Key result fields.** `calculate_portfolio_greeks` returns a dict with `delta`,
`gamma`, `theta`, `vega`, and `rho`. `calculate_var_parametric` and
`calculate_var_monte_carlo` each return a `(VaR, expected_shortfall)` tuple.
`scenario_analysis` returns a dict of `ScenarioResult` objects
(`portfolio_pnl`, `new_portfolio_value`, `individual_pnl`).
`generate_risk_report` returns a `RiskMetrics` dataclass with `portfolio_value`,
`var_95`, `var_99`, the portfolio Greeks, and `risk_contributions`.
