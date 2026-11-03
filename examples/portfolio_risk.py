"""Value-at-Risk of a small option book on two hypothetical underlyings.

Run with:  python examples/portfolio_risk.py
"""

from derivflow import PortfolioRiskAnalyzer

book = PortfolioRiskAnalyzer(risk_free_rate=0.04)
book.add_stock_position("INDEX_A", 1_000, 100.0, volatility=0.18)
book.add_option_position("INDEX_A", -20, 100.0, 95.0, 0.25, "put", volatility=0.22)
book.add_stock_position("INDEX_B", -500, 50.0, volatility=0.25)
book.add_option_position("INDEX_B", 10, 50.0, 55.0, 0.5, "call", volatility=0.27)
book.set_correlation(["INDEX_A", "INDEX_B"], [[1.0, 0.6], [0.6, 1.0]])

print(f"book value: {book.calculate_portfolio_value():,.2f}")
for symbol, exposure in book.dollar_delta_by_underlying().items():
    print(f"  dollar delta {symbol}: {exposure:,.2f}")

print("\n1-day VaR and expected shortfall (losses, currency units):")
for c in (0.95, 0.99):
    var_p, es_p = book.calculate_var_parametric(c)
    var_mc, es_mc = book.calculate_var_monte_carlo(c, num_sims=200_000, random_seed=1)
    print(
        f"  {c:.0%}  delta-normal VaR {var_p:8.2f} ES {es_p:8.2f}   "
        f"full-revaluation MC VaR {var_mc:8.2f} ES {es_mc:8.2f}"
    )

contributions = book.var_contributions(0.99)
print("\ncomponent 99% VaR:", {k: round(v, 2) for k, v in contributions.items()})

shock = book.scenario_analysis({"INDEX_A -10%": {"INDEX_A": -0.10}})["INDEX_A -10%"]
print(f"P&L if INDEX_A falls 10%: {shock.portfolio_pnl:,.2f}")
print("delta hedge for INDEX_A:", round(book.calculate_hedge_ratio("INDEX_A")["hedge_quantity"], 2))
