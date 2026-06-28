"""Portfolio risk analytics: build a portfolio and compute Value-at-Risk.

Run with:  python examples/portfolio_risk.py
"""

from derivflow import create_sample_portfolio

analyzer = create_sample_portfolio()

print(f"Positions:       {len(analyzer.positions)}")
print(f"Portfolio value: {analyzer.calculate_portfolio_value():,.2f}\n")

greeks = analyzer.calculate_portfolio_greeks()
print("Portfolio Greeks:")
for name, value in greeks.items():
    print(f"  {name:6s}: {value:,.4f}")

print("\nValue-at-Risk (parametric delta-normal):")
for confidence in (0.95, 0.99):
    var, es = analyzer.calculate_var_parametric(confidence)
    print(f"  {confidence:.0%} VaR: {var:,.2f} | Expected Shortfall: {es:,.2f}")
