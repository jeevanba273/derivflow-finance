"""Quickstart: price a European option, compute Greeks, and recover implied volatility.

Run with:  python examples/quickstart.py
"""

from derivflow import price_european_option, GreeksCalculator, implied_volatility

S, K, T, r, sigma = 100, 105, 0.25, 0.05, 0.20

call = price_european_option(S, K, T, r, sigma, 'call')
put = price_european_option(S, K, T, r, sigma, 'put')
print(f"European call: {call:.4f}")
print(f"European put:  {put:.4f}")

greeks = GreeksCalculator().calculate_greeks(S, K, T, r, sigma, 'call')
print("\nGreeks (call):")
print(f"  Delta {greeks.delta:.4f} | Gamma {greeks.gamma:.4f} | Vega {greeks.vega:.4f}"
      f" | Theta {greeks.theta:.4f} | Rho {greeks.rho:.4f}")

iv = implied_volatility(call, S, K, T, r, 'call')
print(f"\nImplied volatility recovered from price {call:.4f}: {iv:.4%} (input was {sigma:.0%})")

# Dividend-yield support (generalized Black-Scholes-Merton)
call_div = price_european_option(S, K, T, r, sigma, 'call', q=0.03)
print(f"\nCall with 3% dividend yield: {call_div:.4f} (vs {call:.4f} without)")
