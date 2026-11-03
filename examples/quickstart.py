"""Price a European option, compute its Greeks and recover the implied volatility.

Run with:  python examples/quickstart.py
"""

from derivflow import GreeksCalculator, implied_volatility, price_european_option

# Hull's worked example: S=42, K=40, six months, r=10%, sigma=20%.
S, K, T, r, sigma = 42.0, 40.0, 0.5, 0.10, 0.20

call = price_european_option(S, K, T, r, sigma, "call")
put = price_european_option(S, K, T, r, sigma, "put")
print(f"call {call:.4f}   put {put:.4f}")

g = GreeksCalculator().calculate_greeks(S, K, T, r, sigma, "call")
print(
    f"delta {g.delta:.4f}  gamma {g.gamma:.4f}  vega {g.vega:.4f}/vol pt  "
    f"theta {g.theta:.4f}/day  rho {g.rho:.4f}/rate pt"
)

iv = implied_volatility(call, S, K, T, r, "call")
print(f"implied volatility from the call price: {iv:.10f}")

# Continuous dividend yield q (Black-Scholes-Merton)
print(f"call with q=3%: {price_european_option(S, K, T, r, sigma, 'call', q=0.03):.4f}")
