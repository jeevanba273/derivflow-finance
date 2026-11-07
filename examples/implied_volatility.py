"""Implied volatility: single quotes, a whole smile at once, and quotes with no solution.

Run with:  python examples/implied_volatility.py
"""

import numpy as np

from derivflow_finance import black_scholes_price, implied_volatility, implied_volatility_vectorized

S, T, r, q = 100.0, 0.5, 0.03, 0.01

# A smile priced from a known volatility function, then inverted.
strikes = np.array([70.0, 85.0, 100.0, 115.0, 130.0])
true_vols = 0.20 + 0.4 * np.log(strikes / S) ** 2 - 0.05 * np.log(strikes / S)
prices = np.array(
    [black_scholes_price(S, k, T, r, v, "call", q) for k, v in zip(strikes, true_vols, strict=True)]
)
solved = implied_volatility_vectorized(prices, S, strikes, T, r, "call", q)
for k, v, s in zip(strikes, true_vols, solved, strict=True):
    print(f"K={k:5.1f}  true {v:.6f}  solved {s:.6f}  error {abs(s - v):.1e}")

# A quote below the discounted intrinsic value has no solution.
try:
    implied_volatility(29.0, S, 70.0, T, r, "call", q)
except ValueError as exc:
    print("no solution:", exc)
print("with on_fail='nan':", implied_volatility(29.0, S, 70.0, T, r, "call", q, on_fail="nan"))
