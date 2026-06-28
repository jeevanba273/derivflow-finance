"""Implied volatility: invert option prices back into volatilities.

Run with:  python examples/implied_volatility.py
"""

from derivflow import price_european_option, implied_volatility, implied_vol_from_chain

print("Single-option round trip (price -> implied vol):")
for sigma in (0.15, 0.25, 0.40):
    price = price_european_option(100, 100, 0.5, 0.05, sigma, 'call')
    iv = implied_volatility(price, 100, 100, 0.5, 0.05, 'call')
    print(f"  true sigma {sigma:.2f} -> price {price:.4f} -> implied {iv:.4f}")

print("\nInverting a small option chain (a volatility smile):")
chain = [
    {'strike': k, 'expiry': 0.5, 'option_type': 'call',
     'market_price': price_european_option(
         100, k, 0.5, 0.05, 0.20 + 0.10 * abs(k - 100) / 100, 'call')}
    for k in (90, 95, 100, 105, 110)
]
for row in implied_vol_from_chain(chain, S=100, r=0.05):
    print(f"  K={row['strike']:.0f}  price={row['market_price']:.4f}  IV={row['volatility']:.4f}")
