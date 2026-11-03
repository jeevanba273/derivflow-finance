# Implied volatility

`implied_volatility` inverts the Black-Scholes-Merton price of one option;
`implied_volatility_vectorized` does the same on arrays.

## Method

The quote is converted to its out-of-the-money equivalent with put-call parity, because the
out-of-the-money price is the option's time value. The solver then runs a safeguarded Newton
iteration on the logarithm of that price, starting from the Manaster-Koehler point
sigma0 = sqrt(2 |ln(F/K)| / T) and falling back to bisection inside a bracket. It stops when
the volatility changes by less than 1e-12.

```python
import numpy as np

from derivflow import black_scholes_price, implied_volatility, implied_volatility_vectorized

price = black_scholes_price(100.0, 120.0, 0.25, 0.03, 0.35, "call", 0.01)
print(f"{implied_volatility(price, 100.0, 120.0, 0.25, 0.03, 'call', 0.01):.12f}")

strikes = np.linspace(70, 130, 7)
vols = 0.2 + 0.3 * np.log(strikes / 100.0) ** 2
prices = black_scholes_price(100.0, strikes, 0.5, 0.03, vols, "put", 0.01)
solved = implied_volatility_vectorized(prices, 100.0, strikes, 0.5, 0.03, "put", 0.01)
print(f"max error over the smile: {np.abs(solved - vols).max():.1e}")
```

```text
0.350000000000
max error over the smile: 1.4e-15
```

## When there is no answer

A price outside the no-arbitrage bounds has no implied volatility, and a price whose time
value is below 1e-12 of the price (for example deep in the money with a day to expiry) has
none that double precision can identify. Both raise `ValueError` with the reason;
`on_fail="nan"` returns NaN instead, which the vectorised solver does by default.

```python
try:
    implied_volatility(15.0, 100.0, 80.0, 0.5, 0.03, "call")
except ValueError as exc:
    print(exc)

print(implied_volatility(15.0, 100.0, 80.0, 0.5, 0.03, "call", on_fail="nan"))
```

```text
price 15.0 is at or below the discounted intrinsic value 21.191 (no positive volatility reproduces it)
nan
```

Where vega is very small relative to the spot the price determines the volatility only to a
few digits. On the benchmark set (moneyness 0.7 to 1.3, one week to two years) the solved
volatility is within 3e-10 of the true one wherever vega exceeds 1e-6 S; see
[benchmarks](../benchmarks.md). Results agree with py_vollib's implementation of Jaeckel's
"Let's Be Rational" to 1e-9 in the cross-check tests.

## Option chains

`implied_vol_from_chain` accepts a list of dicts, a pandas DataFrame or a `MarketDataResult`
and returns rows ready for `VolatilitySurface.load_market_data`.

```python
from derivflow import implied_vol_from_chain

chain = [
    {
        "strike": k,
        "expiry": 0.5,
        "option_type": "call",
        "market_price": black_scholes_price(100.0, k, 0.5, 0.03, 0.22, "call"),
    }
    for k in (90.0, 100.0, 110.0)
]
for row in implied_vol_from_chain(chain, S=100.0, r=0.03):
    print(row["strike"], round(row["volatility"], 6))
```

```text
90.0 0.22
100.0 0.22
110.0 0.22
```
