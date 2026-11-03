# European option pricing

`price_european_option` prices one option by name of method; `black_scholes_price` is the
closed form on NumPy arrays.

## Closed form

The Black-Scholes-Merton price with continuous dividend yield `q`:

$$C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2), \qquad P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1),$$

$$d_1 = \frac{\ln(S/K) + (r - q + \sigma^2/2) T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}.$$

```python
from derivflow import black_scholes_price, price_european_option

# Haug (2007): S=60, K=65, T=0.25, r=8%, sigma=30% gives a call of 2.1334
print(round(price_european_option(60, 65, 0.25, 0.08, 0.30, "call"), 4))

# Merton's dividend case: S=100, K=95, T=0.5, r=10%, q=5%, sigma=20% gives a put of 2.4648
print(round(price_european_option(100, 95, 0.5, 0.10, 0.20, "put", q=0.05), 4))
```

```text
2.1334
2.4648
```

## Many options at once

`black_scholes_price` broadcasts its arguments, so a strike ladder or a whole grid is one call.

```python
import numpy as np

strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
calls = black_scholes_price(100.0, strikes, 1.0, 0.03, 0.2, "call", 0.01)
puts = black_scholes_price(100.0, strikes, 1.0, 0.03, 0.2, "put", 0.01)
print(np.round(calls, 4))

# put-call parity: C - P = S e^{-qT} - K e^{-rT}
parity = (calls - puts) - (100.0 * np.exp(-0.01) - strikes * np.exp(-0.03))
print(f"max parity error {np.abs(parity).max():.1e}")
```

```text
[22.3185 14.6592  8.8273  4.8947  2.5216]
max parity error 1.1e-14
```

## Choosing a method

`price_european_option(..., method=...)` accepts `"black_scholes"` (default), `"binomial"`
(100-step tree, see [trees](trees.md)) and `"monte_carlo"` (100,000 draws, see
[Monte Carlo](monte-carlo.md)). The tree and Monte Carlo values approximate the closed form.

```python
for method in ("black_scholes", "binomial"):
    print(method, round(price_european_option(100, 105, 0.5, 0.05, 0.25, "call", method=method), 4))
```

```text
black_scholes 5.9885
binomial 6.0027
```

Expired options (T <= 0) return their intrinsic value. Non-positive spot, strike or volatility
raises `ValueError` naming the parameter.
