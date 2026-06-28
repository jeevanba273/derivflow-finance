# Volatility Surface

`VolatilitySurface` builds and interpolates an implied-volatility surface from
discrete market quotes. It supports cubic-spline / RBF interpolation and an SVI
smile model, and can extract smiles, check basic arbitrage conditions, and
report surface statistics. The `create_sample_surface` helper returns a surface
pre-populated with a realistic synthetic smile for experimentation.

```python
from derivflow import create_sample_surface

surface = create_sample_surface()
surface.build_surface(method="cubic_spline")

vol = surface.interpolate(strike=102, expiry=0.25)
print(f"Interpolated vol: {vol:.2%}")

smile = surface.get_smile(expiry=0.25, num_points=5)
print(smile["strikes"], smile["volatilities"])

stats = surface.surface_statistics()
print(stats["num_points"], stats["mean_volatility"])
```

To build a surface from your own data, pass a list of dicts (or a pandas
DataFrame) with `strike`, `expiry`, and `volatility` keys to
`load_market_data`, then call `build_surface`.

**Key result fields.** `interpolate` returns a `float` volatility. `get_smile`
returns a dict with `strikes`, `volatilities`, and `expiry`.
`surface_statistics` returns a dict including `num_points`, `min_volatility`,
`max_volatility`, `mean_volatility`, and `vol_of_vol`.

## Implied Volatility Solver

When you have market **prices** rather than vols, invert them with the implied
volatility solver. `implied_volatility` solves a single quote with
Newton-Raphson and a Brent fallback; `implied_volatility_vectorized` broadcasts
over array inputs; and `implied_vol_from_chain` inverts a whole option chain.

```python
from derivflow import implied_volatility, implied_vol_from_chain

iv = implied_volatility(price=5.0, S=100, K=100, T=0.5, r=0.03,
                        option_type="call", q=0.0)
print(f"Implied vol: {iv:.4f}")
```

`implied_vol_from_chain` accepts a `MarketDataResult`, a pandas DataFrame, or a
list of dicts. Tabular inputs must provide `strike`, `expiry` (time to expiry in
**years**), `option_type`, and a price column. It returns a list of dicts whose
shape (`strike`, `expiry`, `volatility`, `market_price`, `option_type`) feeds
directly into `VolatilitySurface.load_market_data`:

```python
chain = [
    {"strike": 95,  "expiry": 0.25, "option_type": "call", "market_price": 7.1},
    {"strike": 100, "expiry": 0.25, "option_type": "call", "market_price": 4.2},
    {"strike": 105, "expiry": 0.25, "option_type": "call", "market_price": 2.3},
]
rows = implied_vol_from_chain(chain, S=100, r=0.03, q=0.0)
for row in rows:
    print(row["strike"], row["volatility"])
```

**Key result fields.** `implied_volatility` returns a `float` volatility, or
`np.nan` when the price is outside the no-arbitrage bounds (unless
`on_fail="raise"`). `implied_vol_from_chain` returns a list of dicts with the
keys `strike`, `expiry`, `volatility`, `market_price`, and `option_type`.
