# Quickstart

These short, runnable snippets exercise the main entry points of the library.
Every example uses the real top-level public API.

## Price a European option

`price_european_option` is the fastest way to price a vanilla European call or
put with the analytical Black-Scholes-Merton model. A continuous dividend yield
`q` is supported by the analytical method.

```python
from derivflow import price_european_option

call = price_european_option(S=100, K=105, T=0.25, r=0.05, sigma=0.2, option_type="call")
put = price_european_option(S=100, K=105, T=0.25, r=0.05, sigma=0.2, option_type="put")
div_call = price_european_option(S=100, K=105, T=0.25, r=0.05, sigma=0.2,
                                 option_type="call", q=0.03)

print(f"Call: {call:.4f}  Put: {put:.4f}  Call (q=3%): {div_call:.4f}")
```

## Calculate the Greeks

`GreeksCalculator.calculate_greeks` returns a `GreeksResult` dataclass with
fields such as `delta`, `gamma`, `theta`, `vega`, and `rho`.

```python
from derivflow import GreeksCalculator

calc = GreeksCalculator()  # analytical by default
greeks = calc.calculate_greeks(S=100, K=105, T=0.25, r=0.05, sigma=0.2, option_type="call")

print(f"Delta: {greeks.delta:.4f}")
print(f"Gamma: {greeks.gamma:.4f}")
print(f"Vega:  {greeks.vega:.4f}")
```

## Recover implied volatility

`implied_volatility` inverts an observed option price for its Black-Scholes
implied volatility using Newton-Raphson with a Brent fallback.

```python
from derivflow import implied_volatility, price_european_option

# Price an option, then recover the volatility that produced it.
market_price = price_european_option(S=100, K=100, T=0.5, r=0.03, sigma=0.25, option_type="call")
iv = implied_volatility(market_price, S=100, K=100, T=0.5, r=0.03, option_type="call")

print(f"Recovered implied vol: {iv:.4f}")  # ~0.25
```

## Price a barrier option

`BarrierOptions.price` returns a `BarrierOptionResult` with the `price` and
`probability_survival` fields.

```python
from derivflow import BarrierOptions

barrier = BarrierOptions()
result = barrier.price(S=100, K=105, H=95, T=0.25, r=0.05, sigma=0.3,
                       barrier_type="down_and_out", option_type="call")

print(f"Barrier price:        {result.price:.4f}")
print(f"Survival probability: {result.probability_survival:.2%}")
```

## Build a volatility surface

`create_sample_surface` returns a ready-to-use `VolatilitySurface` populated
with synthetic smile data, ideal for experimentation.

```python
from derivflow import create_sample_surface

surface = create_sample_surface()
surface.build_surface()  # cubic-spline interpolation by default

vol = surface.interpolate(strike=102, expiry=0.25)
print(f"Interpolated vol at K=102, T=0.25: {vol:.2%}")
```
