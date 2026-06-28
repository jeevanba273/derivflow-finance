# Pricing

The core pricing layer offers three methodologies behind a unified interface:
analytical Black-Scholes-Merton (`BlackScholesAnalytical`), a `BinomialTree`
(European and American), and a `MonteCarloEngine`. The `PricingEngine` class
coordinates them, and the `price_european_option` helper is the simplest entry
point.

`price_european_option` accepts a continuous dividend yield `q` when using the
analytical method (`method="black_scholes"`, the default). The binomial and
Monte Carlo engines do not model dividends, so passing a non-zero `q` with those
methods raises a clear error.

```python
from derivflow import PricingEngine, price_european_option

# Convenience helper (analytical, supports dividend yield q)
price = price_european_option(S=100, K=105, T=0.25, r=0.05, sigma=0.2,
                              option_type="call", q=0.02)
print(f"Call price: {price:.4f}")

# Compare methods through the engine
engine = PricingEngine()
comparison = engine.compare_methods(S=100, K=105, T=0.25, r=0.05, sigma=0.2,
                                    option_type="call")
for method, result in comparison.items():
    print(method, result.get("price"))
```

**Key result fields.** `price_european_option` returns a `float`. The
`PricingEngine.compare_methods` call returns a dict keyed by method name; each
value is itself a dict containing at least a `price` key (the Monte Carlo entry
additionally carries `std_error`, `confidence_interval`, and `num_simulations`).
`PricingEngine.validate_put_call_parity` returns a dict with `absolute_error`
and a boolean `parity_satisfied`.
