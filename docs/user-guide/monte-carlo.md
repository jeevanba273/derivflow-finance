# Monte Carlo

`MonteCarloEngine` prices European payoffs by sampling the terminal price exactly,
S_T = S exp((r - q - sigma^2/2) T + sigma sqrt(T) Z), so there is no time-discretisation
error. Each engine has its own `numpy.random.Generator`; NumPy's global random state is never
changed, and a seed makes a run reproducible.

The result reports the estimate, its standard error (sample standard deviation / sqrt(n),
with ddof = 1) and a 95% confidence interval (estimate plus or minus 1.96 standard errors).

## Variance reduction

- `antithetic=True` pairs each draw Z with -Z and averages the pair.
- `control_variate=True` uses the discounted terminal price, whose expectation S e^{-qT} is
  known, with the regression coefficient estimated from the same sample.

```python
from derivflow_finance import MonteCarloEngine, black_scholes_price

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
print(f"closed form {black_scholes_price(S, K, T, r, sigma):.4f}")
for label, kwargs in [
    ("plain", {}),
    ("antithetic", {"antithetic": True}),
    ("control variate", {"control_variate": True}),
]:
    res = MonteCarloEngine(200_000, random_seed=7).price(S, K, T, r, sigma, "call", **kwargs)
    lo, hi = res["confidence_interval"]
    print(
        f"{label:16s} {res['price']:.4f}  s.e. {res['std_error']:.4f}  95% CI [{lo:.4f}, {hi:.4f}]"
    )
```

```text
closed form 10.4506
plain            10.4479  s.e. 0.0329  95% CI [10.3834, 10.5123]
antithetic       10.4276  s.e. 0.0232  95% CI [10.3821, 10.4732]
control variate  10.4377  s.e. 0.0125  95% CI [10.4131, 10.4622]
```

## Standard error and the number of paths

The standard error falls as 1/sqrt(n): a hundred times more paths gives ten times less error.

```python
for n in (10_000, 1_000_000):
    res = MonteCarloEngine(n, random_seed=3).price(S, K, T, r, sigma, "call")
    print(f"{n:9,d} paths: s.e. {res['std_error']:.4f}")
```

```text
   10,000 paths: s.e. 0.1471
1,000,000 paths: s.e. 0.0147
```

A 95% interval excludes the true value in about one run in twenty. Over 400 seeds with 20,000
draws the measured coverage was 95% (plain), 94% (antithetic) and 94% (control variate); see
[validation](../validation.md) (`python benchmarks/validation_report.py`).

## Paths and custom payoffs

`generate_gbm_paths` returns full paths on an equally spaced grid for path-dependent payoffs;
`price(..., payoff_func=f)` accepts any payoff `f(S_T, K)` of the terminal price.

```python
import numpy as np

engine = MonteCarloEngine(100_000, random_seed=1)
digital = engine.price(S, K, T, r, sigma, payoff_func=lambda s, k: (s > k).astype(float))
print(f"digital call {digital['price']:.4f} +- {digital['std_error']:.4f}")

paths = MonteCarloEngine(5, random_seed=1).generate_gbm_paths(S, T, r, sigma, num_steps=4)
print(np.round(paths, 2))
```

```text
digital call 0.5308 +- 0.0015
[[100.   104.3  114.08 118.8  105.07]
 [100.   110.3  116.2  110.96 118.48]
 [100.   104.49 108.42 109.55 116.58]
 [100.    93.6   92.78  89.08  95.29]
 [100.   101.15  98.98  92.22  90.56]]
```
