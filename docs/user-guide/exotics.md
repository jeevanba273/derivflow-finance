# Barrier and Asian options

## Barrier options

`BarrierOptions` prices single-barrier European options (up or down, in or out, calls and
puts, no rebate) under Black-Scholes dynamics without dividends.

- `method="analytical"`: the Reiner-Rubinstein closed form for a continuously monitored
  barrier, as written by Haug (2007), section 4.17.1. It matches QuantLib's
  `AnalyticBarrierEngine` to 2e-14 on 48 test cases.
- `method="monte_carlo"`: exact GBM paths checked against the barrier at each time step, which
  is a discretely monitored barrier.

```python
from derivflow import AsianOptions, BarrierOptions, black_scholes_price

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
barrier = BarrierOptions(random_seed=7)
vanilla = black_scholes_price(S, K, T, r, sigma)
for kind, H in [("down_and_out", 90), ("down_and_in", 90), ("up_and_out", 120)]:
    res = barrier.price(S, K, H, T, r, sigma, kind, "call")
    print(f"{kind:12s} H={H}: {res.price:.4f}, P(no touch) {res.probability_survival:.4f}")

knock_in = barrier.price(S, K, 90, T, r, sigma, "down_and_in").price
knock_out = barrier.price(S, K, 90, T, r, sigma, "down_and_out").price
print(f"in + out - vanilla = {knock_in + knock_out - vanilla:.1e}")
```

```text
down_and_out H=90: 8.6655, P(no touch) 0.4492
down_and_in  H=90: 1.7851, P(no touch) 0.4492
up_and_out   H=120: 1.1761, P(no touch) 0.5873
in + out - vanilla = 0.0e+00
```

`probability_survival` is the risk-neutral probability that the barrier is not touched before
expiry.

### Discrete monitoring

A daily-monitored barrier is touched less often than a continuous one, so the knock-out is
worth more. Broadie, Glasserman and Kou (1997) show that the discrete price is close to the
continuous formula with the barrier moved away from the spot by exp(0.5826 sigma sqrt(dt)).

```python
import numpy as np

mc = barrier.monte_carlo_price(
    S, K, 90, T, r, sigma, "down_and_out", num_sims=1_000_000, num_steps=252
)
shifted = barrier.price(
    S, K, 90 * np.exp(-0.5826 * sigma * np.sqrt(T / 252)), T, r, sigma, "down_and_out"
)
print(f"daily monitoring, Monte Carlo {mc.price:.4f} +- {mc.std_error:.4f}")
print(f"continuous formula at the shifted barrier {shifted.price:.4f}")
```

```text
daily monitoring, Monte Carlo 8.8915 +- 0.0146
continuous formula at the shifted barrier 8.9139
```

## Asian options

`AsianOptions` prices average-price options with n equally spaced fixings at t_i = i T / n
(today's spot is not a fixing).

- Geometric average: closed form, continuous (Kemna and Vorst, 1990) or discrete with n
  fixings. Both match QuantLib's analytic engines to 1.4e-14.
- Arithmetic average: Monte Carlo, by default with the geometric option on the same paths as a
  control variate whose mean is the discrete closed form for the same fixings.

```python
asian = AsianOptions(num_sims=100_000, random_seed=42)
geo = asian.price(S, K, T, r, sigma, "call", asian_type="geometric", num_steps=12)
cv = asian.price(S, K, T, r, sigma, "call", asian_type="arithmetic", num_steps=12)
plain = asian.price(
    S, K, T, r, sigma, "call", asian_type="arithmetic", num_steps=12, use_control_variate=False
)
print(f"geometric, closed form       {geo.price:.4f}")
print(f"arithmetic, control variate  {cv.price:.4f} +- {cv.std_error:.4f}")
print(f"arithmetic, plain            {plain.price:.4f} +- {plain.std_error:.4f}")
```

```text
geometric, closed form       5.9402
arithmetic, control variate  6.1571 +- 0.0008
arithmetic, plain            6.1832 +- 0.0270
```

`sensitivity_analysis` computes finite-difference delta, gamma, vega, rho and theta, reusing
the same random numbers for every bumped revaluation so the Monte Carlo noise largely cancels.

```python
print(asian.sensitivity_analysis(S, K, T, r, sigma, "call", "arithmetic", "spot"))
```

```text
{'delta': 0.589141766162582, 'gamma': 0.03245557391728582}
```
