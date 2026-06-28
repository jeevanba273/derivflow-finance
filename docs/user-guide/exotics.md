# Exotic Options

DERIVFLOW prices two families of path-dependent exotics: barrier options and
Asian (average) options.

## Barrier options

`BarrierOptions` supports the four single-barrier European variants
(`up_and_out`, `up_and_in`, `down_and_out`, `down_and_in`) for calls and puts.
The `analytical` method uses Reiner-Rubinstein closed forms; `monte_carlo`
simulates GBM paths and checks the barrier along each path.

```python
from derivflow import BarrierOptions

barrier = BarrierOptions(random_seed=42)
result = barrier.price(S=100, K=105, H=95, T=0.25, r=0.05, sigma=0.3,
                       barrier_type="down_and_out", option_type="call",
                       method="analytical")

print(f"Price:                {result.price:.4f}")
print(f"Survival probability: {result.probability_survival:.2%}")
```

**Key result fields.** `BarrierOptionResult` carries `price` and
`probability_survival` (the chance the barrier is not breached). The Greek
fields are optional and may be `None`.

## Asian options

`AsianOptions` prices geometric averages analytically and arithmetic averages by
Monte Carlo (using the geometric Asian as a control variate for variance
reduction). With `method="auto"` the engine routes geometric to analytical and
arithmetic to Monte Carlo automatically. The `price_asian_option` and
`compare_asian_types` helpers wrap common workflows.

```python
from derivflow import AsianOptions, price_asian_option, compare_asian_types

asian = AsianOptions(num_sims=50000, random_seed=42)
geo = asian.price(S=100, K=100, T=0.25, r=0.05, sigma=0.3,
                  option_type="call", asian_type="geometric")
print(f"Geometric Asian call: {geo.price:.4f}")

# Quick helper returns just the float price
arith = price_asian_option(S=100, K=100, T=0.25, r=0.05, sigma=0.3,
                           option_type="call", asian_type="arithmetic")
print(f"Arithmetic Asian call: {arith:.4f}")

print(compare_asian_types(S=100, K=100, T=0.25, r=0.05, sigma=0.3, option_type="call"))
```

**Key result fields.** `AsianOptionResult` carries `price` and, for Monte Carlo
results, `std_error`, `confidence_interval`, `num_simulations`, and a
`convergence_info` dict (which reports whether the control variate was used and
the achieved variance-reduction ratio). `compare_asian_types` returns a dict
with `arithmetic_asian`, `geometric_asian`, `price_difference`, and
`relative_difference`.
