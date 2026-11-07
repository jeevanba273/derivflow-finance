# Volatility surfaces

`VolatilitySurface` turns (strike, expiry, implied volatility) quotes into a function of
strike and expiry. Expiries are in years and volatilities are annualised.

| method | what it does |
|---|---|
| `"rbf"` (default) | thin-plate radial basis function on (ln K, T), passes through every quote |
| `"linear"` | piecewise-linear on a Delaunay triangulation of (ln K, T); nearest quote outside the hull |
| `"svi"` | raw SVI (Gatheral, 2004) per expiry, total variance linear in T between expiries |

None of the methods forces the surface to be free of arbitrage.

## SVI

Raw SVI writes the total implied variance w = sigma_imp^2 T of one expiry as

$$w(k) = a + b\left(\rho (k - m) + \sqrt{(k - m)^2 + \sigma^2}\right), \qquad k = \ln(K / K_{\text{ref}}).$$

Each expiry with at least five quotes is fitted by bounded least squares; `reference_price`
sets K_ref (otherwise the median strike of the expiry is used).

```python
import numpy as np

from derivflow_finance import VolatilitySurface

surface = VolatilitySurface(method="svi", reference_price=100.0)
for T, atm in ((0.25, 0.22), (1.0, 0.20)):
    for K in np.linspace(70, 130, 13):
        k = np.log(K / 100.0)
        surface.add_volatility_point(K, T, atm - 0.10 * k + 0.25 * k * k)
surface.build_surface()

for K in (80.0, 100.0, 120.0):
    print(
        f"K={K:5.1f}: T=0.25 {surface.interpolate(K, 0.25):.4f}, T=0.5 {surface.interpolate(K, 0.5):.4f}"
    )
```

```text
K= 80.0: T=0.25 0.2549, T=0.5 0.2418
K=100.0: T=0.25 0.2199, T=0.5 0.2068
K=120.0: T=0.25 0.2103, T=0.5 0.1972
```

## Checking a smile for arbitrage

`check_arbitrage` prices calls from a smile and tests the static no-arbitrage conditions
across strikes: call prices non-increasing with slopes no steeper than -e^{-rT} (vertical
spreads) and convex in strike (butterflies). Calendar arbitrage is not checked.

```python
strikes = np.linspace(70, 130, 25)
smile = np.array([surface.interpolate(K, 0.5) for K in strikes])
print(surface.check_arbitrage(strikes, smile, expiry=0.5, spot=100.0, r=0.03)["arbitrage_free"])

smile[12] += 0.08  # a volatility spike at K = 100
result = surface.check_arbitrage(strikes, smile, expiry=0.5, spot=100.0, r=0.03)
print(result["arbitrage_free"], result["butterfly_arbitrage"])
```

```text
True
False True
```

## Interpolating surfaces

```python
from derivflow_finance import create_sample_surface

sample = create_sample_surface()  # 36 illustrative quotes with noise, fixed seed
sample.build_surface(method="rbf")
print(round(sample.interpolate(100.0, 0.25), 4), round(sample.interpolate(102.5, 0.4), 4))
```

```text
0.2363 0.2465
```

`get_smile` samples one expiry over a strike range and `plot_surface` draws the surface with
matplotlib (extra `[plot]`).
