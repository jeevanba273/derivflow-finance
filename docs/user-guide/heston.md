# Heston Model

`HestonModel` implements the Heston stochastic-volatility model, where variance
follows its own mean-reverting square-root process correlated with the asset.
Parameters are set with `set_parameters` (which warns when the Feller condition
$2\kappa\theta > \sigma^2$ is violated) and stored in a `HestonParameters`
dataclass. Options are priced by Monte Carlo (Euler-Maruyama) or by Fourier
inversion of the characteristic function, and parameters can be calibrated to
market quotes.

```python
from derivflow import HestonModel

heston = HestonModel()
heston.set_parameters(
    v0=0.04,     # initial variance
    kappa=2.0,   # mean-reversion speed
    theta=0.04,  # long-run variance
    sigma=0.3,   # vol of vol
    rho=-0.7,    # leverage correlation
)

mc = heston.price_option(S=100, K=105, T=0.25, r=0.05,
                         option_type="call", method="monte_carlo")
print(f"Heston Monte Carlo price: {mc.price:.4f}")

fourier = heston.price_option(S=100, K=105, T=0.25, r=0.05,
                              option_type="call", method="fourier")
print(f"Heston Fourier price:     {fourier.price:.4f}")
```

To fit the model, pass a list of market observations (each a dict with `K`, `T`,
`price`, and `option_type`) to `calibrate_to_market(market_data, S, r)`.

**Key result fields.** `price_option` returns a `HestonResult` whose primary
field is `price`; the Greek fields (`delta`, `gamma`, `vega`, `theta`, `volga`,
`vanna`) are optional. `calibrate_to_market` returns a dict with a `success`
flag, the fitted `parameters`, and the objective value.
