# Greeks

`GreeksCalculator` returns a `GreeksResult` with ten sensitivities of a European option under
Black-Scholes-Merton. `method="analytical"` (default) uses closed forms; `method="numerical"`
bumps the inputs of a pricing function, which you can replace with your own.

## Units

| Greek | definition | reported per |
|---|---|---|
| delta | dV/dS | 1 unit of spot |
| gamma | d2V/dS2 | 1 unit of spot |
| theta | dV/dt, calendar time (= -dV/dT) | calendar day |
| vega | dV/dsigma | 1 volatility point (0.01) |
| rho | dV/dr | 1 rate point (0.01) |
| volga | d2V/dsigma2 | (1 volatility point)^2 |
| vanna | d2V/dS dsigma | 1 volatility point |
| speed | d3V/dS3 | 1 unit of spot |
| zomma | d3V/dS2 dsigma | 1 volatility point |
| color | d3V/dS2 dt | calendar day |

QuantLib reports vega and rho per unit and theta per year: multiply by 100, 100 and 365 to
compare. The test suite checks every analytical Greek against central finite differences of
an independent price function, and the first-order Greeks against QuantLib.

```python
from derivflow import GreeksCalculator, format_greeks_report

g = GreeksCalculator().calculate_greeks(100.0, 105.0, 0.5, 0.05, 0.25, "call", q=0.02)
print(format_greeks_report(g, 100.0, 105.0, "call"))
```

```text
CALL option, spot 100.00, strike 105.00, OTM
  delta      0.454510  per unit of spot
  gamma      0.022225  per unit of spot
  theta     -0.022008  per calendar day
  vega       0.277817  per vol point
  rho        0.199652  per rate point
  volga   0.000319204  per vol point squared
  vanna    0.00439308  per vol point
  speed  -9.30609e-05  per unit of spot
  zomma  -0.000863479  per vol point
  color   5.92984e-05  per calendar day
```

## Haug's worked examples

Haug (2007) quotes theta per year and rho per unit rate.

```python
calc = GreeksCalculator()
print(round(calc.analytical_greeks(105, 100, 0.5, 0.10, 0.36, "call", 0.10).delta, 4))
print(round(calc.analytical_greeks(55, 60, 0.75, 0.10, 0.30).gamma, 4))
print(round(calc.analytical_greeks(430, 405, 1 / 12, 0.07, 0.20, "put", 0.05).theta * 365, 4))
print(round(calc.analytical_greeks(72, 75, 1.0, 0.09, 0.19).rho * 100, 4))
```

```text
0.5946
0.0278
-31.1924
38.7325
```

## Finite differences

The numerical method uses central differences for delta, gamma, vega and rho and a one-day
forward difference for theta, in the same units as the closed forms.

```python
num = GreeksCalculator("numerical").calculate_greeks(100.0, 105.0, 0.5, 0.05, 0.25, "call")
ana = GreeksCalculator().calculate_greeks(100.0, 105.0, 0.5, 0.05, 0.25, "call")
for name in ("delta", "gamma", "theta", "vega", "rho"):
    print(f"{name:5s} analytical {getattr(ana, name):.6f}  numerical {getattr(num, name):.6f}")
```

```text
delta analytical 0.481580  numerical 0.481552
gamma analytical 0.022544  numerical 0.022538
theta analytical -0.025078  numerical -0.025101
vega  analytical 0.281794  numerical 0.281794
rho   analytical 0.210847  numerical 0.210845
```

`greeks_ladder` evaluates delta, gamma, theta, vega and rho over a range of spots.
