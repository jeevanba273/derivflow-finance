# Greeks

`GreeksCalculator` computes option risk sensitivities. With the default
`analytical` method it returns first-, second-, and third-order Greeks in a
single `GreeksResult` dataclass. A `numerical` method is also available for
custom or exotic pricers via finite differences. A continuous dividend yield `q`
is supported.

```python
from derivflow import GreeksCalculator, format_greeks_report

calc = GreeksCalculator(method="analytical")
greeks = calc.calculate_greeks(S=100, K=105, T=0.25, r=0.05, sigma=0.2,
                               option_type="call", q=0.0)

print(f"Delta: {greeks.delta:.4f}")
print(f"Gamma: {greeks.gamma:.4f}")
print(f"Theta: {greeks.theta:.4f}")   # per calendar day
print(f"Vega:  {greeks.vega:.4f}")    # per 1% vol move
print(f"Rho:   {greeks.rho:.4f}")     # per 1% rate move

# Human-readable summary
print(format_greeks_report(greeks, S=100, K=105, option_type="call"))
```

**Key result fields.** `GreeksResult` exposes the first-order Greeks `delta`,
`gamma`, `theta`, `vega`, `rho`, the second-order `volga` and `vanna`, and the
third-order `speed`, `zomma`, and `color`. Theta and color are scaled per
calendar day; vega, rho, volga, and vanna are scaled per 1% move.
`format_greeks_report` returns a formatted string suitable for printing.
