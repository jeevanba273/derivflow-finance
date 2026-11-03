# Plots

`derivflow.visualization.DerivativesDashboard` builds Plotly figures (extra `[plot]`). Each
`plot_*` method returns a `plotly.graph_objects.Figure`; call `.show()` to display it or
`.write_html(path)` to save it. Prices and Greeks come from the Black-Scholes-Merton functions
of this package.

```python
from derivflow.visualization import DerivativesDashboard

dash = DerivativesDashboard(theme="plotly_white")
figures = {
    "payoff": dash.plot_option_payoff(100.0, 5.0, "call"),
    "spot sensitivity": dash.plot_price_sensitivity(100.0, 100.0, 0.5, 0.03, 0.2, "call", "spot"),
    "greeks": dash.plot_greeks_dashboard(100.0, 100.0, 0.5, 0.03, 0.2, "call"),
    "pnl": dash.plot_pnl_analysis(100.0, 100.0, 0.5, 0.03, 0.2, "call"),
    "sample surface": dash.plot_volatility_surface(),
}
for name, fig in figures.items():
    print(name, type(fig).__name__, len(fig.data), "traces")
```

```text
payoff Figure 2 traces
spot sensitivity Figure 2 traces
greeks Figure 6 traces
pnl Figure 4 traces
sample surface Figure 1 traces
```

`plot_volatility_surface(symbol=...)` and `create_comprehensive_dashboard(symbol)` fetch live
data and need the `[data]` extra and internet access; without a symbol the surface plot uses
`create_sample_surface()`.
