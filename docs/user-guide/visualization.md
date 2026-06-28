# Visualization

`DerivativesDashboard` builds interactive [Plotly](https://plotly.com/python/)
charts for derivatives analytics: option payoff diagrams, price-sensitivity
curves, a multi-panel Greeks dashboard, 3D volatility surfaces, and P&L
analysis. The `quick_payoff_plot` and `quick_greeks_plot` helpers wrap the most
common single-chart workflows.

```{note}
This module requires `plotly` and is only importable when that optional
dependency is installed. Every method returns a Plotly `go.Figure`; call
`.show()` to render it in a browser or notebook.
```

```python
from derivflow import DerivativesDashboard, quick_payoff_plot, quick_greeks_plot

# One-liners
payoff = quick_payoff_plot(strike=105, premium=3.2, option_type="call")
greeks = quick_greeks_plot(S=100, K=105, T=0.25, r=0.05, sigma=0.2,
                           option_type="call")

# Full dashboard object
dash = DerivativesDashboard(theme="plotly_dark")
fig = dash.plot_price_sensitivity(S=100, K=105, T=0.25, r=0.05, sigma=0.2,
                                  option_type="call", param="spot")
fig.show()
```

**Key result.** Every plotting method returns a Plotly `go.Figure`. The
`create_comprehensive_dashboard(symbol)` method (which uses live market data)
returns a dict mapping chart names to figures.
