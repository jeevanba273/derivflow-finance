# DERIVFLOW-FINANCE Notebooks

Runnable tutorial notebooks. Install the package and a Jupyter kernel first:

```bash
pip install -e ".[notebooks]"
jupyter notebook
```

Suggested order:

1. `01_quickstart_pricing_and_greeks.ipynb` - pricing, Greeks, implied volatility, dividends
2. `02_volatility_surface_and_implied_vol.ipynb` - surface construction and IV smile recovery
3. `03_exotic_options_barrier_and_asian.ipynb` - barrier (with in/out parity) and Asian options
4. `04_heston_stochastic_volatility.ipynb` - Heston model Monte Carlo pricing
5. `05_portfolio_risk_and_VaR.ipynb` - portfolio Greeks and Value-at-Risk
6. `06_market_data_and_dashboards.ipynb` - live market data (with offline fallback) and dashboards

Notebook 06 fetches live data via `yfinance`; it falls back to sample values when offline.
The dashboard cells build Plotly figures and need no network.
