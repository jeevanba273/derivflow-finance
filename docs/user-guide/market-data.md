# Market Data

`AdvancedMarketData` pulls live market data through
[`yfinance`](https://pypi.org/project/yfinance/): spot prices, full option
chains, historical volatility, and a risk-free-rate proxy. It also assesses data
quality and detects whether the US market is currently open.

```{note}
This module makes **live network calls** to Yahoo Finance. It requires an
active internet connection, and results depend on data availability, market
hours, and upstream rate limits. It is therefore not part of the offline core
and is only importable when `yfinance` is installed.
```

```python
from derivflow import AdvancedMarketData

md = AdvancedMarketData()

# Current/last spot price
price, timestamp = md.get_current_price("AAPL")
print(f"AAPL: {price:.2f} at {timestamp}")

# Full option chain for the nearest expiry
chain = md.get_options_chain("AAPL")
print(f"{chain.total_options} options, quality: {chain.data_quality}")

# 30-day historical volatility
hv = md.get_historical_volatility("AAPL", days=30)
print(f"Historical vol: {hv:.2%}")
```

The returned `MarketDataResult` can be fed straight into the implied-volatility
solver via `implied_vol_from_chain`, or its `surface_data` (from
`build_volatility_surface_from_market`) into `VolatilitySurface`.

**Key result fields.** `get_current_price` returns a `(price, timestamp)` tuple.
`get_options_chain` returns a `MarketDataResult` with `symbol`, `spot_price`,
`options_data` (a list of `OptionData`), `total_options`, `data_quality`, and
`market_status`. Each `OptionData` carries `strike`, `expiry`, `option_type`,
`last_price`, `bid`, `ask`, `volume`, `open_interest`, and `implied_volatility`.
