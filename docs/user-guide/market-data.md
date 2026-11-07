# Market data

`derivflow_finance.utils.market_data.AdvancedMarketData` fetches data from Yahoo Finance through
`yfinance`. It is an optional extra and the core package never imports it:

```bash
pip install "derivflow-finance[data]"
```

The examples on this page need internet access and are not run by the test suite; the module
itself is tested offline against a stub of `yfinance`.

```python
# not run in the tests: needs internet access
from derivflow_finance.utils.market_data import AdvancedMarketData

md = AdvancedMarketData()
spot, when = md.get_current_price("SPY")
rate = md.get_risk_free_rate()  # 13-week T-bill yield (^IRX) / 100
vol_30d = md.get_historical_volatility("SPY", days=30)
surface_points = md.build_volatility_surface_from_market("SPY", max_expiries=3, r=rate)
print(spot, rate, vol_30d, len(surface_points["surface_data"]), surface_points["dropped"])
```

What the methods do:

- `get_options_chain` returns the calls and puts of one expiry as `OptionData` rows.
- `build_volatility_surface_from_market` solves each quote's implied volatility from the
  bid/ask mid (last trade when there is no two-sided quote) with this package's solver, drops
  quotes without a solution and counts them in `"dropped"`. Yahoo's own figure is kept as
  `vendor_implied_volatility`. Time to expiry is ACT/365 to 16:00 New York time.
- `get_historical_volatility` is the sample standard deviation (ddof = 1) of daily log returns
  times sqrt(252).
- `get_risk_free_rate` uses the 13-week Treasury bill yield as a continuously compounded short
  rate, which is an approximation; it raises when the data are unavailable unless a `default`
  is given.
- `is_market_open` knows weekends and regular hours (09:30 to 16:00 New York), not exchange
  holidays.

Listed US equity options are American, while the implied volatilities here come from the
European Black-Scholes-Merton formula with a flat dividend yield. They are therefore
approximate where early exercise matters: deep in-the-money puts and options on stocks paying
dividends. Out-of-the-money quotes are the safer input for a surface.

Yahoo data are delayed and can be stale or missing, especially outside trading hours. Check
quotes before relying on numbers derived from them.
