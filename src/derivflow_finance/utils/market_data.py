"""
Market data from Yahoo Finance (optional; install with ``pip install derivflow-finance[data]``).

This module is never imported by the core package. It wraps ``yfinance`` to
fetch spot prices, option chains, historical volatility and a short-rate
proxy, and to turn an option chain into (strike, expiry, implied volatility)
points for :class:`derivflow_finance.volatility.surface.VolatilitySurface`.

Conventions and limitations:

- Time to expiry is ACT/365 to 16:00 New York time on the expiry date
  (:func:`derivflow_finance._dates.year_fraction_to_expiry`).
- Implied volatilities are solved with :func:`derivflow_finance.implied_volatility`
  from the bid/ask mid (last trade if there is no two-sided quote), the given
  rate and dividend yield. Quotes with no solution are dropped. Yahoo's own
  implied volatility is kept separately as ``vendor_implied_volatility``.
- Listed US equity options are American; implied volatilities from the
  European formula are approximate where early exercise matters (deep
  in-the-money puts, dividend-paying stocks).
- The risk-free rate proxy is the 13-week US Treasury bill yield (^IRX) / 100,
  used as a continuously compounded rate; this is an approximation.
- Yahoo data are delayed and can be stale or missing, especially outside
  trading hours. :meth:`AdvancedMarketData.is_market_open` knows weekends and
  regular hours but not exchange holidays.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .._dates import new_york, year_fraction_to_expiry
from ..volatility.implied_vol import implied_volatility

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


def _yf() -> Any:
    """Import yfinance lazily so the module imports without it."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - exercised only without yfinance
        raise ImportError(
            "yfinance is required for live market data: pip install derivflow-finance[data]"
        ) from exc
    return yf


@dataclass
class OptionData:
    """Single option quote."""

    symbol: str
    strike: float
    expiry: str
    option_type: str  # 'call' or 'put'
    last_price: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    implied_volatility: float | None = None  # as reported by Yahoo


@dataclass
class MarketDataResult:
    """Option chain for one expiry."""

    symbol: str
    spot_price: float
    options_data: list[OptionData]
    timestamp: datetime
    total_options: int
    data_quality: str
    market_status: str


def _mid_or_last(option: OptionData) -> float:
    if option.bid > 0 and option.ask > 0:
        return 0.5 * (option.bid + option.ask)
    return option.last_price


class AdvancedMarketData:
    """Thin client for Yahoo Finance data used by the analytics."""

    def __init__(self, cache_duration: int = 300, rate_limit_delay: float = 0.5):
        """
        Parameters:
        -----------
        cache_duration : int
            Seconds a fetched price is reused (kept for compatibility).
        rate_limit_delay : float
            Seconds to wait after each option-chain request.
        """
        self.cache_duration = cache_duration
        self.cache: dict[str, Any] = {}
        self.rate_limit_delay = rate_limit_delay

    @staticmethod
    def _safe_float(val: Any, default: float = 0.0) -> float:
        try:
            if val is None or val == "" or pd.isna(val):
                return default
            return float(val)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _safe_int(val: Any, default: int = 0) -> int:
        try:
            if val is None or val == "" or pd.isna(val):
                return default
            return int(float(val))
        except (ValueError, TypeError):
            return default

    def is_market_open(self, now: datetime | None = None) -> tuple[bool, str]:
        """Whether US equity markets are in regular hours (09:30-16:00 New York,
        Monday to Friday). Exchange holidays are not known to this check."""
        tz = new_york()
        now = datetime.now(tz) if now is None else now.astimezone(tz)
        opening = now.replace(hour=9, minute=30, second=0, microsecond=0)
        closing = now.replace(hour=16, minute=0, second=0, microsecond=0)
        if now.weekday() >= 5:
            return False, "Weekend"
        if now < opening:
            return False, "Pre-market"
        if now > closing:
            return False, "After-hours"
        return True, "Open"

    def get_risk_free_rate(self, default: float | None = None) -> float:
        """13-week Treasury bill yield (^IRX) as a decimal.

        Raises ``ValueError`` if it cannot be fetched, unless ``default`` is
        given, in which case ``default`` is returned.
        """
        try:
            hist = _yf().Ticker("^IRX").history(period="5d")
            if not hist.empty:
                return self._safe_float(hist["Close"].iloc[-1]) / 100.0
            raise ValueError("no ^IRX data")
        except Exception as exc:
            if default is not None:
                logger.warning("Using default risk-free rate %s: %s", default, exc)
                return float(default)
            raise ValueError(f"Could not fetch the ^IRX short rate: {exc}") from exc

    def get_current_price(self, symbol: str) -> tuple[float, datetime]:
        """Last traded price (5-minute bars over two days, else daily) and its time."""
        try:
            ticker = _yf().Ticker(symbol)
            hist = ticker.history(period="2d", interval="5m")
            if hist.empty:
                hist = ticker.history(period="5d")
            if hist.empty:
                raise ValueError(f"No price data available for {symbol}")
            return self._safe_float(hist["Close"].iloc[-1]), hist.index[-1].to_pydatetime()
        except Exception as e:
            raise ValueError(f"Error fetching price for {symbol}: {e}") from e

    def _parse_chain(
        self, symbol: str, expiry: str, frame: Any, option_type: str
    ) -> list[OptionData]:
        quotes: list[OptionData] = []
        if frame is None or frame.empty:
            return quotes
        for row in frame.to_dict("records"):
            strike = self._safe_float(row.get("strike"))
            if strike <= 0:
                continue
            vendor_iv = row.get("impliedVolatility")
            quotes.append(
                OptionData(
                    symbol=symbol,
                    strike=strike,
                    expiry=expiry,
                    option_type=option_type,
                    last_price=self._safe_float(row.get("lastPrice")),
                    bid=self._safe_float(row.get("bid")),
                    ask=self._safe_float(row.get("ask")),
                    volume=self._safe_int(row.get("volume")),
                    open_interest=self._safe_int(row.get("openInterest")),
                    implied_volatility=None if pd.isna(vendor_iv) else float(vendor_iv),
                )
            )
        return quotes

    def get_options_chain(self, symbol: str, expiry_date: str | None = None) -> MarketDataResult:
        """Calls and puts for one expiry (the nearest listed one if ``expiry_date``
        is None or not listed)."""
        _, market_status = self.is_market_open()
        try:
            ticker = _yf().Ticker(symbol)
            spot_price, timestamp = self.get_current_price(symbol)
            expiry_dates = list(ticker.options)
            if not expiry_dates:
                raise ValueError(f"No options data available for {symbol}")
            if expiry_date is None:
                target = expiry_dates[0]
            elif expiry_date in expiry_dates:
                target = expiry_date
            else:
                wanted = datetime.strptime(expiry_date, "%Y-%m-%d")
                target = min(
                    expiry_dates,
                    key=lambda x: abs((datetime.strptime(x, "%Y-%m-%d") - wanted).days),
                )
            chain = ticker.option_chain(target)
            options = self._parse_chain(symbol, target, chain.calls, "call")
            options += self._parse_chain(symbol, target, chain.puts, "put")

            if options:
                priced = sum(opt.last_price > 0 for opt in options) / len(options)
                with_iv = sum(bool(opt.implied_volatility) for opt in options) / len(options)
                score = 0.5 * (priced + with_iv)
                if score > 0.7:
                    quality = "Excellent"
                elif score > 0.5:
                    quality = "Good"
                elif score > 0.3:
                    quality = "Fair"
                else:
                    quality = "Limited"
            else:
                quality = "No Data"
            time.sleep(self.rate_limit_delay)
            return MarketDataResult(
                symbol=symbol,
                spot_price=spot_price,
                options_data=options,
                timestamp=timestamp,
                total_options=len(options),
                data_quality=quality,
                market_status=market_status,
            )
        except Exception as e:
            raise ValueError(f"Error fetching options chain for {symbol}: {e}") from e

    def build_volatility_surface_from_market(
        self,
        symbol: str,
        max_expiries: int = 4,
        r: float | None = None,
        q: float = 0.0,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Implied volatility points for the nearest ``max_expiries`` expiries.

        For each quote the price is the bid/ask mid (last trade without a
        two-sided quote) and the volatility is solved with
        :func:`derivflow_finance.implied_volatility` at rate ``r`` (default: the ^IRX
        proxy from :meth:`get_risk_free_rate`) and dividend yield ``q``.
        Quotes without a solution are dropped and counted in ``'dropped'``.

        Returns:
        --------
        dict
            'symbol', 'spot_price', 'risk_free_rate', 'dividend_yield',
            'surface_data' (list of dicts with strike, expiry in years,
            volatility, market_price, option_type, volume, open_interest,
            vendor_implied_volatility), 'dropped', 'timestamp',
            'market_status'.
        """
        try:
            ticker = _yf().Ticker(symbol)
            expiries = list(ticker.options)[:max_expiries]
            spot, _ = self.get_current_price(symbol)
            rate = self.get_risk_free_rate() if r is None else float(r)
            _, market_status = self.is_market_open(now)
        except Exception as e:
            raise ValueError(f"Error building volatility surface: {e}") from e

        points: list[dict[str, Any]] = []
        dropped = 0
        for expiry in expiries:
            try:
                chain = self.get_options_chain(symbol, expiry)
            except ValueError as exc:
                logger.warning("Skipping expiry %s: %s", expiry, exc)
                continue
            T = year_fraction_to_expiry(expiry, now)
            for option in chain.options_data:
                price = _mid_or_last(option)
                vol = (
                    implied_volatility(
                        price, spot, option.strike, T, rate, option.option_type, q, on_fail="nan"
                    )
                    if price > 0 and T > 0
                    else float("nan")
                )
                if not np.isfinite(vol):
                    dropped += 1
                    continue
                points.append(
                    {
                        "strike": option.strike,
                        "expiry": T,
                        "volatility": vol,
                        "market_price": price,
                        "option_type": option.option_type,
                        "volume": option.volume,
                        "open_interest": option.open_interest,
                        "vendor_implied_volatility": option.implied_volatility,
                    }
                )
        logger.info("%s: %d surface points, %d quotes dropped", symbol, len(points), dropped)
        return {
            "symbol": symbol,
            "spot_price": spot,
            "risk_free_rate": rate,
            "dividend_yield": q,
            "surface_data": points,
            "dropped": dropped,
            "timestamp": datetime.now(),
            "market_status": market_status,
        }

    def get_historical_volatility(self, symbol: str, days: int = 30) -> float:
        """Annualised close-to-close volatility: standard deviation (ddof=1) of the
        last ``days`` daily log returns times sqrt(252)."""
        try:
            hist = _yf().Ticker(symbol).history(period=f"{max(days * 2, 60)}d")
            closes = hist["Close"].dropna().to_numpy(dtype=float)
            if closes.size < 6:
                raise ValueError(f"Insufficient historical data for {symbol}")
            closes = closes[-(days + 1) :]
            returns = np.diff(np.log(closes))
            return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        except Exception as e:
            raise ValueError(f"Error calculating historical volatility: {e}") from e

    def market_summary(self, symbol: str) -> dict[str, Any]:
        """Spot, 30-day historical volatility, short rate and nearest-expiry chain
        statistics. Returns {'error': message} if the spot price is unavailable."""
        try:
            _, market_status = self.is_market_open()
            spot_price, timestamp = self.get_current_price(symbol)
            summary: dict[str, Any] = {
                "symbol": symbol,
                "spot_price": spot_price,
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "market_status": market_status,
                "historical_volatility_30d": self.get_historical_volatility(symbol),
                "risk_free_rate": self.get_risk_free_rate(default=float("nan")),
            }
        except Exception as e:
            return {"error": str(e)}
        try:
            chain = self.get_options_chain(symbol)
            vendor_ivs = [o.implied_volatility for o in chain.options_data if o.implied_volatility]
            summary.update(
                {
                    "options_available": len(chain.options_data),
                    "data_quality": chain.data_quality,
                    "average_vendor_implied_volatility": (
                        float(np.mean(vendor_ivs)) if vendor_ivs else None
                    ),
                    "total_volume": sum(o.volume for o in chain.options_data),
                    "total_open_interest": sum(o.open_interest for o in chain.options_data),
                    "expiry_date": chain.options_data[0].expiry if chain.options_data else None,
                }
            )
        except Exception as exc:
            summary.update(
                {"options_available": 0, "data_quality": "Unavailable", "note": str(exc)}
            )
        return summary
