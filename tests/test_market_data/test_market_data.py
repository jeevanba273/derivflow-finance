"""Market data client. Offline tests use a stub yfinance module; tests that hit
Yahoo Finance are marked 'network' and are skipped unless --run-network is given."""

import sys
import types
from datetime import datetime

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("zoneinfo")

from derivflow._dates import new_york, year_fraction_to_expiry  # noqa: E402
from derivflow.core.pricing_engine import black_scholes_price  # noqa: E402
from derivflow.utils.market_data import AdvancedMarketData  # noqa: E402

try:
    new_york()
except Exception:  # pragma: no cover - Windows without tzdata
    pytest.skip("IANA time zone data not available", allow_module_level=True)

SPOT, RATE, SIGMA = 100.0, 0.04, 0.30
NOW = datetime(2026, 3, 2, 12, 0)  # a Monday, New York time
EXPIRY = "2026-04-17"


def _stub_yfinance(closes=None, irx=4.2):
    """A minimal stand-in for yfinance with deterministic data."""
    T = year_fraction_to_expiry(EXPIRY, NOW)
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    call_prices = [black_scholes_price(SPOT, k, T, RATE, SIGMA, "call") for k in strikes]
    calls = pd.DataFrame(
        {
            "strike": strikes + [60.0],
            # last row: priced below intrinsic value, so no volatility solves it
            "bid": [p - 0.01 for p in call_prices] + [30.0],
            "ask": [p + 0.01 for p in call_prices] + [30.0],
            "lastPrice": call_prices + [30.0],
            "volume": [10] * 6,
            "openInterest": [100] * 6,
            "impliedVolatility": [np.nan] * 6,  # vendor IV missing
        }
    )
    puts = calls.iloc[0:0]
    closes = closes if closes is not None else [SPOT] * 70

    class Ticker:
        def __init__(self, symbol):
            self.symbol = symbol
            self.options = (EXPIRY,)

        def history(self, period=None, interval=None):
            if self.symbol == "^IRX":
                return pd.DataFrame({"Close": [irx]}, index=pd.to_datetime(["2026-03-02"]))
            idx = pd.date_range("2025-12-01", periods=len(closes), freq="D")
            return pd.DataFrame({"Close": closes}, index=idx)

        def option_chain(self, expiry):
            return types.SimpleNamespace(calls=calls, puts=puts)

    return types.SimpleNamespace(Ticker=Ticker)


@pytest.fixture
def stub(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", _stub_yfinance())
    return AdvancedMarketData(rate_limit_delay=0.0)


def test_surface_points_are_solved_from_prices_not_defaulted(stub):
    result = stub.build_volatility_surface_from_market("TEST", r=RATE, now=NOW)
    vols = [p["volatility"] for p in result["surface_data"]]
    assert len(vols) == 5 and result["dropped"] == 1
    # Mid prices are the model prices, so the solved volatility is the input one.
    np.testing.assert_allclose(vols, SIGMA, atol=1e-9)
    assert all(p["vendor_implied_volatility"] is None for p in result["surface_data"])


def test_short_rate_proxy_is_the_13_week_bill(stub):
    assert stub.get_risk_free_rate() == pytest.approx(0.042)


def test_historical_volatility_uses_252_days(monkeypatch):
    rng = np.random.default_rng(0)
    log_ret = rng.normal(0.0, 0.01, 69)
    closes = list(100.0 * np.exp(np.concatenate([[0.0], np.cumsum(log_ret)])))
    monkeypatch.setitem(sys.modules, "yfinance", _stub_yfinance(closes=closes))
    vol = AdvancedMarketData().get_historical_volatility("TEST", days=30)
    assert vol == pytest.approx(log_ret[-30:].std(ddof=1) * np.sqrt(252), rel=1e-12)


@pytest.mark.parametrize(
    ("when", "expected"),
    [
        (datetime(2026, 3, 7, 12, 0), (False, "Weekend")),
        (datetime(2026, 3, 4, 9, 0), (False, "Pre-market")),
        (datetime(2026, 3, 4, 10, 0), (True, "Open")),
        (datetime(2026, 3, 4, 17, 0), (False, "After-hours")),
    ],
)
def test_is_market_open(when, expected):
    assert AdvancedMarketData().is_market_open(when.replace(tzinfo=new_york())) == expected


def test_year_fraction_is_act_365_to_the_close():
    assert year_fraction_to_expiry("2026-01-02", datetime(2026, 1, 1, 16, 0)) == pytest.approx(
        1 / 365
    )
    assert year_fraction_to_expiry("2026-01-01", datetime(2026, 1, 2, 9, 0)) == 0.0


@pytest.mark.network
def test_get_current_price_live():
    price, _ = AdvancedMarketData().get_current_price("SPY")
    assert price > 0


@pytest.mark.network
def test_historical_volatility_live():
    assert AdvancedMarketData().get_historical_volatility("SPY", days=30) > 0


def test_module_imports_and_constructs_without_yfinance():
    """yfinance is imported only when data is actually fetched."""
    import subprocess
    import textwrap

    code = textwrap.dedent(
        """
        import sys
        class _Block:
            def find_spec(self, name, path=None, target=None):
                if name == "yfinance" or name.startswith("yfinance."):
                    raise ImportError("simulated missing yfinance")
                return None
        sys.meta_path.insert(0, _Block())
        from derivflow.utils.market_data import AdvancedMarketData
        AdvancedMarketData()
        print("IMPORT_OK")
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "IMPORT_OK" in result.stdout
