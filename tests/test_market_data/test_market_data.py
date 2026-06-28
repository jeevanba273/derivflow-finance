"""Market data: constructor initialization (the fixed __init__) and live calls."""

import pytest

from derivflow.utils.market_data import AdvancedMarketData


def test_constructor_initializes_state():
    # Regression: the constructor was previously named `_init_` and never ran,
    # leaving these attributes unset (AttributeError on first use).
    md = AdvancedMarketData()
    assert md.cache == {}
    assert md.cache_duration == 300
    assert md.rate_limit_delay == 0.5


def test_is_market_open_returns_status():
    md = AdvancedMarketData()
    is_open, status = md.is_market_open()
    assert is_open in (True, False)
    assert isinstance(status, str) and status


@pytest.mark.integration
def test_get_current_price_live():
    md = AdvancedMarketData()
    try:
        price, _ = md.get_current_price('AAPL')
    except Exception:
        pytest.skip("network/market data unavailable")
    assert price > 0


@pytest.mark.integration
def test_historical_volatility_live():
    md = AdvancedMarketData()
    try:
        vol = md.get_historical_volatility('AAPL', days=30)
    except Exception:
        pytest.skip("network/market data unavailable")
    assert vol > 0


def test_imports_without_yfinance():
    """Regression: derivflow must import on a minimal install lacking yfinance
    (yfinance is loaded lazily only when live data is actually fetched)."""
    import subprocess
    import sys
    import textwrap
    code = textwrap.dedent(
        '''
        import sys
        class _Block:
            def find_spec(self, name, path=None, target=None):
                if name == "yfinance" or name.startswith("yfinance."):
                    raise ImportError("simulated-missing: yfinance")
                return None
        sys.meta_path.insert(0, _Block())
        for m in [m for m in sys.modules if m == "yfinance" or m.startswith("yfinance.")]:
            del sys.modules[m]
        import derivflow
        from derivflow.portfolio import PortfolioRiskAnalyzer
        from derivflow.utils.market_data import AdvancedMarketData
        AdvancedMarketData()  # constructor must not require yfinance
        print("IMPORT_OK")
        '''
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "IMPORT_OK" in result.stdout
