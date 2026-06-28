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
