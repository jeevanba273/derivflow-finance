"""
derivflow: option pricing and risk calculations in Python.

The core package depends only on NumPy and SciPy. Live market data
(``derivflow.utils.market_data``, extra ``[data]``) and Plotly dashboards
(``derivflow.visualization``, extra ``[plot]``) are optional and are only
imported when one of their names is first accessed.

Example
-------
>>> from derivflow import price_european_option
>>> round(price_european_option(S=42, K=40, T=0.5, r=0.10, sigma=0.20, option_type="call"), 2)
4.76
"""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Any

try:
    __version__ = _pkg_version("derivflow-finance")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0+unknown"

__author__ = "Jeevan B A"
__license__ = "MIT"

from .core.pricing_engine import (
    BinomialTree,
    BlackScholesAnalytical,
    MonteCarloEngine,
    PricingEngine,
    black_scholes_price,
    price_european_option,
)
from .exotic.asian_options import (
    AsianOptionResult,
    AsianOptions,
    AsianType,
    AverageType,
    compare_asian_types,
    price_asian_option,
)
from .exotic.barrier_options import BarrierOptionResult, BarrierOptions, BarrierType
from .greeks.calculator import GreeksCalculator, GreeksResult, format_greeks_report
from .models.heston import HestonModel, HestonParameters, HestonResult
from .portfolio.portfolio_risk import (
    PortfolioRiskAnalyzer,
    Position,
    RiskMetrics,
    ScenarioResult,
    create_sample_portfolio,
)
from .volatility.implied_vol import (
    implied_vol_from_chain,
    implied_volatility,
    implied_volatility_vectorized,
)
from .volatility.surface import (
    VolatilityPoint,
    VolatilitySurface,
    VolatilitySurfaceResult,
    create_sample_surface,
)

# Optional names resolved lazily (PEP 562) so that ``import derivflow`` never
# imports yfinance, pandas or plotly.
_LAZY_EXPORTS = {
    "AdvancedMarketData": "derivflow.utils.market_data",
    "OptionData": "derivflow.utils.market_data",
    "MarketDataResult": "derivflow.utils.market_data",
    "DerivativesDashboard": "derivflow.visualization.dashboard",
    "quick_payoff_plot": "derivflow.visualization.dashboard",
    "quick_greeks_plot": "derivflow.visualization.dashboard",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'derivflow' has no attribute {name!r}")
    module = importlib.import_module(module_name)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


def get_package_info() -> dict[str, str]:
    """Return the package name, installed version, author, licence and project URLs."""
    return {
        "name": "derivflow-finance",
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "source": "https://github.com/jeevanba273/derivflow-finance",
        "documentation": "https://derivflow-finance.readthedocs.io",
    }


__all__ = [
    "AsianOptionResult",
    "AsianOptions",
    "AsianType",
    "AverageType",
    "BarrierOptionResult",
    "BarrierOptions",
    "BarrierType",
    "BinomialTree",
    "BlackScholesAnalytical",
    "GreeksCalculator",
    "GreeksResult",
    "HestonModel",
    "HestonParameters",
    "HestonResult",
    "MonteCarloEngine",
    "PortfolioRiskAnalyzer",
    "Position",
    "PricingEngine",
    "RiskMetrics",
    "ScenarioResult",
    "VolatilityPoint",
    "VolatilitySurface",
    "VolatilitySurfaceResult",
    "__version__",
    "black_scholes_price",
    "compare_asian_types",
    "create_sample_portfolio",
    "create_sample_surface",
    "format_greeks_report",
    "get_package_info",
    "implied_vol_from_chain",
    "implied_volatility",
    "implied_volatility_vectorized",
    "price_asian_option",
    "price_european_option",
]
