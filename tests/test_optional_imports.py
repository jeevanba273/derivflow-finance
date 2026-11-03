"""The core package must import with only NumPy and SciPy loaded."""

import subprocess
import sys
import textwrap

OPTIONAL = ("yfinance", "pandas", "plotly", "matplotlib")


def test_core_import_does_not_load_optional_dependencies():
    code = textwrap.dedent(
        f"""
        import sys
        import derivflow
        from derivflow import (
            price_european_option, GreeksCalculator, BarrierOptions, AsianOptions,
            HestonModel, PortfolioRiskAnalyzer, VolatilitySurface, implied_volatility,
        )
        loaded = [m for m in {OPTIONAL!r} if m in sys.modules]
        print("LOADED=" + ",".join(loaded))
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "LOADED=\n" in result.stdout or result.stdout.strip() == "LOADED=", result.stdout


def test_optional_names_resolve_lazily():
    import derivflow

    assert "AdvancedMarketData" in dir(derivflow)
    assert derivflow.AdvancedMarketData.__name__ == "AdvancedMarketData"
