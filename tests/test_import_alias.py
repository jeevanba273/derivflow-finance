"""The deprecated ``derivflow`` import name (0.4 series) resolves to ``derivflow_finance``.

Each check runs in a fresh interpreter so that the module cache of the test
session does not hide what a user's first import does.
"""

import importlib.util
import subprocess
import sys
import textwrap

import pytest


def _run(code: str) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return result


def test_new_name_imports_without_any_warning():
    _run(
        """
        import sys, warnings
        warnings.simplefilter("error")
        import derivflow_finance
        from derivflow_finance import price_european_option
        assert "derivflow" not in sys.modules
        """
    )


def test_old_name_warns_once_at_the_import_and_is_the_same_package():
    _run(
        """
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import derivflow
            import derivflow
            from derivflow import price_european_option
        import derivflow_finance
        future = [w for w in caught if issubclass(w.category, FutureWarning)]
        assert len(future) == 1, [str(w.message) for w in caught]
        assert "derivflow_finance" in str(future[0].message)
        assert "0.5.0" in str(future[0].message)
        assert future[0].filename == "<string>", future[0].filename
        assert derivflow is derivflow_finance
        assert derivflow.__version__ == derivflow_finance.__version__
        assert price_european_option is derivflow_finance.price_european_option
        """
    )


def test_old_submodule_imports_return_the_new_module_objects():
    _run(
        """
        import sys, warnings
        warnings.simplefilter("ignore", FutureWarning)
        from derivflow.core.pricing_engine import black_scholes_price
        import derivflow.volatility.implied_vol as iv_old
        from derivflow.portfolio import risk_measures
        from derivflow._dates import year_fraction_to_expiry
        import derivflow_finance.core.pricing_engine as pe
        import derivflow_finance.volatility.implied_vol as iv_new
        import derivflow_finance._dates as dates
        assert black_scholes_price is pe.black_scholes_price
        assert iv_old is iv_new
        assert risk_measures is sys.modules["derivflow_finance.portfolio.risk_measures"]
        assert year_fraction_to_expiry is dates.year_fraction_to_expiry
        aliases = [n for n in sys.modules if n == "derivflow" or n.startswith("derivflow.")]
        assert len(aliases) > 10, aliases
        for name in aliases:
            new = "derivflow_finance" + name[len("derivflow"):]
            module = sys.modules[name]
            assert sys.modules[new] is module, name
            assert module.__name__ == new, (name, module.__name__)
            assert module.__spec__.name == new, (name, module.__spec__.name)
        """
    )


@pytest.mark.skipif(importlib.util.find_spec("pandas") is None, reason="needs pandas")
def test_old_name_for_a_lazily_loaded_optional_module():
    _run(
        """
        import sys, warnings
        warnings.simplefilter("ignore", FutureWarning)
        from derivflow.utils.market_data import AdvancedMarketData
        import derivflow_finance.utils.market_data as md
        assert AdvancedMarketData is md.AdvancedMarketData
        assert sys.modules["derivflow.utils.market_data"] is md
        assert md.__spec__.name == "derivflow_finance.utils.market_data", md.__spec__.name
        """
    )


def test_unknown_old_submodule_raises_module_not_found():
    _run(
        """
        import warnings
        warnings.simplefilter("ignore", FutureWarning)
        try:
            import derivflow.no_such_module
        except ModuleNotFoundError:
            pass
        else:
            raise AssertionError("expected ModuleNotFoundError")
        """
    )


def test_pickles_that_reference_the_old_name_still_load():
    _run(
        """
        import pickle, warnings
        warnings.simplefilter("ignore", FutureWarning)
        import derivflow_finance
        old = b"cderivflow.core.pricing_engine\\nblack_scholes_price\\n."
        assert pickle.loads(old) is derivflow_finance.core.pricing_engine.black_scholes_price
        """
    )
