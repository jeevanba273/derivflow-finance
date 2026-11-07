"""PricingEngine and the core engines: dispatch, parity checks and errors."""

import pytest

from derivflow_finance import BinomialTree, MonteCarloEngine, PricingEngine
from derivflow_finance.core.pricing_engine import black_scholes_price, price_european_option

ARGS = {"S": 100.0, "K": 105.0, "T": 0.5, "r": 0.04, "sigma": 0.25}


def _seeded_engine(seed=11):
    engine = PricingEngine()
    engine.methods["monte_carlo"] = MonteCarloEngine(random_seed=seed)
    return engine


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_monte_carlo_put_call_parity_is_judged_against_its_standard_errors(seed):
    """A Monte Carlo call and put come from different draws, so the identity
    holds only to Monte Carlo error; the check must allow for it."""
    res = _seeded_engine(seed).validate_put_call_parity(**ARGS, method="monte_carlo", q=0.01)
    assert res["absolute_error"] > 1e-6  # not exact ...
    assert res["parity_satisfied"]  # ... but within the estimates' error
    assert res["tolerance"] > res["absolute_error"]


@pytest.mark.parametrize("method", ["black_scholes", "binomial"])
def test_exact_methods_satisfy_parity_to_rounding(method):
    res = PricingEngine().validate_put_call_parity(**ARGS, method=method, q=0.01)
    assert res["parity_satisfied"] and res["tolerance"] == 1e-6
    assert res["absolute_error"] < 1e-10


def test_compare_methods_prices_every_method_and_reports_failures():
    engine = _seeded_engine()
    res = engine.compare_methods(**ARGS)
    assert set(res) == {"black_scholes", "binomial", "monte_carlo"}
    bs = black_scholes_price(*ARGS.values(), "call")
    assert res["black_scholes"]["price"] == pytest.approx(bs, abs=1e-12)
    assert abs(res["binomial"]["price"] - bs) < 0.02  # 100-step tree
    mc = res["monte_carlo"]
    assert abs(mc["price"] - bs) < 4 * mc["std_error"]
    partial = engine.compare_methods(**ARGS, methods=["black_scholes", "lattice"])
    assert "price" in partial["black_scholes"]
    assert "Unknown pricing method" in partial["lattice"]["error"]


def test_price_option_dispatch_and_caching():
    engine = _seeded_engine()
    with pytest.raises(ValueError, match="Unknown pricing method"):
        engine.price_option("lattice", **ARGS)
    first = engine.price_option("black_scholes", **ARGS)
    assert engine.price_option("black_scholes", **ARGS) == first
    assert any(key.startswith("black_scholes") for key in engine.results_cache)
    # Monte Carlo estimates are not cached: the generator moves on between calls
    a = engine.price_option("monte_carlo", **ARGS)["price"]
    b = engine.price_option("monte_carlo", **ARGS)["price"]
    assert a != b
    assert not any(key.startswith("monte_carlo") for key in engine.results_cache)


@pytest.mark.parametrize("method", ["binomial", "monte_carlo"])
def test_price_european_option_with_other_methods(method):
    price = price_european_option(*ARGS.values(), "put", method=method, q=0.01)
    assert isinstance(price, float)
    assert price == pytest.approx(black_scholes_price(*ARGS.values(), "put", 0.01), abs=0.15)


def test_binomial_tree_input_errors_and_expiry():
    tree = BinomialTree()
    with pytest.raises(ValueError, match="option_type"):
        tree.price(100, 100, 1.0, 0.05, 0.2, "straddle")
    with pytest.raises(ValueError, match="positive"):
        tree.price(-1.0, 100, 0.0, 0.05, 0.2)
    assert tree.price(110, 100, 0.0, 0.05, 0.2, "call") == 10.0
    assert tree.price(110, 100, 0.0, 0.05, 0.2, "put") == 0.0
    with pytest.raises(ValueError, match="risk-neutral probability"):
        BinomialTree(steps=1).price(100, 100, 1.0, 1.0, 0.01)


def test_monte_carlo_and_vectorised_input_errors():
    with pytest.raises(ValueError, match="num_sims"):
        MonteCarloEngine(num_sims=1)
    with pytest.raises(ValueError, match="option_type"):
        MonteCarloEngine(num_sims=10).price(100, 100, 1.0, 0.05, 0.2, "straddle")
    with pytest.raises(ValueError, match="option_type"):
        black_scholes_price(100, 100, 1.0, 0.05, 0.2, "straddle")
