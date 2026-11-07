"""PricingEngine and the core engines: dispatch, parity checks and errors."""

import pytest

from derivflow_finance import MonteCarloEngine, PricingEngine

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
