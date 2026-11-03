"""Cox-Ross-Rubinstein tree: textbook value, convergence and exercise invariants."""

import itertools

import numpy as np
import pytest

from derivflow.core.pricing_engine import BinomialTree


def test_hull_five_step_american_put():
    # Hull, Options, Futures, and Other Derivatives: American put with S=K=50,
    # r=10%, sigma=40%, T=5 months on a 5-step tree is worth 4.49.
    price = BinomialTree(5).price(50, 50, 5 / 12, 0.10, 0.40, "put", american=True)
    assert round(price, 2) == 4.49


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("q", [0.0, 0.03])
def test_european_converges_to_black_scholes(bs_ref, option_type, q):
    args = (100.0, 105.0, 1.0, 0.05, 0.25)
    truth = bs_ref(*args, option_type, q)
    errors = [
        abs(BinomialTree(n).price(*args, option_type, q=q) - truth) for n in (50, 200, 800, 3200)
    ]
    # CRR error is O(1/n) with an oscillating constant: check the bound and the trend.
    for n, err in zip((50, 200, 800, 3200), errors, strict=True):
        assert err < 3.0 / n  # measured n * error ranges from 0.9 to 2.3 here
    assert errors[-1] < errors[0] / 10
    assert errors[-1] < 1e-3


CASES = list(itertools.product([80.0, 100.0, 120.0], [0.25, 1.0, 3.0], [0.15, 0.4], [0.0, 0.08]))


@pytest.mark.parametrize(("S", "T", "sigma", "r"), CASES)
def test_american_put_at_least_european_and_intrinsic(S, T, sigma, r):
    tree = BinomialTree(400)
    am = tree.price(S, 100.0, T, r, sigma, "put", american=True)
    eu = tree.price(S, 100.0, T, r, sigma, "put", american=False)
    assert am >= eu - 1e-12
    assert am >= max(100.0 - S, 0.0) - 1e-12


@pytest.mark.parametrize(("S", "T", "sigma", "r"), CASES)
def test_american_call_equals_european_without_dividends(S, T, sigma, r):
    tree = BinomialTree(400)
    am = tree.price(S, 100.0, T, r, sigma, "call", american=True)
    eu = tree.price(S, 100.0, T, r, sigma, "call", american=False)
    assert am == pytest.approx(eu, abs=1e-12)


def test_dividends_make_early_exercise_of_calls_valuable():
    tree = BinomialTree(500)
    am = tree.price(100, 90, 1.0, 0.03, 0.2, "call", american=True, q=0.08)
    eu = tree.price(100, 90, 1.0, 0.03, 0.2, "call", american=False, q=0.08)
    assert am > eu + 0.1


def test_expired_option_is_intrinsic_value():
    assert BinomialTree(10).price(110, 100, 0.0, 0.05, 0.2, "call") == 10.0
    assert BinomialTree(10).price(110, 100, 0.0, 0.05, 0.2, "put") == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [{"sigma": 0.0}, {"sigma": -0.2}, {"S": -1.0}, {"K": 0.0}, {"T": np.nan}],
)
def test_invalid_inputs_raise(kwargs):
    args = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2} | kwargs
    with pytest.raises(ValueError):
        BinomialTree(50).price(**args)


def test_invalid_steps_raise():
    with pytest.raises(ValueError):
        BinomialTree(0)
