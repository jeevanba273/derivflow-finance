"""Vectorised Black-Scholes-Merton pricer: agrees with the scalar pricer."""

import numpy as np
import pytest

from derivflow_finance.core.pricing_engine import BlackScholesAnalytical, black_scholes_price


def test_matches_scalar_pricer_elementwise():
    rng = np.random.default_rng(0)
    n = 2000
    S = rng.uniform(50, 150, n)
    K = rng.uniform(50, 150, n)
    T = rng.uniform(1 / 365, 3, n)
    r = rng.uniform(-0.01, 0.08, n)
    q = rng.uniform(0.0, 0.05, n)
    sigma = rng.uniform(0.05, 1.0, n)
    bs = BlackScholesAnalytical()
    for option_type in ("call", "put"):
        vec = black_scholes_price(S, K, T, r, sigma, option_type, q)
        ref = np.array(
            [
                bs.price(*args, option_type, qi)
                for *args, qi in zip(S, K, T, r, sigma, q, strict=True)
            ]
        )
        np.testing.assert_allclose(vec, ref, rtol=1e-12, atol=1e-12)


def test_put_call_parity_vectorised():
    K = np.linspace(50, 150, 101)
    c = black_scholes_price(100.0, K, 0.75, 0.03, 0.25, "call", 0.01)
    p = black_scholes_price(100.0, K, 0.75, 0.03, 0.25, "put", 0.01)
    np.testing.assert_allclose(
        c - p, 100 * np.exp(-0.01 * 0.75) - K * np.exp(-0.03 * 0.75), atol=1e-10
    )


def test_broadcasting_and_scalar_input():
    out = black_scholes_price(100.0, np.array([[90.0], [110.0]]), np.array([0.5, 1.0]), 0.05, 0.2)
    assert out.shape == (2, 2)
    assert isinstance(black_scholes_price(42, 40, 0.5, 0.1, 0.2), float)
    assert black_scholes_price(42, 40, 0.5, 0.1, 0.2) == pytest.approx(4.7594, abs=1e-4)


def test_expired_options_return_intrinsic_value():
    out = black_scholes_price(np.array([90.0, 110.0]), 100.0, 0.0, 0.05, 0.2, "call")
    np.testing.assert_allclose(out, [0.0, 10.0])


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [({"sigma": -0.2}, "sigma"), ({"S": 0.0}, "S"), ({"K": np.array([100.0, -1.0])}, "K")],
)
def test_invalid_inputs_raise_like_the_scalar_pricer(kwargs, name):
    args = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2} | kwargs
    with pytest.raises(ValueError, match=name):
        black_scholes_price(**args)


def test_zero_volatility_gives_discounted_forward_intrinsic():
    assert black_scholes_price(100.0, 90.0, 1.0, 0.05, 0.0) == pytest.approx(
        100.0 - 90.0 * np.exp(-0.05)
    )
