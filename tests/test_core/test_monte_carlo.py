"""Monte Carlo engine: confidence intervals, variance reduction, reproducibility."""

import numpy as np
import pytest

from derivflow.core.pricing_engine import MonteCarloEngine

CASES = [
    (100.0, 100.0, 1.0, 0.05, 0.2, "call", 0.0),
    (100.0, 120.0, 0.5, 0.03, 0.35, "call", 0.02),
    (100.0, 90.0, 2.0, 0.01, 0.25, "put", 0.0),
]


@pytest.mark.parametrize("method", ["plain", "antithetic", "control_variate"])
@pytest.mark.parametrize(("S", "K", "T", "r", "sigma", "ot", "q"), CASES)
def test_confidence_interval_contains_black_scholes(bs_ref, method, S, K, T, r, sigma, ot, q):
    eng = MonteCarloEngine(num_sims=200_000, random_seed=2024)
    res = eng.price(
        S, K, T, r, sigma, ot, q=q,
        antithetic=method == "antithetic", control_variate=method == "control_variate",
    )
    truth = bs_ref(S, K, T, r, sigma, ot, q)
    lo, hi = res["confidence_interval"]
    # A 95% interval misses about 1 time in 20; allow 3 standard errors so the
    # fixed-seed test is not fragile, and report the interval for inspection.
    assert abs(res["price"] - truth) < 3 * res["std_error"], (lo, hi, truth)


def test_variance_reduction_lowers_the_standard_error():
    args = (100.0, 100.0, 1.0, 0.05, 0.2, "call")
    plain = MonteCarloEngine(100_000, random_seed=1).price(*args)
    anti = MonteCarloEngine(100_000, random_seed=1).price(*args, antithetic=True)
    cv = MonteCarloEngine(100_000, random_seed=1).price(*args, control_variate=True)
    assert anti["std_error"] < 0.8 * plain["std_error"]
    assert cv["std_error"] < 0.5 * plain["std_error"]
    assert anti["num_simulations"] == plain["num_simulations"] == cv["num_simulations"]


def test_standard_error_shrinks_like_one_over_sqrt_n():
    args = (100.0, 100.0, 1.0, 0.05, 0.2, "call")
    small = MonteCarloEngine(10_000, random_seed=3).price(*args)["std_error"]
    large = MonteCarloEngine(1_000_000, random_seed=3).price(*args)["std_error"]
    assert large / small == pytest.approx(0.1, rel=0.05)


def test_seed_is_reproducible_and_does_not_touch_global_state():
    before = np.random.get_state()[1].copy()
    a = MonteCarloEngine(5_000, random_seed=11).price(100, 100, 1.0, 0.05, 0.2)
    b = MonteCarloEngine(5_000, random_seed=11).price(100, 100, 1.0, 0.05, 0.2)
    assert a["price"] == b["price"]
    assert np.array_equal(np.random.get_state()[1], before)


def test_path_shape_and_initial_value():
    eng = MonteCarloEngine(num_sims=1000, random_seed=1)
    paths = eng.generate_gbm_paths(100, 1.0, 0.05, 0.2, num_steps=50)
    assert paths.shape == (1000, 51)
    assert np.allclose(paths[:, 0], 100.0)


def test_paths_match_the_exact_gbm_recurrence():
    """Vectorised paths equal the step-by-step exact GBM recurrence for the same shocks."""
    num_sims, num_steps, seed = 2000, 60, 7
    new_paths = MonteCarloEngine(num_sims, random_seed=seed).generate_gbm_paths(
        100, 1.0, 0.05, 0.2, num_steps=num_steps
    )
    Z = np.random.default_rng(seed).standard_normal((num_sims, num_steps))
    dt = 1.0 / num_steps
    ref = np.zeros((num_sims, num_steps + 1))
    ref[:, 0] = 100
    for t in range(1, num_steps + 1):
        ref[:, t] = ref[:, t - 1] * np.exp(
            (0.05 - 0.5 * 0.2**2) * dt + 0.2 * np.sqrt(dt) * Z[:, t - 1]
        )
    assert np.allclose(new_paths, ref, atol=1e-10)
