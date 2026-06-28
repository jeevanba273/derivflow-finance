"""Monte Carlo engine: statistical accuracy and the vectorization regression lock."""

import numpy as np
import pytest

from derivflow.core.pricing_engine import MonteCarloEngine


@pytest.mark.slow
def test_mc_within_standard_errors(atm_params, bs_ref, tol):
    eng = MonteCarloEngine(num_sims=200000, random_seed=tol['mc_seed'])
    res = eng.price(option_type='call', **atm_params)
    truth = bs_ref(option_type='call', **atm_params)
    assert abs(res['price'] - truth) < tol['mc_sigma'] * res['std_error']
    lo, hi = res['confidence_interval']
    assert lo < truth < hi


def test_path_shape_and_initial_value():
    eng = MonteCarloEngine(num_sims=1000, random_seed=1)
    paths = eng.generate_gbm_paths(100, 1.0, 0.05, 0.2, num_steps=50)
    assert paths.shape == (1000, 51)
    assert np.allclose(paths[:, 0], 100.0)


def test_vectorization_matches_reference_loop():
    """Lock the cumsum vectorization to the original per-step recurrence."""
    num_sims, num_steps, seed = 2000, 60, 7
    eng = MonteCarloEngine(num_sims=num_sims, random_seed=seed)

    np.random.seed(seed)
    new_paths = eng.generate_gbm_paths(100, 1.0, 0.05, 0.2, num_steps=num_steps)

    # Reference: identical shocks, original explicit loop.
    np.random.seed(seed)
    dt = 1.0 / num_steps
    Z = np.random.standard_normal((num_sims, num_steps))
    ref = np.zeros((num_sims, num_steps + 1))
    ref[:, 0] = 100
    for t in range(1, num_steps + 1):
        ref[:, t] = ref[:, t - 1] * np.exp(
            (0.05 - 0.5 * 0.2 ** 2) * dt + 0.2 * np.sqrt(dt) * Z[:, t - 1])

    assert np.allclose(new_paths, ref, atol=1e-10)
