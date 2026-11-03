"""Seeded Monte Carlo classes are reproducible and never touch NumPy's global RNG."""

import numpy as np

from derivflow import AsianOptions, BarrierOptions, create_sample_surface


def _global_state():
    return np.random.get_state()[1].copy()  # noqa: NPY002 - checking the global state


def test_barrier_monte_carlo():
    before = _global_state()
    a = BarrierOptions(random_seed=5).monte_carlo_price(
        100, 100, 90, 1.0, 0.05, 0.2, "down_and_out", num_sims=5_000, num_steps=50
    )
    b = BarrierOptions(random_seed=5).monte_carlo_price(
        100, 100, 90, 1.0, 0.05, 0.2, "down_and_out", num_sims=5_000, num_steps=50
    )
    assert a.price == b.price
    assert a.std_error > 0
    assert np.array_equal(_global_state(), before)


def test_asian_monte_carlo():
    before = _global_state()
    a = AsianOptions(num_sims=5_000, random_seed=5).price(100, 100, 1.0, 0.05, 0.2)
    b = AsianOptions(num_sims=5_000, random_seed=5).price(100, 100, 1.0, 0.05, 0.2)
    assert a.price == b.price
    assert np.array_equal(_global_state(), before)


def test_sample_surface():
    before = _global_state()
    s1 = [p.volatility for p in create_sample_surface().vol_data]
    s2 = [p.volatility for p in create_sample_surface().vol_data]
    assert s1 == s2
    assert np.array_equal(_global_state(), before)


def test_asian_sensitivities_use_common_random_numbers():
    """Bumped revaluations reuse the same draws, so the finite-difference gamma
    is stable across seeds instead of being dominated by Monte Carlo noise."""
    gammas = [
        AsianOptions(num_sims=50_000, random_seed=seed).sensitivity_analysis(
            100, 100, 1.0, 0.05, 0.2, "call", "arithmetic", "spot"
        )["gamma"]
        for seed in (1, 2, 3)
    ]
    assert all(g > 0 for g in gammas)
    assert max(gammas) - min(gammas) < 1e-3
