"""Batched path simulation: bounded batches and results identical to one batch."""

from itertools import pairwise

import numpy as np
import pytest

import derivflow_finance._batching as batching
from derivflow_finance import AsianOptions, BarrierOptions


@pytest.mark.parametrize(("num_paths", "num_steps"), [(1, 1), (10, 252), (100_001, 12), (7, 10**7)])
def test_batches_cover_every_path_once_within_the_budget(num_paths, num_steps):
    bounds = list(batching.path_batches(num_paths, num_steps))
    assert bounds[0][0] == 0 and bounds[-1][1] == num_paths
    assert all(a[1] == b[0] for a, b in pairwise(bounds))
    sizes = [stop - start for start, stop in bounds]
    assert all(1 <= size <= max(1, batching.PATH_BUDGET // num_steps) for size in sizes)


def _barrier():
    return BarrierOptions(random_seed=11).monte_carlo_price(
        100.0, 100.0, 115.0, 0.5, 0.05, 0.25, "up_and_out", "put", num_sims=5_003, num_steps=50
    )


def _asian(use_cv):
    return AsianOptions(num_sims=5_003, random_seed=21).price_arithmetic_asian_mc(
        100.0, 95.0, 1.0, 0.05, 0.2, "call", 52, use_cv
    )


@pytest.mark.parametrize("budget", [1, 97, 1_000])
def test_small_batches_give_identical_results(monkeypatch, budget):
    whole = (_barrier(), _asian(True), _asian(False))
    monkeypatch.setattr(batching, "PATH_BUDGET", budget)
    batched = (_barrier(), _asian(True), _asian(False))
    for a, b in zip(whole, batched, strict=True):
        assert a.price == b.price and a.std_error == b.std_error
    assert whole[0].probability_survival == batched[0].probability_survival


def test_the_generator_continues_across_calls_as_before(monkeypatch):
    """Two calls on one engine draw consecutive random numbers, batched or not."""
    engine = AsianOptions(num_sims=3_001, random_seed=5)
    first = [engine.price(100, 100, 1.0, 0.05, 0.2).price for _ in range(2)]
    monkeypatch.setattr(batching, "PATH_BUDGET", 50)
    engine = AsianOptions(num_sims=3_001, random_seed=5)
    second = [engine.price(100, 100, 1.0, 0.05, 0.2).price for _ in range(2)]
    assert first == second and first[0] != first[1]
    assert np.isfinite(first).all()
