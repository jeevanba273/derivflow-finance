"""Barrier Monte Carlo for all eight barrier types.

- Against an independent simulation written here (exact GBM on the same
  monitoring dates, a separate random generator): the two estimates must agree
  within their combined standard error. This checks the up-barrier and put
  branches of the simulation.
- Against the continuous closed form with the Broadie-Glasserman-Kou (1997)
  shift H exp(+-0.5826 sigma sqrt(dt)) for regular barriers (payoff zero at the
  barrier), where that correction is accurate; it is less accurate for reverse
  barriers (up-and-out calls, down-and-out puts with the strike inside), so they
  are only checked against the simulation.
"""

import warnings

import numpy as np
import pytest

from derivflow_finance import BarrierOptions

S, T, R, SIGMA, STEPS, PATHS = 100.0, 0.5, 0.05, 0.25, 50, 100_000
TYPES = ["down_and_out", "down_and_in", "up_and_out", "up_and_in"]


def _independent_mc(K, H, barrier_type, option_type, seed):
    rng = np.random.default_rng(seed)
    dt = T / STEPS
    log_s = np.log(S) + np.cumsum(
        (R - SIGMA**2 / 2) * dt + SIGMA * np.sqrt(dt) * rng.standard_normal((PATHS, STEPS)),
        axis=1,
    )
    paths = np.exp(log_s)
    if barrier_type.startswith("up"):
        hit = (paths >= H).any(axis=1)
    else:
        hit = (paths <= H).any(axis=1)
    final = paths[:, -1]
    payoff = np.maximum(final - K, 0) if option_type == "call" else np.maximum(K - final, 0)
    payoff = payoff * (hit if barrier_type.endswith("in") else ~hit)
    discounted = np.exp(-R * T) * payoff
    return discounted.mean(), discounted.std(ddof=1) / np.sqrt(PATHS)


@pytest.mark.parametrize("barrier_type", TYPES)
@pytest.mark.parametrize(
    ("option_type", "K"), [("call", 100.0), ("put", 100.0), ("call", 85.0), ("put", 120.0)]
)
def test_monte_carlo_matches_an_independent_simulation(barrier_type, option_type, K):
    H = 115.0 if barrier_type.startswith("up") else 90.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # strike on the far side of the barrier
        ours = BarrierOptions(random_seed=3).monte_carlo_price(
            S, K, H, T, R, SIGMA, barrier_type, option_type, num_sims=PATHS, num_steps=STEPS
        )
    ref, ref_se = _independent_mc(K, H, barrier_type, option_type, seed=97)
    assert abs(ours.price - ref) < 4 * np.hypot(ours.std_error, ref_se)


@pytest.mark.parametrize(
    ("barrier_type", "option_type", "H"),
    [
        ("down_and_out", "call", 90.0),
        ("down_and_in", "call", 90.0),
        ("up_and_out", "put", 115.0),
        ("up_and_in", "put", 115.0),
    ],
)
def test_monte_carlo_matches_the_continuity_corrected_closed_form(barrier_type, option_type, H):
    shift = 0.5826 * SIGMA * np.sqrt(T / STEPS)
    shifted = H * np.exp(shift if barrier_type.startswith("up") else -shift)
    ref = (
        BarrierOptions()
        .analytical_price(S, 100.0, shifted, T, R, SIGMA, barrier_type, option_type)
        .price
    )
    mc = BarrierOptions(random_seed=5).monte_carlo_price(
        S, 100.0, H, T, R, SIGMA, barrier_type, option_type, num_sims=PATHS, num_steps=STEPS
    )
    assert abs(mc.price - ref) < 4 * mc.std_error + 0.01 * ref


def test_monte_carlo_and_dispatch_errors():
    barrier = BarrierOptions(random_seed=1)
    with pytest.raises(ValueError, match="T must be positive"):
        barrier.monte_carlo_price(S, 100, 90, 0.0, R, SIGMA, "down_and_out")
    with pytest.raises(ValueError, match="Unknown method"):
        barrier.price(S, 100, 90, 1.0, R, SIGMA, "down_and_out", method="pde")
    mc = barrier.price(S, 100, 90, 1.0, R, SIGMA, "down_and_out", method="monte_carlo")
    assert mc.std_error > 0
