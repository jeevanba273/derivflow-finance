"""
DERIVFLOW-FINANCE: Performance benchmarks
=========================================

Measures real, machine-local performance for the headline operations and
substantiates the README's claims. Run with:

    python benchmarks/run_benchmarks.py
    python benchmarks/run_benchmarks.py --json

All timings use a fixed seed and report the median of several repetitions.
The Monte Carlo section also reports the before/after speedup of the vectorized
exact-GBM path generator versus the original Python time-step loop, asserting
the two produce identical paths (hardware-independent evidence of the win).
"""

import argparse
import json
import time
from statistics import median

import numpy as np

from derivflow.core.pricing_engine import (
    BlackScholesAnalytical, MonteCarloEngine, price_european_option,
)
from derivflow.exotic.asian_options import AsianOptions
from derivflow._numba_utils import HAS_NUMBA


def _median_time(func, reps=5):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        func()
        times.append(time.perf_counter() - t0)
    return median(times)


def _old_loop_paths(S0, T, r, sigma, num_sims, num_steps, Z):
    """Original per-step Python loop (reference for the speedup comparison)."""
    dt = T / num_steps
    paths = np.zeros((num_sims, num_steps + 1))
    paths[:, 0] = S0
    for t in range(1, num_steps + 1):
        paths[:, t] = paths[:, t - 1] * np.exp(
            (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1])
    return paths


def _new_cumsum_paths(S0, T, r, sigma, num_sims, num_steps, Z):
    """Vectorized exact-GBM (same scheme as MonteCarloEngine.generate_gbm_paths)."""
    dt = T / num_steps
    increments = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.zeros((num_sims, num_steps + 1))
    np.cumsum(increments, axis=1, out=log_paths[:, 1:])
    return S0 * np.exp(log_paths)


def benchmark_black_scholes():
    bs = BlackScholesAnalytical()
    n = 50000
    rng = np.random.default_rng(0)
    strikes = rng.uniform(80, 120, n)

    def run():
        for k in strikes:
            bs.price(100, k, 0.5, 0.05, 0.2, 'call')

    elapsed = _median_time(run, reps=3)
    return {'n_options': n, 'seconds': elapsed, 'options_per_sec': n / elapsed}


def benchmark_monte_carlo():
    num_sims, num_steps = 10000, 252
    np.random.seed(42)
    eng = MonteCarloEngine(num_sims=num_sims, random_seed=42)

    def run():
        np.random.seed(42)
        eng.generate_gbm_paths(100, 1.0, 0.05, 0.2, num_steps=num_steps)

    latency = _median_time(run, reps=5)

    # Before/after speedup with identical inputs (and identical outputs).
    np.random.seed(42)
    Z = np.random.standard_normal((num_sims, num_steps))
    old_t = _median_time(lambda: _old_loop_paths(100, 1.0, 0.05, 0.2, num_sims, num_steps, Z), reps=5)
    new_t = _median_time(lambda: _new_cumsum_paths(100, 1.0, 0.05, 0.2, num_sims, num_steps, Z), reps=5)
    identical = bool(np.allclose(
        _old_loop_paths(100, 1.0, 0.05, 0.2, num_sims, num_steps, Z),
        _new_cumsum_paths(100, 1.0, 0.05, 0.2, num_sims, num_steps, Z),
        atol=1e-10))
    assert identical, "Vectorized GBM paths diverge from the reference loop"
    return {
        'num_sims': num_sims, 'num_steps': num_steps,
        'paths_10k_seconds': latency,
        'old_loop_seconds': old_t, 'new_cumsum_seconds': new_t,
        'speedup_x': old_t / new_t, 'outputs_identical': identical,
    }


def benchmark_asian_control_variate():
    asian = AsianOptions(num_sims=50000, random_seed=42)
    res = asian.price_arithmetic_asian_mc(100, 100, 1.0, 0.05, 0.2, 'call',
                                          num_steps=50, use_control_variate=True)
    info = res.convergence_info or {}
    return {
        'price': res.price,
        'std_error': res.std_error,
        'variance_reduction_ratio': info.get('variance_reduction_ratio'),
    }


def main():
    parser = argparse.ArgumentParser(description="DERIVFLOW-FINANCE benchmarks")
    parser.add_argument('--json', action='store_true', help="emit results as JSON")
    args = parser.parse_args()

    results = {
        'numba_available': HAS_NUMBA,
        'black_scholes': benchmark_black_scholes(),
        'monte_carlo': benchmark_monte_carlo(),
        'asian_control_variate': benchmark_asian_control_variate(),
    }

    if args.json:
        print(json.dumps(results, indent=2))
        return

    bs = results['black_scholes']
    mc = results['monte_carlo']
    asn = results['asian_control_variate']
    print("DERIVFLOW-FINANCE Benchmarks")
    print("=" * 50)
    print("numba available: %s" % HAS_NUMBA)
    print("\nBlack-Scholes pricing:")
    print("  {:d} options in {:.3f}s -> {:,.0f} options/sec".format(
        bs['n_options'], bs['seconds'], bs['options_per_sec']))
    print("\nMonte Carlo (10k paths x 252 steps):")
    print("  latency: %.4fs" % mc['paths_10k_seconds'])
    print("  vectorization speedup: %.1fx (old %.4fs -> new %.4fs), outputs identical: %s"
          % (mc['speedup_x'], mc['old_loop_seconds'], mc['new_cumsum_seconds'], mc['outputs_identical']))
    print("\nAsian arithmetic (control variate):")
    print("  price: %.4f  std_error: %.5f  variance_reduction_ratio: %s"
          % (asn['price'], asn['std_error'], asn['variance_reduction_ratio']))


if __name__ == "__main__":
    main()
