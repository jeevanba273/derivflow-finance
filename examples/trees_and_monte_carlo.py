"""Binomial tree (European and American) and Monte Carlo with variance reduction.

Run with:  python examples/trees_and_monte_carlo.py
"""

from derivflow import BinomialTree, MonteCarloEngine, black_scholes_price

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
bs = black_scholes_price(S, K, T, r, sigma, "put")
print(f"Black-Scholes European put: {bs:.6f}")

print("\nCRR tree, European put error and American put:")
for steps in (50, 200, 800, 3200):
    tree = BinomialTree(steps)
    eu = tree.price(S, K, T, r, sigma, "put")
    am = tree.price(S, K, T, r, sigma, "put", american=True)
    print(f"  {steps:5d} steps  European error {eu - bs:+.6f}  American {am:.6f}")

print("\nMonte Carlo, European call, 100,000 draws, seed 1:")
truth = black_scholes_price(S, K, T, r, sigma, "call")
for label, kwargs in [
    ("plain", {}),
    ("antithetic", {"antithetic": True}),
    ("control variate", {"control_variate": True}),
]:
    res = MonteCarloEngine(100_000, random_seed=1).price(S, K, T, r, sigma, "call", **kwargs)
    lo, hi = res["confidence_interval"]
    print(
        f"  {label:16s} {res['price']:.4f}  s.e. {res['std_error']:.4f}  "
        f"95% CI [{lo:.4f}, {hi:.4f}]  contains {truth:.4f}: {lo < truth < hi}"
    )
