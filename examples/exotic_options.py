"""Barrier and Asian options.

Run with:  python examples/exotic_options.py
"""

import numpy as np

from derivflow import AsianOptions, BarrierOptions, black_scholes_price

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
vanilla = black_scholes_price(S, K, T, r, sigma, "call")
print(f"vanilla call {vanilla:.4f}")

barrier = BarrierOptions(random_seed=7)
print("\nContinuously monitored barrier calls (closed form):")
for bt, H in [("down_and_out", 90), ("down_and_in", 90), ("up_and_out", 120), ("up_and_in", 120)]:
    res = barrier.price(S, K, H, T, r, sigma, bt, "call")
    print(f"  {bt:13s} H={H}: {res.price:.4f}  P(no touch) {res.probability_survival:.4f}")
out = barrier.price(S, K, 90, T, r, sigma, "down_and_out").price
knock_in = barrier.price(S, K, 90, T, r, sigma, "down_and_in").price
print(f"  in + out = {knock_in + out:.10f} (vanilla {vanilla:.10f})")

# Daily monitoring is a discrete barrier; Broadie-Glasserman-Kou: shift H down by
# exp(-0.5826 sigma sqrt(dt)) to approximate it with the continuous formula.
mc = barrier.monte_carlo_price(
    S, K, 90, T, r, sigma, "down_and_out", num_sims=1_000_000, num_steps=252
)
H_shift = 90 * np.exp(-0.5826 * sigma * np.sqrt(T / 252))
bgk = barrier.price(S, K, H_shift, T, r, sigma, "down_and_out").price
print(
    f"\ndown-and-out, daily monitoring: Monte Carlo {mc.price:.4f} +- {mc.std_error:.4f}, "
    f"shifted-barrier closed form {bgk:.4f}"
)

asian = AsianOptions(num_sims=100_000, random_seed=42)
print("\nAsian calls with 12 monthly fixings:")
geo = asian.price(S, K, T, r, sigma, "call", asian_type="geometric", num_steps=12)
cv = asian.price(S, K, T, r, sigma, "call", asian_type="arithmetic", num_steps=12)
plain = asian.price(
    S, K, T, r, sigma, "call", asian_type="arithmetic", num_steps=12, use_control_variate=False
)
print(f"  geometric, closed form       {geo.price:.4f}")
print(f"  arithmetic, control variate  {cv.price:.4f} +- {cv.std_error:.4f}")
print(f"  arithmetic, plain MC         {plain.price:.4f} +- {plain.std_error:.4f}")
ratio = cv.convergence_info["variance_reduction_ratio"]
print(f"  variance reduction ratio of the control variate: {ratio:.0f}")
