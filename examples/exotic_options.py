"""Exotic options: barrier and Asian pricing.

Run with:  python examples/exotic_options.py
"""

from derivflow import BarrierOptions, AsianOptions, price_european_option

S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

vanilla = price_european_option(S, K, T, r, sigma, 'call')
print(f"Vanilla call: {vanilla:.4f}\n")

print("Barrier options (call):")
barrier = BarrierOptions()
for bt in ('down_and_out', 'down_and_in', 'up_and_out', 'up_and_in'):
    H = 90 if bt.startswith('down') else 110
    res = barrier.price(S, K, H, T, r, sigma, bt, 'call')
    print(f"  {bt:13s} H={H}: price {res.price:.4f} | survival {res.probability_survival:.2%}")

# in + out reconstructs the vanilla
di = barrier.price(S, K, 90, T, r, sigma, 'down_and_in', 'call').price
do = barrier.price(S, K, 90, T, r, sigma, 'down_and_out', 'call').price
print(f"  down_in + down_out = {di + do:.4f} (vanilla {vanilla:.4f})\n")

print("Asian options (call, averaging reduces effective volatility):")
asian = AsianOptions(num_sims=50000, random_seed=42)
geo = asian.price(S, K, T, r, sigma, 'call', asian_type='geometric')
arith = asian.price(S, K, T, r, sigma, 'call', asian_type='arithmetic')
print(f"  geometric (analytical):  {geo.price:.4f}")
print(f"  arithmetic (MC + control variate): {arith.price:.4f}")
if arith.convergence_info:
    vrr = arith.convergence_info.get('variance_reduction_ratio')
    if vrr:
        print(f"  variance reduction ratio: {vrr:.0f}x")
